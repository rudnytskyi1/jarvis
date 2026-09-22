"""Общий профиль между домами (ТЗ F-212).

One hub, several rooms, one person who may live in two of them: the flag
``memberships.share_identity`` decides whether their face and voice are used
outside the room they were registered in. The tests follow the rule through the
module and through a real room whose identity layer is wired to the database.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hub import migrations_runner
from hub.shared_identity import (
    Membership,
    describe,
    filter_visible,
    memberships_of,
    primary_home,
    set_share_identity,
    share_command,
    shared_with,
    visible_in,
    visible_people,
)


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    for home_id, name in (("livingroom", "Living room"), ("kitchen", "Kitchen"),
                          ("garage", "Garage")):
        conn.execute("INSERT INTO homes(home_id, name) VALUES (?, ?)", (home_id, name))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anna', 'Анна')")
    # Макс lives in the living room; Anna is a member of two rooms.
    conn.execute("INSERT INTO memberships(person_id, home_id, role, created_at)"
                 " VALUES ('p-max', 'livingroom', 'admin', '2026-01-01T10:00:00')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role, created_at)"
                 " VALUES ('p-anna', 'kitchen', 'user', '2026-02-01T10:00:00')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role, created_at)"
                 " VALUES ('p-anna', 'garage', 'guest', '2026-03-01T10:00:00')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


# --- the rule ---------------------------------------------------------------


def test_the_memberships_of_a_person_are_read_oldest_first(hub_db):
    rows = memberships_of(hub_db, "p-anna")
    assert [row.home_id for row in rows] == ["kitchen", "garage"]
    assert isinstance(rows[0], Membership) and rows[0].role == "user"
    assert primary_home(hub_db, "p-anna") == "kitchen"
    assert memberships_of(hub_db, "nobody") == [] and primary_home(hub_db, "nobody") is None


def test_a_person_is_always_recognized_in_their_own_house(hub_db):
    assert visible_in(hub_db, "p-max", "livingroom")
    assert visible_in(hub_db, "p-anna", "kitchen")


def test_a_second_house_sees_the_profile_only_with_consent(hub_db):
    # F-212: «иначе — только в своём».
    assert not visible_in(hub_db, "p-anna", "garage")
    assert set_share_identity(hub_db, "p-anna", "kitchen", True)
    assert visible_in(hub_db, "p-anna", "garage"), "разрешил — узнают во всех домах, где он в членстве"
    assert shared_with(hub_db, "p-anna") == ["kitchen"]


def test_taking_the_consent_back_hides_the_profile_again(hub_db):
    set_share_identity(hub_db, "p-anna", "kitchen", True)
    assert set_share_identity(hub_db, "p-anna", "kitchen", False)
    assert not visible_in(hub_db, "p-anna", "garage")
    assert shared_with(hub_db, "p-anna") == []


def test_consent_never_creates_membership(hub_db):
    set_share_identity(hub_db, "p-anna", "kitchen", True)
    # Макс never joined the garage - sharing his profile would not put him there.
    set_share_identity(hub_db, "p-max", "garage", True)
    assert not visible_in(hub_db, "p-max", "garage")
    assert visible_people(hub_db, "garage") == {"p-anna"} == filter_visible(
        hub_db, "garage", ["p-anna", "p-max"])


def test_a_home_sees_exactly_its_visible_members(hub_db):
    assert visible_people(hub_db, "livingroom") == {"p-max"}
    assert visible_people(hub_db, "kitchen") == {"p-anna"}
    assert visible_people(hub_db, "nowhere") == set()
    set_share_identity(hub_db, "p-anna", "kitchen", True)
    assert visible_people(hub_db, "garage") == {"p-anna"}


def test_changing_the_flag_of_a_foreign_room_does_nothing(hub_db):
    assert not set_share_identity(hub_db, "p-max", "garage", True)
    assert not set_share_identity(hub_db, "", "kitchen", True)
    assert not set_share_identity(hub_db, "p-max", "", True)


def test_a_person_without_any_membership_has_no_flag_to_consult(hub_db):
    """Imported data (no membership row) keeps the behaviour of the phases before F-212."""
    hub_db.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-legacy', 'Legacy')")
    hub_db.commit()
    assert visible_in(hub_db, "p-legacy", "garage")
    assert "p-legacy" in visible_people(hub_db, "garage")


def test_a_change_is_audited_once(hub_db):
    audit = _Audit()
    assert set_share_identity(hub_db, "p-anna", "kitchen", True, audit=audit, actor="p-anna")
    # Asking for the same value again is not a new event.
    assert not set_share_identity(hub_db, "p-anna", "kitchen", True, audit=audit)
    assert [row["action"] for row in audit.rows] == ["identity.share"]
    assert set_share_identity(hub_db, "p-anna", "kitchen", False, audit=audit)
    assert [row["action"] for row in audit.rows] == ["identity.share", "identity.unshare"]
    assert audit.rows[0]["target"] == "p-anna" and audit.rows[0]["home_id"] == "kitchen"


def test_the_person_own_words_are_what_changes_the_flag():
    assert share_command("Rowan, share my identity with the other rooms") is True
    assert share_command("Rowan, разреши узнавать меня в других домах") is True
    assert share_command("Rowan, не узнавай меня в других домах") is False
    assert share_command("Rowan, stop sharing my profile") is False
    assert share_command("Rowan, compartir mi identidad") is None
    # A refusal wins when both appear, and ordinary speech is not consent.
    assert share_command("share my identity, no, stop sharing") is False
    assert share_command("Rowan, включи свет") is None
    assert share_command("") is None


def test_the_description_matches_the_table(hub_db):
    set_share_identity(hub_db, "p-anna", "kitchen", True)
    described = describe(hub_db, "p-anna")
    assert described["primary_home"] == "kitchen"
    assert described["shared_with"] == ["kitchen"]
    assert [room["home_id"] for room in described["rooms"]] == ["kitchen", "garage"]
    assert described["rooms"][0]["share_identity"] is True


# --- the room ---------------------------------------------------------------


def _connection(hub_db, monkeypatch, *, home_id: str = "livingroom", speaker: str = "Макс"):
    from hub import app as hub_app
    from hub.room_state import RoomState

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: _Audit())
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = home_id
    connection.session = None
    connection.room = RoomState()
    connection._speaker_name = speaker
    connection._speaker_role = "user"
    connection._speaker_score = 0.9
    connection._track_face_match = {}
    connection._track_voice_match = {}
    return connection


def test_a_room_sees_its_own_member_but_not_an_unshared_outsider(hub_db, monkeypatch):
    livingroom = _connection(hub_db, monkeypatch, home_id="livingroom")
    assert livingroom._person_visible_here("p-max")
    assert not livingroom._person_visible_here("p-anna")
    garage = _connection(hub_db, monkeypatch, home_id="garage", speaker="Анна")
    assert not garage._person_visible_here("p-anna"), "Анна не разрешила делиться"
    set_share_identity(hub_db, "p-anna", "kitchen", True)
    assert garage._person_visible_here("p-anna")
    assert not garage._person_visible_here("p-max")
    assert not garage._person_visible_here("")


def test_without_a_database_the_phase_one_behaviour_stays(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection.home_id = ""
    assert connection._person_visible_here("p-anna")


def test_an_unreadable_database_says_no(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    hub_db.close()
    assert not connection._person_visible_here("p-max"), "согласие, которое нельзя доказать, — не согласие"


def test_the_person_can_allow_and_refuse_sharing_by_voice(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, speaker="Макс")
    assert asyncio.run(connection._share_identity_turn("Rowan, включи свет")) is None
    answer = asyncio.run(connection._share_identity_turn(
        "Rowan, разреши узнавать меня в других домах"))
    assert answer is not None and "may recognise" in answer
    assert hub_db.execute("SELECT share_identity FROM memberships WHERE person_id='p-max'"
                          ).fetchone()[0] == 1
    again = asyncio.run(connection._share_identity_turn(
        "Rowan, разреши узнавать меня в других домах"))
    assert again is not None and "already allowed" in again
    refusal = asyncio.run(connection._share_identity_turn(
        "Rowan, не узнавай меня в других домах"))
    assert refusal is not None and "only in this room" in refusal
    assert hub_db.execute("SELECT share_identity FROM memberships WHERE person_id='p-max'"
                          ).fetchone()[0] == 0


def test_a_stranger_cannot_change_anybodys_consent(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, speaker="")
    answer = asyncio.run(connection._share_identity_turn("Rowan, share my profile"))
    assert answer is not None and "recognised" in answer
    assert hub_db.execute("SELECT MAX(share_identity) FROM memberships").fetchone()[0] == 0


def _fuse(hub_db, monkeypatch, home_id: str, *, name: str = "p-anna") -> Any:
    """Run one fusion pass of a live track whose face matched ``name``."""
    import time

    from hub import app as hub_app
    from hub.identity_fusion import BeliefStore, IdentityHysteresis

    store = BeliefStore(hub_db)
    monkeypatch.setattr(hub_app, "_identity_beliefs", store)
    monkeypatch.setattr(hub_app, "_body_embeddings", False)
    connection = _connection(hub_db, monkeypatch, home_id=home_id)
    connection.room.update([{"id": "t1", "box": [0.2, 0.1, 0.6, 0.9]}])
    connection._track_face_match["t1"] = (name, 0.9, time.monotonic())
    connection._identity_hysteresis = IdentityHysteresis()
    connection._fuse_identities()
    return store.get("t1")


def test_the_fusion_ignores_a_name_that_was_not_shared_with_the_room(hub_db, monkeypatch):
    belief = _fuse(hub_db, monkeypatch, "garage")
    assert belief is not None and belief.person_id is None
    assert belief.p == 0.0, "a hidden profile is not weak evidence, it is no evidence"
    # The person allowed it: the very same signal now names the track.
    set_share_identity(hub_db, "p-anna", "kitchen", True)
    named = _fuse(hub_db, monkeypatch, "garage")
    assert named is not None and named.person_id == "p-anna" and named.p > 0.8


def test_the_fusion_still_names_the_member_of_its_own_room(hub_db, monkeypatch):
    belief = _fuse(hub_db, monkeypatch, "kitchen")
    assert belief is not None and belief.person_id == "p-anna" and belief.p > 0.8
