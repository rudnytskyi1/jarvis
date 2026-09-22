"""ТЗ F-606: «разреши ему музыку» — гостевое окно, которое кончается само."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import guest_access
from hub.guest_access import GuestGrantStore
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.session import Session

OWNER = "p-owner"
GUEST = "p-guest"
OTHER = "p-other"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (OWNER, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (GUEST, "Кай"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (OTHER, "Марина"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role, share_identity,"
                 " share_presence) VALUES (?,?,?,0,0)", (OWNER, "livingroom", "admin"))
    # ТЗ F-210 writes exactly this: the guest becomes a member of the room with
    # the ``guest`` role.
    conn.execute("INSERT INTO memberships(person_id, home_id, role, share_identity,"
                 " share_presence) VALUES (?,?,?,0,0)", (GUEST, "livingroom", "guest"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role, share_identity,"
                 " share_presence) VALUES (?,?,?,0,0)", (OTHER, "livingroom", "guest"))
    conn.commit()
    yield conn
    conn.close()


def _roles(*, owner: str = "admin", guest: str = "guest"):
    table = {"Антон": owner, "Кай": guest, "Марина": guest}
    registry = Mock()
    registry.role_of = lambda name: table.get(str(name or ""), "unknown")
    return registry


def _connection(hub_db, monkeypatch, *, speaker: str = "Антон", role: str = "admin",
                present: tuple[str, ...] = ("Кай",), store=None):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_voices", _roles())
    monkeypatch.setattr(hub_app, "_guest_grants", store if store is not None else False)
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = "livingroom"
    conn.cfg = Config()
    conn.session = Session(client_id="room-pc", devices=[], history_turns=2)
    conn._speaker_name = speaker
    conn._speaker_role = role
    conn._speaker_score = 0.9
    conn._reply_language = "ru"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn.presence = hub_app.PresenceTracker(30.0)
    if present:
        conn.presence.note_faces(list(present))
    return conn


# --- разбор фразы ---------------------------------------------------------


@pytest.mark.parametrize("text,name", [
    ("разреши ему музыку", ""),
    ("Разреши Каю музыку.", "Каю"),
    ("разреши музыку Каю", "Каю"),
    ("allow him music", ""),
    ("allow Max to play music", "Max"),
    ("let her play music", ""),
    ("permítele música", ""),
    ("permite a Max música", "Max"),
])
def test_the_owner_s_sentence_is_understood(text, name):
    request = guest_access.grant_command(text)
    assert request is not None and request.capability == guest_access.MUSIC
    assert request.name == name


@pytest.mark.parametrize("text", [
    "скажи Максу, что я иду",
    "разреши мне музыку",
    "включи музыку",
    "who is at home?",
    "add Max to my contacts",
])
def test_other_sentences_stay_with_their_own_turns(text):
    # "разреши мне музыку" is not a grant either: the closed pronoun list stops
    # the parser from treating "мне" as the guest's name.
    request = guest_access.grant_command(text)
    assert request is None or request.name != "мне"


def test_a_grant_opens_music_and_nothing_else():
    assert guest_access.grant_for_tool("device_set", {"capability": "media_play"}) == "music"
    assert guest_access.grant_for_tool("pc_control", {"command": "media_play_pause"}) == "music"
    assert guest_access.grant_for_tool("pc_control", {"command": "volume_set"}) == "music"
    assert guest_access.grant_for_tool("pc_control", {"command": "open_app"}) is None
    assert guest_access.grant_for_tool("run_command", {"command": "dir"}) is None


# --- хранилище ------------------------------------------------------------


def test_a_window_lives_until_its_deadline_and_then_goes_away(hub_db):
    store = GuestGrantStore(hub_db)
    moment = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
    grant = store.grant(home_id="livingroom", guest_person_id=GUEST, capability="music",
                        granted_by=OWNER, window_s=1800, now=moment)
    assert grant.active(now=moment)
    assert store.allows("livingroom", GUEST, "music", now=moment) is True
    assert store.expires_at("livingroom", GUEST, "music", now=moment) == grant.expires_at
    # 31 minutes later the same row is simply not a grant any more.
    later = moment + timedelta(minutes=31)
    assert store.active("livingroom", GUEST, now=later) == []
    assert store.allows("livingroom", GUEST, "music", now=later) is False
    assert store.expires_at("livingroom", GUEST, "music", now=later) == ""
    assert len(store.history("livingroom")) == 1, "the past is still readable"
    assert store.prune(now=later) == 1
    assert store.history("livingroom") == []


def test_a_window_is_one_person_one_home_and_one_capability(hub_db):
    store = GuestGrantStore(hub_db)
    store.grant(home_id="livingroom", guest_person_id=GUEST, capability="music",
                granted_by=OWNER, window_s=600)
    assert store.allows("livingroom", GUEST, "music") is True
    assert store.allows("livingroom", OTHER, "music") is False, "one guest, not every guest"
    assert store.allows("kitchen", GUEST, "music") is False, "one room, not every room"
    assert store.allows("livingroom", GUEST, "light") is False
    with pytest.raises(ValueError):
        store.grant(home_id="livingroom", guest_person_id=GUEST, capability="teleport")


# --- живой ход ------------------------------------------------------------


def test_the_owner_grants_music_to_the_guest_in_the_room(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    answer = asyncio.run(conn._guest_grant_turn("разреши ему музыку", "ru"))
    assert answer is not None and "Кай" in answer and "30 мин" in answer
    grant = store.history("livingroom")[0]
    assert grant.guest_person_id == GUEST
    assert grant.capability == "music" and grant.granted_by == OWNER
    assert grant.active() is True


def test_the_owner_may_name_the_guest(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    conn = _connection(hub_db, monkeypatch, present=("Кай", "Марина"), store=store)
    answer = asyncio.run(conn._guest_grant_turn("разреши Марине музыку", "ru"))
    assert answer is not None and "Марина" in answer
    assert store.history("livingroom")[0].guest_person_id == OTHER


def test_two_guests_and_no_name_is_a_question_not_a_guess(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    conn = _connection(hub_db, monkeypatch, present=("Кай", "Марина"), store=store)
    answer = asyncio.run(conn._guest_grant_turn("разреши ему музыку", "ru"))
    assert "несколько гостей" in answer
    assert store.history("livingroom") == [], "nobody was granted anything"


def test_no_guest_in_the_room_is_said_honestly(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    conn = _connection(hub_db, monkeypatch, present=("Антон",), store=store)
    answer = asyncio.run(conn._guest_grant_turn("разреши ему музыку", "ru"))
    assert "нет гостя" in answer
    assert store.history("livingroom") == []


def test_a_name_the_room_does_not_see_is_refused(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    conn = _connection(hub_db, monkeypatch, present=("Кай",), store=store)
    answer = asyncio.run(conn._guest_grant_turn("разреши Лене музыку", "ru"))
    assert "Лене" in answer and "не вижу" in answer
    assert store.history("livingroom") == []


def test_only_the_owner_may_extend_a_guest_s_access(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    conn = _connection(hub_db, monkeypatch, speaker="Кай", role="guest", store=store)
    answer = asyncio.run(conn._guest_grant_turn("разреши ему музыку", "ru"))
    assert "только хозяин" in answer
    assert store.history("livingroom") == []


def test_an_unrecognised_voice_cannot_grant_anything(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    conn = _connection(hub_db, monkeypatch, speaker="unknown", role="unknown", store=store)
    answer = asyncio.run(conn._guest_grant_turn("разреши ему музыку", "ru"))
    assert "узнать ваш голос" in answer
    assert store.history("livingroom") == []


def test_a_grant_is_audited(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    audit = Mock()
    monkeypatch.setattr(hub_app, "_audit", audit)
    asyncio.run(conn._guest_grant_turn("разреши ему музыку", "ru"))
    record = audit.record.call_args
    assert record.kwargs["action"] == "guest.grant"
    assert record.kwargs["actor"] == OWNER and record.kwargs["target"] == GUEST
    assert record.kwargs["result"] == "ok"
    assert record.kwargs["detail"]["capability"] == "music"
    assert record.kwargs["detail"]["expires_at"]


# --- окно открывает ровно музыку и само закрывается ----------------------


def test_the_window_lets_the_guest_play_music_but_not_the_rest_of_the_pc(
        hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    monkeypatch.setattr(hub_app, "_devices", False)
    conn = _connection(hub_db, monkeypatch, store=store)
    conn._speaker_name, conn._speaker_role = "Кай", "guest"
    before = asyncio.run(conn._permission_check("pc_control", {"command": "media_play_pause"}))
    assert before is not None and "компьютер" in before
    store.grant(home_id="livingroom", guest_person_id=GUEST, capability="music",
                granted_by=OWNER, window_s=600)
    after = asyncio.run(conn._permission_check("pc_control", {"command": "media_play_pause"}))
    assert after is None, "the owner's window is what lets this through"
    still = asyncio.run(conn._permission_check("pc_control", {"command": "open_app",
                                                              "value": "steam"}))
    assert still is not None and "компьютер" in still


def test_the_window_closes_itself_when_it_runs_out(hub_db, monkeypatch):
    store = GuestGrantStore(hub_db)
    monkeypatch.setattr(hub_app, "_devices", False)
    conn = _connection(hub_db, monkeypatch, store=store)
    conn._speaker_name, conn._speaker_role = "Кай", "guest"
    store.grant(home_id="livingroom", guest_person_id=GUEST, capability="music",
                granted_by=OWNER, window_s=0)
    refusal = asyncio.run(conn._permission_check("pc_control",
                                                {"command": "media_play_pause"}))
    assert refusal is not None and "компьютер" in refusal


def test_the_window_is_configured(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch, store=GuestGrantStore(hub_db))
    assert conn._guest_grant_window() == 1800.0
    conn.cfg.server.identity.guest.grant_window_s = 600
    assert conn._guest_grant_window() == 600.0
