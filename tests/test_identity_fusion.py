"""Слияние голоса, лица и тела в одного человека (ТЗ F-206).

The three signals are pure numbers here - that is the point of the module: the
face and voice engines, the body store and the camera all feed it, and the
answer it gives is one row of ``identity_belief`` with the numbers behind it.
The D-06 context rule and the once-a-second hub pass get their own tests.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest

from common.config import load_config
from hub import app as hub_app
from hub import migrations_runner
from hub.decision_points import identity_heuristic
from hub.identity_fusion import (
    FACE_THRESHOLD,
    FUSION_INTERVAL_S,
    VOICE_MARGIN,
    VOICE_THRESHOLD,
    Belief,
    BeliefStore,
    Signal,
    body_signal,
    fuse,
)
from hub.reid import BodyEmbeddingStore
from hub.room_state import RoomState


def _vector(seed: int = 0, *, dim: int = 512) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('legacy-max', 'Макс')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('legacy-drew', 'Drew')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES ('legacy-max',"
                 " 'livingroom', 'admin')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


# --- the numbers of the ТЗ ------------------------------------------------------


def test_the_thresholds_are_the_ones_the_spec_states():
    # ТЗ F-206: лицо cos ≥ 0,45; голос 0,40 и отрыв 0,15; раз в секунду.
    assert FACE_THRESHOLD == 0.45
    assert VOICE_THRESHOLD == 0.40
    assert VOICE_MARGIN == 0.15
    assert FUSION_INTERVAL_S == 1.0


def test_a_confidence_is_how_far_past_the_bar_a_signal_is():
    assert Signal("face", "max", 0.45).confidence(0.45) == 0.0
    assert Signal("face", "max", 1.0).confidence(0.45) == 1.0
    assert Signal("face", "max", 0.4).confidence(0.45) == 0.0
    assert Signal("face", "max", 0.725).confidence(0.45) == pytest.approx(0.5)


# --- the fusion -----------------------------------------------------------------


def test_no_signal_means_nobody():
    belief = fuse(track_id="a:1", signals=[], now=1.0)
    assert belief.person_id is None and belief.p == 0.0
    assert belief.sources["reason"] == "no signal past its threshold"


def test_one_good_face_is_enough_to_believe():
    belief = fuse(track_id="a:1", signals=[Signal("face", "max", 0.9)], now=1.0)
    assert belief.person_id == "max"
    # 1 - (0.9-0.45)/(1-0.45) = 0.818: past the 0.8 the hysteresis wants (F-207).
    assert belief.p == pytest.approx(0.818, abs=0.001)
    assert belief.sources["face"] == 0.9


def test_a_face_that_barely_clears_its_bar_is_not_an_identity():
    belief = fuse(track_id="a:1", signals=[Signal("face", "max", 0.5)], now=1.0)
    assert belief.person_id == "max" and belief.p < 0.1


def test_a_signal_below_its_bar_is_not_evidence():
    belief = fuse(track_id="a:1", signals=[Signal("face", "max", 0.44),
                                           Signal("body", "max", 0.49)], now=1.0)
    assert belief.person_id is None and belief.p == 0.0


def test_two_weak_signals_reinforce_each_other():
    one = fuse(track_id="a:1", signals=[Signal("face", "max", 0.6)], now=1.0)
    both = fuse(track_id="a:1", signals=[Signal("face", "max", 0.6),
                                         Signal("body", "max", 0.6)], now=1.0)
    assert both.p > one.p and both.person_id == "max"
    # Each kind is measured against its OWN bar (the face's 0.45, the body's 0.5).
    face_c, body_c = (0.6 - 0.45) / 0.55, (0.6 - 0.5) / 0.5
    assert both.p == pytest.approx(1 - (1 - face_c) * (1 - body_c), abs=0.001)
    assert both.sources["body"] == 0.6


def test_a_voice_needs_its_lead_over_the_runner_up():
    # Two voices that close together are not evidence at all (ТЗ F-206).
    unclear = fuse(track_id="a:1", signals=[Signal("voice", "max", 0.5),
                                            Signal("voice", "drew", 0.45)], now=1.0)
    assert unclear.person_id is None and unclear.p == 0.0
    clear = fuse(track_id="a:1", signals=[Signal("voice", "max", 0.71),
                                          Signal("voice", "drew", 0.45)], now=1.0)
    assert clear.person_id == "max" and clear.sources["voice"] == 0.71


def test_the_body_bar_comes_from_the_config():
    belief = fuse(track_id="a:1", signals=[Signal("body", "max", 0.6)], now=1.0,
                  thresholds={"face": 0.45, "voice": 0.4, "body": 0.7})
    assert belief.person_id is None


def test_two_people_within_the_margin_go_to_d06():
    signals = [Signal("face", "max", 0.9), Signal("face", "drew", 0.88)]
    alone = fuse(track_id="a:1", signals=signals, now=1.0)
    assert alone.person_id is None and alone.sources["ambiguous"]
    assert alone.p > 0.0, "the numbers are kept even when nobody is named"
    # The context of D-06 is the only thing that separates them here.
    with_context = fuse(track_id="a:1", signals=signals, expected=("drew",), now=1.0)
    assert with_context.person_id == "drew"
    assert with_context.sources["context"] == "expected here / already in the room"
    # Both expected: still nobody, a coin toss is not an answer.
    both = fuse(track_id="a:1", signals=signals, expected=("max", "drew"), now=1.0)
    assert both.person_id is None


def test_a_clear_lead_ignores_the_context():
    belief = fuse(track_id="a:1", signals=[Signal("face", "max", 0.9),
                                           Signal("face", "drew", 0.6)],
                  expected=("drew",), now=1.0)
    assert belief.person_id == "max"


def test_a_belief_explains_itself():
    belief = fuse(track_id="a:1", signals=[Signal("face", "max", 0.9),
                                           Signal("voice", "max", 0.71)], now=1.0)
    assert "face 0.90" in belief.explains() and "voice 0.71" in belief.explains()
    assert fuse(track_id="a:2", signals=[], now=1.0).explains() == "no signal"


# --- D-06 ------------------------------------------------------------------------


def test_d06_answers_the_question_the_fusion_asks():
    assert identity_heuristic([("max", 0.9)]) == ("max", "clear lead")
    assert identity_heuristic([("max", 0.9), ("drew", 0.5)]) == ("max", "clear lead")
    assert identity_heuristic([("max", 0.9), ("drew", 0.85)]) == (None, "ambiguous")
    assert identity_heuristic([("max", 0.9), ("drew", 0.85)], present=("drew",)) == ("drew", "context")
    assert identity_heuristic([]) == (None, "no candidate")


# --- the body signal ---------------------------------------------------------------


def test_the_body_signal_comes_from_the_vectors_of_that_day(hub_db):
    bodies = BodyEmbeddingStore(hub_db)
    person = _vector(1)
    bodies.save(track_id="a:1", vector=person, session_day="2026-09-21",
                person_id="legacy-max", home_id="livingroom")
    bodies.save(track_id="a:2", vector=person, session_day="2026-09-21", home_id="livingroom")
    signal = body_signal(bodies, track_id="a:2", day="2026-09-21")
    assert signal is not None and signal.kind == "body" and signal.person_id == "legacy-max"
    # Yesterday's clothes say nothing about today (ТЗ F-206).
    assert body_signal(bodies, track_id="a:2", day="2026-09-22") is None
    assert body_signal(None, track_id="a:2", day="2026-09-21") is None
    assert body_signal(bodies, track_id="missing", day="2026-09-21") is None


def test_the_body_signal_is_silent_when_nobody_looks_like_that_track(hub_db):
    bodies = BodyEmbeddingStore(hub_db)
    bodies.save(track_id="a:1", vector=_vector(2), session_day="2026-09-21",
                person_id="legacy-max", home_id="livingroom")
    bodies.save(track_id="a:2", vector=_vector(3), session_day="2026-09-21", home_id="livingroom")
    assert body_signal(bodies, track_id="a:2", day="2026-09-21") is None


# --- the table ---------------------------------------------------------------------


def test_a_belief_is_stored_once_per_track(hub_db):
    store = BeliefStore(hub_db)
    first = Belief(track_id="a:1", person_id="legacy-max", p=0.82,
                   sources={"face": 0.9, "p": 0.82}, at=100.0)
    assert store.save(first, home_id="livingroom") is True
    assert store.get("a:1") == first
    second = Belief(track_id="a:1", person_id=None, p=0.1, sources={"reason": "no signal"}, at=101.0)
    assert store.save(second, home_id="livingroom") is True
    assert store.count() == 1, "a track has ONE current belief"
    assert store.get("a:1").p == 0.1 and store.get("a:1").person_id is None
    assert store.get("missing") is None


def test_a_belief_of_a_person_that_does_not_exist_stores_no_link(hub_db):
    store = BeliefStore(hub_db)
    belief = Belief(track_id="a:1", person_id="ghost", p=0.9, sources={}, at=1.0)
    assert store.save(belief, home_id="livingroom") is True
    assert store.get("a:1").person_id is None


def test_the_beliefs_of_a_home_come_back_strongest_first(hub_db):
    store = BeliefStore(hub_db)
    store.save(Belief("a:1", "legacy-max", 0.9, {}, 1.0), home_id="livingroom")
    store.save(Belief("a:2", "legacy-drew", 0.3, {}, 1.0), home_id="livingroom")
    assert [belief.track_id for belief in store.for_home("livingroom")] == ["a:1", "a:2"]
    assert [belief.track_id for belief in store.for_home("livingroom", min_p=0.5)] == ["a:1"]
    assert store.for_home("other") == []
    assert store.forget("a:1") == 1
    assert store.forget("a:1") == 0
    assert store.count(home_id="livingroom") == 1


# --- the hub pass -------------------------------------------------------------------


def _room(tracks) -> RoomState:
    room = RoomState()
    room.update(tracks, now=time.monotonic())
    return room


def _connection(hub_db, room):
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = room
    connection._track_face_match = {}
    connection._track_voice_match = {}
    connection._track_face_state = {}
    connection._track_mouth = {}
    connection._pending_pin = None
    connection._pin_opened = None
    connection._pending_challenge = None
    connection._challenge_opened = None
    connection._speaker_name = "unknown"
    connection._speaker_role = "unknown"
    connection._speaker_score = 0.0
    return connection


class _Audit:
    """A recording stand-in for the hub's ``audit`` table (F-706)."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs) -> None:
        self.rows.append(kwargs)


def _wire(monkeypatch, hub_db, tmp_path, beliefs, bodies=None):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_identity_beliefs", beliefs)
    monkeypatch.setattr(hub_app, "_body_embeddings", bodies or False)


def test_the_hub_writes_the_belief_of_a_live_track(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, time.monotonic())

    connection._fuse_identities()

    belief = store.get("a:1")
    assert belief is not None and belief.person_id == "legacy-max" and belief.p > 0.8
    assert belief.sources["face"] == 0.9


def test_a_signal_older_than_its_window_is_not_evidence(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection._track_face_match["a:1"] = ("legacy-max", 0.9,
                                           time.monotonic() - hub_app.LIVE_SIGNAL_TTL_S - 1)

    connection._fuse_identities()

    belief = store.get("a:1")
    assert belief is not None and belief.person_id is None and belief.p == 0.0


def test_the_body_of_today_is_part_of_the_belief(hub_db, tmp_path, monkeypatch):
    bodies = BodyEmbeddingStore(hub_db)
    person = _vector(4)
    from hub.reid import session_day_of

    bodies.save(track_id="earlier", vector=person, session_day=session_day_of(),
                person_id="legacy-max", home_id="livingroom")
    bodies.save(track_id="a:1", vector=person, session_day=session_day_of(),
                home_id="livingroom")
    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db), bodies)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))

    connection._fuse_identities()

    belief = BeliefStore(hub_db).get("a:1")
    assert belief is not None and belief.person_id == "legacy-max"
    assert belief.sources["body"] >= 0.99


def test_a_track_that_left_takes_its_belief_with_it(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, time.monotonic())
    connection._fuse_identities()
    assert store.count() == 1

    connection.room.update([], now=time.monotonic() + 10.0)  # the track left
    connection._fuse_identities()

    assert store.count() == 0 and connection._track_face_match == {}


def test_the_context_is_the_memberships_and_the_named_tracks(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]},
                                            {"id": "a:2", "box": [0.7, 0.1, 0.9, 0.9]}]))
    hub_db.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen,"
                   " person_id) VALUES ('a:2','livingroom','pc-1','now','now','legacy-drew')")
    hub_db.commit()

    expected, present = connection._identity_context()

    assert "legacy-max" in expected and "legacy-drew" not in expected
    assert "legacy-drew" in present


def test_the_loop_fuses_every_live_track(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    monkeypatch.setattr(hub_app, "IDENTITY_POLL_S", 0.01)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, time.monotonic())
    connection._identity_task = None

    async def run() -> None:
        task = asyncio.create_task(connection._identity_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())

    assert store.count() == 1 and store.get("a:1").person_id == "legacy-max"


def test_starting_the_fusion_respects_the_config_and_the_loop(hub_db, tmp_path, monkeypatch):
    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    connection = _connection(hub_db, _room([]))
    connection.cfg = type("Cfg", (), {"server": type("S", (), {
        "identity": type("I", (), {"enabled": False})()})()})()
    connection._identity_task = None
    connection._start_identity_task()
    assert connection._identity_task is None, "disabled in the config means no task"

    connection.cfg.server.identity.enabled = True
    monkeypatch.setattr(hub_app, "IDENTITY_POLL_S", 0.01)

    async def run() -> None:
        connection._start_identity_task()
        await asyncio.sleep(0.02)
        assert connection._identity_task is not None
        # A second call is a no-op, not a second loop.
        first = connection._identity_task
        connection._start_identity_task()
        assert connection._identity_task is first
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)

    asyncio.run(run())


# --- the hysteresis of F-207 ---------------------------------------------------------


def _belief(person: str | None, p: float) -> Belief:
    return Belief(track_id="a:1", person_id=person, p=p, sources={}, at=1.0)


def test_the_hysteresis_counts_the_runs_the_spec_states():
    from hub.identity_fusion import IdentityHysteresis

    hysteresis = IdentityHysteresis()
    assert (hysteresis.recognize_p, hysteresis.lose_p) == (0.8, 0.5)
    assert (hysteresis.recognize_runs, hysteresis.lose_runs) == (2, 3)
    # One strong pass is not enough (ТЗ F-207: дважды подряд).
    assert hysteresis.observe("a:1", _belief("max", 0.82)) == "pending"
    assert hysteresis.recognized("a:1") is None
    assert hysteresis.observe("a:1", _belief("max", 0.9)) == "recognized"
    assert hysteresis.recognized("a:1") == "max"
    # Already recognized and still strong: nothing new happens.
    assert hysteresis.observe("a:1", _belief("max", 0.95)) == "unchanged"


def test_a_strong_pass_for_another_person_restarts_the_count():
    from hub.identity_fusion import IdentityHysteresis

    hysteresis = IdentityHysteresis()
    hysteresis.observe("a:1", _belief("max", 0.9))
    assert hysteresis.observe("a:1", _belief("drew", 0.9)) == "pending"
    assert hysteresis.recognized("a:1") is None
    assert hysteresis.observe("a:1", _belief("drew", 0.9)) == "recognized"


def test_a_weak_pass_in_the_middle_breaks_the_run():
    from hub.identity_fusion import IdentityHysteresis

    hysteresis = IdentityHysteresis()
    hysteresis.observe("a:1", _belief("max", 0.9))
    assert hysteresis.observe("a:1", _belief("max", 0.6)) == "unchanged"
    assert hysteresis.observe("a:1", _belief("max", 0.9)) == "pending"
    assert hysteresis.recognized("a:1") is None


def test_a_person_is_lost_after_three_weak_passes():
    from hub.identity_fusion import IdentityHysteresis

    hysteresis = IdentityHysteresis()
    hysteresis.observe("a:1", _belief("max", 0.9))
    hysteresis.observe("a:1", _belief("max", 0.9))
    assert hysteresis.observe("a:1", _belief("max", 0.4)) == "fading"
    assert hysteresis.observe("a:1", _belief("max", 0.4)) == "fading"
    assert hysteresis.recognized("a:1") == "max"
    assert hysteresis.observe("a:1", _belief("max", 0.2)) == "lost"
    assert hysteresis.recognized("a:1") is None
    # A never-recognized track has nothing to lose.
    assert hysteresis.observe("a:2", _belief(None, 0.0)) == "unchanged"
    assert hysteresis.observe("a:2", _belief(None, 0.1)) == "unchanged"


def test_a_strong_pass_resets_the_losing_run():
    from hub.identity_fusion import IdentityHysteresis

    hysteresis = IdentityHysteresis()
    hysteresis.observe("a:1", _belief("max", 0.9))
    hysteresis.observe("a:1", _belief("max", 0.9))
    hysteresis.observe("a:1", _belief("max", 0.3))
    hysteresis.observe("a:1", _belief("max", 0.3))
    assert hysteresis.observe("a:1", _belief("max", 0.9)) == "unchanged"
    assert hysteresis.observe("a:1", _belief("max", 0.3)) == "fading"
    assert hysteresis.recognized("a:1") == "max"


def test_a_visit_is_greeted_once_per_track():
    from hub.identity_fusion import IdentityHysteresis

    hysteresis = IdentityHysteresis()
    assert hysteresis.should_greet("a:1") is False
    hysteresis.observe("a:1", _belief("max", 0.9))
    hysteresis.observe("a:1", _belief("max", 0.9))
    assert hysteresis.should_greet("a:1") is True
    assert hysteresis.should_greet("a:1") is False, "приветствие — один раз за визит"
    assert hysteresis.greeted("a:1") is True
    # A DIFFERENT person on the same track deserves their own greeting.
    hysteresis.observe("a:1", _belief("drew", 0.9))
    hysteresis.observe("a:1", _belief("drew", 0.9))
    assert hysteresis.should_greet("a:1") is True
    hysteresis.forget("a:1")
    assert hysteresis.recognized("a:1") is None and hysteresis.live() == []


def test_the_hub_confirms_a_name_only_after_two_passes(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, time.monotonic())

    connection._fuse_identities()
    assert hub_db.execute("SELECT COUNT(*) FROM tracks WHERE person_id IS NOT NULL").fetchone()[0] == 0

    connection._fuse_identities()
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == \
        "legacy-max"
    assert connection._identity_hysteresis.greeted("a:1") is True


def test_the_hub_loses_a_name_after_three_weak_passes(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, time.monotonic())
    connection._fuse_identities()
    connection._fuse_identities()
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == \
        "legacy-max"

    connection._track_face_match.clear()   # nobody is recognized any more
    for _ in range(3):
        connection._fuse_identities()

    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] is None


def test_fresh_direct_evidence_keeps_the_name(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, time.monotonic())
    connection._fuse_identities()
    connection._fuse_identities()
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == \
        "legacy-max"

    # The voice is fresh but too weak for the fusion's bar, so the belief says
    # "nobody" - and the witness that IS still there keeps the name.
    connection._track_face_match.clear()
    connection._track_voice_match["a:1"] = ("legacy-max", 0.35, time.monotonic())
    for _ in range(3):
        connection._fuse_identities()

    assert store.get("a:1").p == 0.0
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == \
        "legacy-max"


def test_stale_evidence_loses_the_name(hub_db, tmp_path, monkeypatch):
    store = BeliefStore(hub_db)
    _wire(monkeypatch, hub_db, tmp_path, store)
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, time.monotonic())
    connection._fuse_identities()
    connection._fuse_identities()

    # Nothing was seen or heard for longer than the window: the name goes.
    connection._track_face_match["a:1"] = ("legacy-max", 0.9,
                                           time.monotonic() - hub_app.LIVE_SIGNAL_TTL_S - 1)
    for _ in range(3):
        connection._fuse_identities()

    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] is None


# --- the admin gate and the spoken PIN (F-208) ---------------------------------------


def test_a_privileged_action_needs_a_second_witness():
    from hub.identity_fusion import ADMIN_FACE_THRESHOLD, ADMIN_VOICE_THRESHOLD, admin_gate

    # ТЗ F-208: голос ≥ 0,65 И (лицо ≥ 0,55 или тело того же дня).
    assert (ADMIN_VOICE_THRESHOLD, ADMIN_FACE_THRESHOLD) == (0.65, 0.55)
    assert admin_gate(person_id="legacy-max", voice_score=0.7, face_score=0.6).allowed is True
    assert admin_gate(person_id="legacy-max", voice_score=0.7,
                      body_linked_today=True).allowed is True
    weak_voice = admin_gate(person_id="legacy-max", voice_score=0.6, face_score=0.9)
    assert weak_voice.allowed is False and weak_voice.code == "voice"
    without_second = admin_gate(person_id="legacy-max", voice_score=0.7)
    assert without_second.allowed is False and without_second.code == "second_factor"
    nobody = admin_gate(person_id=None, voice_score=0.9, face_score=0.9)
    assert nobody.allowed is False and nobody.code == "no_person"
    assert without_second.factors["voice"] == 0.7


def test_a_phone_needs_its_own_bar_and_the_pin():
    from hub.identity_fusion import admin_gate

    assert admin_gate(person_id="legacy-max", voice_score=0.7, face_score=0.6,
                      channel="phone").code == "pin"
    assert admin_gate(person_id="legacy-max", voice_score=0.7, channel="phone",
                      pin_ok=True).allowed is True
    assert admin_gate(person_id="legacy-max", voice_score=0.6, channel="phone",
                      pin_ok=True, phone_threshold=0.65).code == "voice"


def test_a_spoken_pin_is_understood_in_three_languages():
    from hub.identity_fusion import VoicePin

    pin = VoicePin()
    assert pin.digits("1 2 3 4") == "1234"
    assert pin.digits("один два три четыре") == "1234"
    assert pin.digits("uno dos tres cuatro") == "1234"
    assert pin.digits("my PIN is 12 34, please") == "1234"
    assert pin.digits("yes please") == ""
    # A PIN has to be a real PIN.
    assert pin.valid("1234") is True and pin.valid("123") is False
    assert pin.valid("123456789") is False and pin.valid("abcd") is False


def test_a_pin_is_stored_only_as_a_salted_hash(hub_db):
    from hub.identity_fusion import VoicePin

    pin = VoicePin()
    assert pin.set_pin(hub_db, "legacy-max", "1234") is True
    stored = pin.stored_for(hub_db, "legacy-max")
    assert stored is not None and stored.startswith("pbkdf2$")
    assert "1234" not in stored, "the digits are never written down"
    assert pin.verify_hash("1234", stored) is True
    assert pin.verify_hash("4321", stored) is False
    assert pin.set_pin(hub_db, "legacy-max", "12") is False
    assert pin.set_pin(hub_db, "nobody", "1234") is False


def test_three_wrong_pins_lock_the_person_out(hub_db):
    from hub.identity_fusion import VoicePin

    pin = VoicePin(max_failures=3, lockout_s=300.0)
    pin.set_pin(hub_db, "legacy-max", "1234")
    assert pin.verify(hub_db, "legacy-max", "hello there", now=100.0) == (False, "not_a_pin")
    assert pin.verify(hub_db, "legacy-max", "9999", now=101.0) == (False, "wrong")
    assert pin.locked("legacy-max", now=102.0) is False, "two tries are not a lockout"
    assert pin.verify(hub_db, "legacy-max", "8888", now=102.0) == (False, "wrong")
    assert pin.locked("legacy-max", now=103.0) is True
    assert pin.verify(hub_db, "legacy-max", "1234", now=104.0) == (False, "locked")
    assert pin.locked("legacy-max", now=500.0) is False, "the lockout expires"
    assert pin.verify(hub_db, "legacy-max", "1234", now=501.0) == (True, "ok")
    assert pin.failures("legacy-max") == 0, "a good PIN clears the counter"


def test_a_person_without_a_pin_cannot_use_a_phone_for_admin(hub_db):
    from hub.identity_fusion import VoicePin

    assert VoicePin().verify(hub_db, "legacy-max", "1234") == (False, "no_pin")


def test_the_hub_asks_for_the_pin_of_a_phone_admin_call(hub_db, tmp_path, monkeypatch):
    from hub.identity_fusion import VoicePin

    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    VoicePin().set_pin(hub_db, "legacy-max", "1234")
    connection = _connection(hub_db, _room([]))
    connection.cfg = load_config("config.example.yaml")
    connection.cfg.server.identity.enabled = True
    connection.session = type("Session", (), {
        "identity": type("I", (), {"kind": "phone"})(),
        "client_id": "phone-1"})()
    connection._speaker_role = "admin"
    connection._speaker_name = "Макс"
    connection._speaker_score = 0.9

    denial = asyncio.run(connection._admin_strength_check("run_command", {"command": "dir"}))

    assert denial and "PIN" in denial
    assert connection._pending_pin is not None
    assert connection._pending_pin["person_id"] == "legacy-max"
    assert connection._pin_opened and "PIN" in connection._pin_opened
    # The room hears the hub's own question (the say step of this turn).
    assert hub_db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0


def test_a_room_admin_action_passes_with_a_face(hub_db, tmp_path, monkeypatch):
    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection.cfg = load_config("config.example.yaml")
    connection.cfg.server.identity.enabled = True
    connection._speaker_role = "admin"
    connection._speaker_name = "Макс"
    connection._speaker_score = 0.9
    connection._track_face_match["a:1"] = ("legacy-max", 0.7, time.monotonic())
    hub_db.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen,"
                   " person_id) VALUES ('a:1','livingroom','pc-1','now','now','legacy-max')")
    hub_db.commit()

    assert asyncio.run(connection._admin_strength_check("run_command", {})) is None


def test_a_room_admin_action_without_a_second_witness_asks_for_the_word(
        hub_db, tmp_path, monkeypatch):
    """F-208 refuses; F-214 (P2-24) offers its own second factor instead.

    The ТЗ asks for a challenge word for privileged actions, and the default
    configuration runs it when the F-208 witnesses are missing: the room is
    asked to say a random word, and the call waits for it - which is stronger
    than a refusal, not weaker, because a recording cannot answer a word it
    never heard.
    """
    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection.cfg = load_config("config.example.yaml")
    connection.cfg.server.identity.enabled = True
    connection._speaker_role = "admin"
    connection._speaker_name = "Макс"
    connection._speaker_score = 0.9

    denial = asyncio.run(connection._admin_strength_check("run_command", {}))

    assert denial and connection._pending_challenge is not None
    assert denial == connection._challenge_opened
    assert connection._pending_challenge["tool"] == "run_command"
    assert connection._pending_pin is None, "the room has a camera - no PIN is asked"


def test_without_the_challenge_the_f208_refusal_stands(hub_db, tmp_path, monkeypatch):
    """`anti_spoofing.challenge: off` keeps the phase-2 behaviour exactly."""
    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection.cfg = load_config("config.example.yaml")
    connection.cfg.server.identity.enabled = True
    connection.cfg.server.identity.anti_spoofing.challenge = "off"
    connection._speaker_role = "admin"
    connection._speaker_name = "Макс"
    connection._speaker_score = 0.9

    denial = asyncio.run(connection._admin_strength_check("run_command", {}))

    assert denial and "face" in denial
    assert connection._pending_challenge is None
    assert connection._pending_pin is None


def test_the_permission_check_applies_the_admin_gate(hub_db, tmp_path, monkeypatch):
    """F-208 rides the SAME place D-07 does, so an allowed role is not enough."""
    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    connection = _connection(hub_db, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))
    connection.cfg = load_config("config.example.yaml")
    connection.cfg.server.identity.enabled = True
    connection._speaker_role = "admin"
    connection._speaker_name = "Макс"
    connection._speaker_score = 0.9
    connection._wake_words = lambda: []
    monkeypatch.setattr(hub_app, "_decision_chain", lambda *a, **k: None)

    denial = asyncio.run(connection._permission_check("run_command", {"command": "dir"}))

    # ТЗ F-214 (P2-24): the privileged call does not run on the role alone.
    assert denial and connection._pending_challenge is not None
    # The same speaker WITH the second witness gets through.
    hub_db.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen,"
                   " person_id) VALUES ('a:1','livingroom','pc-1','now','now','legacy-max')")
    hub_db.commit()
    connection._track_face_match["a:1"] = ("legacy-max", 0.7, time.monotonic())
    assert asyncio.run(connection._permission_check("run_command", {"command": "dir"})) is None


def test_the_pin_question_is_answered_by_the_next_utterance(hub_db, tmp_path, monkeypatch):
    from hub.identity_fusion import VoicePin

    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    VoicePin().set_pin(hub_db, "legacy-max", "1234")
    connection = _connection(hub_db, _room([]))
    connection.cfg = load_config("config.example.yaml")
    spoken: list[str] = []

    async def speak(voice, say_text, **kwargs):
        spoken.append(str(say_text))

    monkeypatch.setattr(connection, "_speak_confirmation", speak)
    monkeypatch.setattr(connection, "_execute_tool", _fake_tool)
    monkeypatch.setattr(hub_app, "_pin", None)
    audit = _Audit()
    monkeypatch.setattr(hub_app, "_audit", audit)
    connection._pending_pin = {"person_id": "legacy-max", "tool": "run_command",
                               "args": {}, "expires": time.monotonic() + 20}

    answered = asyncio.run(connection._resolve_pin(
        "один два три четыре", voice=None, session=None, started_at=None,
        language="ru", stt_ms=1, t_start=time.perf_counter()))

    assert answered is True
    assert spoken and "Done" in spoken[0]
    assert connection._pending_pin is None
    assert [row["action"] for row in audit.rows] == ["confirm.pin"]
    assert audit.rows[0]["result"] == "ok"
    assert "1234" not in str(audit.rows[0]), "the digits never reach the audit table"


async def _fake_tool(tool, args):
    _fake_tool.calls.append((tool, args))
    return {"ok": True, "reply": "Done. The command ran."}


_fake_tool.calls = []


def test_a_wrong_pin_asks_again_and_a_third_one_locks(hub_db, tmp_path, monkeypatch):
    from hub.identity_fusion import VoicePin

    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    pin = VoicePin(max_failures=3, lockout_s=300.0)
    pin.set_pin(hub_db, "legacy-max", "1234")
    monkeypatch.setattr(hub_app, "_pin", pin)
    audit = _Audit()
    monkeypatch.setattr(hub_app, "_audit", audit)
    connection = _connection(hub_db, _room([]))
    connection.cfg = load_config("config.example.yaml")
    spoken: list[str] = []

    async def speak(voice, say_text, **kwargs):
        spoken.append(str(say_text))

    monkeypatch.setattr(connection, "_speak_confirmation", speak)

    async def answer(text: str) -> None:
        connection._pending_pin = {"person_id": "legacy-max", "tool": "run_command",
                                   "args": {}, "expires": time.monotonic() + 20}
        await connection._resolve_pin(text, voice=None, session=None, started_at=None,
                                      language="en", stt_ms=1, t_start=time.perf_counter())

    asyncio.run(answer("9999"))
    assert connection._pending_pin is not None, "a wrong PIN may be tried again"
    assert "again" in spoken[-1]
    asyncio.run(answer("8888"))
    asyncio.run(answer("7777"))
    assert connection._pending_pin is None
    assert "Too many wrong PINs" in spoken[-1]
    assert [row["result"] for row in audit.rows] == ["denied", "denied", "denied"]
    assert all(row["action"] == "confirm.pin" for row in audit.rows)


def test_an_expired_pin_question_does_nothing(hub_db, tmp_path, monkeypatch):
    _wire(monkeypatch, hub_db, tmp_path, BeliefStore(hub_db))
    monkeypatch.setattr(hub_app, "_pin", None)
    connection = _connection(hub_db, _room([]))
    connection.cfg = load_config("config.example.yaml")
    spoken: list[str] = []

    async def speak(voice, say_text, **kwargs):
        spoken.append(str(say_text))

    monkeypatch.setattr(connection, "_speak_confirmation", speak)
    connection._pending_pin = {"person_id": "legacy-max", "tool": "run_command",
                               "args": {}, "expires": time.monotonic() - 1}

    assert asyncio.run(connection._resolve_pin(
        "1 2 3 4", voice=None, session=None, started_at=None, language="en",
        stt_ms=1, t_start=time.perf_counter())) is True
    assert connection._pending_pin is None
    assert "ran out" in spoken[-1]
