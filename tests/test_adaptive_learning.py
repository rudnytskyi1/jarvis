"""Адаптивное дообучение профиля (ТЗ F-211).

The rules of the ТЗ are checked twice: as the pure decision of one vector (the
caps, the two similarity bars and the 0,9 confidence), and as the review of a
whole profile in the real hub database.
"""
from __future__ import annotations

import numpy as np
import pytest

from hub import migrations_runner
from hub.adaptive_learning import (
    BODY_PER_DAY,
    FACE_MAX_VECTORS,
    MIN_P,
    VOICE_MAX_VECTORS,
    AdaptiveLearning,
    Candidate,
    cap_for,
    caps_summary,
    cosine,
    decide,
)


def _vector(seed: int = 0, *, dim: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _axis(index: int, *, dim: int = 8, value: float = 1.0) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = value
    return vector


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    for person_id, name in (("p-max", "Макс"), ("p-drew", "Drew")):
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?, ?)",
                     (person_id, name))
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _store(conn, table: str, person_id: str, vector, *, track_id: str = "t1",
           day: str | None = None, quality: float | None = None, row_id: str | None = None) -> str:
    import uuid

    if track_id:
        conn.execute("INSERT OR IGNORE INTO tracks(track_id, home_id, first_seen, last_seen)"
                     " VALUES (?, 'livingroom', '2026-09-21T10:00:00', '2026-09-21T10:00:00')",
                     (track_id,))
    row = row_id or uuid.uuid4().hex
    values = np.asarray(vector, dtype=np.float32).ravel()
    if table == "body_embeddings":
        conn.execute("INSERT INTO body_embeddings(id, person_id, track_id, session_day, vector,"
                     " dim, quality) VALUES (?,?,?,?,?,?,?)",
                     (row, person_id, track_id, day, values.tobytes(), values.size, quality))
    else:
        conn.execute(f"INSERT INTO {table}(id, person_id, track_id, vector, dim, quality)"
                     " VALUES (?,?,?,?,?,?)",
                     (row, person_id, track_id, values.tobytes(), values.size, quality))
    conn.commit()
    return row


# --- the rules of F-211 -----------------------------------------------------


def test_the_limits_are_the_ones_the_spec_states():
    # ТЗ F-211: до 12 векторов лица, до 8 голоса, тело по дням, порог p ≥ 0,9.
    assert FACE_MAX_VECTORS == 12
    assert VOICE_MAX_VECTORS == 8
    assert MIN_P == 0.9
    assert cap_for("face") == 12 and cap_for("voice") == 8 and cap_for("body") == BODY_PER_DAY
    assert caps_summary() == {"face": 12, "voice": 8, "body": BODY_PER_DAY}


def test_a_cosine_is_a_measure_not_a_guess():
    assert cosine(_axis(0), _axis(0)) == pytest.approx(1.0)
    assert cosine(_axis(0), _axis(1)) == pytest.approx(0.0)
    assert cosine(_axis(0), _axis(0, value=-1.0)) == pytest.approx(-1.0)
    # Vectors that cannot be compared (different dimensions, junk) are not a match.
    assert cosine(_axis(0), _axis(0, dim=16)) == -1.0
    assert cosine([], _axis(0)) == -1.0
    assert cosine("nonsense", _axis(0)) == -1.0


def test_a_vector_of_an_unconfirmed_person_is_never_learned():
    verdict = decide(kind="face", p=0.85, vector=_axis(0))
    assert verdict.rejected and verdict.reason == "low_confidence"
    # 0,9 itself is enough: the boundary belongs to the learner.
    assert decide(kind="face", p=0.9, vector=_axis(0)).accepted


def test_a_vector_that_is_not_a_vector_is_refused():
    assert decide(kind="voice", p=0.99, vector=[]).reason == "invalid_vector"
    assert decide(kind="voice", p=0.99, vector="nonsense").reason == "invalid_vector"


def test_a_vector_indistinguishable_from_another_person_is_refused():
    other = _axis(1)
    verdict = decide(kind="face", p=0.99, vector=_axis(1), others=[other])
    assert verdict.rejected and verdict.reason == "conflict"
    # A clearly different direction is fine.
    assert decide(kind="face", p=0.99, vector=_axis(2), others=[other]).accepted


def test_the_persons_own_vector_is_not_stored_twice():
    own = _axis(3)
    assert decide(kind="voice", p=0.99, vector=_axis(3), own=[own]).reason == "duplicate"


def test_conflict_is_checked_before_the_limits():
    verdict = decide(kind="face", p=0.99, vector=_axis(1), others=[_axis(1)], kept=99)
    assert verdict.reason == "conflict"


def test_a_full_profile_replaces_its_oldest_vector():
    verdict = decide(kind="face", p=0.99, vector=_axis(4), kept=FACE_MAX_VECTORS,
                     own=[_axis(5)])
    assert verdict.accepted and verdict.reason == "replaced_oldest" and verdict.removed == 1
    assert decide(kind="voice", p=0.99, vector=_axis(4), kept=VOICE_MAX_VECTORS).removed == 1


def test_the_limits_come_from_the_config_when_it_says_so():
    assert decide(kind="face", p=0.99, vector=_axis(6), kept=3, cap=3).reason == "replaced_oldest"
    assert decide(kind="voice", p=0.99, vector=_axis(6), kept=2, cap=3).reason == "stored"


# --- the profile in the hub database ---------------------------------------


def test_a_new_vector_is_written_for_a_confirmed_person(hub_db):
    learner = AdaptiveLearning(hub_db)
    verdicts = learner.learn("p-max", 0.95, [Candidate("face", _axis(0), quality=0.9)])
    assert [item.reason for item in verdicts] == ["stored"]
    row = hub_db.execute("SELECT person_id, dim, quality FROM face_embeddings").fetchone()
    assert row == ("p-max", 8, 0.9)


def test_nobody_learns_for_a_person_that_does_not_exist(hub_db):
    learner = AdaptiveLearning(hub_db)
    assert learner.learn("ghost", 0.99, [Candidate("face", _axis(0))]) == []
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == 0


def test_a_low_confidence_vector_is_not_written_at_all(hub_db):
    learner = AdaptiveLearning(hub_db)
    verdicts = learner.learn("p-max", 0.7, [Candidate("voice", _axis(1))])
    assert [item.reason for item in verdicts] == ["low_confidence"]
    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0


def test_a_review_removes_a_vector_that_fits_somebody_else(hub_db):
    learner = AdaptiveLearning(hub_db)
    # The same direction as Drew's face, filed under Макс: too ambiguous to keep.
    _store(hub_db, "face_embeddings", "p-drew", _axis(0), track_id="t9")
    ambiguous = _store(hub_db, "face_embeddings", "p-max", _axis(0) * 0.99, track_id="t1")
    own = _store(hub_db, "face_embeddings", "p-max", _axis(2), track_id="t1")
    verdicts = learner.review("p-max")
    reasons = {item.reason for item in verdicts}
    assert "conflict" in reasons and "kept" in reasons
    left = {row[0] for row in hub_db.execute("SELECT id FROM face_embeddings")}
    assert ambiguous not in left and own in left
    assert "t9" in {row[0] for row in hub_db.execute("SELECT track_id FROM face_embeddings")}


def test_a_review_keeps_only_the_newest_vectors_of_each_kind(hub_db):
    learner = AdaptiveLearning(hub_db)
    ids = [_store(hub_db, "face_embeddings", "p-max", _axis(index % 8), row_id=f"f{index}")
           for index in range(14)]
    verdicts = learner.review("p-max")
    removed = [item for item in verdicts if item.removed]
    assert len(removed) == 14 - FACE_MAX_VECTORS
    assert all(item.reason == "surplus" for item in removed)
    left = {row[0] for row in hub_db.execute("SELECT id FROM face_embeddings ORDER BY rowid")}
    assert left == set(ids[-FACE_MAX_VECTORS:]), "the newest angles survive"


def test_the_body_is_capped_per_day_and_never_mixed_across_days(hub_db):
    learner = AdaptiveLearning(hub_db)
    for index in range(BODY_PER_DAY + 2):
        _store(hub_db, "body_embeddings", "p-max", _axis(index), day="2026-09-20",
               row_id=f"d{index}")
    _store(hub_db, "body_embeddings", "p-max", _axis(0), day="2026-09-21", row_id="today")
    learner.review("p-max")
    rows = hub_db.execute("SELECT id, session_day FROM body_embeddings ORDER BY rowid").fetchall()
    assert len(rows) == BODY_PER_DAY + 1
    assert ("today", "2026-09-21") in [(row[0], row[1]) for row in rows]
    days = {row[1] for row in rows}
    assert days == {"2026-09-20", "2026-09-21"}


def test_the_voice_limit_is_the_one_of_the_spec(hub_db):
    learner = AdaptiveLearning(hub_db)
    for index in range(VOICE_MAX_VECTORS + 3):
        _store(hub_db, "voice_embeddings", "p-max", _axis(index % 8), row_id=f"v{index}")
    learner.review("p-max")
    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings WHERE person_id='p-max'"
                          ).fetchone()[0] == VOICE_MAX_VECTORS


def test_a_review_of_a_clean_profile_changes_nothing(hub_db):
    learner = AdaptiveLearning(hub_db)
    kept = _store(hub_db, "face_embeddings", "p-max", _axis(3))
    _store(hub_db, "face_embeddings", "p-drew", _axis(5), track_id="t9")
    verdicts = learner.review("p-max")
    assert all(not item.removed for item in verdicts)
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == 2
    assert hub_db.execute("SELECT id FROM face_embeddings WHERE person_id='p-max'"
                          ).fetchone()[0] == kept


def test_a_review_without_a_person_or_without_vectors_is_a_no_op(hub_db):
    learner = AdaptiveLearning(hub_db)
    assert learner.review("") == []
    assert learner.review("ghost") == []


def test_the_profile_keeps_the_vectors_of_the_other_person_intact(hub_db):
    learner = AdaptiveLearning(hub_db)
    for index in range(FACE_MAX_VECTORS + 2):
        _store(hub_db, "face_embeddings", "p-max", _axis(index % 8), row_id=f"f{index}")
    drew = _store(hub_db, "face_embeddings", "p-drew", _axis(9 % 8), track_id="t9", row_id="drew")
    learner.review("p-max")
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings WHERE person_id='p-drew'"
                          ).fetchone()[0] == 1
    assert hub_db.execute("SELECT id FROM face_embeddings WHERE id='drew'").fetchone()[0] == drew


def test_the_track_profile_is_what_the_room_already_saw(hub_db):
    learner = AdaptiveLearning(hub_db)
    _store(hub_db, "face_embeddings", None, _axis(0), track_id="t1")
    _store(hub_db, "voice_embeddings", None, _axis(1), track_id="t1")
    _store(hub_db, "body_embeddings", None, _axis(2), track_id="t1", day="2026-09-21")
    _store(hub_db, "body_embeddings", None, _axis(3), track_id="t2", day="2026-09-21")
    candidates = learner.track_profile("t1", day="2026-09-21")
    assert {item.kind for item in candidates} == {"face", "voice", "body"}
    assert all(item.track_id == "t1" for item in candidates)
    assert [item.kind for item in candidates].count("body") == 1
    assert learner.track_profile("") == []


def test_a_summary_of_the_pass_is_json_able(hub_db):
    import json

    learner = AdaptiveLearning(hub_db)
    verdicts = learner.learn("p-max", 0.95, [Candidate("face", _axis(0)),
                                             Candidate("voice", [])])
    summary = learner.stamp(verdicts)
    assert summary["learned"] == 1 and summary["skipped"] == 1
    assert json.dumps(summary), "the summary has to survive a log line"


def test_the_caps_of_the_config_are_honoured(hub_db):
    learner = AdaptiveLearning(hub_db, face_max_vectors=2, body_per_day=1,
                               conflict_similarity=0.5, duplicate_similarity=0.99)
    for index in range(4):
        _store(hub_db, "face_embeddings", "p-max", _axis(index), row_id=f"f{index}")
    _store(hub_db, "body_embeddings", "p-max", _axis(4), day="2026-09-21", row_id="b1")
    _store(hub_db, "body_embeddings", "p-max", _axis(5), day="2026-09-21", row_id="b2")
    learner.review("p-max")
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == 2
    assert hub_db.execute("SELECT COUNT(*) FROM body_embeddings").fetchone()[0] == 1
    assert hub_db.execute("SELECT id FROM body_embeddings").fetchone()[0] == "b2"


def test_a_learned_vector_is_a_real_row_of_the_schema(hub_db):
    learner = AdaptiveLearning(hub_db)
    _store(hub_db, "face_embeddings", None, _axis(6), track_id="t4", row_id="known-track")
    hub_db.execute("DELETE FROM face_embeddings")
    learner.learn("p-max", 0.95, [Candidate("body", _axis(0), quality=0.5,
                                            track_id="t4", day="2026-09-21")])
    row = hub_db.execute("SELECT person_id, track_id, session_day, dim FROM body_embeddings"
                         ).fetchone()
    assert row == ("p-max", "t4", "2026-09-21", 8)
    # And the profile reads it back as a comparable vector.
    profile = learner.profile("p-max", "body", day="2026-09-21")
    assert len(profile) == 1 and cosine(profile[0][1], _axis(0)) == pytest.approx(1.0)


def test_a_vector_of_a_track_the_hub_never_stored_keeps_no_reference(hub_db):
    learner = AdaptiveLearning(hub_db)
    learner.learn("p-max", 0.95, [Candidate("face", _axis(0), track_id="ghost-track")])
    row = hub_db.execute("SELECT track_id FROM face_embeddings").fetchone()
    assert row == (None,), "a dangling track_id would break the foreign key"


# --- the hub (ТЗ F-211 in the running connection) ---------------------------


def _connection(hub_db, monkeypatch, *, learner=None, enabled: bool = True):
    """A room whose identity layer is wired to the real database."""
    from common.config import Config
    from hub import app as hub_app
    from hub.room_state import RoomState

    cfg = Config()
    cfg.server.identity.learning.enabled = enabled
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_learning", learner if learner is not None else False)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = RoomState()
    connection.cfg = cfg
    return connection


def test_the_hub_reviews_a_profile_at_the_adaptive_confidence(hub_db, monkeypatch):
    learner = AdaptiveLearning(hub_db)
    for index in range(FACE_MAX_VECTORS + 2):
        _store(hub_db, "face_embeddings", "p-max", _axis(index % 8), row_id=f"f{index}")
    connection = _connection(hub_db, monkeypatch, learner=learner)
    connection._review_profile("p-max", 0.95)
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == FACE_MAX_VECTORS


def test_the_hub_leaves_a_profile_alone_below_the_adaptive_confidence(hub_db, monkeypatch):
    learner = AdaptiveLearning(hub_db)
    for index in range(FACE_MAX_VECTORS + 2):
        _store(hub_db, "face_embeddings", "p-max", _axis(index % 8), row_id=f"f{index}")
    connection = _connection(hub_db, monkeypatch, learner=learner)
    # 0,8 is enough for the name of F-207, not for the profile of F-211.
    connection._review_profile("p-max", 0.8)
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == FACE_MAX_VECTORS + 2


def test_the_hub_reviews_nothing_when_learning_is_switched_off(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, enabled=False)
    connection._review_profile("p-max", 0.99)  # must not raise without a learner
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == 0


def test_a_confirmed_track_triggers_the_review(hub_db, monkeypatch):
    """The hook is in ``_confirm_track_identity``, not only in the method."""
    from hub.identity_fusion import Belief

    learner = AdaptiveLearning(hub_db)
    for index in range(FACE_MAX_VECTORS + 1):
        _store(hub_db, "face_embeddings", "p-max", _axis(index % 8), row_id=f"f{index}")
    connection = _connection(hub_db, monkeypatch, learner=learner)
    connection._identity_hysteresis = None
    belief = Belief(track_id="t1", person_id="p-max", p=0.95, sources={})
    connection._confirm_track_identity("t1", "p-max", belief)
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == FACE_MAX_VECTORS
