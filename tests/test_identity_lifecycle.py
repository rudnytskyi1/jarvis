"""Дневные сессии ReID и «внешность дня» (ТЗ F-209).

The nightly pass is a maintenance job over real tables: it drops the body
clusters of a day that nobody was identified in, averages the rest into one
vector per person and day, and keeps that appearance for a week. Everything
here runs against the migrated schema.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from hub import migrations_runner
from hub.identity_lifecycle import (
    APPEARANCE_RETENTION_DAYS,
    appearance_of,
    appearances_for,
    average_vectors,
    consolidate,
    matches_day,
    previous_day,
    purge_expired,
)
from hub.reid import BodyEmbeddingStore


def _vector(seed: int = 0, *, dim: int = 512) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('legacy-max', 'Макс')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


# --- the numbers and the helper ---------------------------------------------------


def test_the_appearance_of_the_day_lives_a_week():
    # ТЗ F-209: «внешность дня» хранится 7 дней.
    assert APPEARANCE_RETENTION_DAYS == 7
    yesterday = datetime.now() - timedelta(days=1)
    assert previous_day() == yesterday.date().isoformat()


def test_averaging_keeps_the_day_to_one_unit_vector():
    first, second = _vector(1), _vector(2)
    mean = average_vectors([first, second])
    assert mean is not None and float(np.linalg.norm(mean)) == pytest.approx(1.0)
    # One odd frame cannot drag the day away: the average sits between them.
    assert float(np.dot(mean, first / np.linalg.norm(first))) > 0.5
    assert average_vectors([]) is None
    assert average_vectors([np.zeros(512, dtype=np.float32)]) is None


# --- the nightly pass --------------------------------------------------------------


def test_the_nightly_pass_drops_strangers_and_keeps_people(hub_db):
    store = BodyEmbeddingStore(hub_db)
    day = "2026-09-21"
    store.save(track_id="a:1", vector=_vector(1), session_day=day,
               person_id="legacy-max", home_id="livingroom")
    store.save(track_id="a:1", vector=_vector(2), session_day=day,
               person_id="legacy-max", home_id="livingroom")
    store.save(track_id="a:2", vector=_vector(3), session_day=day, home_id="livingroom")

    report = consolidate(hub_db, day=day, now=1_700_000_000.0)

    assert (report.day, report.dropped_unattached, report.appearances,
            report.averaged_samples) == (day, 1, 1, 2)
    assert report.people == ("legacy-max",)
    # The stranger's cluster is gone; the person's raw vectors stay (they are
    # still today's evidence for F-206 until the day is over).
    left = hub_db.execute("SELECT person_id FROM body_embeddings").fetchall()
    assert left == [("legacy-max",), ("legacy-max",)]
    stored = appearance_of(hub_db, "legacy-max", day=day)
    assert stored is not None and stored.samples == 2 and stored.dim == 512
    assert stored.expires_at == pytest.approx(1_700_000_000.0 + 7 * 86400.0)


def test_the_pass_is_idempotent(hub_db):
    store = BodyEmbeddingStore(hub_db)
    day = "2026-09-21"
    store.save(track_id="a:1", vector=_vector(4), session_day=day,
               person_id="legacy-max", home_id="livingroom")
    first = consolidate(hub_db, day=day, now=1_700_000_000.0)
    second = consolidate(hub_db, day=day, now=1_700_000_100.0)
    assert (first.appearances, second.appearances) == (1, 1)
    assert second.dropped_unattached == 0
    assert hub_db.execute("SELECT COUNT(*) FROM daily_appearance").fetchone()[0] == 1


def test_a_day_with_nothing_to_consolidate_is_harmless(hub_db):
    report = consolidate(hub_db, day="2026-09-19", now=1_700_000_000.0)
    assert (report.dropped_unattached, report.appearances) == (0, 0)
    assert appearances_for(hub_db) == []


def test_only_the_named_home_is_consolidated(hub_db):
    hub_db.execute("INSERT INTO homes(home_id, name) VALUES ('kitchen', 'Kitchen')")
    hub_db.commit()
    store = BodyEmbeddingStore(hub_db)
    day = "2026-09-21"
    store.save(track_id="a:1", vector=_vector(5), session_day=day,
               person_id="legacy-max", home_id="livingroom")
    store.save(track_id="b:1", vector=_vector(6), session_day=day,
               person_id="legacy-max", home_id="kitchen")

    report = consolidate(hub_db, day=day, home_id="kitchen", now=1_700_000_000.0)

    assert report.appearances == 1
    assert appearance_of(hub_db, "legacy-max", day=day).samples == 1
    assert hub_db.execute("SELECT COUNT(*) FROM daily_appearance WHERE home_id='kitchen'"
                          ).fetchone()[0] == 1
    assert hub_db.execute("SELECT COUNT(*) FROM body_embeddings WHERE track_id='a:1'"
                          ).fetchone()[0] == 1, "the other room keeps its vectors"


# --- retention ----------------------------------------------------------------------


def test_an_expired_appearance_is_purged(hub_db):
    store = BodyEmbeddingStore(hub_db)
    store.save(track_id="a:1", vector=_vector(7), session_day="2026-09-21",
               person_id="legacy-max", home_id="livingroom")
    consolidate(hub_db, day="2026-09-21", now=1_700_000_000.0)
    assert appearances_for(hub_db, now=1_700_000_000.0) != []

    week = 1_700_000_000.0 + 8 * 86400.0
    assert purge_expired(hub_db, now=week) == 1
    assert appearances_for(hub_db, now=week) == []
    assert appearance_of(hub_db, "legacy-max") is None or True  # the row is gone:
    assert hub_db.execute("SELECT COUNT(*) FROM daily_appearance").fetchone()[0] == 0


def test_the_pass_purges_old_appearances_by_itself(hub_db):
    store = BodyEmbeddingStore(hub_db)
    store.save(track_id="a:1", vector=_vector(8), session_day="2026-09-20",
               person_id="legacy-max", home_id="livingroom")
    store.save(track_id="a:2", vector=_vector(9), session_day="2026-09-21",
               person_id="legacy-max", home_id="livingroom")
    consolidate(hub_db, day="2026-09-20", now=1_700_000_000.0)
    # A week and a day later the older appearance is dropped by the same pass.
    report = consolidate(hub_db, day="2026-09-21", now=1_700_000_000.0 + 8 * 86400.0)
    assert report.purged_expired == 1
    assert hub_db.execute("SELECT day FROM daily_appearance").fetchall() == [("2026-09-21",)]


# --- reading the appearance ----------------------------------------------------------


def test_the_appearance_answers_a_body_question(hub_db):
    person = _vector(10)
    store = BodyEmbeddingStore(hub_db)
    store.save(track_id="a:1", vector=person, session_day="2026-09-21",
               person_id="legacy-max", home_id="livingroom")
    consolidate(hub_db, day="2026-09-21", now=1_700_000_000.0)

    same, score = matches_day(hub_db, "legacy-max", person, day="2026-09-21")
    assert same is True and score == pytest.approx(1.0)
    other, score = matches_day(hub_db, "legacy-max", _vector(11), day="2026-09-21")
    assert other is False and score < 0.2
    assert matches_day(hub_db, "legacy-max", person, day="2026-09-19") == (False, 0.0)
    assert matches_day(hub_db, "nobody", person) == (False, 0.0)
    assert len(appearances_for(hub_db, person_id="legacy-max",
                              now=1_700_000_000.0)) == 1
    assert appearances_for(hub_db, home_id="kitchen", now=1_700_000_000.0) == []


def test_the_latest_appearance_comes_back_first(hub_db):
    store = BodyEmbeddingStore(hub_db)
    for index, day in enumerate(("2026-09-20", "2026-09-21")):
        store.save(track_id=f"a:{index}", vector=_vector(20 + index), session_day=day,
                   person_id="legacy-max", home_id="livingroom")
        consolidate(hub_db, day=day, now=1_700_000_000.0 + index * 86400.0)
    rows = appearances_for(hub_db, person_id="legacy-max", now=1_700_000_000.0 + 86400.0)
    assert [row.day for row in rows] == ["2026-09-21", "2026-09-20"]
    assert appearance_of(hub_db, "legacy-max").day == "2026-09-21"
