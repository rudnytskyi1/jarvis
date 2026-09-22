"""Ручная разметка неопознанных треков (ТЗ F-216).

The queue and the label are checked against the real schema: the point of the
feature is that a click changes what the hub knows (the track gets a person,
the samples of that day stop carrying nobody) AND leaves the label behind as
data for the thresholds of 15.6.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pytest

from hub import migrations_runner
from hub.labelling import (
    LabelResult,
    crop_file,
    day_of,
    day_window,
    label,
    labels,
    queue,
    summary,
)

DAY = "2026-09-21"
OTHER_DAY = "2026-09-20"


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


def _vector(seed: int) -> bytes:
    return np.random.default_rng(seed).standard_normal(8).astype(np.float32).tobytes()


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    crop_dir = tmp_path / "homes" / "livingroom" / "body"
    crop_dir.mkdir(parents=True)
    crop = crop_dir / "t-unknown-1.jpg"
    crop.write_bytes(b"\xff\xd8\xff")
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-drew', 'Drew')")
    # Неопознанный трек сегодняшнего дня, неопознанный вчерашний и уже названный.
    for track_id, seen, person in (("t-unknown-1", f"{DAY}T10:00:00", None),
                                   ("t-unknown-old", f"{OTHER_DAY}T10:00:00", None),
                                   ("t-known", f"{DAY}T11:00:00", "p-drew")):
        conn.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen,"
                     " person_id) VALUES (?,?,?,?,?,?)",
                     (track_id, "livingroom", "pc-1", seen, seen, person))
    start, _end = day_window(DAY)
    conn.execute("INSERT INTO body_crops(crop_id, home_id, client_id, track_id, ts, width,"
                 " height, path) VALUES ('c1','livingroom','pc-1','t-unknown-1',?,0,640,?)",
                 (start + 3600, str(crop)))
    conn.execute("INSERT INTO face_embeddings(id, person_id, track_id, vector, dim, quality)"
                 " VALUES ('f1',NULL,'t-unknown-1',?,8,0.7)", (_vector(1),))
    conn.execute("INSERT INTO body_embeddings(id, person_id, track_id, session_day, vector,"
                 " dim, quality) VALUES ('b1',NULL,'t-unknown-1',?,?,8,0.6)",
                 (DAY, _vector(2)))
    conn.execute("INSERT INTO body_embeddings(id, person_id, track_id, session_day, vector,"
                 " dim, quality) VALUES ('b-old',NULL,'t-unknown-1',?,?,8,0.5)",
                 (OTHER_DAY, _vector(3)))
    conn.execute("INSERT INTO voice_embeddings(id, person_id, track_id, vector, dim)"
                 " VALUES ('v1',NULL,'t-unknown-1',?,8)", (_vector(4),))
    conn.execute("INSERT INTO identity_belief(track_id, home_id, person_id, p, sources_json, at)"
                 " VALUES ('t-unknown-1','livingroom',NULL,0.41,?,123.0)",
                 (json.dumps({"voice": 0.52, "p": 0.41}),))
    conn.commit()
    try:
        yield conn, crop
    finally:
        conn.close()


def test_the_day_helpers_agree_with_the_clock():
    assert day_of("2026-09-21T10:00:00") == DAY
    assert day_of(datetime(2026, 9, 21, 23, 30)) == DAY
    start, end = day_window(DAY)
    assert day_of(start + 60) == DAY and end - start == 86400.0


# --- очередь ---------------------------------------------------------------


def test_the_queue_holds_the_unrecognized_tracks_of_the_day(hub_db):
    conn, _crop = hub_db
    found = queue(conn, day=DAY)
    assert [track.track_id for track in found] == ["t-unknown-1"]
    track = found[0]
    assert track.home_id == "livingroom" and track.faces == 1 and track.bodies == 2
    assert track.voices == 1 and track.quality == pytest.approx(0.7)
    assert [crop.crop_id for crop in track.crops] == ["c1"]
    assert track.belief["p"] == pytest.approx(0.41)
    assert track.belief["sources"]["voice"] == pytest.approx(0.52)
    assert json.dumps(track.summary())


def test_a_named_track_and_another_day_stay_out_of_the_queue(hub_db):
    conn, _crop = hub_db
    today = [track.track_id for track in queue(conn, day=DAY)]
    assert "t-known" not in today
    yesterday = [track.track_id for track in queue(conn, day=OTHER_DAY)]
    assert yesterday == ["t-unknown-old"]


def test_the_queue_can_be_limited_and_can_skip_the_crops(hub_db):
    conn, _crop = hub_db
    assert len(queue(conn, day=DAY, limit=1)) == 1
    assert queue(conn, day=DAY, with_crops=False)[0].crops == ()
    assert queue(conn, day=DAY, home_id="somewhere-else") == []
    assert queue(conn, day=DAY, home_id="livingroom")[0].track_id == "t-unknown-1"


# --- клик ------------------------------------------------------------------


def test_a_click_names_the_track_links_the_samples_and_keeps_the_label(hub_db):
    conn, _crop = hub_db
    audit = _Audit()
    result = label(conn, "t-unknown-1", "p-max", day=DAY, actor="p-max",
                   crop_id="c1", audit=audit, now=777.0)

    assert result.ok and result.display_name == "Макс"
    assert (result.faces_linked, result.bodies_linked, result.voices_linked) == (1, 1, 1)
    assert conn.execute("SELECT person_id FROM tracks WHERE track_id='t-unknown-1'"
                        ).fetchone()[0] == "p-max"
    assert conn.execute("SELECT person_id FROM face_embeddings WHERE id='f1'"
                        ).fetchone()[0] == "p-max"
    assert conn.execute("SELECT person_id FROM voice_embeddings WHERE id='v1'"
                        ).fetchone()[0] == "p-max"
    assert conn.execute("SELECT person_id FROM body_embeddings WHERE id='b1'"
                        ).fetchone()[0] == "p-max"
    # Вчерашняя одежда к сегодняшней метке отношения не имеет.
    assert conn.execute("SELECT person_id FROM body_embeddings WHERE id='b-old'"
                        ).fetchone()[0] is None
    kept = labels(conn, day=DAY)
    assert len(kept) == 1 and kept[0]["person_id"] == "p-max" and kept[0]["crop_id"] == "c1"
    assert kept[0]["actor"] == "p-max" and kept[0]["at"] == 777.0
    assert audit.rows and audit.rows[0]["action"] == "identity.label"
    assert audit.rows[0]["detail"]["faces_linked"] == 1
    # Метка убирает трек из очереди: он больше не неопознанный.
    assert queue(conn, day=DAY) == []


def test_the_owner_can_correct_a_label_and_both_clicks_stay_as_data(hub_db):
    conn, _crop = hub_db
    assert label(conn, "t-unknown-1", "p-max", day=DAY).ok
    second = label(conn, "t-unknown-1", "p-drew", day=DAY, actor="p-max")
    assert second.ok and second.display_name == "Drew"
    assert conn.execute("SELECT person_id FROM tracks WHERE track_id='t-unknown-1'"
                        ).fetchone()[0] == "p-drew"
    assert len(labels(conn, day=DAY)) == 2, "обе метки — данные для порогов"
    assert summary(conn)["per_person"]["p-max"] == 1


def test_a_click_about_nobody_known_changes_nothing(hub_db):
    conn, _crop = hub_db
    missing_track = label(conn, "t-nope", "p-max")
    assert not missing_track.ok and missing_track.note == "no such track"
    missing_person = label(conn, "t-unknown-1", "p-nope", day=DAY)
    assert not missing_person.ok and missing_person.note == "no such person"
    assert conn.execute("SELECT person_id FROM tracks WHERE track_id='t-unknown-1'"
                        ).fetchone()[0] is None
    assert labels(conn) == []


def test_the_label_can_leave_the_samples_alone(hub_db):
    conn, _crop = hub_db
    result = label(conn, "t-unknown-1", "p-max", day=DAY, link=False)
    assert result.ok and result.faces_linked == 0
    assert conn.execute("SELECT person_id FROM face_embeddings WHERE id='f1'"
                        ).fetchone()[0] is None
    assert conn.execute("SELECT person_id FROM tracks WHERE track_id='t-unknown-1'"
                        ).fetchone()[0] == "p-max"


def test_the_labels_are_the_data_of_the_fifteenth_sixth_set(hub_db):
    conn, _crop = hub_db
    assert summary(conn) == {"total": 0, "per_day": {}, "per_person": {}}
    label(conn, "t-unknown-1", "p-max", day=DAY, now=100.0)
    label(conn, "t-unknown-old", "p-max", day=OTHER_DAY, now=200.0)
    counted = summary(conn)
    assert counted["total"] == 2 and counted["per_day"][DAY] == 1
    assert counted["per_person"] == {"p-max": 2}
    assert summary(conn, day=OTHER_DAY)["total"] == 1
    newest_first = [row["track_id"] for row in labels(conn, person_id="p-max")]
    assert newest_first == ["t-unknown-old", "t-unknown-1"]
    assert isinstance(LabelResult(True), LabelResult)


# --- кроп ------------------------------------------------------------------


def test_the_panel_may_show_the_crop_that_lives_under_the_data_directory(hub_db, tmp_path):
    conn, crop = hub_db
    assert crop_file(conn, "c1", tmp_path) == crop
    assert crop_file(conn, "nope", tmp_path) is None
    assert crop_file(conn, "c1", tmp_path / "camera") is None, "чужой корень - не наш файл"
    conn.execute("UPDATE body_crops SET path=? WHERE crop_id='c1'",
                 (str(tmp_path.parent / "outside.jpg"),))
    assert crop_file(conn, "c1", tmp_path) is None
