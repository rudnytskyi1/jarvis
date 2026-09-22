"""ТЗ F-304: the media TTL task in the hub's scheduler and its audit report."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub.audit import AuditLog
from hub.media import MediaStore, MediaTtlTask
from hub.scheduler import Job, Scheduler

NOW = datetime.fromisoformat("2026-09-21T12:00:00+00:00")
OLD = (NOW - timedelta(days=10)).timestamp()
FRESH = (NOW - timedelta(hours=1)).timestamp()


def connect(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    return conn


def fixed_clock():
    return lambda: NOW


def store_for(conn, tmp_path, **kwargs):
    return MediaStore(conn, tmp_path, media_ttl_days=3, clip_ttl_days=7,
                      clock=fixed_clock(), **kwargs)


def test_the_task_expires_frames_and_crops_by_media_ttl_and_clips_by_seven_days(tmp_path):
    conn = connect(tmp_path)
    try:
        store = store_for(conn, tmp_path)
        _, old_crop = store.save_bytes("livingroom", "crop", b"crop", ts=OLD)
        _, old_clip = store.save_bytes("livingroom", "clip", b"clip", ts=OLD)
        _, fresh_frame = store.save_bytes("livingroom", "frame", b"frame", ts=FRESH)
        # Клип в четыре дня старше TTL кадров, но младше семи дней ТЗ 15.4.
        four_days = (NOW - timedelta(days=4)).timestamp()
        _, kept_clip = store.save_bytes("kitchen", "clip", b"clip", ts=four_days)

        report = MediaTtlTask(store, audit=AuditLog(conn), clock=fixed_clock()).run()

        assert report["expired_rows"] == 2
        assert report["deleted_files"] == 2
        assert report["media_ttl_days"] == 3
        assert report["clip_ttl_days"] == 7
        assert not old_crop.exists()
        assert not old_clip.exists()
        assert fresh_frame.exists()
        assert kept_clip.exists()
        left = sorted(Path(row[0]).name for row in conn.execute("SELECT path FROM media"))
        assert left == sorted([fresh_frame.name, kept_clip.name])
    finally:
        conn.close()


def test_the_pass_writes_one_audit_row_with_what_it_removed(tmp_path):
    conn = connect(tmp_path)
    try:
        store = store_for(conn, tmp_path)
        audit = AuditLog(conn)
        store.save_bytes("livingroom", "frame", b"old", ts=OLD)
        store.save_bytes("livingroom", "clip", b"old", ts=OLD)

        MediaTtlTask(store, audit=audit, clock=fixed_clock()).run()

        events = audit.events(action=MediaTtlTask.NAME)
        assert len(events) == 1
        assert events[0]["result"] == "ok"
        assert events[0]["target"] == "media"
        detail = events[0]["detail"]
        assert detail["expired_rows"] == 2
        assert detail["deleted_files"] == 2
        assert detail["media_ttl_days"] == 3
        assert detail["clip_ttl_days"] == 7
    finally:
        conn.close()


def test_a_pass_that_removed_nothing_leaves_the_audit_alone(tmp_path):
    conn = connect(tmp_path)
    try:
        store = store_for(conn, tmp_path)
        audit = AuditLog(conn)
        store.save_bytes("livingroom", "frame", b"fresh", ts=FRESH)

        report = MediaTtlTask(store, audit=audit, clock=fixed_clock()).run()

        assert report["expired_rows"] == 0
        assert audit.events() == []
    finally:
        conn.close()


def test_the_task_keeps_embeddings_and_presence_events(tmp_path):
    """ТЗ 15.4: «эмбеддинги и события — до „забудь меня“»."""
    conn = connect(tmp_path)
    try:
        store = store_for(conn, tmp_path)
        store.save_bytes("livingroom", "frame", b"old", ts=OLD)
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
        conn.execute("INSERT INTO body_embeddings(id, person_id, vector, dim) "
                     "VALUES ('b1', 'p-max', X'0000', 1)")
        conn.execute("INSERT INTO face_embeddings(id, person_id, vector, dim) "
                     "VALUES ('f1', 'p-max', X'00', 1)")
        conn.execute("INSERT INTO presence_events(event_id, home_id, kind, person_id, ts) "
                     "VALUES ('e1', 'livingroom', 'person_entered', 'p-max', ?)", (OLD,))
        conn.commit()

        MediaTtlTask(store, audit=AuditLog(conn), clock=fixed_clock()).run()

        assert conn.execute("SELECT count(*) FROM media").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM body_embeddings").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM face_embeddings").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM presence_events").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_file_outside_the_home_media_root_is_reported_but_not_deleted(tmp_path):
    conn = connect(tmp_path)
    try:
        store = store_for(conn, tmp_path)
        outside = tmp_path / "somewhere-else.jpg"
        outside.write_bytes(b"not ours")
        store.save_bytes("livingroom", "frame", b"old", ts=OLD)
        conn.execute("INSERT INTO media(media_ref, home_id, path, kind, ts, expires_at) "
                     "VALUES ('stray', 'livingroom', ?, 'frame', ?, '2020-01-01T00:00:00+00:00')",
                     (str(outside), OLD))
        conn.commit()

        report = MediaTtlTask(store, audit=AuditLog(conn), clock=fixed_clock()).run()

        assert report["expired_rows"] == 2
        assert report["deleted_files"] == 1
        assert report["unsafe_skipped"] == 1
        assert outside.read_bytes() == b"not ours"
    finally:
        conn.close()


def test_a_broken_database_becomes_a_failed_report_instead_of_a_crash(tmp_path, monkeypatch):
    conn = connect(tmp_path)
    try:
        conn.execute("DROP TABLE media")
        conn.commit()
        audit = AuditLog(conn)
        monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
        task = MediaTtlTask(store_for(conn, tmp_path), audit=audit, clock=fixed_clock())

        scheduler = Scheduler(on_error=hub_app._scheduled_job_failed)
        scheduler.add(Job(name=MediaTtlTask.NAME, interval_s=60, run=task.run))

        assert asyncio.run(scheduler.run_once(MediaTtlTask.NAME)) is None
        assert scheduler.failures(MediaTtlTask.NAME) == 1
        events = audit.events(action=MediaTtlTask.NAME)
        assert [entry["result"] for entry in events] == ["failed"]
        assert "media" in events[0]["detail"]["error"]
    finally:
        conn.close()


# --- the hub schedules the task ---------------------------------------------


def hub_with_media(tmp_path, monkeypatch, cfg):
    conn = connect(tmp_path)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_media", None)
    monkeypatch.setattr(hub_app, "_audit", AuditLog(conn))
    return conn, hub_app._media_ttl_scheduler(cfg)


def test_the_hub_schedules_the_media_ttl_task(tmp_path, monkeypatch):
    conn, scheduler = hub_with_media(tmp_path, monkeypatch, Config())
    try:
        assert scheduler is not None
        jobs = scheduler.jobs()
        assert [job.name for job in jobs] == [MediaTtlTask.NAME]
        assert jobs[0].interval_s == 3600.0

        # Просроченное, записанное в тот же корень data, что видит хаб.
        MediaStore(conn, tmp_path / "data", clock=fixed_clock()).save_bytes(
            "livingroom", "crop", b"old", ts=OLD)
        report = asyncio.run(scheduler.run_once(MediaTtlTask.NAME))
        assert report["expired_rows"] == 1
        assert conn.execute("SELECT count(*) FROM media").fetchone()[0] == 0
        assert AuditLog(conn).events(action=MediaTtlTask.NAME)[0]["result"] == "ok"
    finally:
        conn.close()


def test_the_interval_comes_from_the_config(tmp_path, monkeypatch):
    cfg = Config.model_validate({"server": {"media": {"cleanup_interval_s": 120}}})
    conn, scheduler = hub_with_media(tmp_path, monkeypatch, cfg)
    try:
        assert scheduler is not None
        assert scheduler.jobs()[0].interval_s == 120.0
    finally:
        conn.close()


def test_zero_interval_keeps_only_the_startup_pass(tmp_path, monkeypatch):
    cfg = Config.model_validate({"server": {"media": {"cleanup_interval_s": 0}}})
    conn, scheduler = hub_with_media(tmp_path, monkeypatch, cfg)
    try:
        assert scheduler is None
    finally:
        conn.close()


def test_a_hub_without_its_database_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_gateway", None)
    monkeypatch.setattr(hub_app, "_media", None)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    assert hub_app._media_ttl_scheduler(Config()) is None


def test_the_health_endpoint_publishes_the_schedule(monkeypatch):
    scheduler = Scheduler()
    scheduler.add(Job(name=MediaTtlTask.NAME, interval_s=3600.0, run=lambda: {}))
    monkeypatch.setattr(hub_app, "_scheduler", scheduler)

    assert asyncio.run(hub_app.health())["scheduler"] == [{
        "name": MediaTtlTask.NAME,
        "interval_s": 3600.0,
        "threaded": False,
        "running": False,
        "runs": 0,
        "failures": 0,
        "last_report": {},
    }]

    monkeypatch.setattr(hub_app, "_scheduler", None)
    assert hub_app._scheduler_snapshot() == []
