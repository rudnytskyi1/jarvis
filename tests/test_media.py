"""Room media storage and TTL cleanup (ТЗ 4.6, F-304)."""
from __future__ import annotations

from datetime import datetime

import pytest

from common.config import Config
from hub import migrations_runner
from hub.media import MediaStore


def connect(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    return conn


def fixed_clock(iso="2026-09-21T12:00:00+00:00"):
    return lambda: datetime.fromisoformat(iso)


def test_default_media_config_and_home_id_validation():
    cfg = Config.model_validate({})
    assert cfg.server.media.media_ttl_days == 3
    assert cfg.server.media.clip_ttl_days == 7
    with pytest.raises(ValueError):
        MediaStore.validate_home_id("../escape")


def test_save_bytes_writes_to_dated_home_dir_and_registers_row(tmp_path):
    conn = connect(tmp_path)
    store = MediaStore(conn, tmp_path, clock=fixed_clock())
    try:
        ref, path = store.save_bytes(
            "livingroom",
            "frame",
            b"jpeg",
            ts=datetime.fromisoformat("2026-09-21T12:00:00+00:00").timestamp(),
        )
        assert path == tmp_path / "homes" / "livingroom" / "media" / "2026-09-21" / path.name
        assert path.read_bytes() == b"jpeg"
        row = conn.execute(
            "SELECT media_ref, home_id, kind, expires_at FROM media WHERE media_ref=?",
            (ref,),
        ).fetchone()
        assert row[0] == ref
        assert row[1] == "livingroom"
        assert row[2] == "frame"
        assert row[3] == "2026-09-24T12:00:00+00:00"
    finally:
        conn.close()


def test_clips_use_seven_day_ttl(tmp_path):
    conn = connect(tmp_path)
    store = MediaStore(conn, tmp_path, clock=fixed_clock())
    try:
        ref, _ = store.save_bytes(
            "livingroom",
            "clip",
            b"mp4",
            ts=datetime.fromisoformat("2026-09-21T12:00:00+00:00").timestamp(),
        )
        row = conn.execute("SELECT expires_at FROM media WHERE media_ref=?", (ref,)).fetchone()
        assert row[0] == "2026-09-28T12:00:00+00:00"
    finally:
        conn.close()


def test_register_refuses_paths_outside_the_home_media_root(tmp_path):
    conn = connect(tmp_path)
    store = MediaStore(conn, tmp_path, clock=fixed_clock())
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    try:
        with pytest.raises(ValueError):
            store.register("livingroom", "frame", outside)
    finally:
        conn.close()


def test_cleanup_deletes_only_expired_files_and_rows(tmp_path):
    conn = connect(tmp_path)
    store = MediaStore(conn, tmp_path, clock=fixed_clock())
    now = datetime.fromisoformat("2026-09-21T12:00:00+00:00")
    old_ts = datetime.fromisoformat("2026-09-01T12:00:00+00:00").timestamp()
    current_ts = now.timestamp()
    try:
        _, old_path = store.save_bytes("livingroom", "frame", b"old", ts=old_ts)
        current_ref, current_path = store.save_bytes("livingroom", "frame", b"new", ts=current_ts)
        assert old_path.exists()

        result = store.cleanup_expired(now=now)

        assert result["expired_rows"] == 1
        assert result["deleted_files"] == 1
        assert not old_path.exists()
        assert current_path.exists()
        rows = list(conn.execute("SELECT media_ref FROM media ORDER BY media_ref"))
        assert [row[0] for row in rows] == [current_ref]
    finally:
        conn.close()


def test_unknown_media_kind_is_rejected(tmp_path):
    conn = connect(tmp_path)
    store = MediaStore(conn, tmp_path, clock=fixed_clock())
    try:
        with pytest.raises(ValueError):
            store.save_bytes("livingroom", "movie", b"x")
    finally:
        conn.close()
