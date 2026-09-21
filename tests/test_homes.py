"""Seeding the homes table from config (ТЗ sections 4.2, 4.7)."""
from __future__ import annotations

from hub import migrations_runner
from hub.homes import ensure_home, home_config_rev, sync_homes_from_config


def cfg_homes(**overrides):
    from common.config import HomeConfig, QuietHoursConfig

    data = dict(home_id="livingroom", name="Living room", tz="America/Chicago",
                owner_person_id="anton")
    data.update(overrides)
    home = HomeConfig(**{k: v for k, v in data.items() if k != "quiet"})
    if "quiet" in overrides:
        home.quiet_hours = QuietHoursConfig(**overrides["quiet"])
    return home


def connect(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('anton', 'Anton')")
    conn.commit()
    return conn


def test_ensure_home_is_idempotent(tmp_path):
    conn = connect(tmp_path)
    try:
        ensure_home(conn, "a", name="First")
        ensure_home(conn, "a", name="Second")
        row = conn.execute("SELECT name FROM homes WHERE home_id='a'").fetchone()
        assert row[0] == "First", "an existing room must never be overwritten by a seed"
    finally:
        conn.close()


def test_sync_creates_then_updates_and_bumps_the_revision(tmp_path):
    conn = connect(tmp_path)
    try:
        assert sync_homes_from_config(conn, [cfg_homes()]) == ["livingroom"]
        assert home_config_rev(conn, "livingroom") == 1
        assert sync_homes_from_config(conn, [cfg_homes()]) == [], "an unchanged config changes nothing"
        assert sync_homes_from_config(conn, [cfg_homes(name="Big room")]) == ["livingroom"]
        assert home_config_rev(conn, "livingroom") == 2
        assert conn.execute("SELECT name FROM homes WHERE home_id='livingroom'").fetchone()[0] == "Big room"
    finally:
        conn.close()


def test_a_room_removed_from_config_is_kept(tmp_path):
    conn = connect(tmp_path)
    try:
        sync_homes_from_config(conn, [cfg_homes()])
        sync_homes_from_config(conn, [])
        assert conn.execute("SELECT COUNT(*) FROM homes").fetchone()[0] == 1
    finally:
        conn.close()


def test_an_unknown_owner_is_stored_as_null_instead_of_crashing(tmp_path):
    conn = connect(tmp_path)
    try:
        sync_homes_from_config(conn, [cfg_homes(owner_person_id="not-registered")])
        assert conn.execute("SELECT owner_person_id FROM homes WHERE home_id='livingroom'").fetchone()[0] is None
    finally:
        conn.close()
