"""The hub schema migrations apply once, in order, and roll back on failure."""
from __future__ import annotations

import sqlite3

import pytest

from hub import migrations_runner as runner

REQUIRED_TABLES = {
    "homes", "persons", "clients", "memberships", "contacts", "tracks",
    "body_crops",
    "identity_belief",
    "identity_labels",
    "daily_appearance",
    "voice_embeddings", "face_embeddings", "body_embeddings", "presence_events",
    "devices", "scenes", "rules", "skills", "skill_state", "dialog_turns",
    "memories", "reminders", "polls", "poll_answers", "objects_index",
    "decisions", "api_usage", "audit", "media", "schema_version",
}


def _tables(conn):
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_fresh_database_reaches_the_full_schema(tmp_path):
    conn = runner.connect(str(tmp_path / "hub.db"))
    try:
        assert runner.migrate(conn) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert REQUIRED_TABLES <= _tables(conn)
        assert [row[0] for row in conn.execute("SELECT version FROM schema_version")] == \
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        assert "observed" in columns
        assert "preset" in {row[1] for row in conn.execute("PRAGMA table_info(scenes)")}
    finally:
        conn.close()


def test_migrate_is_idempotent(tmp_path):
    conn = runner.connect(str(tmp_path / "hub.db"))
    try:
        runner.migrate(conn)
        assert runner.migrate(conn) == []
        assert len(list(conn.execute("SELECT * FROM schema_version"))) == 9
    finally:
        conn.close()


def test_foreign_keys_are_enforced(tmp_path):
    conn = runner.connect(str(tmp_path / "hub.db"))
    try:
        runner.migrate(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO clients(client_id, home_id, kind, token_hash) "
                "VALUES ('c1', 'missing', 'room_pc', 'x')"
            )
    finally:
        conn.close()


def test_failed_migration_rolls_back_and_stops(tmp_path):
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "0001_ok.py").write_text(
        "VERSION = 1\nNAME = 'ok'\ndef apply(conn):\n    conn.execute('CREATE TABLE first(a INTEGER)')\n",
        encoding="utf-8",
    )
    (directory / "0002_broken.py").write_text(
        "VERSION = 2\nNAME = 'broken'\ndef apply(conn):\n"
        "    conn.execute('CREATE TABLE second(a INTEGER)')\n"
        "    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    conn = runner.connect(str(tmp_path / "hub.db"))
    try:
        with pytest.raises(RuntimeError):
            runner.migrate(conn, directory)
        tables = _tables(conn)
        assert "first" in tables            # migration 1 committed
        assert "second" not in tables       # migration 2 rolled back
        assert [row[0] for row in conn.execute("SELECT version FROM schema_version")] == [1]
    finally:
        conn.close()
