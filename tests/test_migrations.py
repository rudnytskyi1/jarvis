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
    "device_state_events", "intercom_messages", "poll_asks", "guest_grants",
    "person_preferences", "person_scene_favourites",
    "push_subscriptions", "push_outbox",
    "digest_runs",
    "turn_events",
}


def _tables(conn):
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_fresh_database_reaches_the_full_schema(tmp_path):
    conn = runner.connect(str(tmp_path / "hub.db"))
    try:
        assert runner.migrate(conn) == list(range(1, 32))
        assert REQUIRED_TABLES <= _tables(conn)
        assert [row[0] for row in conn.execute("SELECT version FROM schema_version")] == \
            list(range(1, 32))
        # ТЗ F-704: отчёт дня перечисляет неполные ходы, а не только удачи.
        assert "degraded" in {row[1] for row in conn.execute(
            "PRAGMA table_info(dialog_turns)")}
        # ТЗ F-309: дом помнит отпечаток зон кадра, поэтому смена ТОЛЬКО
        # маски тоже будит комнату патчем.
        assert "zones_rev" in {row[1] for row in conn.execute("PRAGMA table_info(homes)")}
        columns = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        assert "observed" in columns
        assert "preset" in {row[1] for row in conn.execute("PRAGMA table_info(scenes)")}
        reminder_columns = {row[1] for row in conn.execute("PRAGMA table_info(reminders)")}
        assert {"delivery_state", "delivery_note", "attempts", "trigger_kind"} <= reminder_columns
        rule_columns = {row[1] for row in conn.execute("PRAGMA table_info(rules)")}
        assert {"name", "last_fired_at", "author_person_id"} <= rule_columns
        history_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(device_state_events)")}
        assert {"event_id", "device_id", "home_id", "capability", "value_json",
                "source", "ts"} <= history_columns
        object_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(objects_index)")}
        assert "zone" in object_columns
        contact_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(contacts)")}
        assert {"person_a", "person_b", "status", "created_at", "requested_by",
                "confirmed_at", "blocked_by", "share_presence_a",
                "share_presence_b"} <= contact_columns
        intercom_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(intercom_messages)")}
        assert {"message_id", "home_id", "origin_home", "from_person", "to_person",
                "text", "kind", "status", "created_at", "delivered_at",
                "reply_to", "pushed_at"} <= intercom_columns
        poll_columns = {row[1] for row in conn.execute("PRAGMA table_info(polls)")}
        assert {"options_json", "audience_json", "status", "closed_at",
                "summarized_at"} <= poll_columns
        answer_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(poll_answers)")}
        assert "home_id" in answer_columns
        ask_columns = {row[1] for row in conn.execute("PRAGMA table_info(poll_asks)")}
        assert {"poll_id", "person_id", "asked_at"} <= ask_columns
        grant_columns = {row[1] for row in conn.execute("PRAGMA table_info(guest_grants)")}
        assert {"grant_id", "home_id", "guest_person_id", "capability", "granted_by",
                "granted_at", "expires_at"} <= grant_columns
        pref_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(person_preferences)")}
        assert {"person_id", "language", "voice", "wake_word", "style",
                "updated_at"} <= pref_columns
        fav_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(person_scene_favourites)")}
        assert {"person_id", "name", "created_at"} <= fav_columns
        sub_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(push_subscriptions)")}
        assert {"subscription_id", "person_id", "kind", "endpoint", "keys_json",
                "created_at", "last_seen_at"} <= sub_columns
        outbox_columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(push_outbox)")}
        assert {"message_id", "person_id", "home_id", "kind", "title", "body",
                "created_at", "delivered_at", "state"} <= outbox_columns
        client_columns = {row[1] for row in conn.execute("PRAGMA table_info(clients)")}
        assert "person_id" in client_columns
    finally:
        conn.close()


def test_migrate_is_idempotent(tmp_path):
    conn = runner.connect(str(tmp_path / "hub.db"))
    try:
        runner.migrate(conn)
        assert runner.migrate(conn) == []
        assert len(list(conn.execute("SELECT * FROM schema_version"))) == 31
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
