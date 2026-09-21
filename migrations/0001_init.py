"""0001 - initial hub schema (ТЗ section 14).

Vectors are stored as float32 little-endian BLOBs in ordinary tables, so the hub
runs without the sqlite-vec extension. When sqlite-vec is available a later
migration adds matching virtual tables keyed by these rows' primary keys; the
metadata columns here stay authoritative either way.
"""
from __future__ import annotations

import sqlite3

VERSION = 1
NAME = "init"

_TABLES = {
    "homes": """
        CREATE TABLE homes (
            home_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            tz TEXT NOT NULL DEFAULT 'America/Chicago',
            quiet_hours_json TEXT NOT NULL DEFAULT '{"start": "", "end": ""}',
            owner_person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            settings_json TEXT NOT NULL DEFAULT '{}',
            config_rev INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "persons": """
        CREATE TABLE persons (
            person_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            preferred_language TEXT,
            settings_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "clients": """
        CREATE TABLE clients (
            client_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('room_pc', 'phone', 'sensor_node')),
            token_hash TEXT NOT NULL,
            caps_json TEXT NOT NULL DEFAULT '{}',
            version TEXT,
            hw TEXT,
            last_seen TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "memberships": """
        CREATE TABLE memberships (
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('admin', 'trusted', 'user', 'guest')),
            share_identity INTEGER NOT NULL DEFAULT 0,
            share_presence INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (person_id, home_id)
        )""",
    "contacts": """
        CREATE TABLE contacts (
            person_a TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            person_b TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'blocked')),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (person_a, person_b),
            CHECK (person_a <> person_b)
        )""",
    "tracks": """
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            client_id TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            p REAL,
            sources_json TEXT NOT NULL DEFAULT '{}'
        )""",
    "voice_embeddings": """
        CREATE TABLE voice_embeddings (
            id TEXT PRIMARY KEY,
            person_id TEXT REFERENCES persons(person_id) ON DELETE CASCADE,
            track_id TEXT REFERENCES tracks(track_id) ON DELETE SET NULL,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            quality REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "face_embeddings": """
        CREATE TABLE face_embeddings (
            id TEXT PRIMARY KEY,
            person_id TEXT REFERENCES persons(person_id) ON DELETE CASCADE,
            track_id TEXT REFERENCES tracks(track_id) ON DELETE SET NULL,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            quality REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "body_embeddings": """
        CREATE TABLE body_embeddings (
            id TEXT PRIMARY KEY,
            person_id TEXT REFERENCES persons(person_id) ON DELETE CASCADE,
            track_id TEXT REFERENCES tracks(track_id) ON DELETE SET NULL,
            session_day TEXT,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            quality REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "presence_events": """
        CREATE TABLE presence_events (
            event_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            track_id TEXT,
            ts REAL NOT NULL,
            zone TEXT,
            media_ref TEXT
        )""",
    "devices": """
        CREATE TABLE devices (
            device_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            zone TEXT,
            kind TEXT NOT NULL,
            capabilities_json TEXT NOT NULL DEFAULT '[]',
            adapter TEXT NOT NULL,
            adapter_config_json TEXT NOT NULL DEFAULT '{}',
            state_json TEXT NOT NULL DEFAULT '{}',
            restricted INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT
        )""",
    "scenes": """
        CREATE TABLE scenes (
            scene_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            steps_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "rules": """
        CREATE TABLE rules (
            rule_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            trigger_json TEXT NOT NULL,
            conditions_json TEXT NOT NULL DEFAULT '{}',
            actions_json TEXT NOT NULL DEFAULT '[]',
            enabled INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "skills": """
        CREATE TABLE skills (
            skill_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL CHECK (scope IN ('hub', 'home')),
            home_id TEXT REFERENCES homes(home_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            version TEXT NOT NULL DEFAULT '0.1.0',
            enabled INTEGER NOT NULL DEFAULT 0,
            manifest_json TEXT NOT NULL DEFAULT '{}'
        )""",
    "skill_state": """
        CREATE TABLE skill_state (
            skill_id TEXT NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
            home_id TEXT NOT NULL DEFAULT '',
            key TEXT NOT NULL,
            value_json TEXT NOT NULL DEFAULT 'null',
            PRIMARY KEY (skill_id, home_id, key)
        )""",
    "dialog_turns": """
        CREATE TABLE dialog_turns (
            turn_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            utterance_id TEXT,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            ts REAL NOT NULL,
            model TEXT,
            tokens INTEGER
        )""",
    "memories": """
        CREATE TABLE memories (
            memory_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL CHECK (scope IN ('person', 'home', 'hub')),
            owner_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            vector BLOB,
            dim INTEGER,
            weight REAL NOT NULL DEFAULT 1.0,
            expires_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "reminders": """
        CREATE TABLE reminders (
            reminder_id TEXT PRIMARY KEY,
            person_id TEXT REFERENCES persons(person_id) ON DELETE CASCADE,
            home_id TEXT REFERENCES homes(home_id) ON DELETE CASCADE,
            due_at TEXT NOT NULL,
            text TEXT NOT NULL,
            delivered_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "polls": """
        CREATE TABLE polls (
            poll_id TEXT PRIMARY KEY,
            author_person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            home_id TEXT,
            question TEXT NOT NULL,
            deadline TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "poll_answers": """
        CREATE TABLE poll_answers (
            poll_id TEXT NOT NULL REFERENCES polls(poll_id) ON DELETE CASCADE,
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            answer TEXT NOT NULL,
            at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (poll_id, person_id)
        )""",
    "objects_index": """
        CREATE TABLE objects_index (
            id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            ts REAL NOT NULL,
            label TEXT NOT NULL,
            bbox_json TEXT NOT NULL DEFAULT '[]',
            vector BLOB,
            dim INTEGER,
            media_ref TEXT
        )""",
    "decisions": """
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            provider TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            value_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            latency_ms INTEGER NOT NULL,
            outcome TEXT,
            at REAL NOT NULL
        )""",
    "api_usage": """
        CREATE TABLE api_usage (
            request_id TEXT PRIMARY KEY,
            month TEXT NOT NULL,
            model TEXT NOT NULL,
            amount_micro INTEGER NOT NULL,
            settled INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER,
            output_tokens INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    "audit": """
        CREATE TABLE audit (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            actor_person_id TEXT,
            home_id TEXT,
            action TEXT NOT NULL,
            target TEXT,
            result TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}'
        )""",
    "media": """
        CREATE TABLE media (
            media_ref TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            kind TEXT NOT NULL,
            ts REAL NOT NULL,
            expires_at TEXT
        )""",
}

_INDEXES = (
    "CREATE INDEX idx_clients_home ON clients(home_id)",
    "CREATE INDEX idx_memberships_home ON memberships(home_id)",
    "CREATE INDEX idx_tracks_home ON tracks(home_id)",
    "CREATE INDEX idx_devices_home ON devices(home_id)",
    "CREATE INDEX idx_scenes_home ON scenes(home_id)",
    "CREATE INDEX idx_rules_home ON rules(home_id)",
    "CREATE INDEX idx_dialog_turns_ts ON dialog_turns(home_id, ts)",
    "CREATE INDEX idx_objects_index_home ON objects_index(home_id)",
    "CREATE INDEX idx_media_home ON media(home_id)",
    "CREATE INDEX idx_voice_embeddings_person ON voice_embeddings(person_id)",
    "CREATE INDEX idx_face_embeddings_person ON face_embeddings(person_id)",
    "CREATE INDEX idx_body_embeddings_person ON body_embeddings(person_id)",
    "CREATE INDEX idx_body_embeddings_day ON body_embeddings(session_day)",
    "CREATE INDEX idx_presence_events_ts ON presence_events(home_id, ts)",
    "CREATE INDEX idx_memories_scope ON memories(scope, owner_id)",
    "CREATE INDEX idx_decisions_type ON decisions(type, at)",
    "CREATE INDEX idx_audit_ts ON audit(ts)",
    "CREATE INDEX idx_reminders_due ON reminders(due_at)",
)


def apply(conn: sqlite3.Connection) -> None:
    for statement in _TABLES.values():
        conn.execute(statement)
    for statement in _INDEXES:
        conn.execute(statement)
