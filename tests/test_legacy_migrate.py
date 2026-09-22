"""Legacy single-room store import into the hub database (ТЗ section 4.6)."""
from __future__ import annotations

import json
import struct

import pytest

from hub import migrations_runner
from hub.legacy_migrate import (
    LEGACY_HOME_ID,
    migrate_dialogs,
    migrate_legacy,
    migrate_memory,
    migrate_people,
    person_id_for,
)


def connect(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    return conn


def write_people(data_dir):
    people = {
        "voice_model": "speechbrain/spkrec-ecapa-voxceleb",
        "people": {
            "Drew": {"role": "admin", "voice_embeddings": [[0.1, -0.2, 0.3]], "face_embeddings": []},
            "John": {"role": "admin", "voice_embeddings": [], "face_embeddings": [[0.5, 0.6]]},
        },
    }
    (data_dir / "people.json").write_text(json.dumps(people), encoding="utf-8")


def test_migrate_people_imports_persons_roles_and_embeddings(tmp_path):
    conn = connect(tmp_path)
    write_people(tmp_path)
    try:
        assert migrate_people(conn, tmp_path) == 2
        assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 2
        assert conn.execute(
            "SELECT role FROM memberships WHERE person_id=? AND home_id=?",
            (person_id_for("Drew"), LEGACY_HOME_ID),
        ).fetchone()[0] == "admin"
        vector = conn.execute(
            "SELECT vector, dim FROM voice_embeddings WHERE person_id=?",
            (person_id_for("Drew"),),
        ).fetchone()
        assert vector[1] == 3
        assert struct.unpack("<3f", vector[0]) == pytest.approx((0.1, -0.2, 0.3))
        assert conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id=?",
            (person_id_for("John"),),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_migrate_memory_resolves_tombstones_and_scopes(tmp_path):
    conn = connect(tmp_path)
    (tmp_path / "memory.jsonl").write_text(
        "\n".join([
            json.dumps({"ts": "2026-09-17T03:11:13", "fact": "room fact"}),
            json.dumps({"ts": "2026-09-18T00:00:00", "person": "Anton", "fact": "pref",
                        "key": "apps.browser", "value": "Chrome", "author": "Anton"}),
            json.dumps({"ts": "2026-09-18T00:00:01", "person": "Anton", "fact": "old", "id": "abc"}),
            json.dumps({"_op": "edit", "target": "abc", "person": "Anton",
                        "fact": "new", "ts": "2026-09-18T00:00:02"}),
            json.dumps({"ts": "2026-09-18T00:00:03", "person": "Anton", "fact": "drop", "id": "xyz"}),
            json.dumps({"_op": "delete", "target": "xyz", "person": "Anton",
                        "ts": "2026-09-18T00:00:04"}),
        ]),
        encoding="utf-8",
    )
    try:
        assert migrate_memory(conn, tmp_path) == 3
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3
        assert conn.execute(
            "SELECT COUNT(*) FROM memories WHERE scope='home' AND owner_id=? AND text='room fact'",
            (LEGACY_HOME_ID,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT kind FROM memories WHERE scope='person' AND text='pref'",
        ).fetchone()[0] == "preference"
        # A legacy "fact" is stored as the typed kind it really is, so the
        # memory read never has to refuse a row (F-414).
        assert conn.execute(
            "SELECT kind FROM memories WHERE text='room fact'",
        ).fetchone()[0] == "home_fact"
        assert conn.execute(
            "SELECT kind FROM memories WHERE text='new'",
        ).fetchone()[0] == "person_fact"
        assert conn.execute(
            "SELECT COUNT(*) FROM memories WHERE text='old' OR text='drop'",
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM persons WHERE person_id=?",
            (person_id_for("Anton"),),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_migrate_dialogs_writes_user_and_assistant_turns(tmp_path):
    conn = connect(tmp_path)
    dialogs = tmp_path / "dialogs"
    dialogs.mkdir()
    (dialogs / "2026-09-20.jsonl").write_text(
        "\n".join([
            json.dumps({"ts": "2026-09-20T00:18:27", "client_id": "livingroom",
                        "transcript": "hello", "speaker": "Anton", "reply": "hi"}),
            json.dumps({"ts": "2026-09-20T00:18:40", "client_id": "livingroom",
                        "transcript": "who?", "speaker": "unknown", "reply": "you"}),
        ]),
        encoding="utf-8",
    )
    try:
        assert migrate_dialogs(conn, tmp_path) == 4
        rows = list(conn.execute(
            "SELECT role, text, person_id, home_id FROM dialog_turns ORDER BY ts",
        ))
        assert [row[0] for row in rows] == ["user", "assistant", "user", "assistant"]
        assert rows[0][2] == person_id_for("Anton")
        assert rows[2][2] is None
        assert rows[0][3] == LEGACY_HOME_ID
    finally:
        conn.close()


def test_migrate_legacy_is_idempotent_and_missing_files_are_safe(tmp_path):
    conn = connect(tmp_path)
    write_people(tmp_path)
    (tmp_path / "memory.jsonl").write_text(
        json.dumps({"ts": "2026-09-17T03:11:13", "fact": "room fact"}),
        encoding="utf-8",
    )
    try:
        first = migrate_legacy(conn, tmp_path)
        second = migrate_legacy(conn, tmp_path)
        assert first == {"people": 2, "memory": 1, "dialogs": 0}
        assert second == {"people": 2, "memory": 1, "dialogs": 0}
        assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 2
    finally:
        conn.close()
