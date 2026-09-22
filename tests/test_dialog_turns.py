"""P3-17 (F-414): dialogues are read from ``dialog_turns``, not from the archive.

ТЗ 9.4 keeps dialogues in the table of section 14; every turn is already
written there next to its readable jsonl twin. The reader here rebuilds the
question/answer pairs, selects a person by ``person_id`` and never invents a
history - and the flag ``server.memory.dialogs_from_db`` decides whether the
hub uses it or stays with the archive store it used before.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.config import Config, MemoryConfig
from hub import app, dialog_turns
from hub.homes import ensure_home
from hub.legacy_migrate import migrate_dialogs
from hub.migrations_runner import connect, migrate
from hub.session import Session
from hub.utterances import record_dialog_turns


def hub_db(tmp_path):
    """A real hub database with the schema of section 14, applied."""
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    return conn


def person(conn, name, person_id=None):
    """Register a person, the way the identity pipeline does."""
    conn.execute("INSERT OR IGNORE INTO persons(person_id, display_name) VALUES (?, ?)",
                 (person_id or f"p-{name.casefold()}", name))
    conn.commit()
    return person_id or f"p-{name.casefold()}"


def room(conn, home_id="livingroom"):
    ensure_home(conn, home_id, name=home_id)
    conn.commit()
    return home_id


def turn(conn, *, uid, question, answer, ts=1000.0, home="livingroom", speaker="Anton"):
    """One real turn of the table, written by the hub's own writer."""
    return record_dialog_turns(conn, home_id=home, utterance_id=uid, question=question,
                               answer=answer, ts=ts, speaker=speaker)


def stamp(base, seconds):
    return (base + timedelta(seconds=seconds)).isoformat(timespec="seconds")


BASE = datetime(2026, 9, 20, 19, 0, 0)


# --- the pairs of the table -------------------------------------------------


def test_a_live_turn_comes_back_as_a_question_and_its_answer(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        stored = turn(conn, uid="01ARZ3NDEKTSV4RRFFQ69G5FAV",
                      question="Where are my keys?", answer="In the top drawer.",
                      ts=BASE.timestamp())
        assert stored == ["01ARZ3NDEKTSV4RRFFQ69G5FAV:user",
                          "01ARZ3NDEKTSV4RRFFQ69G5FAV:assistant"]
        rows = dialog_turns.DialogTurns(tmp_path / "hub.db").recent("Anton")
        assert [(row["question"], row["answer"]) for row in rows] == [
            ("Where are my keys?", "In the top drawer.")]
        assert rows[0]["ts"] == "2026-09-20T19:00:00"
        assert rows[0]["person"] == "p-anton"
        assert rows[0]["id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV:user"
    finally:
        conn.close()


def test_the_history_walks_the_room_from_the_oldest_turn(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        for number in range(3):
            turn(conn, uid=f"u{number}", question=f"Question {number}.",
                 answer=f"Answer {number}.", ts=BASE.timestamp() + number)
        rows = dialog_turns.DialogTurns(tmp_path / "hub.db").recent("Anton")
        assert [row["question"] for row in rows] == [
            "Question 0.", "Question 1.", "Question 2."]
    finally:
        conn.close()


def test_a_question_nobody_answered_is_reported_as_interrupted(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        conn.execute("INSERT INTO dialog_turns(turn_id, home_id, person_id, role, text, ts)"
                     " VALUES ('t1:user', 'livingroom', 'p-anton', 'user', 'Are you there?', ?)",
                     (BASE.timestamp(),))
        conn.commit()
        rows = dialog_turns.DialogTurns(tmp_path / "hub.db").recent("Anton")
        assert rows[0]["answer"] == dialog_turns.INTERRUPTED
    finally:
        conn.close()


def test_a_persons_history_follows_them_into_another_room(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn, "livingroom")
        room(conn, "dorm-max")
        person(conn, "Anton")
        turn(conn, uid="a", question="In the living room.", answer="Yes.",
             ts=BASE.timestamp(), home="livingroom")
        turn(conn, uid="b", question="In Max's room.", answer="Also yes.",
             ts=BASE.timestamp() + 5, home="dorm-max")
        rows = dialog_turns.DialogTurns(tmp_path / "hub.db").recent("Anton")
        assert [row["question"] for row in rows] == ["In the living room.", "In Max's room."]
    finally:
        conn.close()


def test_one_persons_turns_never_come_back_for_another(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        person(conn, "Max")
        turn(conn, uid="a", question="Anton's question.", answer="Anton's answer.",
             ts=BASE.timestamp(), speaker="Anton")
        turn(conn, uid="b", question="Max's question.", answer="Max's answer.",
             ts=BASE.timestamp() + 5, speaker="Max")
        reader = dialog_turns.DialogTurns(tmp_path / "hub.db")
        assert [row["question"] for row in reader.recent("Anton")] == ["Anton's question."]
        assert [row["question"] for row in reader.recent("Max")] == ["Max's question."]
    finally:
        conn.close()


def test_an_unknown_name_has_no_history_at_all(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        turn(conn, uid="a", question="Hello.", answer="Hi.", ts=BASE.timestamp())
        reader = dialog_turns.DialogTurns(tmp_path / "hub.db")
        assert reader.recent("Nobody") == []
        assert reader.recall("Nobody", "hello") == []
        assert reader.recent("") == []
    finally:
        conn.close()


def test_recent_keeps_the_newest_turns_and_respects_the_limit(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        for number in range(6):
            turn(conn, uid=f"u{number}", question=f"Question {number}.",
                 answer=f"Answer {number}.", ts=BASE.timestamp() + number)
        reader = dialog_turns.DialogTurns(tmp_path / "hub.db")
        assert [row["question"] for row in reader.recent("Anton", 2)] == [
            "Question 4.", "Question 5."]
        assert len(reader.recent("Anton", 100)) == 6
    finally:
        conn.close()


def test_recall_ranks_by_terms_and_honours_the_dates(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        turn(conn, uid="a", question="Where are my keys?",
             answer="Keys, keys, keys: in the top drawer.", ts=BASE.timestamp())
        turn(conn, uid="b", question="What is the weather?",
             answer="Cold.", ts=BASE.timestamp() + 3600)
        turn(conn, uid="c", question="Did I take my keys to class?",
             answer="You did.", ts=BASE.timestamp() + 7200)
        reader = dialog_turns.DialogTurns(tmp_path / "hub.db")
        hits = reader.recall("Anton", "keys")
        assert [row["question"] for row in hits] == [
            "Did I take my keys to class?", "Where are my keys?"], "a tie goes to the newest"
        # Two terms: the turn that carries both wins over the one that carries
        # one, whatever the order of the rows.
        ranked = reader.recall("Anton", "keys drawer")
        assert [row["question"] for row in ranked] == [
            "Where are my keys?", "Did I take my keys to class?"]
        assert [row["question"] for row in reader.recall("Anton", "keys drawer", limit=1)] == [
            "Where are my keys?"]
        assert reader.recall("Anton", "weather")[0]["question"] == "What is the weather?"
        assert reader.recall("Anton", "keys", since=stamp(BASE, 3600)) == [
            row for row in hits if row["question"] == "Did I take my keys to class?"]
        assert reader.recall("Anton", "keys", until=stamp(BASE, 1800)) == [
            row for row in hits if row["question"] == "Where are my keys?"]
        assert reader.recall("Anton", "xylophone zebra") == []
    finally:
        conn.close()


def test_a_database_without_the_schema_is_an_empty_history(tmp_path):
    Path(tmp_path / "hub.db").write_bytes(b"")
    reader = dialog_turns.DialogTurns(tmp_path / "hub.db")
    assert reader.recent("Anton") == []
    assert reader.recall("Anton", "hello") == []


def test_the_pairing_model_is_strict():
    with pytest.raises(Exception):
        dialog_turns.Turn(turn_id="t", epoch=1.0, bogus=1)
    row = dialog_turns.Turn(turn_id="t", person_id="p", epoch=1.0, ts="", question="q",
                            answer="a").as_row()
    assert row == {"id": "t", "person": "p", "ts": "", "question": "q", "answer": "a"}


# --- the tail of the archive ------------------------------------------------


def _write_line(directory: Path, name: str, entry: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_the_tail_of_the_archive_is_imported_once(tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        _write_line(tmp_path / "dialogs", "2026-09-20.jsonl", {
            "ts": stamp(BASE, 0), "client_id": "livingroom", "transcript": "Old question",
            "speaker": "Anton", "reply": "Old answer"})
        assert migrate_dialogs(conn, tmp_path) == 2, "the tail is a question and an answer"
        assert migrate_dialogs(conn, tmp_path) == 0, "the second run changes nothing"
        reader = dialog_turns.DialogTurns(tmp_path / "hub.db")
        assert [(row["question"], row["answer"]) for row in reader.recent("Anton")] == [
            ("Old question", "Old answer")]
        assert conn.execute("SELECT count(*) FROM dialog_turns").fetchone()[0] == 2
    finally:
        conn.close()


def test_a_turn_the_table_already_has_is_not_imported_twice(tmp_path):
    """The jsonl line is the readable twin of the rows, not a second turn."""
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        uid = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        turn(conn, uid=uid, question="Where are my keys?", answer="In the top drawer.",
             ts=BASE.timestamp())
        _write_line(tmp_path / "dialogs", "2026-09-20.jsonl", {
            "ts": stamp(BASE, 0), "client_id": "livingroom", "utterance_id": uid,
            "transcript": "Where are my keys?", "speaker": "Anton",
            "reply": "In the top drawer."})
        assert migrate_dialogs(conn, tmp_path) == 0
        rows = dialog_turns.DialogTurns(tmp_path / "hub.db").recent("Anton")
        assert len(rows) == 1, "one turn in the table must stay one turn in history"
        # A line written by an older client (no utterance id) is still imported.
        _write_line(tmp_path / "dialogs", "2026-09-20.jsonl", {
            "ts": stamp(BASE, 60), "client_id": "livingroom",
            "transcript": "Older client", "speaker": "Anton", "reply": "Still stored"})
        assert migrate_dialogs(conn, tmp_path) == 2
        assert [row["question"] for row in
                dialog_turns.DialogTurns(tmp_path / "hub.db").recent("Anton")] == [
            "Where are my keys?", "Older client"]
    finally:
        conn.close()


def test_the_lookup_indexes_the_reader_needs_exist(tmp_path):
    conn = hub_db(tmp_path)
    try:
        names = {str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='dialog_turns'")}
        assert {"idx_dialog_turns_person", "idx_dialog_turns_utterance"} <= names
    finally:
        conn.close()


# --- the flag ---------------------------------------------------------------


def test_the_flag_chooses_the_reader(monkeypatch, tmp_path):
    cfg = Config()
    monkeypatch.setattr(app, "get_config", lambda: cfg)
    monkeypatch.setattr(app, "_hub_db_path", lambda: tmp_path / "hub.db")
    archive = SimpleNamespace()
    monkeypatch.setattr(app, "_conversations", archive)
    assert app._dialog_reader() is archive, "without the flag nothing changes"
    cfg.server.memory.dialogs_from_db = True
    reader = app._dialog_reader()
    assert isinstance(reader, dialog_turns.DialogTurns)
    assert reader.db_path.endswith("hub.db")


def test_the_recall_tool_reads_the_table_when_the_flag_is_on(monkeypatch, tmp_path):
    conn = hub_db(tmp_path)
    try:
        room(conn)
        person(conn, "Anton")
        turn(conn, uid="a", question="Where are my keys?",
             answer="In the top drawer.", ts=BASE.timestamp())
        cfg = Config()
        cfg.server.permissions_enabled = False
        cfg.server.identity.enabled = False
        cfg.server.memory.dialogs_from_db = True
        monkeypatch.setattr(app, "get_config", lambda: cfg)
        monkeypatch.setattr(app, "_hub_db_path", lambda: tmp_path / "hub.db")

        class ArchiveTrap:
            """Any use of the archive store with the flag on is a failure."""

            def __getattr__(self, name):
                raise AssertionError(f"the archive store was used ({name})")

        monkeypatch.setattr(app, "_conversations", ArchiveTrap())
        connection = app.Connection(SimpleNamespace(client=None), cfg)
        connection.home_id = "livingroom"
        connection.session = Session("room-pc", [], 8)
        connection._speaker_name, connection._speaker_role = "Anton", "admin"

        result = asyncio.run(connection._execute_tool(
            "recall_conversation", {"query": "keys", "person": "me"}))
        assert result["ok"] is True and result["person"] == "Anton"
        assert [(row["question"], row["answer"]) for row in result["exchanges"]] == [
            ("Where are my keys?", "In the top drawer.")]
    finally:
        conn.close()


def test_the_dialog_flag_is_off_by_default():
    assert MemoryConfig().dialogs_from_db is False
    assert Config().server.memory.dialogs_from_db is False
