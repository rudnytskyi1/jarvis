"""«Забудь меня» (ТЗ F-213).

The point of the test suite is the same as the point of the feature: after the
deletion there has to be nothing left, and before the spoken "yes" there has to
be nothing deleted. Both halves run against the real schema, the real memory
store and the real dialogue archive.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any

import numpy as np
import pytest

from hub import migrations_runner
from hub.conversations import Conversations
from hub.forget_me import (
    ForgetReport,
    cancelled,
    confirmation,
    erase,
    forget_requested,
    inventory,
    question,
)
from hub.storage import Memory


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


class _Registry:
    """The phase-1 people registry, as far as F-213 uses it."""

    def __init__(self, people: tuple[str, ...] = ()) -> None:
        self.people = list(people)
        self.deleted: list[str] = []

    def admin_profile(self, action: str, name: str, **kwargs: Any) -> tuple[str, str]:
        if action != "delete" or name not in self.people:
            raise ValueError(f"no profile {name!r}")
        self.people.remove(name)
        self.deleted.append(name)
        return "user", "deleted"


class _Archive:
    def __init__(self, events: int = 3, folders: int = 2) -> None:
        self.answers = {"events": events, "folders": folders, "person_id": "p-max"}
        self.forgotten: list[str] = []

    def forget(self, person: str) -> dict[str, Any]:
        self.forgotten.append(person)
        return dict(self.answers)


def _vector(seed: int = 0, *, dim: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"\xff\xd8\xff")
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-other', 'Другой')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES"
                 " ('p-max', 'livingroom', 'admin')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES"
                 " ('p-other', 'livingroom', 'user')")
    conn.execute("INSERT INTO tracks(track_id, home_id, person_id, first_seen, last_seen)"
                 " VALUES ('t1', 'livingroom', 'p-max', 'now', 'now')")
    conn.execute("INSERT INTO tracks(track_id, home_id, person_id, first_seen, last_seen)"
                 " VALUES ('t2', 'livingroom', 'p-other', 'now', 'now')")
    for index, (person, track) in enumerate((("p-max", "t1"), ("p-max", "t1"),
                                            ("p-other", "t2"))):
        conn.execute("INSERT INTO voice_embeddings(id, person_id, track_id, vector, dim)"
                     " VALUES (?,?,?,?,?)",
                     (f"v{index}", person, track, _vector(index).tobytes(), 8))
    conn.execute("INSERT INTO face_embeddings(id, person_id, track_id, vector, dim)"
                 " VALUES ('f1','p-max','t1',?,8)", (_vector(3).tobytes(),))
    conn.execute("INSERT INTO body_embeddings(id, person_id, track_id, session_day, vector, dim)"
                 " VALUES ('b1','p-max','t1','2026-09-21',?,8)", (_vector(4).tobytes(),))
    conn.execute("INSERT INTO daily_appearance(person_id, day, home_id, vector, dim, samples,"
                 " expires_at) VALUES ('p-max','2026-09-20','livingroom',?,8,3,'2026-09-27')",
                 (_vector(5).tobytes(),))
    conn.execute("INSERT INTO presence_events(event_id, home_id, kind, person_id, ts)"
                 " VALUES ('e1','livingroom','person_entered','p-max',1.0)")
    conn.execute("INSERT INTO dialog_turns(turn_id, home_id, person_id, role, text, ts)"
                 " VALUES ('d1','livingroom','p-max','user','привет',1.0)")
    conn.execute("INSERT INTO body_crops(crop_id, home_id, client_id, track_id, ts, width,"
                 " height, path) VALUES ('c1','livingroom','pc-1','t1',1.0,0,640,?)",
                 (str(crop),))
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


# --- the words --------------------------------------------------------------


def test_the_request_is_recognised_in_three_languages():
    assert forget_requested("Rowan, забудь меня")
    assert forget_requested("Rowan AI, forget me")
    assert forget_requested("Rowan, olvídame")
    assert forget_requested("Rowan, delete all my data")
    assert not forget_requested("Rowan, включи свет")
    assert not forget_requested("")


def test_somebody_asking_not_to_forget_is_not_asking_to_forget():
    assert not forget_requested("Rowan, don't forget me")
    assert not forget_requested("Rowan, не забывай меня")
    assert not forget_requested("Rowan, no me olvides")


def test_the_question_names_what_goes_and_says_it_is_irreversible():
    asked = question("ru", window_s=8)
    assert "8" in asked and "необратимо" in asked
    assert "голос" in asked and "диалог" in asked
    assert "cannot be undone" in question("en")
    assert "No se puede deshacer" in question("es")
    assert cancelled("en").startswith("Alright")
    # The F-113 helper of the same window is what the hub holds open.
    request = confirmation("ru", window_s=8)
    assert request.window_s == 8 and not request.expired()


# --- what is there, and what is gone ---------------------------------------


def test_the_inventory_counts_the_person_before_anything_is_deleted(hub_db):
    counted = inventory(hub_db, "p-max")
    assert counted.vectors == 4, "two voice + one face + one body"
    assert counted.crops == 1 and counted.mentions == 2 and counted.appearances == 1
    assert counted.tracks == 1 and counted.memberships == 1
    assert counted.total == 9
    assert inventory(hub_db, "") == type(counted)()


def test_the_deletion_removes_every_row_of_the_person(hub_db, tmp_path):
    audit, archive = _Audit(), _Archive()
    memory = Memory(tmp_path)
    memory.add("Макс любит чай", "Макс")
    memory.add("Про комнату", "")
    conversations = Conversations(tmp_path)
    conversations.append("Макс", "2026-09-21T10:00:00", "привет", "привет")
    conversations.append("Другой", "2026-09-21T10:00:01", "привет", "привет")
    registry = _Registry(("Макс",))
    crop = tmp_path / "crop.jpg"
    assert crop.exists(), "the fixture wrote the crop the row points at"

    report = erase(hub_db, "p-max", memory=memory, conversations=conversations,
                   archive=archive, registry=registry, audit=audit, actor="p-max")

    assert report.ok and report.display_name == "Макс"
    assert (report.vectors, report.crops, report.mentions, report.appearances) == (4, 1, 2, 1)
    assert report.memory == 1 and report.conversations == 1
    assert report.memberships == 1 and report.registry is True
    assert report.archive == {"events": 3, "folders": 2, "person_id": "p-max"}
    assert not crop.exists(), "the crop file is gone too"
    # Nothing of Макс is left anywhere.
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE person_id='p-max'").fetchone()[0] == 0
    assert hub_db.execute("SELECT COUNT(*) FROM memberships WHERE person_id='p-max'").fetchone()[0] == 0
    for table in ("voice_embeddings", "face_embeddings", "body_embeddings", "daily_appearance",
                  "presence_events", "dialog_turns"):
        assert hub_db.execute(f"SELECT COUNT(*) FROM {table} WHERE person_id='p-max'"
                              ).fetchone()[0] == 0, table
    assert hub_db.execute("SELECT COUNT(*) FROM body_crops WHERE crop_id='c1'").fetchone()[0] == 0
    # And nothing of anybody ELSE was touched.
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE person_id='p-other'").fetchone()[0] == 1
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='t2'").fetchone()[0] == "p-other"
    assert conversations.recent("Другой")
    assert memory.facts("") and not memory.facts("Макс")


def test_the_memories_table_is_erased_with_the_person(hub_db, tmp_path):
    """The table reaches the prompt (P3-16) and the history (P3-17): a person
    who is forgotten must not survive in it - and the room's own facts stay."""
    hub_db.execute("INSERT INTO memories(memory_id, scope, owner_id, kind, text)"
                   " VALUES ('m1','person','макс','person_fact','Любит чай')")
    hub_db.execute("INSERT INTO memories(memory_id, scope, owner_id, kind, text)"
                   " VALUES ('m2','home','livingroom','home_fact','Чайник на столе')")
    hub_db.execute("INSERT INTO memories(memory_id, scope, owner_id, kind, text)"
                   " VALUES ('m3','person','Другой','person_fact','Любит кофе')")
    hub_db.commit()
    assert inventory(hub_db, "p-max").memory == 1, "counted before anything is deleted"
    report = erase(hub_db, "p-max", memory=Memory(tmp_path))
    assert report.memory == 1
    left = [row[0] for row in hub_db.execute("SELECT memory_id FROM memories ORDER BY memory_id")]
    assert left == ["m2", "m3"], "only the person's own facts are gone"


def test_a_track_stays_but_no_longer_names_anybody(hub_db):
    erase(hub_db, "p-max")
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='t1'").fetchone()[0] is None
    assert hub_db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 2


def test_the_deletion_is_audited_with_the_numbers(hub_db):
    audit = _Audit()
    erase(hub_db, "p-max", audit=audit, actor="p-max")
    assert audit.rows and audit.rows[0]["action"] == "identity.forget"
    assert audit.rows[0]["result"] == "ok" and audit.rows[0]["target"] == "p-max"
    assert audit.rows[0]["detail"]["vectors"] == 4


def test_a_person_without_an_id_is_a_no_op_that_says_so(hub_db):
    report = erase(hub_db, "")
    assert not report.ok and report.note == "no person"
    assert hub_db.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 2


def test_the_report_is_spoken_with_the_real_numbers():
    report = ForgetReport(person_id="p-max", display_name="Макс", vectors=4, crops=1,
                          mentions=2, conversations=1, memory=1)
    spoken = report.spoken("ru")
    assert "4" in spoken and "нельзя" in spoken
    assert "deleted everything" in report.spoken("en")
    assert json.dumps(report.summary())


def test_a_database_error_reports_a_failure_and_keeps_the_person(hub_db):
    class Broken:
        def execute(self, sql, *args):
            if "DELETE FROM voice_embeddings" in str(sql):
                raise sqlite3.OperationalError("disk I/O error")
            return hub_db.execute(sql, *args)

        def commit(self):
            hub_db.commit()

        def rollback(self):
            hub_db.rollback()

    report = erase(Broken(), "p-max")  # type: ignore[arg-type]
    assert not report.ok and "database error" in report.note
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE person_id='p-max'").fetchone()[0] == 1
    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings WHERE person_id='p-max'"
                          ).fetchone()[0] == 2


# --- the room ---------------------------------------------------------------


def _config():
    from common.config import Config

    return Config()


def _connection(hub_db, monkeypatch, *, speaker: str = "Макс"):
    from hub import app as hub_app
    from hub.room_state import RoomState

    audit = _Audit()
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_training_archive", None)
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_body_crop_store", lambda: None)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = RoomState()
    connection._speaker_name = speaker
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection._pending_forget = None
    connection.cfg = _config()
    # ``_speak_confirmation`` is the real room code: it holds the reply lock,
    # sends the bubble to the socket and closes the turn's metrics. Only the
    # things that leave the hub are stubbed, so the test sees the real line.
    connection._reply_lock = asyncio.Lock()
    connection.utterance_id = "utt-1"
    connection.spoken: list[str] = []
    connection.sent: list[dict[str, Any]] = []

    async def _say(voice, say_text, **kwargs):
        connection.spoken.append(str(say_text))

    async def _send_json(payload):
        connection.sent.append(payload)

    connection._stream_tts = _say
    connection.send_json = _send_json
    connection._finish_utterance = lambda **kwargs: None

    async def _log_dialog(*args, **kwargs):
        return None

    connection._log_dialog = _log_dialog
    return connection, audit


async def _capture_speech(voice, text, **kwargs):
    return None


def test_the_room_asks_before_it_deletes_anything(hub_db, monkeypatch):
    connection, _audit = _connection(hub_db, monkeypatch)
    turn = asyncio.run(connection._forget_turn("Rowan, забудь меня", "ru"))
    assert turn is not None and "необратимо" in turn
    assert connection._pending_forget is not None
    # Nothing is gone yet - the question is only open.
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE person_id='p-max'").fetchone()[0] == 1
    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 3


def test_a_stranger_saying_forget_me_changes_nothing(hub_db, monkeypatch):
    connection, _audit = _connection(hub_db, monkeypatch, speaker="")
    turn = asyncio.run(connection._forget_turn("Rowan, forget me", "en"))
    assert turn is not None and "did not recognise" in turn
    assert connection._pending_forget is None


def test_the_no_keeps_the_person_and_the_yes_deletes_them(hub_db, monkeypatch):
    connection, audit = _connection(hub_db, monkeypatch)
    spoke: list[str] = []

    async def capture(voice, text, **kwargs):
        spoke.append(text)

    connection._stream_tts = capture
    asyncio.run(connection._forget_turn("Rowan, забудь меня", "ru"))
    assert asyncio.run(connection._resolve_forget(
        "нет", voice=None, session=None, started_at=None, language="ru", stt_ms=0, t_start=0.0))
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE person_id='p-max'").fetchone()[0] == 1
    assert spoke and "ничего не удаляю" in spoke[-1]

    asyncio.run(connection._forget_turn("Rowan, забудь меня", "ru"))
    assert asyncio.run(connection._resolve_forget(
        "да", voice=None, session=None, started_at=None, language="ru", stt_ms=0, t_start=0.0))
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE person_id='p-max'").fetchone()[0] == 0
    assert spoke and "Готово" in spoke[-1]
    assert [row for row in audit.rows if row["action"] == "identity.forget"]


def test_an_expired_window_deletes_nothing(hub_db, monkeypatch):
    connection, _audit = _connection(hub_db, monkeypatch)
    spoke: list[str] = []

    async def capture(voice, text, **kwargs):
        spoke.append(text)

    connection._stream_tts = capture
    asyncio.run(connection._forget_turn("Rowan, забудь меня", "ru"))
    connection._pending_forget["confirmation"].opened_at -= 100.0
    assert asyncio.run(connection._resolve_forget(
        "да", voice=None, session=None, started_at=None, language="ru", stt_ms=0, t_start=0.0))
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE person_id='p-max'").fetchone()[0] == 1
    assert spoke and "window ran out" in spoke[-1]


def test_an_utterance_without_a_pending_question_is_not_an_answer(hub_db, monkeypatch):
    connection, _audit = _connection(hub_db, monkeypatch)
    assert not asyncio.run(connection._resolve_forget(
        "да", voice=None, session=None, started_at=None, language="ru", stt_ms=0, t_start=0.0))
    assert asyncio.run(connection._forget_turn("Rowan, включи свет", "ru")) is None
    assert connection._pending_forget is None
