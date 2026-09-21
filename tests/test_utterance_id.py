"""One utterance, one id: frames, logs, database rows and metrics (ТЗ 4.5)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common import protocol as proto
from common.config import Config
from common.ids import is_ulid, new_ulid
from hub import app as hub_app
from hub import main as hub_main
from hub import migrations_runner as runner
from hub.session import Session
from hub.utterances import UtteranceMetrics, record_dialog_turns, resolve_person_id


@pytest.fixture(autouse=True)
def fresh_metrics(monkeypatch):
    """Every test gets its own registry instead of the hub-wide one."""
    metrics = UtteranceMetrics()
    monkeypatch.setattr(hub_app, "_utterance_metrics", metrics)
    return metrics


# --- the client mints the id ---------------------------------------------


class _FakeVad:
    def __init__(self, audio: bytes) -> None:
        self.audio = audio

    async def record(self, _read_frame, pre_roll=None, lead_in_s=None, on_audio=None):
        if on_audio is not None:
            await on_audio(self.audio)
        return self.audio


def _room_client(audio: bytes):
    from client.main import JarvisClient

    client = JarvisClient.__new__(JarvisClient)
    client.sample_rate = 16000
    client.frame_bytes = 3200
    client.ws = SimpleNamespace(send_json=AsyncMock(), send_bytes=AsyncMock(), connected=True)
    client._wire_lock = asyncio.Lock()
    client._enrollment_until = 0.0
    client._recording_live = False
    client._live_turn_id = ""
    client.enrollment_vad = SimpleNamespace()
    client.vad = _FakeVad(audio)
    client._enter_conversation = Mock()
    client._leave_conversation = Mock()
    client._beep = AsyncMock()
    client._receive_response = AsyncMock(return_value="ok")
    return client


def test_the_client_sends_one_ulid_on_every_frame_of_the_utterance():
    async def scenario():
        client = _room_client(b"\0\1" * 1600)
        await client._handle_utterance(b"", 0.4)
        return client

    client = asyncio.run(scenario())
    frames = [call.args[0] for call in client.ws.send_json.call_args_list]
    start = next(frame for frame in frames if frame["type"] == proto.MSG_UTTERANCE_START)
    end = next(frame for frame in frames if frame["type"] == proto.MSG_UTTERANCE_END)

    assert is_ulid(start["utterance_id"])
    assert end["utterance_id"] == start["utterance_id"] == client._live_turn_id
    # The audio is still one unbroken stream between start and end.
    assert client.ws.send_bytes.await_count == 1


def test_two_utterances_never_share_an_id():
    async def scenario():
        ids = []
        for _ in range(2):
            client = _room_client(b"\0\1" * 1600)
            await client._handle_utterance(b"", 0.4)
            ids.append(client._live_turn_id)
        return ids

    first, second = asyncio.run(scenario())
    assert first != second
    assert is_ulid(first) and is_ulid(second)


# --- the hub adopts it ----------------------------------------------------


def _connection(**attributes):
    conn = hub_app.Connection(SimpleNamespace(client=None), Config())
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.send_json = AsyncMock()
    for name, value in attributes.items():
        setattr(conn, name, value)
    return conn


def test_the_hub_keeps_the_id_the_client_sent(fresh_metrics):
    conn = _connection()
    announced = new_ulid()
    conn._on_utterance_start({"utterance_id": announced, "sr": 16000})

    assert conn.utterance_id == announced
    assert fresh_metrics.active_id() == announced
    assert fresh_metrics.snapshot()["total"] == 1


def test_a_v1_client_without_an_id_gets_one_from_the_hub(fresh_metrics):
    conn = _connection()
    conn._on_utterance_start({"sr": 16000})

    assert is_ulid(conn.utterance_id)
    assert fresh_metrics.active_id() == conn.utterance_id


def test_the_hub_counts_the_room_the_utterance_came_from(fresh_metrics):
    conn = _connection()
    conn.home_id = "livingroom"
    conn._on_utterance_start({"sr": 16000})

    assert fresh_metrics.snapshot()["by_home"] == {"livingroom": 1}


def test_the_id_is_stamped_on_the_log_records_of_the_turn(caplog):
    conn = _connection()
    announced = new_ulid()
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        conn._on_utterance_start({"utterance_id": announced, "sr": 16000})

    record = next(item for item in caplog.records if "started by" in item.getMessage())
    assert record.utterance_id == announced


def test_the_plain_log_format_always_has_the_field():
    handler = logging.StreamHandler()
    handler.addFilter(hub_main.UtteranceIdFilter())
    handler.setFormatter(logging.Formatter(hub_main.LOG_FORMAT, datefmt=hub_main.LOG_DATE_FORMAT))
    record = logging.LogRecord("jarvis.server.app", logging.INFO, __file__, 1, "hello", (), None)

    assert handler.filter(record) is record
    assert record.utterance_id == "-"
    assert "-] jarvis.server.app: hello" in handler.format(record)


def test_the_reply_and_the_action_frames_carry_the_id(monkeypatch, fresh_metrics):
    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(
        generate=AsyncMock(side_effect=AssertionError("the model must not be called")),
        verify=AsyncMock(side_effect=AssertionError("no verifier")),
    ))
    monkeypatch.setattr(hub_app, "_stt",
                        SimpleNamespace(transcribe_pcm=lambda *a: ("volume 30", "en")))
    monkeypatch.setattr(hub_app, "_tts", object())
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app.speaker_mod, "check_permission",
                        lambda *a, **k: "Permission denied for this speaker.")

    async def scenario():
        conn = _connection()
        conn.utterance_id = new_ulid()
        conn.send_json = AsyncMock()
        conn._stream_tts = AsyncMock()
        conn._log_dialog = AsyncMock()
        conn._run_client_action = AsyncMock(side_effect=AssertionError("denied"))
        fresh_metrics.started(conn.utterance_id, home_id="livingroom")
        await conn._handle_utterance(b"\0" * 1600)
        return conn

    conn = asyncio.run(scenario())
    transcript = next(call.args[0] for call in conn.send_json.call_args_list
                      if call.args[0]["type"] == proto.MSG_TRANSCRIPT)
    said = next(call.args[0] for call in conn.send_json.call_args_list
                if call.args[0]["type"] == proto.MSG_SAY)
    assert transcript["utterance_id"] == conn.utterance_id
    assert said["utterance_id"] == conn.utterance_id

    trace = fresh_metrics.last()
    assert trace is not None and trace["utterance_id"] == conn.utterance_id
    assert trace["ok"] is True
    assert set(trace["stages_ms"]) == {"stt", "llm", "tts", "total"}
    assert trace["stages_ms"]["total"] >= trace["stages_ms"]["stt"]


def test_a_failed_utterance_is_marked_failed(fresh_metrics):
    conn = _connection()
    conn.utterance_id = new_ulid()

    async def scenario():
        fresh_metrics.started(conn.utterance_id)
        await hub_app.Connection._process_utterance(conn, b"\0" * 1600)

    # No STT engine is loaded, so the turn fails inside _handle_utterance.
    asyncio.run(scenario())
    trace = fresh_metrics.last()
    assert trace is not None and trace["ok"] is False
    assert fresh_metrics.snapshot()["failed"] == 1


# --- the durable record ---------------------------------------------------


def _db(tmp_path):
    conn = runner.connect(str(tmp_path / "hub.db"))
    runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    return conn


def test_the_turn_is_stored_in_dialog_turns_under_the_utterance_id(tmp_path):
    conn = _db(tmp_path)
    try:
        utterance_id = new_ulid()
        stored = record_dialog_turns(conn, home_id="livingroom", utterance_id=utterance_id,
                                     question="turn the light off", answer="Done.", ts=1.5)
        assert stored == [f"{utterance_id}:user", f"{utterance_id}:assistant"]
        rows = conn.execute(
            "SELECT role, text, utterance_id, ts FROM dialog_turns ORDER BY role"
        ).fetchall()
        assert rows == [("assistant", "Done.", utterance_id, 1.5),
                        ("user", "turn the light off", utterance_id, 1.5)]
    finally:
        conn.close()


def test_storing_the_same_utterance_twice_does_not_duplicate_it(tmp_path):
    conn = _db(tmp_path)
    try:
        utterance_id = new_ulid()
        arguments = dict(home_id="livingroom", utterance_id=utterance_id,
                         question="hi", answer="hello", ts=2.0)
        record_dialog_turns(conn, **arguments)
        record_dialog_turns(conn, **arguments)
        count = conn.execute("SELECT COUNT(*) FROM dialog_turns").fetchone()[0]
        assert count == 2
    finally:
        conn.close()


def test_an_unknown_room_is_skipped_instead_of_breaking_the_turn(tmp_path):
    conn = _db(tmp_path)
    try:
        assert record_dialog_turns(conn, home_id="nowhere", utterance_id=new_ulid(),
                                   question="hi", answer="hello", ts=1.0) == []
        assert conn.execute("SELECT COUNT(*) FROM dialog_turns").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_stranger_speaker_is_stored_without_a_person_row(tmp_path):
    conn = _db(tmp_path)
    try:
        assert resolve_person_id(conn, "Nobody") is None
        record_dialog_turns(conn, home_id="livingroom", utterance_id=new_ulid(),
                            question="hi", answer="hello", ts=1.0, person_id="missing-person")
        assert conn.execute("SELECT person_id FROM dialog_turns").fetchone()[0] is None
        assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_known_speaker_is_linked_to_their_person_row(tmp_path):
    conn = _db(tmp_path)
    try:
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
        conn.commit()
        assert resolve_person_id(conn, "anton") == "p-anton"
        record_dialog_turns(conn, home_id="livingroom", utterance_id=new_ulid(),
                            question="hi", answer="hello", ts=1.0,
                            person_id=resolve_person_id(conn, "Anton"))
        row = conn.execute("SELECT person_id FROM dialog_turns WHERE role='user'").fetchone()
        assert row[0] == "p-anton"
    finally:
        conn.close()


def test_the_hub_writes_the_turns_it_answers(monkeypatch, tmp_path):
    path = tmp_path / "hub.db"
    conn = runner.connect(str(path))
    runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.commit()
    try:
        # The archive opens its own connection in the worker thread; the hub
        # connection itself only tells us that the database is in play.
        monkeypatch.setattr(hub_app, "_hub_conn", SimpleNamespace())
        monkeypatch.setattr(hub_app, "_hub_db_path", lambda: path)
        room = _connection()
        room.home_id = "livingroom"
        room.utterance_id = new_ulid()
        room._speaker_name = "Anton"

        asyncio.run(room._store_dialog_turns(datetime(2026, 9, 21, 10, 0, 0),
                                             "hello there", "Hi!"))
        rows = conn.execute(
            "SELECT role, text, utterance_id, person_id FROM dialog_turns ORDER BY role"
        ).fetchall()
        assert rows == [("assistant", "Hi!", room.utterance_id, "p-anton"),
                        ("user", "hello there", room.utterance_id, "p-anton")]
    finally:
        conn.close()


def test_health_publishes_the_utterance_traces(fresh_metrics):
    fresh_metrics.started("01ARZ3NDEKTSV4RRFFQ69G5FAV", home_id="livingroom")
    fresh_metrics.finished("01ARZ3NDEKTSV4RRFFQ69G5FAV",
                           stages={"stt": 400, "llm": 900, "tts": 600, "total": 2000})

    payload = asyncio.run(hub_app.health())
    block = payload["utterances"]
    assert block["total"] == 1
    assert block["last"]["utterance_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert block["last"]["stages_ms"]["total"] == 2000
