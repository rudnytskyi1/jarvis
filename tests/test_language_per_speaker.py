"""Язык по говорящему: поле, подсказка Whisper, инструкция модели (ТЗ F-106)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub.decider import RulesDecider  # noqa: F401  (imported for parity with the other suites)
from hub.languages import (
    allowed,
    effective,
    instruction,
    language_of,
    name,
    normalize,
    set_preferred_language,
    stt_hint,
)
from hub.session import Session
from hub.speaker import PEOPLE_FILENAME, VOICE_KEY, VoiceRegistry, normalize_people

# --- the field --------------------------------------------------------------


@pytest.mark.parametrize(("given", "expected"), [
    ("ru", "ru"), ("RU", "ru"), ("ru-RU", "ru"), ("russian", "ru"),
    ("Русский", "ru"), ("en_US", "en"), ("English", "en"),
    ("es-ES", "es"), ("Spanish", "es"), ("", None), (None, None), ("klingon", None),
])
def test_language_codes_are_normalised(given, expected):
    assert normalize(given) == expected


def test_the_names_and_instructions_are_the_ones_the_model_needs():
    assert name("ru") == "Russian"
    assert instruction("ru") == "Answer in Russian."
    assert instruction("klingon") == ""


def test_a_stranger_is_limited_to_the_rooms_whitelist():
    assert allowed("ru", ["en", "ru", "es"]) == "ru"
    assert allowed("de", ["en", "ru", "es"]) is None
    assert allowed("de", []) == "de", "an empty whitelist means every language"


def test_the_answer_language_is_preferred_then_detected_then_configured():
    assert effective(preferred="es", detected="ru", configured="en",
                     whitelist=["en", "ru", "es"]) == "es"
    assert effective(preferred=None, detected="ru", configured="en",
                     whitelist=["en", "ru", "es"]) == "ru"
    assert effective(preferred=None, detected="ru", configured="en",
                     whitelist=["en", "es"]) == "en", "not a language of this room"
    assert effective(preferred=None, detected="de", configured=None,
                     whitelist=["en", "ru", "es"]) is None


def test_the_whisper_hint_only_exists_after_identification():
    assert stt_hint(preferred=None, configured=None) is None
    assert stt_hint(preferred=None, configured="en") == "en"
    assert stt_hint(preferred="ru", configured="en") == "ru"


# --- the two stores ---------------------------------------------------------


def _registry(tmp_path, people: dict) -> VoiceRegistry:
    path = tmp_path / PEOPLE_FILENAME
    path.write_text(json.dumps({"people": people}), encoding="utf-8")
    return VoiceRegistry(data_dir=tmp_path, enabled=False)


def test_the_registry_keeps_a_copy_for_the_voice_pipeline(tmp_path):
    registry = _registry(tmp_path, {"Anton": {"role": "admin", VOICE_KEY: []}})
    assert registry.language_of("Anton") is None
    assert registry.set_language("Anton", "ru") == "ru"
    assert registry.language_of("Anton") == "ru"
    # ...and it survives a restart: the file is the store.
    assert VoiceRegistry(data_dir=tmp_path, enabled=False).language_of("Anton") == "ru"


def test_an_unknown_profile_is_not_silently_created(tmp_path):
    registry = _registry(tmp_path, {"Anton": {"role": "admin", VOICE_KEY: []}})
    with pytest.raises(ValueError):
        registry.set_language("Nobody", "ru")


def test_the_database_row_is_canonical(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO persons(person_id, display_name, preferred_language) "
                 "VALUES ('p-1', 'Anton', 'es')")
    conn.commit()
    assert language_of(None, conn, "Anton") == "es"
    assert language_of(None, conn, "anton") == "es", "names are matched case-insensitively"
    assert language_of(None, conn, "Nobody") is None


def test_writing_updates_both_stores(tmp_path):
    registry = _registry(tmp_path, {"Anton": {"role": "admin", VOICE_KEY: []}})
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-1', 'Anton')")
    conn.commit()
    assert set_preferred_language(registry, conn, "Anton", "ru") == "ru"
    assert registry.language_of("Anton") == "ru"
    row = conn.execute("SELECT preferred_language FROM persons WHERE person_id='p-1'").fetchone()
    assert row[0] == "ru"
    assert language_of(registry, conn, "Anton") == "ru"


def test_a_typo_is_refused_instead_of_stored(tmp_path):
    registry = _registry(tmp_path, {"Anton": {"role": "admin", VOICE_KEY: []}})
    with pytest.raises(ValueError):
        set_preferred_language(registry, None, "Anton", "klingon")
    assert registry.language_of("Anton") is None


def test_the_registry_shape_survives_the_new_field():
    people = normalize_people({"Anton": {"role": "admin", VOICE_KEY: [],
                                         "preferred_language": "ru"}})
    assert people["Anton"]["preferred_language"] == "ru"


# --- the pipeline -----------------------------------------------------------


class _Socket:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.frames: list[dict] = []
        self.client_state = hub_app.WebSocketState.CONNECTED
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)


def _turn(tmp_path, monkeypatch, *, heard: str, detected: str, preferred: str | None = None,
          voice_known: bool = True, whitelist: list[str] | None = None,
          configured: str | None = None):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    hints: list[str | None] = []

    def transcribe_pcm(pcm, rate, language):
        hints.append(language)
        return heard, detected

    voices = SimpleNamespace(
        enabled=True,
        identify=lambda *args: (("Anton", "admin", 0.9) if voice_known else ("unknown", "unknown", 0.0)),
        language_of=lambda who: preferred,
        people=lambda: {"Anton": "admin"},
    )
    monkeypatch.setattr(hub_app, "_voices", voices)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(transcribe_pcm=transcribe_pcm))
    calls: list[list[dict]] = []

    async def generate(messages, run_tool):
        calls.append([dict(message) for message in messages])
        return SimpleNamespace(text="Done.", tool_calls=[], rounds=1, history=list(messages))

    async def verify(history, answer, run_tool):
        return SimpleNamespace(text=answer, tool_calls=[], rounds=0, history=list(history))

    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(generate=generate, verify=verify))
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000,
                                                        synth=lambda part: b"\0\1" * 8))
    cfg = Config()
    if whitelist is not None:
        cfg.server.stt.allowed_languages = whitelist
    cfg.server.stt.language = configured
    socket = _Socket()
    connection = hub_app.Connection(socket, cfg)
    connection.ws = socket
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    return connection, hints, calls


def test_a_known_speaker_hears_the_answer_in_their_language(tmp_path, monkeypatch):
    connection, hints, calls = _turn(tmp_path, monkeypatch, heard="включи свет",
                                     detected="ru", preferred="ru")
    asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    assert hints == [None], "the first turn cannot know the speaker yet"
    assert "Answer in Russian." in calls[0][-1]["content"]
    assert connection._reply_language == "ru"


def test_the_next_utterance_pins_whisper_to_that_language(tmp_path, monkeypatch):
    connection, hints, _calls = _turn(tmp_path, monkeypatch, heard="включи свет",
                                      detected="ru", preferred="ru")
    asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    assert hints == [None, "ru"], "ТЗ F-106: the hint arrives after identification"


def test_a_stranger_is_answered_in_the_language_the_room_spoke(tmp_path, monkeypatch):
    connection, _hints, calls = _turn(tmp_path, monkeypatch, heard="hola rowan",
                                      detected="es", voice_known=False)
    asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    assert "Answer in Spanish." in calls[0][-1]["content"]
    assert connection._stt_language is None, "nothing is pinned for a stranger"


def test_a_language_the_room_does_not_speak_falls_back(tmp_path, monkeypatch):
    connection, _hints, calls = _turn(tmp_path, monkeypatch, heard="hallo",
                                      detected="de", voice_known=False,
                                      whitelist=["en", "ru", "es"], configured="en")
    asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    assert "Answer in" not in calls[0][-1]["content"], "English is the default"
    assert connection._reply_language == "en"


def test_the_trace_records_both_languages(tmp_path, monkeypatch):
    connection, _hints, _calls = _turn(tmp_path, monkeypatch, heard="включи свет",
                                       detected="ru", preferred="ru")
    entries: list[dict] = []
    monkeypatch.setattr(hub_app, "_dialogs", SimpleNamespace(append=entries.append))
    asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    assert entries and entries[-1]["reply_language"] == "ru"
    assert entries[-1]["preferred_language"] == "ru"
