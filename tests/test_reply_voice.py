"""ТЗ F-607: голос ответа едет с человеком, а не живёт в настройках дома."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common.config import Config
from hub import app
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.person_preferences import PersonPreferencesStore
from hub.session import Session
from hub.tts import TtsEngine

AMY = "p-amy"
MAX = "p-max"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.commit()
    yield conn
    conn.close()


def _engine(*, speaker: str = "en_0", voices: tuple[str, ...] = ("en_0", "en_5", "ru_1")):
    engine = TtsEngine(SimpleNamespace(engine="kokoro", language="en", model_id="v3_en",
                                       speaker=speaker, sample_rate=24000,
                                       kokoro_model_path="x", kokoro_voices_path="y"))
    engine._model = SimpleNamespace(speakers=list(voices))
    return engine


def _connection(hub_db, monkeypatch, *, speaker: str = "Антон", store=None):
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_voices", None)
    monkeypatch.setattr(app, "_preferences",
                        store if store is not None else PersonPreferencesStore(hub_db))
    conn = app.Connection.__new__(app.Connection)
    conn.home_id = "livingroom"
    conn.cfg = Config()
    conn.session = Session(client_id="room-pc", devices=[], history_turns=2)
    conn._speaker_name = speaker
    conn._speaker_role = "user"
    conn._reply_language = "en"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._reply_lock = asyncio.Lock()
    return conn


# --- the engine ------------------------------------------------------------


def test_an_engine_shares_its_model_with_another_voice():
    engine = _engine()
    other = engine.with_voice("en_5")
    assert other is not engine and other.speaker == "en_5"
    assert other._model is engine._model, "the model is not loaded twice"
    assert engine.speaker == "en_0", "the room's own engine is untouched"


def test_an_engine_keeps_its_voice_when_it_cannot_produce_the_requested_one():
    engine = _engine()
    assert engine.with_voice("klingon_7") is engine
    assert engine.with_voice("") is engine
    assert engine.with_voice("en_0") is engine
    assert engine.knows_voice("en_5") is True
    assert engine.knows_voice("klingon_7") is False


def test_an_engine_without_the_list_is_not_a_free_for_all():
    engine = _engine()
    engine._model = SimpleNamespace()  # no speakers, no get_voices
    assert engine.voices() == []
    assert engine.knows_voice("en_5") is False
    assert engine.with_voice("en_5") is engine


def test_kokoro_style_engines_are_asked_for_their_voice_list():
    engine = _engine()
    engine._model = SimpleNamespace(get_voices=lambda: ["af_sarah", "af_sky"])
    assert engine.voices() == ["af_sarah", "af_sky"]
    assert engine.with_voice("af_sky").speaker == "af_sky"


# --- the person's voice ----------------------------------------------------


def test_the_reply_uses_the_voice_of_the_person_who_spoke(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, voice="en_5")
    engine = _engine()
    conn = _connection(hub_db, monkeypatch, store=store)
    assert conn._reply_voice(engine).speaker == "en_5"
    assert engine.speaker == "en_0"


def test_two_people_get_their_own_voices_in_the_same_room(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, voice="en_5")
    store.set(MAX, voice="ru_1")
    engine = _engine()
    conn = _connection(hub_db, monkeypatch, speaker="Антон", store=store)
    assert conn._reply_voice(engine).speaker == "en_5"
    conn._speaker_name = "Макс"
    assert conn._reply_voice(engine).speaker == "ru_1"


def test_a_person_without_a_voice_keeps_the_rooms_voice(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch)
    engine = _engine()
    assert conn._reply_voice(engine) is engine


def test_a_voice_the_engine_cannot_produce_does_not_mute_the_room(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, voice="klingon_7")
    conn = _connection(hub_db, monkeypatch, store=store)
    engine = _engine()
    assert conn._reply_voice(engine) is engine


def test_an_unrecognised_voice_keeps_the_rooms_voice(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch, speaker="unknown")
    engine = _engine()
    assert conn._reply_voice(engine) is engine


def test_a_proactive_line_uses_the_voice_of_the_person_it_is_for(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    store.set(MAX, voice="ru_1")
    conn = _connection(hub_db, monkeypatch, speaker="Антон", store=store)
    engine = _engine()
    monkeypatch.setattr(app, "_tts", engine)
    assert asyncio.run(conn._say_proactive("Макс, пора выходить", name="Макс")) is True
    spoken_voice = conn._stream_tts.call_args.args[0]
    assert spoken_voice.speaker == "ru_1"


# --- through a real turn ---------------------------------------------------


def test_a_real_turn_is_spoken_in_the_voice_of_the_speaker(hub_db, monkeypatch, tmp_path):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, voice="en_5")
    engine = _engine()
    brain = SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(text="Okay.", history=[],
                                                        tool_calls=[])),
        verify=AsyncMock())
    voices = Mock()
    voices.enabled = True
    voices.identify_ex = lambda *a: ("Антон", "user", 0.9, None)
    voices.role_of = lambda name: "user"
    voices.people = lambda: {"Антон": "user", "Макс": "user"}
    voices.voice_profiles = lambda: {}
    voices.face_profiles = lambda: {}
    monkeypatch.setattr(app, "_llm", brain)
    monkeypatch.setattr(app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: ("hello", "en")))
    monkeypatch.setattr(app, "_voices", voices)
    monkeypatch.setattr(app, "_tts", engine)
    monkeypatch.setattr(app, "_conversations", None)
    monkeypatch.setattr(app, "_memory", None)
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_audit", None)
    monkeypatch.setattr(app, "_preferences", store)
    for name in ("_polls", "_intercom", "_contacts", "_scenes", "_objects",
                 "_device_states", "_switches"):
        monkeypatch.setattr(app, name, None)
    for name in ("_devices", "_tools", "_interhome_limits", "_media"):
        monkeypatch.setattr(app, name, False)
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn.home_id = "livingroom"
    conn.cfg.server.diarization.enabled = False
    conn.cfg.server.llm.verify_actions = False
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.send_json = AsyncMock()
    conn._announce_speaker = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn.presence = SimpleNamespace(present=lambda: {"Антон"}, unknown_count=0)
    asyncio.run(conn._handle_utterance(b"\0" * 3200 * 2))
    spoken_voice = conn._stream_tts.call_args.args[0]
    assert spoken_voice is not engine and spoken_voice.speaker == "en_5"
    assert engine.speaker == "en_0"
