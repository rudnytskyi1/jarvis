"""Деградация комнаты без хаба (ТЗ 4.8).

ТЗ 4.8 просит шесть вещей: локальный faster-whisper (small или base), локальные
команды и сцены, кэш заранее синтезированных TTS-фраз, «мозг оффлайн» в HUD и
голосом через 3 с, реконнект с экспоненциальной задержкой и досылку
накопленных событий присутствия после восстановления.

Проверяется здесь настоящий код: правила оффлайн-режима, кэш фраз на диске,
буфер присутствия, обёртка локального распознавания, политика реконнекта в
транспорте, предсинтез на хабе и локальный ход клиента. Живого железа
(микрофона, камеры, отсутствующего хаба) в песочнице нет.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from client import ws_client as ws_mod
from client.local_stt import LocalStt
from client.main import JarvisClient
from client.offline import (
    OfflineMode,
    backoff_delay,
    language_of,
    offline_notice,
    prefetch_phrases,
)
from client.presence_buffer import PresenceBuffer
from client.tts_cache import PhraseCache
from common import protocol
from hub import app as hub_app

# --- правила оффлайн-режима --------------------------------------------------


def _mode(**overrides: Any):
    clock = {"now": 100.0}
    config = SimpleNamespace(enabled=True, after_s=3.0, backoff_base_s=1.0,
                             backoff_factor=2.0, backoff_max_s=30.0)
    for key, value in overrides.items():
        setattr(config, key, value)
    return OfflineMode(config, clock=lambda: clock["now"]), clock


def test_the_room_is_told_only_after_three_seconds_without_the_hub():
    mode, clock = _mode()
    assert mode.offline() is False
    mode.link_down()
    clock["now"] += 1.0
    assert mode.offline() is False and mode.take_notice() is False
    clock["now"] += 2.5
    assert mode.offline() is True
    assert mode.take_notice() is True
    assert mode.take_notice() is False, "одного раза на простой достаточно"
    clock["now"] += 60.0
    assert mode.take_notice() is False, "простой тот же — повторно не говорим"
    mode.link_up()
    assert mode.offline() is False and mode.down_s() == 0.0
    mode.link_down()
    clock["now"] += 3.0
    assert mode.take_notice() is True, "новый простой — новое предупреждение"


def test_offline_mode_can_be_switched_off():
    mode, clock = _mode(enabled=False)
    mode.link_down()
    clock["now"] += 300.0
    assert mode.offline() is False and mode.take_notice() is False


def test_the_reconnect_delay_grows_and_stops_at_the_ceiling():
    assert [backoff_delay(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert backoff_delay(9) == 30.0, "потолок держит комнату от штурма хаба"
    assert backoff_delay(1, base_s=3.0, factor=2.0, max_s=30.0) == 3.0
    mode, _clock = _mode()
    assert mode.delay_for(3) == 4.0
    assert mode.attempts() == 0, "чистая функция попытки не считает"
    assert mode.next_delay() == 1.0 and mode.attempts() == 1


def test_the_line_the_room_hears_exists_in_three_languages():
    assert language_of("ru-RU") == "ru" and language_of("es") == "es"
    assert language_of("") == "en"
    assert offline_notice("ru").startswith("Хаб недоступен")
    assert offline_notice("en").startswith("The hub")
    assert offline_notice("es").startswith("El concentrador")
    ids = {item["id"] for item in prefetch_phrases()}
    assert ids == {"offline.ru", "offline.en", "offline.es"}
    assert all(item["text"] for item in prefetch_phrases())


# --- локальное распознавание -------------------------------------------------


class _FakeWhisper:
    """Stand-in for faster-whisper: one segment for every call."""

    def __init__(self, text: str = "включи свет", *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[Any] = []

    def transcribe(self, audio, **kwargs):
        if self.fail:
            raise RuntimeError("no weights")
        self.calls.append((audio, kwargs))
        return [SimpleNamespace(text=self.text)], SimpleNamespace(language="ru")


def test_the_local_recognizer_uses_faster_whisper_on_this_machine():
    built: list[tuple[str, str, str]] = []
    whisper = _FakeWhisper()

    def factory(name: str, device: str, compute_type: str) -> Any:
        built.append((name, device, compute_type))
        return whisper

    stt = LocalStt(SimpleNamespace(enabled=True, model="small", device="auto",
                                   compute_type="int8", language="ru", timeout_s=10.0),
                   model_factory=factory)
    assert stt.available is False
    assert stt.load() is True
    assert built == [("small", "cuda", "int8")], "auto пробует карту первой"
    audio = b"\x01\x00" * 16000
    assert stt.transcribe(audio, 16000) == "включи свет"
    assert whisper.calls[0][1]["language"] == "ru"


def test_a_machine_without_cuda_still_gets_a_local_recognizer():
    attempts: list[str] = []

    def factory(_name: str, device: str, _compute_type: str) -> Any:
        attempts.append(device)
        if device == "cuda":
            raise RuntimeError("no CUDA device")
        return _FakeWhisper()

    stt = LocalStt(SimpleNamespace(enabled=True, model="base", device="auto"),
                   model_factory=factory)
    assert stt.load() is True
    assert attempts == ["cuda", "cpu"]
    assert stt.transcribe(b"\x00\x00" * 800, 16000) == "включи свет"


def test_without_a_local_model_the_client_admits_it():
    def factory(*_args: Any) -> Any:
        raise ImportError("faster_whisper is not installed")

    stt = LocalStt(SimpleNamespace(enabled=True, model="base", device="auto"),
                   model_factory=factory)
    assert stt.load() is False
    assert stt.available is False
    assert "could not be loaded" in stt.reason or "faster_whisper" in stt.reason
    assert stt.transcribe(b"\x00\x00" * 800, 16000) is None


def test_switching_local_stt_off_is_not_a_silent_failure():
    stt = LocalStt(SimpleNamespace(enabled=False, model="base"))
    assert stt.load() is False
    assert "switched off" in stt.reason
    assert stt.transcribe(b"\x00\x00", 16000) is None


def test_an_empty_phrase_is_not_called_a_transcription():
    stt = LocalStt(SimpleNamespace(enabled=True, model="base"),
                   model_factory=lambda *args: _FakeWhisper(""))
    assert stt.transcribe(b"", 16000) == ""


# --- кэш заранее синтезированных фраз ---------------------------------------


def test_a_phrase_the_hub_sent_earlier_is_played_without_the_hub(tmp_path):
    cache = PhraseCache(tmp_path / "phrases")
    assert cache.enabled is True
    assert cache.has("offline.ru") is False
    assert cache.store("offline.ru", "Хаб недоступен, работаю локально.", b"pcm-bytes", 48000)
    assert cache.get("offline.ru") == (b"pcm-bytes", 48000)
    assert cache.text_of("offline.ru").startswith("Хаб недоступен")
    assert cache.ids() == ("offline.ru",)
    # Фраза переживает перезапуск клиента: она лежит на диске этого ПК.
    again = PhraseCache(tmp_path / "phrases")
    assert again.get("offline.ru") == (b"pcm-bytes", 48000)
    assert again.get("offline.en") is None


def test_the_phrase_cache_admits_missing_audio_and_a_missing_file(tmp_path):
    cache = PhraseCache(tmp_path / "phrases")
    assert cache.store("", "текст", b"pcm", 48000) is False
    assert cache.store("empty", "", b"", 48000) is False
    assert cache.store("gone", "текст", b"pcm", 48000) is True
    (tmp_path / "phrases" / cache._index["gone"]["file"]).unlink()
    assert cache.get("gone") is None and cache.load_errors == 1


def test_a_cache_that_is_switched_off_keeps_nothing(tmp_path):
    cache = PhraseCache(tmp_path / "phrases", enabled=False)
    assert cache.store("offline.ru", "текст", b"pcm", 48000) is False
    assert cache.get("offline.ru") is None


def test_a_broken_index_is_not_a_crash(tmp_path):
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    cache = PhraseCache(tmp_path)
    assert cache.ids() == ()
    assert cache.store("offline.ru", "текст", b"pcm", 48000) is True
    assert cache.get("offline.ru") == (b"pcm", 48000)


# --- буфер событий присутствия ----------------------------------------------


def test_presence_frames_are_kept_while_the_hub_is_gone():
    buffer = PresenceBuffer(3)
    assert buffer.add({"type": "camera_state", "persons": 1, "ts": 111.0}) is True
    assert buffer.add({"type": "tracks", "tracks": []}) is True
    assert buffer.add({"type": "camera_frame", "id": "x"}) is False, "кадры не копим"
    assert buffer.add("nonsense") is False
    assert buffer.add({"type": "camera_state", "persons": 2}) is True
    assert buffer.add({"type": "camera_state", "persons": 3}) is True
    assert buffer.dropped == 1, "буфер ограничен: старое вытесняется"
    rows = buffer.take()
    assert [row["type"] for row in rows] == ["tracks", "camera_state", "camera_state"]
    assert [row["persons"] for row in rows if row["type"] == "camera_state"] == [2, 3]
    assert all(row["replay"] is True and row["ts"] for row in rows)
    assert len(buffer) == 0 and buffer.take() == []


def test_a_buffer_of_zero_keeps_nothing():
    buffer = PresenceBuffer(0)
    assert buffer.add({"type": "camera_state", "persons": 1}) is False
    assert buffer.stats()["buffered"] == 0


def test_the_first_stamp_of_a_buffered_frame_survives():
    buffer = PresenceBuffer(4)
    buffer.add({"type": "camera_state", "persons": 1, "ts": 42.5})
    assert buffer.take()[0]["ts"] == 42.5


# --- протокол ----------------------------------------------------------------


def test_the_offline_frames_are_part_of_the_protocol():
    assert protocol.MSG_TTS_PREFETCH == "tts_prefetch"
    assert protocol.MSG_TTS_PHRASE == "tts_phrase"
    assert protocol.MSG_OFFLINE_HINT == "offline_hint"
    assert protocol.MSG_TTS_PREFETCH in protocol.CLIENT_MESSAGE_TYPES
    assert protocol.MSG_TTS_PHRASE in protocol.SERVER_MESSAGE_TYPES
    assert protocol.MSG_OFFLINE_HINT in protocol.SERVER_MESSAGE_TYPES
    assert "MSG_TTS_PREFETCH" in protocol.__all__
    assert "MSG_TTS_PHRASE" in protocol.__all__
    assert "MSG_OFFLINE_HINT" in protocol.__all__


def test_the_offline_hint_frame_has_reason_and_eta():
    hint = protocol.OfflineHint(reason="shutdown", eta_s=30.0)
    assert hint.type == "offline_hint" and hint.reason == "shutdown" and hint.eta_s == 30.0
    with pytest.raises(Exception):
        protocol.OfflineHint(reason="x", eta_s=-1)


# --- транспорт: экспоненциальная задержка ------------------------------------


def test_the_transport_waits_longer_after_every_failed_attempt(monkeypatch):
    waits: list[float] = []
    seen: list[tuple[float, int]] = []
    holder: dict[str, Any] = {}

    async def failing(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("no route to host")

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) >= 4:
            holder["client"]._should_stop = lambda: True

    client = ws_mod.WSClient(
        "ws://hub:8765/ws", {"type": "hello"}, reconnect_delay=3.0,
        should_stop=lambda: False,
        backoff=backoff_delay,
        before_retry=lambda delay, attempt: seen.append((delay, attempt)),
    )
    holder["client"] = client
    monkeypatch.setattr(ws_mod, "ws_connect", failing)
    monkeypatch.setattr(client, "_sleep", fake_sleep)
    with pytest.raises(ws_mod.WSDisconnected):
        asyncio.run(client.ensure_connected())
    assert waits == [1.0, 2.0, 4.0, 8.0]
    assert seen == [(1.0, 1), (2.0, 2), (4.0, 3), (8.0, 4)]
    assert client.attempts == 4


def test_an_async_retry_hook_is_awaited(monkeypatch):
    calls: list[str] = []
    holder: dict[str, Any] = {}

    async def failing(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("down")

    async def hook(delay: float, attempt: int) -> None:
        calls.append(f"{attempt}:{delay:.0f}")
        holder["client"]._should_stop = lambda: True

    async def fake_sleep(_seconds: float) -> None:
        return None

    client = ws_mod.WSClient("ws://hub/ws", {}, should_stop=lambda: False,
                             backoff=lambda attempt: 2.0, before_retry=hook)
    holder["client"] = client
    monkeypatch.setattr(ws_mod, "ws_connect", failing)
    monkeypatch.setattr(client, "_sleep", fake_sleep)
    with pytest.raises(ws_mod.WSDisconnected):
        asyncio.run(client.ensure_connected())
    assert calls == ["1:2"]


def test_the_default_transport_still_waits_the_old_three_seconds(monkeypatch):
    waits: list[float] = []
    holder: dict[str, Any] = {}

    async def failing(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("down")

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        holder["client"]._should_stop = lambda: True

    client = ws_mod.WSClient("ws://hub/ws", {}, reconnect_delay=3.0,
                             should_stop=lambda: False)
    holder["client"] = client
    monkeypatch.setattr(ws_mod, "ws_connect", failing)
    monkeypatch.setattr(client, "_sleep", fake_sleep)
    with pytest.raises(ws_mod.WSDisconnected):
        asyncio.run(client.ensure_connected())
    assert waits == [3.0]


def test_a_broken_backoff_policy_falls_back_to_the_fixed_delay(monkeypatch):
    def broken(_attempt: int) -> float:
        raise RuntimeError("policy")

    waits: list[float] = []
    holder: dict[str, Any] = {}

    async def failing(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("down")

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        holder["client"]._should_stop = lambda: True

    client = ws_mod.WSClient("ws://hub/ws", {}, reconnect_delay=3.0,
                             should_stop=lambda: False, backoff=broken)
    holder["client"] = client
    monkeypatch.setattr(ws_mod, "ws_connect", failing)
    monkeypatch.setattr(client, "_sleep", fake_sleep)
    with pytest.raises(ws_mod.WSDisconnected):
        asyncio.run(client.ensure_connected())
    assert waits == [3.0]


# --- хаб: предсинтез и досылка ----------------------------------------------


class _Ws:
    from starlette.websockets import WebSocketState as _State

    client_state = _State.CONNECTED


class _Voice:
    """Stand-in for the hub's TTS engine."""

    available = True
    sample_rate = 48000

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.asked: list[str] = []

    def synth(self, text: str) -> bytes:
        self.asked.append(text)
        if self.fail:
            raise RuntimeError("the synthesizer is unhappy")
        return b"pcm:" + text.encode("utf-8")[:8]


def _hub_room():
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.ws = _Ws()
    connection._audio_lock = asyncio.Lock()
    connection._hub_status_sent = None
    frames: list[dict[str, Any]] = []
    blobs: list[bytes] = []

    async def send_json(payload: dict[str, Any]) -> None:
        frames.append(payload)

    async def send_bytes(data: bytes) -> None:
        blobs.append(bytes(data))

    async def queue_frame(payload: dict[str, Any], **_kwargs: Any) -> bool:
        frames.append(payload)
        return True

    connection.send_json = send_json
    connection.send_bytes = send_bytes
    connection.queue_frame = queue_frame
    connection.frames = frames
    connection.blobs = blobs
    return connection


def test_the_hub_synthesizes_the_lines_a_room_needs_offline(monkeypatch):
    voice = _Voice()
    monkeypatch.setattr(hub_app, "_tts", voice)
    connection = _hub_room()
    asyncio.run(connection._on_tts_prefetch({"phrases": prefetch_phrases()}))
    headers = [row for row in connection.frames if row["type"] == protocol.MSG_TTS_PHRASE]
    assert [row["id"] for row in headers] == ["offline.ru", "offline.en", "offline.es"]
    assert all("error" not in row for row in headers)
    assert all(row["rate"] == 48000 and row["bytes"] > 0 for row in headers)
    assert len(connection.blobs) == 3 and all(connection.blobs)
    assert voice.asked == [item["text"] for item in prefetch_phrases()]


def test_without_tts_the_hub_answers_with_an_error_and_no_audio(monkeypatch):
    monkeypatch.setattr(hub_app, "_tts", None)
    connection = _hub_room()
    asyncio.run(connection._on_tts_prefetch({"phrases": prefetch_phrases()[:1]}))
    (header,) = [row for row in connection.frames if row["type"] == protocol.MSG_TTS_PHRASE]
    assert header["error"] == "tts unavailable" and header["text"]
    assert connection.blobs == [], "тишину нельзя выдавать за готовую фразу"


def test_a_failed_synthesis_is_not_a_cached_phrase(monkeypatch):
    monkeypatch.setattr(hub_app, "_tts", _Voice(fail=True))
    connection = _hub_room()
    asyncio.run(connection._on_tts_prefetch({"phrases": prefetch_phrases()[:1]}))
    (header,) = [row for row in connection.frames if row["type"] == protocol.MSG_TTS_PHRASE]
    assert header["error"] and connection.blobs == []


def test_a_request_without_phrases_is_ignored(monkeypatch):
    monkeypatch.setattr(hub_app, "_tts", _Voice())
    connection = _hub_room()
    asyncio.run(connection._on_tts_prefetch({}))
    asyncio.run(connection._on_tts_prefetch({"phrases": []}))
    asyncio.run(connection._on_tts_prefetch({"phrases": [{"id": "", "text": "x"}, 7]}))
    assert connection.frames == []


def test_the_hub_tells_every_room_it_is_going_away(monkeypatch):
    rooms = [_hub_room(), _hub_room()]
    monkeypatch.setattr(hub_app, "_connections", set(rooms))
    told = asyncio.run(hub_app.announce_offline_hint("shutdown", 30.0))
    assert told == 2
    for room in rooms:
        frame = room.frames[0]
        assert frame["type"] == protocol.MSG_OFFLINE_HINT
        assert frame["reason"] == "shutdown" and frame["eta_s"] == 30.0


class _Alerts:
    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def observe(self, **kwargs: Any) -> None:
        self.seen.append(kwargs)


def _state_connection():
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = SimpleNamespace(client_id="pc-1")
    connection.room = hub_app.RoomState()
    connection.camera_state = None
    connection.presence = hub_app.PresenceTracker(ttl_s=30.0)
    connection._presence_has_tracks = False
    connection._track_zones = {}
    return connection


def test_a_replayed_state_keeps_the_rooms_own_clock(monkeypatch):
    import time

    alerts = _Alerts()
    monkeypatch.setattr(hub_app, "_presence_alerts", alerts)
    connection = _state_connection()
    stamp = time.time() - 120
    connection._on_camera_state({"persons": 2, "objects": {}, "ts": stamp, "replay": True})
    assert connection.camera_state["ts"] == pytest.approx(stamp, abs=0.5)
    assert alerts.seen == [], "старое событие не поднимает тревогу"
    assert connection.presence is not None
    connection._on_camera_state({"persons": 2, "objects": {}, "ts": stamp})
    assert connection.camera_state["ts"] > stamp + 100, "без replay время ставит хаб"
    assert alerts.seen and alerts.seen[0]["persons"] == 2


def test_a_replayed_state_from_the_future_is_just_now(monkeypatch):
    import time

    monkeypatch.setattr(hub_app, "_presence_alerts", None)
    connection = _state_connection()
    connection._on_camera_state({"persons": 1, "objects": {}, "ts": time.time() + 10_000,
                                 "replay": True})
    assert connection.camera_state["ts"] <= time.time() + 1


def test_a_replayed_track_frame_reaches_the_room_state(monkeypatch):
    import time

    monkeypatch.setattr(hub_app, "_presence_alerts", None)
    connection = _state_connection()
    connection._on_camera_state({"persons": 1, "objects": {},
                                 "tracks": [{"id": "t1", "box": [0.1, 0.1, 0.4, 0.6]}],
                                 "ts": time.time() - 60, "replay": True})
    assert "t1" in connection.room.tracks
    assert connection._presence_has_tracks is True


# --- клиент: предсинтез, досылка, локальный ход ------------------------------


class _ClientWs:
    """Records what the client sends; can break after ``fail_after`` frames."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.json: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.fail_after is not None and len(self.json) >= self.fail_after:
            raise ws_mod.WSDisconnected("the socket died again")
        self.json.append(payload)


class _ClientOverlay:
    def __init__(self) -> None:
        self.hub: list[dict[str, Any]] = []
        self.states: list[str] = []

    def hub_state(self, payload: dict[str, Any]) -> None:
        self.hub.append(payload)

    def set_state(self, state: str) -> None:
        self.states.append(state)


class _ClientAudio:
    def __init__(self) -> None:
        self.played: list[tuple[int, bytes]] = []

    async def open(self, rate: int) -> None:
        self.rate = rate

    async def write(self, pcm: bytes) -> None:
        self.played.append((int(getattr(self, "rate", 0)), bytes(pcm)))

    async def drain(self) -> None:
        return None


def _client(tmp_path, **overrides: Any):
    client = JarvisClient.__new__(JarvisClient)
    client._stopping = False
    client.sample_rate = 16000
    client.ccfg = SimpleNamespace(camera=SimpleNamespace(language="ru"),
                                  server_url="ws://hub/ws")
    client.offline = OfflineMode(SimpleNamespace(enabled=True, after_s=3.0))
    client.phrase_cache = PhraseCache(tmp_path / "phrases")
    client.presence_buffer = PresenceBuffer(4)
    client.ws = _ClientWs()
    client.overlay = _ClientOverlay()
    client.audio_out = _ClientAudio()
    client._pending_phrase = None
    client.local_stt = SimpleNamespace(enabled=False, timeout_s=1.0, reason="switched off")
    client.statuses: list[str] = []
    client._show_status = lambda text, ttl: client.statuses.append(str(text))
    for key, value in overrides.items():
        setattr(client, key, value)
    return client


def test_the_client_keeps_the_line_the_hub_pre_synthesized(tmp_path):
    client = _client(tmp_path)
    client._on_tts_phrase({"type": protocol.MSG_TTS_PHRASE, "id": "offline.ru",
                           "text": offline_notice("ru"), "rate": 48000, "bytes": 3})
    assert client._pending_phrase["id"] == "offline.ru"
    client._on_phrase_audio(b"pcm")
    assert client._pending_phrase is None
    assert client.phrase_cache.get("offline.ru") == (b"pcm", 48000)


def test_a_line_the_hub_could_not_make_is_not_kept(tmp_path):
    client = _client(tmp_path)
    client._on_tts_phrase({"type": protocol.MSG_TTS_PHRASE, "id": "offline.ru",
                           "text": offline_notice("ru"), "error": "tts unavailable"})
    assert client._pending_phrase is None
    client._on_phrase_audio(b"")
    assert client.phrase_cache.ids() == ()


def test_the_client_asks_only_for_the_lines_it_does_not_have(tmp_path):
    client = _client(tmp_path)
    client.phrase_cache.store("offline.ru", offline_notice("ru"), b"pcm", 48000)
    asyncio.run(client._request_phrases())
    (request,) = client.ws.json
    assert request["type"] == protocol.MSG_TTS_PREFETCH
    assert [item["id"] for item in request["phrases"]] == ["offline.en", "offline.es"]
    client.ws.json.clear()
    asyncio.run(client._request_phrases())
    assert client.ws.json, "фразы ещё не пришли"
    for item in prefetch_phrases():
        client.phrase_cache.store(item["id"], item["text"], b"pcm", 48000)
    client.ws.json.clear()
    asyncio.run(client._request_phrases())
    assert client.ws.json == [], "когда всё есть, хаб не дёргается"


def test_the_client_replays_what_the_room_saw_without_the_hub(tmp_path):
    client = _client(tmp_path)
    client.presence_buffer.add({"type": "camera_state", "persons": 1, "ts": 10.0})
    client.presence_buffer.add({"type": "tracks", "tracks": [{"id": "t1"}], "ts": 11.0})
    assert asyncio.run(client._flush_presence()) == 2
    assert [row["type"] for row in client.ws.json] == ["camera_state", "tracks"]
    assert all(row["replay"] is True for row in client.ws.json)
    assert len(client.presence_buffer) == 0


def test_a_flush_that_breaks_keeps_the_rest_of_the_buffer(tmp_path):
    client = _client(tmp_path)
    client.ws = _ClientWs(fail_after=1)
    client.presence_buffer.add({"type": "camera_state", "persons": 1, "ts": 10.0})
    client.presence_buffer.add({"type": "camera_state", "persons": 2, "ts": 11.0})
    assert asyncio.run(client._flush_presence()) == 1
    assert len(client.presence_buffer) == 1
    assert client.presence_buffer.take()[0]["persons"] == 2


def test_the_client_plays_the_cached_line_when_the_hub_is_gone(tmp_path):
    client = _client(tmp_path)
    client.phrase_cache.store("offline.ru", offline_notice("ru"), b"pcm-line", 48000)
    asyncio.run(client._say_offline_notice())
    assert client.audio_out.played == [(48000, b"pcm-line")]
    assert client.statuses == []


def test_without_cached_audio_the_room_still_reads_the_line(tmp_path):
    client = _client(tmp_path)
    asyncio.run(client._say_offline_notice())
    assert client.audio_out.played == []
    assert client.statuses == [offline_notice("ru")]


def test_the_offline_mode_marks_the_hub_offline_on_the_screen(tmp_path):
    client = _client(tmp_path)
    asyncio.run(client._enter_offline_mode())
    assert client.overlay.hub == [{"state": "offline"}]
    assert client.statuses == [offline_notice("ru")]


def test_an_offline_hint_switches_the_room_to_local_at_once(tmp_path):
    client = _client(tmp_path)
    client.phrase_cache.store("offline.ru", offline_notice("ru"), b"pcm-line", 48000)
    asyncio.run(client._on_offline_hint({"reason": "restart", "eta_s": 30.0}))
    assert client.overlay.hub == [{"state": "offline"}]
    assert client.audio_out.played == [(48000, b"pcm-line")]
    assert client.offline.down_s() >= 0.0


def test_a_local_turn_runs_the_command_without_the_hub(tmp_path):
    client = _client(tmp_path)
    client.local_stt = SimpleNamespace(enabled=True, timeout_s=5.0, reason="")
    client.local_stt.transcribe = lambda pcm, rate: "включи свет"
    ran: list[str] = []

    async def record(_self, pre_roll):
        return b"\x00\x00" * 800

    async def run_local(text: str):
        ran.append(text)
        return SimpleNamespace(kind="light", spoken="Свет включён", ok=True, heard=text)

    client.vad = SimpleNamespace(record=record)
    client._run_local_command = run_local
    asyncio.run(client._local_turn(b""))
    assert ran == ["включи свет"]
    assert client.overlay.states[-1] == "idle"


def test_a_local_turn_without_a_local_model_admits_it(tmp_path):
    client = _client(tmp_path)
    ran: list[str] = []

    async def record(_self, pre_roll):
        return b"\x00\x00" * 800

    async def run_local(text: str):
        ran.append(text)
        return None

    client.vad = SimpleNamespace(record=record)
    client._run_local_command = run_local
    asyncio.run(client._local_turn(b""))
    assert ran == []
    assert client.statuses == [offline_notice("ru")]


def test_a_local_command_the_client_does_not_know_is_answered_honestly(tmp_path):
    client = _client(tmp_path)
    client.local_stt = SimpleNamespace(enabled=True, timeout_s=5.0, reason="")
    client.local_stt.transcribe = lambda pcm, rate: "закажи пиццу"
    client.phrase_cache.store("offline.ru", offline_notice("ru"), b"pcm-line", 48000)

    async def record(_self, pre_roll):
        return b"\x00\x00" * 800

    async def run_local(_text: str):
        return None

    client.vad = SimpleNamespace(record=record)
    client._run_local_command = run_local
    asyncio.run(client._local_turn(b""))
    assert client.audio_out.played == [(48000, b"pcm-line")]


def test_nothing_heard_locally_is_not_treated_as_a_command(tmp_path):
    client = _client(tmp_path)
    beeps: list[tuple[float, int]] = []

    async def record(_self, pre_roll):
        return b""

    async def beep(freq: float, ms: int, volume: float = 0.3) -> None:
        beeps.append((freq, ms))

    client.vad = SimpleNamespace(record=record)
    client._beep = beep
    asyncio.run(client._local_turn(b""))
    assert beeps, "пустая реплика — это писк, а не команда"
