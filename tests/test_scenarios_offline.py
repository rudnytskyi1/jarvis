"""ТЗ сценарий 7 (фаза 2): хаб недоступен — комната работает сама и говорит об этом.

Сценарий целиком: связь с хабом пропала → через 3 с (правило ТЗ 4.8) на экране
«мозг оффлайн» и произносится строка, которую хаб принёс ЗАРАНЕЕ (своей TTS у
комнатного ПК нет) → человек говорит «выключи свет» → команду выполняют
СОБСТВЕННЫЕ руки клиента (F-117), а не хаб. Всё это идёт по настоящему коду
клиента (`_enter_offline_mode`, `_local_turn`, `parse_local_command`,
`LocalRunner`); подставлены микрофон, локальное распознавание и само железо
(устройство-выключатель), потому что их в песочнице нет.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from client.main import JarvisClient
from client.offline import OfflineMode, offline_notice
from client.tts_cache import PhraseCache
from common import client_config as client_config_mod
from common import protocol


class _Overlay:
    def __init__(self) -> None:
        self.hub: list[dict] = []
        self.states: list[str] = []

    def hub_state(self, payload: dict) -> None:
        self.hub.append(payload)

    def set_state(self, state: str) -> None:
        self.states.append(state)


class _Audio:
    def __init__(self) -> None:
        self.played: list[tuple[int, bytes]] = []
        self.rate = 0

    async def open(self, rate: int) -> None:
        self.rate = int(rate)

    async def write(self, pcm: bytes) -> None:
        self.played.append((self.rate, bytes(pcm)))

    async def drain(self) -> None:
        return None


class _Vad:
    """The microphone: it hands over one recorded utterance."""

    async def record(self, read, pre_roll: bytes = b"") -> bytes:
        return b"\x00\x01" * 800


class _Dispatcher:
    """The client's own hands, with the wall switch itself stubbed out."""

    def __init__(self) -> None:
        self.actions: list[dict] = []

    async def execute(self, action: dict) -> tuple[bool, str | None, str | None]:
        self.actions.append(dict(action))
        return True, None, f"{action['args'].get('device')} {action['args'].get('state')}"


def client(tmp_path, *, language: str = "ru") -> JarvisClient:
    """A room client with its engines and hardware stubbed."""
    room = JarvisClient.__new__(JarvisClient)
    room._stopping = False
    room.sample_rate = 16000
    room.ccfg = SimpleNamespace(
        server_url="ws://hub/ws",
        camera=SimpleNamespace(language=language),
        wakeword=client_config_mod.WakewordConfig(),
    )
    room.offline = OfflineMode(SimpleNamespace(enabled=True, after_s=3.0))
    room.phrase_cache = PhraseCache(tmp_path / "phrases")
    room.overlay = _Overlay()
    room.audio_out = _Audio()
    room.vad = _Vad()
    room.dispatcher = _Dispatcher()
    room.registry = SimpleNamespace(names=lambda: ["Свет"])
    room.room_config = {"scenes": []}
    room.room_language = language  # type: ignore[attr-defined]
    room.local_stt = SimpleNamespace(enabled=True, timeout_s=5.0,
                                     transcribe=lambda pcm, rate: "выключи свет")
    room._pending_phrase = None
    room.statuses: list[str] = []
    room._show_status = lambda text, ttl: room.statuses.append(str(text))
    room._read_frame = lambda: b""
    room._reply_pcm = b""
    return room


def _cache_the_notice(room: JarvisClient, *, code: str = "ru", pcm: bytes = b"\x11\x22") -> None:
    """The hub's pre-synthesis, exactly as the client receives it (P2-35)."""
    room._on_tts_phrase({"type": protocol.MSG_TTS_PHRASE, "id": f"offline.{code}",
                         "text": offline_notice(code), "rate": 48000, "bytes": len(pcm)})
    room._on_phrase_audio(pcm)


def heard(room: JarvisClient, *, code: str = "ru") -> list[str]:
    """What the room perceived: the cached line out loud, or the line on screen."""
    if room.audio_out.played:
        return [offline_notice(code)] * len(room.audio_out.played)
    return list(room.statuses)


def test_the_room_says_it_is_on_its_own_and_keeps_working(tmp_path):
    room = client(tmp_path)
    _cache_the_notice(room)

    asyncio.run(room._enter_offline_mode())
    assert room.overlay.hub == [{"state": "offline"}], "на экране «мозг оффлайн»"
    assert room.audio_out.played == [(48000, b"\x11\x22")], (
        "строка произносится из кэша, а не выдумывается")

    asyncio.run(room._local_turn(b""))

    assert room.dispatcher.actions == [{
        "id": "local-off",
        "tool": "set_light",
        "args": {"device": "Свет", "state": "off"},
    }], "команду выполнила комната, а не хаб"
    assert room.statuses[-1] == "Свет off", "человек видит, что именно случилось"
    assert room.overlay.states[-1] == "idle"


def test_the_room_never_pretends_to_have_understood_without_the_hub(tmp_path):
    """Фраза не локальная — комната честно говорит про хаб, а не гадает."""
    room = client(tmp_path)
    _cache_the_notice(room)
    room.local_stt = SimpleNamespace(enabled=True, timeout_s=5.0,
                                     transcribe=lambda pcm, rate: "какая сегодня погода")

    asyncio.run(room._local_turn(b""))

    assert room.dispatcher.actions == [], "нечего было выполнять"
    assert heard(room) == [offline_notice("ru")], (
        "комната говорит, что работает без хаба, а не выдумывает ответ")


def test_without_the_cached_line_the_room_still_says_it_on_screen(tmp_path):
    room = client(tmp_path)
    asyncio.run(room._enter_offline_mode())
    assert room.audio_out.played == [], "звука нет — и это не притворство"
    assert room.statuses == [offline_notice("ru")]


def test_the_client_only_obeys_devices_it_actually_knows(tmp_path):
    """«Включи свет» без камеры-устройства у клиента — не команда, а отказ."""
    room = client(tmp_path)
    _cache_the_notice(room)
    room.registry = SimpleNamespace(names=lambda: [])

    asyncio.run(room._local_turn(b""))

    assert room.dispatcher.actions == []
    assert heard(room) == [offline_notice("ru")], (
        "комната не делает вид, что выключатель ей знаком")
