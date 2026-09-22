"""How long the cinema scene takes by voice (ТЗ сценарий 1, бюджет 2 с).

The acceptance criterion is not that ``SceneRunner`` can carry out a list of
steps — ``tests/test_scene_e2e.py`` covers that — but that saying "Rowan,
выключи свет и включи фильм" in a room is answered inside 2 seconds. The part
the hub controls is measured here on the real turn pipeline: real
``Connection``, real ``SceneStore`` presets, real device store and the same
frame the client receives. The model is a stub that fails the measurement if
the turn ever needs it, because a round of the local LLM is what eats this
budget — the sentence names a scene, so the model must not be asked at all.

The room's own STT and TTS engines are stubs: what they cost is the stand's
number, not this machine's, so their time enters as parameters (``--stt-ms``,
``--tts-ms``, both 0 by default) and the report says plainly whether the
engines are included. With the defaults the report is the hub's own work —
the thing the sandbox can honestly measure.

Usage::

    python scripts/measure_scene_latency.py
    python scripts/measure_scene_latency.py --repeats 7 --stt-ms 400 --tts-ms 300
    python scripts/measure_scene_latency.py --language es
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import protocol as proto  # noqa: E402
from common.config import Config, load_config  # noqa: E402
from common.ids import new_ulid  # noqa: E402
from hub import app as hub_app  # noqa: E402
from hub import migrations_runner  # noqa: E402
from hub.devices import Device, DeviceStore, DeviceTools  # noqa: E402
from hub.scenes import SceneStore  # noqa: E402
from hub.session import Session  # noqa: E402
from scripts.measure_overflow import summarize  # noqa: E402

#: The acceptance criterion of phase 2 (ТЗ 16, сценарий 1).
SCENE_BUDGET_S = 2.0

#: The sentence of the criterion and the same request in the other two
#: languages the hub understands (ТЗ 1: en/ru/es).
PHRASES: dict[str, str] = {
    "ru": "Rowan, выключи свет и включи фильм",
    "en": "Rowan, turn off the light and start the movie",
    "es": "Rowan, apaga la luz y pon una película",
}

HOME = "livingroom"

#: Where the throwaway databases are made. The repository's own scratch folder
#: is used when there is one: the system temp directory is not always writable
#: (the sandbox of the project denies it), and a measurement must not fail over
#: where its database lives.
SCRATCH = REPO_ROOT / ".tmp"


def workdir() -> str:
    """A fresh directory for one room's database.

    The name is made here instead of with ``tempfile.mkdtemp`` because the
    sandbox of this project refuses writes into a directory that ``mkdtemp``
    just made - the measurement is not the thing that should fail over that.
    """
    for base in (SCRATCH, Path.cwd()):
        if not base.is_dir():
            continue
        path = base / f"scene-latency-{os.urandom(4).hex()}"
        try:
            path.mkdir()
        except OSError:
            continue
        return str(path)
    raise RuntimeError("nowhere to put the measurement database")


def json_value(value: Any) -> Any:
    """A capability value the way it reaches the report as JSON."""
    return list(value) if isinstance(value, tuple) else value


class _Socket:
    """Just enough websocket for one room to receive its frames."""

    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)
        self.client_state = hub_app.WebSocketState.CONNECTED
        self.frames: list[dict] = []

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        self.client_state = hub_app.WebSocketState.DISCONNECTED

    def said(self) -> list[str]:
        return [frame["text"] for frame in self.frames if frame.get("type") == proto.MSG_SAY]


class _Adapter:
    """A switch that remembers what the scene asked it to do."""

    name = "mqtt"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    async def set(self, device, capability, value):
        self.calls.append((device.id, capability, value))
        return {capability: value}

    async def read(self, device, capability):
        return None


class _Stt:
    """The hub's STT engine: the transcript of the criterion, and its cost."""

    def __init__(self, text: str, *, cost_ms: float = 0.0) -> None:
        self.text = text
        self.cost_ms = max(0.0, float(cost_ms))

    def transcribe_pcm(self, pcm, sample_rate, language=None):
        if self.cost_ms:
            time.sleep(self.cost_ms / 1000.0)
        return self.text, "ru"


def _no_model():
    """A model round that must never happen: a scene is not a model request."""

    async def generate(*args, **kwargs):
        raise AssertionError("the scene turn asked the model")

    async def verify(*args, **kwargs):
        raise AssertionError("the scene turn asked the verifier")

    return SimpleNamespace(generate=generate, verify=verify)


class _Room:
    """One room built on the real stores, with the engines stubbed."""

    def __init__(self, directory: str, phrase: str, *, stt_ms: float = 0.0,
                 tts_ms: float = 0.0) -> None:
        conn = migrations_runner.connect(str(Path(directory) / "hub.db"))
        migrations_runner.migrate(conn)
        conn.execute("INSERT INTO homes(home_id, name) VALUES (?, ?)", (HOME, "Living room"))
        conn.commit()
        self.devices = DeviceStore(conn)
        for device_id, name, capability in (("lr-lamp", "Ceiling lamp", "on_off"),
                                            ("lr-strip", "LED strip", "color_rgb"),
                                            ("lr-tv", "TV", "on_off")):
            self.devices.save(Device(id=device_id, home_id=HOME, name=name, aliases=[],
                                     kind="light", capabilities=[capability], adapter="mqtt",
                                     adapter_config={"switch": device_id}))
        self.scenes = SceneStore(conn)
        self.scenes.ensure_presets(HOME)
        self.adapter = _Adapter()
        self.socket = _Socket()
        cfg = Config()
        cfg.server.permissions_enabled = False
        self.connection = hub_app.Connection(self.socket, cfg)
        self.connection.peer = "pc-1:5100"
        self.connection.home_id = HOME
        self.connection.utterance_id = new_ulid()
        self.connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
        self.connection._run_client_action = self._no_pc_action
        self.connection._stream_tts = self._tts(tts_ms)
        self.phrase = phrase
        self._stt_ms = stt_ms

    async def _no_pc_action(self, *args, **kwargs) -> dict[str, Any]:
        raise AssertionError("the cinema preset has no PC step")

    def _tts(self, tts_ms: float):
        async def stream(*args, **kwargs) -> None:
            if tts_ms:
                await asyncio.sleep(tts_ms / 1000.0)
        return stream

    async def one_turn(self) -> float:
        """Drive one real turn and return how long the hub took."""
        started = time.perf_counter()
        await self.connection._handle_utterance(b"\x01" * 16000)
        return time.perf_counter() - started

    def install(self) -> None:
        """Point the hub's module globals at this room (see ``_stubbed``)."""
        hub_app._stt = _Stt(self.phrase, cost_ms=self._stt_ms)
        hub_app._llm = _no_model()
        hub_app._tts = SimpleNamespace(sample_rate=48000)
        hub_app._voices = None
        hub_app._memory = None
        hub_app._dialogs = None
        hub_app._conversations = None
        hub_app._hub_conn = None
        hub_app._decider = None
        hub_app._decision_log = False
        hub_app._gpu = None
        hub_app._gpu_off = True
        hub_app._device_store = lambda: self.devices
        hub_app._device_tools = lambda: DeviceTools(self.devices, {"mqtt": self.adapter})
        hub_app._scene_store = lambda: self.scenes

    @property
    def applied(self) -> list[tuple[str, str, object]]:
        return list(self.adapter.calls)


#: The hub's module globals a room replaces, and ``_stubbed`` puts back.
_GLOBALS = ("_stt", "_llm", "_tts", "_voices", "_memory", "_dialogs", "_conversations",
            "_hub_conn", "_decider", "_decision_log", "_gpu", "_gpu_off", "_device_store",
            "_device_tools", "_scene_store")


class _stubbed:
    """Install a room's globals and put the hub's own back afterwards."""

    def __init__(self, room: _Room) -> None:
        self.room = room
        self.saved: dict[str, Any] = {}

    def __enter__(self) -> _Room:
        for name in _GLOBALS:
            self.saved[name] = getattr(hub_app, name, None)
        self.room.install()
        return self.room

    def __exit__(self, *exc_info) -> None:
        for name, value in self.saved.items():
            setattr(hub_app, name, value)


def measure(phrase: str = "", *, language: str = "ru", repeats: int = 5,
            stt_ms: float = 0.0, tts_ms: float = 0.0) -> dict[str, Any]:
    """Time the real turn for one sentence and report it against the budget.

    The sentence runs in a fresh room every time (fresh database, fresh
    devices), because a second turn in the room of a finished one would measure
    the caching of the first.
    """
    text = phrase or PHRASES.get(language, PHRASES["ru"])
    turns: list[float] = []
    answered: list[bool] = []
    answer = ""
    applied: list[tuple[str, str, object]] = []
    for _ in range(max(1, int(repeats))):
        directory = workdir()
        try:
            room = _Room(directory, text, stt_ms=stt_ms, tts_ms=tts_ms)
            with _stubbed(room):
                turns.append(asyncio.run(room.one_turn()))
            said = room.socket.said()
            answered.append(bool(said))
            answer = said[-1] if said else ""
            applied = room.applied
        finally:
            shutil.rmtree(directory, ignore_errors=True)
    engines_s = (max(0.0, float(stt_ms)) + max(0.0, float(tts_ms))) / 1000.0
    total = [value + engines_s for value in turns]
    report: dict[str, Any] = {
        "criterion": "ТЗ сценарий 1: сцена «кино» по голосовой команде",
        "language": language,
        "phrase": text,
        "budget_s": SCENE_BUDGET_S,
        "repeats": len(turns),
        "hub_turn_s": summarize(turns),
        "stt_ms": float(stt_ms),
        "tts_ms": float(tts_ms),
        "engines_included": bool(stt_ms or tts_ms),
        "total_s": summarize(total),
        "answer": answer,
        "answered_turns": sum(answered),
        "applied_steps": [[device, capability, json_value(value)]
                          for device, capability, value in applied],
    }
    report["within_budget"] = bool(total) and all(answered) and max(total) <= SCENE_BUDGET_S
    report["seconds_left_s"] = round(SCENE_BUDGET_S - max(total), 3) if total else 0.0
    if total and not all(answered):
        # The injected engine cost passed a stage budget of the hub itself
        # (ТЗ 15.1: 700 ms from the end of speech to the transcript), so the
        # turn ended in words instead of a scene. That is the designed
        # degradation showing up in a measurement, not a slow scene.
        report["note"] = ("a turn ended without an answer: the injected engine cost overran "
                          "a stage budget of the hub (ТЗ 15.1), so the room heard an error")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_scene_latency", description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--language", default="ru", choices=sorted(PHRASES),
                        help="which of the three sentences of the criterion to say")
    parser.add_argument("--phrase", default="", help="say this instead of the preset phrase")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--stt-ms", type=float, default=0.0,
                        help="the room's STT cost (0 = not measured on this machine)")
    parser.add_argument("--tts-ms", type=float, default=0.0,
                        help="the room's first-audio cost (0 = not measured here)")
    args = parser.parse_args(argv)
    load_config(args.config)  # the same --config surface as the other measurements
    report = measure(args.phrase, language=args.language, repeats=args.repeats,
                     stt_ms=args.stt_ms, tts_ms=args.tts_ms)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
