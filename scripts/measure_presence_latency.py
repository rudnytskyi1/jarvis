"""Приветствие по имени и фото незнакомца — с секундомером (ТЗ 15.3, сценарии 2 и 4).

ТЗ 15.3 даёт два бюджета: «вход человека → приветствие ≤ 2 с» и «незнакомец →
уведомление в Telegram ≤ 5 с». Оба меряются здесь по настоящему коду хаба:

``greeting``
    Друг заходит СПИНОЙ к камере (кадр без лица — приветствия быть не должно),
    потом поворачивается. Секундомер идёт от кадра с лицом до кадра ``say`` в
    комнате и включает настоящий цикл приветствий хаба (`_greeting_loop`,
    опрос 0,15 с), настоящие `RoomState`/`PresenceTracker` и удержание личности
    на треке (F-204). Подставлены только движок лиц и TTS; модель вызвать
    нельзя — приветствие скриптовое (F-302).

``stranger``
    Незнакомое лицо в комнате без владельца: настоящий `PresenceAlerts`
    (настоящая БД, настоящий рабочий цикл, правило `target: unknown`) и
    подставной транспорт Telegram. Секундомер идёт от ПЕРВОГО кадра, где
    появился незнакомец, до вызова отправки фото, поэтому в число входит и
    требование ТЗ «человек стабильно в кадре» (``min_stable_s``, по умолчанию
    2 с).

Usage::

    python scripts/measure_presence_latency.py
    python scripts/measure_presence_latency.py --repeats 3 --min-stable-s 2
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import protocol as proto  # noqa: E402
from common.config import Config, load_config  # noqa: E402
from hub import app as hub_app  # noqa: E402
from hub import migrations_runner  # noqa: E402
from hub.presence_alerts import PresenceAlerts  # noqa: E402
from hub.session import Session  # noqa: E402
from scripts.measure_overflow import summarize  # noqa: E402

#: ТЗ 15.3: «вход человека → приветствие ≤ 2 с».
GREETING_BUDGET_S = 2.0
#: ТЗ 15.3: «незнакомец → уведомление в Telegram ≤ 5 с».
ALERT_BUDGET_S = 5.0
#: ТЗ F-702: незнакомец должен постоять в кадре, прежде чем тревога уйдёт.
DEFAULT_MIN_STABLE_S = 2.0
#: The room of the scenarios, its owner's Telegram id and the friend's name.
HOME = "livingroom"
OWNER = 8322835915
GROUP = -10012345678
FRIEND = "Max"

BODY = {"id": "room:1", "box": [0.1, 0.1, 0.5, 0.9]}

SCRATCH = REPO_ROOT / ".tmp"


def jpeg(colour: tuple[int, int, int]) -> bytes:
    """A real (tiny) JPEG: the hub decodes the burst, so a fake blob would not do."""
    buffer = BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


#: The friend facing the camera, the friend's back, and a stranger.
FRIEND_JPEG, BACK_JPEG, STRANGER_JPEG = jpeg((10, 20, 30)), jpeg((40, 50, 60)), jpeg((70, 80, 90))
FRIEND_FACE = [1.0] + [0.0] * 511
STRANGER_FACE = [0.0, 1.0] + [0.0] * 510


def workdir() -> str:
    """A fresh directory for one measurement (see measure_scene_latency)."""
    for base in (SCRATCH, Path.cwd()):
        if not base.is_dir():
            continue
        path = base / f"presence-latency-{os.urandom(4).hex()}"
        try:
            path.mkdir()
        except OSError:
            continue
        return str(path)
    raise RuntimeError("nowhere to put the measurement database")


class Socket:
    """Just enough room to receive what the hub says."""

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


class Provider:
    """The Telegram stand-in: it acknowledges every photo it is handed."""

    ready = True

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_image(self, data, mime, caption, filename, **kwargs):
        self.sent.append({"kind": "image", "caption": caption, "bytes": len(data),
                          "chat_id": kwargs.get("private_reply_to_user_id")})
        return {"ok": True, "chat_id": self.sent[-1]["chat_id"],
                "message_id": len(self.sent) + 5}

    async def send_video(self, data, mime, caption, filename, **kwargs):  # pragma: no cover
        raise AssertionError("the scenario is about a photo")


def photo(embedding: list[float]) -> dict:
    """One located face, in the shape the face engine returns."""
    return {"embedding": embedding, "box": [0.2, 0.12, 0.3, 0.25], "score": 0.99}


class Engine:
    """A face engine that finds a face only in the jpegs it was given.

    ``known`` maps a jpeg to the name its face belongs to; a jpeg that is not
    in either mapping (or a profile that is not enrolled) stays a stranger.
    """

    available = True
    name = "stub"

    def __init__(self, faces: dict[bytes, list[dict]], known: dict[bytes, str] | None = None) -> None:
        self.faces = faces
        self.known = dict(known or {})

    def located_faces(self, jpeg: bytes) -> list[dict]:
        return [dict(item) for item in self.faces.get(jpeg, [])]

    def match(self, embedding, profiles):
        for jpeg, name in self.known.items():
            found = self.faces.get(jpeg) or []
            if found and found[0]["embedding"] == embedding:
                return name, 0.9
        return None, 0.2


def frame(jpeg: bytes, tracks: list[dict] | None = None) -> Any:
    """One presence frame of a room's camera burst."""
    return hub_app.ImageFrame(jpeg, 640, 480, 640, 480, source="camera", tracks=tracks,
                              received_at=time.monotonic())


async def no_tts(*args, **kwargs) -> None:
    """The room's TTS: the engine cost is the stand's number, not this one's."""
    return None


async def no_model(*args, **kwargs):
    """A greeting is scripted (F-302): asking the model here must fail loudly."""
    raise AssertionError("the scripted greeting asked the model")


class room:
    """A real hub connection on a real (temporary) hub database.

    The database is part of the room because the hub stores what it sees: the
    face of a live track (F-204), the presence event (F-301) and the alert
    delivery all live there. A measurement on a hub without a database would
    measure a hub that cannot store anything — that is not this system.
    """

    def __init__(self, engine: Engine, *, alerts: PresenceAlerts | None = None) -> None:
        self.engine = engine
        self.alerts = alerts
        #: Filled by ``__enter__`` so a service built before the room (the alert
        #: service is) can still find the connection it belongs to.
        self.holder: dict[str, Any] = {}
        self.saved: dict[str, Any] = {}
        self.directory: str | None = None

    def __enter__(self) -> tuple[hub_app.Connection, Socket]:
        from hub.auth import ClientTokenStore
        from hub.gateway import Gateway

        # Everything the hub builds lazily on its database is reset per room:
        # a store cached from the previous room would still point at the
        # previous database, which is now gone.
        for name in ("_face", "_voices", "_llm", "_tts", "_memory", "_dialogs",
                     "_conversations", "_presence_alerts", "_hub_conn", "_hub_db_path",
                     "_gateway", "_gateway_failed", "_face_tracks", "_body_embeddings",
                     "_presence_events", "_presence"):
            self.saved[name] = getattr(hub_app, name, None)
        self.directory = self.directory or workdir()
        conn = migrations_runner.connect(str(Path(self.directory) / "hub.db"))
        migrations_runner.migrate(conn)
        conn.execute("INSERT OR REPLACE INTO homes(home_id, name) VALUES (?, ?)",
                     (HOME, "Living room"))
        conn.execute("INSERT OR REPLACE INTO persons(person_id, display_name) VALUES (?, ?)",
                     ("p-" + FRIEND.casefold(), FRIEND))
        conn.commit()
        self.conn = conn
        hub_app._hub_db_path = lambda: Path(self.directory or "") / "hub.db"
        hub_app._gateway = Gateway(ClientTokenStore(conn))
        hub_app._gateway_failed = False
        hub_app._hub_conn = conn
        for name in ("_face_tracks", "_body_embeddings", "_presence", "_presence_events"):
            setattr(hub_app, name, None)
        socket = Socket()
        cfg = Config()
        cfg.server.permissions_enabled = False
        # The gallery is not what these scenarios are about, and its folder is
        # the hub's own data folder.
        cfg.server.face.appearance_enabled = False
        connection = hub_app.Connection(socket, cfg)
        connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
        connection.home_id = HOME
        connection.workplace_name = "Living room"
        connection.camera_state = {"persons": 1}
        connection._stream_tts = no_tts
        hub_app._face = self.engine
        hub_app._voices = SimpleNamespace(face_profiles=lambda: {FRIEND: [[1.0, 0.0]]})
        hub_app._llm = SimpleNamespace(generate=no_model, verify=no_model)
        hub_app._tts = SimpleNamespace(sample_rate=48000)
        hub_app._memory = None
        hub_app._dialogs = None
        hub_app._conversations = None
        hub_app._presence_alerts = self.alerts
        self.connection = connection
        self.holder["connection"] = connection
        return connection, socket

    def __exit__(self, *exc_info) -> None:
        for name, value in self.saved.items():
            setattr(hub_app, name, value)
        with contextlib.suppress(Exception):
            self.conn.close()
        if self.directory:
            shutil.rmtree(self.directory, ignore_errors=True)
            self.directory = None


async def speak_greeting(connection: hub_app.Connection, socket: Socket, *,
                         timeout: float = 4.0) -> float:
    """Run the shipped greeting loop until the room is spoken to; return the wait."""
    started = time.perf_counter()
    task = asyncio.create_task(connection._greeting_loop())
    try:
        while time.perf_counter() - started < timeout:
            if socket.said():
                return time.perf_counter() - started
            await asyncio.sleep(0.02)
        raise AssertionError(f"nobody was greeted within {timeout:.1f} s")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def silent_for(connection: hub_app.Connection, seconds: float) -> None:
    """Run the shipped greeting loop for a while, expecting no speech at all."""
    task = asyncio.create_task(connection._greeting_loop())
    try:
        await asyncio.sleep(seconds)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _greeting_turn(connection: hub_app.Connection, socket: Socket) -> float:
    """Сценарий 2: спина, поворот, приветствие по имени. Возвращает секунды."""
    await connection._match_presence([frame(BACK_JPEG, tracks=[BODY])])
    if socket.said():
        raise AssertionError("a back is not a stranger and not a name")
    await silent_for(connection, 0.4)
    if socket.said():
        raise AssertionError("a back to the camera earned a greeting")
    await connection._match_presence([frame(FRIEND_JPEG, tracks=[BODY])])
    return await speak_greeting(connection, socket)


def measure_greeting(*, repeats: int = 3) -> dict[str, Any]:
    """Вход человека (со спины) → приветствие по имени: сколько секунд."""
    waits: list[float] = []
    lines: list[str] = []
    for _ in range(max(1, int(repeats))):
        engine = Engine({FRIEND_JPEG: [photo(FRIEND_FACE)]}, {FRIEND_JPEG: FRIEND})
        with room(engine) as (connection, socket):
            waits.append(asyncio.run(_greeting_turn(connection, socket)))
            lines = socket.said()
    report = {
        "criterion": "ТЗ сценарий 2 / 15.3: вход человека → приветствие по имени",
        "budget_s": GREETING_BUDGET_S,
        "friend": FRIEND,
        "repeats": len(waits),
        "waits_s": summarize(waits),
        "said": lines,
        "named": bool(lines) and all(FRIEND in line for line in lines),
    }
    report["within_budget"] = bool(waits) and max(waits) <= GREETING_BUDGET_S
    return report


async def _stranger_turn(connection: hub_app.Connection, alerts: PresenceAlerts,
                         provider: Provider, *, gap_s: float = 0.3,
                         timeout: float = 9.0) -> float:
    """Сценарий 4: незнакомец в кадре → фото в Telegram. Возвращает секунды."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        await connection._match_presence([frame(STRANGER_JPEG, tracks=[BODY])])
        await alerts.drain()
        if provider.sent:
            return time.perf_counter() - started
        await asyncio.sleep(gap_s)
    raise AssertionError("the stranger never reached Telegram")


async def _stranger_repeat(directory: str, provider: Provider, min_stable_s: float) -> float:
    """One run of scenario 4: the alert service lives inside the event loop."""
    alerts = PresenceAlerts(directory, lambda: provider, lambda client_id=None: None,
                            OWNER, GROUP)
    try:
        alerts.save_rule({"enabled": True, "target": "unknown", "media": "photo",
                          "destination": "owner", "min_stable_s": float(min_stable_s),
                          "cooldown_s": 10})
        alerts.start()
        engine = Engine({STRANGER_JPEG: [photo(STRANGER_FACE)]})
        room_stub = room(engine, alerts=alerts)
        alerts.get_room = lambda client_id=None: room_stub.holder.get("connection")
        with room_stub as (connection, _socket):
            return await _stranger_turn(connection, alerts, provider)
    finally:
        await alerts.close()


def measure_stranger_alert(*, repeats: int = 3,
                           min_stable_s: float = DEFAULT_MIN_STABLE_S) -> dict[str, Any]:
    """Незнакомец в комнате без владельца → фото в Telegram: сколько секунд."""
    waits: list[float] = []
    deliveries: list[dict] = []
    for _ in range(max(1, int(repeats))):
        directory = workdir()
        provider = Provider()
        try:
            waits.append(asyncio.run(_stranger_repeat(directory, provider,
                                                      float(min_stable_s))))
            deliveries = provider.sent
        finally:
            shutil.rmtree(directory, ignore_errors=True)
    report = {
        "criterion": "ТЗ сценарий 4 / 15.3: незнакомец → фото в Telegram",
        "budget_s": ALERT_BUDGET_S,
        "min_stable_s": float(min_stable_s),
        "repeats": len(waits),
        "waits_s": summarize(waits),
        "deliveries": deliveries,
        "delivered": bool(deliveries) and deliveries[-1]["chat_id"] == OWNER,
    }
    report["within_budget"] = bool(waits) and max(waits) <= ALERT_BUDGET_S
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_presence_latency", description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-stable-s", type=float, default=DEFAULT_MIN_STABLE_S,
                        help="how long the stranger must stay in frame before the alert")
    args = parser.parse_args(argv)
    load_config(args.config)  # the same --config surface as the other measurements
    report = {
        "greeting": measure_greeting(repeats=args.repeats),
        "stranger_alert": measure_stranger_alert(repeats=args.repeats,
                                                 min_stable_s=args.min_stable_s),
    }
    report["within_budget"] = bool(report["greeting"]["within_budget"]
                                   and report["stranger_alert"]["within_budget"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
