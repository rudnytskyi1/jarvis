"""Приватность камеры (ТЗ F-303).

«Rowan, перестань смотреть» must really stop the frames — on the CLIENT, where
the frames come from — and must be visible in the HUD, because a promise nobody
can check is not privacy. The microphone keeps working: otherwise the same
voice could not bring the camera back.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from client.camera import CameraService
from client.overlay import OverlayHUD
from client.privacy import PrivacyMode
from common import protocol
from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub import privacy as privacy_mod
from hub.presence_state import PresenceLog, PresenceState
from hub.room_state import RoomState

# --- слова -------------------------------------------------------------------


@pytest.mark.parametrize("said", [
    "Rowan, перестань смотреть", "перестань смотреть", "не смотри на меня",
    "выключи камеру", "прекрати подглядывать", "Rowan, stop watching",
    "don't watch me", "turn the camera off", "no me mires", "apaga la cámara",
    "deja de mirar",
])
def test_the_request_to_stop_watching_is_understood(said):
    assert privacy_mod.privacy_command(said) is True


@pytest.mark.parametrize("said", [
    "Rowan, смотри снова", "включи камеру", "можешь снова смотреть",
    "watch again", "look again", "turn the camera back on", "mira otra vez",
    "enciende la cámara",
])
def test_the_request_to_watch_again_is_understood(said):
    assert privacy_mod.privacy_command(said) is False


@pytest.mark.parametrize("said", [
    "Rowan, посмотри, кто там", "look at the room", "что ты видишь?",
    "включи свет", "привет", "", None,
])
def test_ordinary_speech_is_not_a_privacy_command(said):
    assert privacy_mod.privacy_command(said) is None
    assert privacy_mod.looks_like_look_request(said) is (said in (
        "Rowan, посмотри, кто там", "look at the room"))


def test_the_confirmation_says_what_changed_in_each_language():
    assert "выключена" in privacy_mod.confirmation(True, "ru")
    assert "watching again" in privacy_mod.confirmation(False, "en")
    assert "apagada" in privacy_mod.confirmation(True, "es", changed=False)
    assert privacy_mod.confirmation(False, "ru", changed=False).startswith("Камера и так")


# --- клиент ------------------------------------------------------------------


def _camera(**overrides) -> CameraService:
    camera = CameraService.__new__(CameraService)
    camera.privacy = PrivacyMode("ru")
    camera._enabled = True
    camera._loop = None
    camera._send_json = None
    camera._send_bytes = None
    camera._send_lock = None
    camera._stop_event = __import__("threading").Event()
    camera._tracks_message = True
    camera._sent_state = (1, ())
    camera._sent_state_at = 0.0
    camera._last_tracks_at = 0.0
    camera._sent_track_ids = ("a:1",)
    camera._sent_tracks_at = 0.0
    camera._track_reports = []
    camera._tracks = []
    camera._cv2 = None
    camera._presence_pending = __import__("threading").Event()
    camera._last_presence_push = 0.0
    camera.face_check_interval_s = 0.5
    for key, value in overrides.items():
        setattr(camera, key, value)
    return camera


def test_privacy_stops_every_kind_of_camera_traffic():
    camera = _camera()
    sent: list[Any] = []
    camera._submit = lambda coro: sent.append(coro) or coro.close()

    assert camera.set_privacy(True, reason="voice") is True
    assert camera.privacy.on and not camera.privacy.allows_frames
    assert camera.privacy.indicator() == "Камера выключена"
    camera._publish_state(2, {"bottle": 1})
    camera._publish_tracks()
    camera._publish_body_crops(object())
    camera._maybe_push_presence(object())
    assert camera._tracks == [] and camera._track_reports == [], "детектор молчит"

    async def scenario():
        await camera.serve_request("c1", burst=3)

    asyncio.run(scenario())
    assert camera.set_privacy(True) is False, "повтор не событие"
    assert camera.set_privacy(False) is True and camera.privacy.indicator() == ""


def test_a_requested_frame_is_refused_in_privacy_mode():
    camera = _camera()
    errors: list[tuple[str, str]] = []

    async def send_error(request_id, error):
        errors.append((request_id, error))

    camera._send_error = send_error
    camera._capture_burst_sync = lambda *a, **k: pytest.fail("никаких снимков в приватном режиме")
    camera.set_privacy(True, reason="voice")
    asyncio.run(camera.serve_request("c1"))
    assert errors == [("c1", "the camera is off (privacy mode)")]


def test_the_privacy_state_is_announced_to_the_hub():
    camera = _camera()
    states: list[dict[str, Any]] = []

    async def send_state(payload):
        states.append(payload)

    camera._send_state = send_state
    camera._loop = object()
    camera._submit = lambda coro: asyncio.run(coro)
    camera.set_privacy(True, reason="voice")
    assert states and states[0]["privacy"] is True and states[0]["persons"] == 0
    states.clear()
    camera.announce_privacy()
    assert states and states[0]["privacy"] is True, "переподключение говорит снова"


def test_the_hud_shows_the_camera_off_badge():
    hud = OverlayHUD.__new__(OverlayHUD)
    hud.enabled = True
    hud._bridge = Mock()
    hud._camera_off = ""
    hud.camera_privacy("Камера выключена")
    hud._bridge.camera_changed.emit.assert_called_once_with("Камера выключена")
    hud.camera_privacy("")
    assert hud._bridge.camera_changed.emit.call_args.args == ("",)
    hud.enabled = False
    hud.camera_privacy("Камера выключена")
    assert hud._bridge.camera_changed.emit.call_count == 2


def test_the_hud_page_defines_the_badge_and_its_hook():
    page = (Path(__file__).resolve().parents[1]
            / "client" / "overlay_web" / "hud.html").read_text(encoding="utf-8")
    assert 'id="camera-off"' in page and 'id="camera-off-label"' in page
    assert "opts.camera_off" in page
    assert "camera-off.visible" in page


def test_the_hello_says_what_the_camera_really_has():
    from client.main import build_hello

    plain = build_hello(SimpleNamespace(client_id="office-1", devices=[]))
    assert plain["privacy"] is False
    private = build_hello(SimpleNamespace(client_id="office-1", devices=[]), privacy=True)
    assert private["privacy"] is True
    assert protocol.MSG_PRIVACY == "privacy"
    assert protocol.Privacy(on=True, reason="voice").on is True
    assert protocol.Privacy(on=True, reason="voice").type == "privacy"
    assert protocol.Hello(privacy=True).privacy is True


# --- хаб ---------------------------------------------------------------------


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


def _connection(hub_db, monkeypatch, *, speaker="Макс", role: str | None = "admin"):
    from hub.auth import ClientTokenStore
    from hub.gateway import Gateway

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_gateway", Gateway(ClientTokenStore(hub_db)))
    monkeypatch.setattr(hub_app, "_presence", PresenceState())
    monkeypatch.setattr(hub_app, "_presence_events", PresenceLog(hub_db))
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.cfg = Config()
    connection._privacy = False
    connection._speaker_name = speaker
    connection._speaker_role = role or ""
    connection._speaker_score = 0.9 if speaker else 0.0
    connection._speaker_language = "ru"
    connection._reply_language = "ru"
    connection.room = RoomState()
    connection.camera_state = None
    connection._presence_has_tracks = False
    connection.presence = hub_app.PresenceTracker(ttl_s=30.0)
    sent: list[dict[str, Any]] = []

    async def send_json(payload):
        sent.append(payload)

    connection.send_json = send_json
    connection.sent = sent
    return connection


def test_the_room_switches_its_camera_off_and_says_so(hub_db, monkeypatch):
    audit = _Audit()
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    connection = _connection(hub_db, monkeypatch)

    spoken = asyncio.run(connection._privacy_turn("Rowan, перестань смотреть"))
    assert "камера выключена" in spoken
    assert connection._privacy is True
    assert connection.sent == [{"type": protocol.MSG_PRIVACY, "on": True, "reason": "voice"}]
    assert audit.rows == [{"action": "privacy.on", "actor": "p-max", "target": "pc-1:5100",
                           "home_id": "livingroom",
                           "detail": {"reason": "voice", "source": "camera"}}]
    assert asyncio.run(connection._privacy_turn("Rowan, посмотри, кто там")) is None


def test_an_ordinary_reply_never_reaches_the_camera(hub_db, monkeypatch):
    audit = _Audit()
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    connection = _connection(hub_db, monkeypatch)
    assert asyncio.run(connection._privacy_turn("включи свет")) is None
    assert connection.sent == [] and audit.rows == []
    assert connection._privacy is False


def test_the_camera_comes_back_and_the_second_request_is_just_words(hub_db, monkeypatch):
    audit = _Audit()
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    connection = _connection(hub_db, monkeypatch)
    asyncio.run(connection._privacy_turn("перестань смотреть"))
    again = asyncio.run(connection._privacy_turn("Rowan, перестань смотреть"))
    assert again.startswith("Камера уже выключена") and len(audit.rows) == 1
    back = asyncio.run(connection._privacy_turn("Rowan, смотри снова"))
    assert "снова смотрит" in back and connection._privacy is False
    assert [row["action"] for row in audit.rows] == ["privacy.on", "privacy.off"]


def test_switching_the_camera_back_on_needs_a_recognised_speaker(hub_db, monkeypatch):
    audit = _Audit()
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    connection = _connection(hub_db, monkeypatch, speaker="", role=None)
    connection._privacy = True
    refused = asyncio.run(connection._privacy_turn("смотри снова"))
    assert "did not recognise" in refused and connection._privacy is True
    assert audit.rows == [] and connection.sent == []
    # выключить камеру может любой в комнате: это безопасная сторона
    connection._privacy = False
    assert "выключена" in asyncio.run(connection._privacy_turn("перестань смотреть"))
    assert connection.sent[-1]["on"] is True


def test_a_presence_frame_is_dropped_while_the_camera_is_off(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection._privacy = True
    connection._presence_burst_id = ""
    connection._presence_burst_frames = []
    connection._on_presence_frame = lambda frames: pytest.fail("кадр не должен дойти")
    frame = SimpleNamespace(id="p1", seq=1, of=1)
    connection._buffer_presence_frame(frame)
    assert connection._presence_burst_frames == []
    connection._privacy = False
    connection._on_presence_frame = lambda frames: frames.append("seen") or None and None
    seen: list[str] = []
    connection._on_presence_frame = lambda frames: seen.append("matched")
    connection._buffer_presence_frame(frame)
    assert seen == ["matched"]


def test_the_client_reports_and_the_hub_believes_the_camera_state(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection._on_camera_state({"privacy": True, "persons": 0})
    assert connection._privacy is True
    connection._on_camera_state({"persons": 1})
    assert connection._privacy is True, "сообщение без поля ничего не меняет"
    connection._on_camera_state({"privacy": False, "persons": 1})
    assert connection._privacy is False


def test_a_reconnected_private_client_tells_the_hub_at_hello(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    payload = {"privacy": True, "capabilities": [], "devices": [],
               "workplace_name": "Room", "client_id": "pc-1"}

    async def authorize(_payload):
        return True

    connection._authorize = authorize
    connection.send_json = _recorder()

    async def send_room_config():
        return None

    async def send_release():
        return None

    connection._send_room_config = send_room_config
    connection._send_release = send_release
    connection._start_greeting_task = lambda: None
    connection._start_identity_task = lambda: None
    monkeypatch.setattr(hub_app, "_telegram_access", None)
    connection.presence_text = lambda: "Room now: nobody"
    asyncio.run(connection._on_hello(payload))
    assert connection._privacy is True


def _recorder():
    async def record(payload):
        return None

    return record


def test_the_sight_status_knows_the_camera_is_off(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection._privacy = True
    connection.room.tracks.clear()
    connection.camera_state = {"persons": 0, "objects": {}, "ts": time.time()}
    assert connection._sight_status(()) == "empty"
    assert connection._privacy is True
