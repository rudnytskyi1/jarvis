"""HUD v2 (ТЗ F-708).

What the room screen has to show while a conversation runs: whether the brain
is listening / thinking / speaking (already there), the live transcript of the
current utterance, who is speaking, the follow-up window, the state of the
brain itself (online / queue / offline) and the camera (on / off) - plus the
names over the tracks when somebody asks to SEE the room camera.

Everything here is checked on the real classes: the hub's status frame and its
broadcast, the queue hook that produces it, the client's routing of the frame,
the labels the hub puts over the picture, and the hooks both HUD pages define.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from client.main import JarvisClient
from common import protocol
from hub import app as hub_app
from hub import camera_view
from hub.gpu_queue import PRIORITY_UTTERANCE, GpuQueue
from hub.room_state import RoomState

REPO_ROOT = Path(__file__).resolve().parents[1]


def page(name: str) -> str:
    return (REPO_ROOT / "client" / "overlay_web" / name).read_text(encoding="utf-8")


# --- страница ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["hud.html", "chat.html"])
def test_both_hud_pages_can_show_the_hub_and_the_track_names(name):
    text = page(name)
    assert 'id="hub"' in text and "hub-label" in text
    assert 'id="track-layer"' in text and "track-label" in text
    assert "window.hudHub" in text and "window.hudTracks" in text
    # Очередь и оффлайн — разные состояния, а не одно «не онлайн».
    assert ".queue" in text and ".offline" in text


@pytest.mark.parametrize("name", ["hud.html", "chat.html"])
def test_the_pages_clear_the_live_transcript_at_the_end_of_a_turn(name):
    assert "clear" in page(name)


def test_the_hud_page_still_carries_the_camera_badge_and_the_followup_bar():
    text = page("hud.html")
    assert 'id="camera-off"' in text and "opts.camera_off" in text
    assert 'id="transcript"' in text and "window.hudFollowup" in text


# --- протокол ----------------------------------------------------------------


def test_the_hub_status_frame_is_part_of_the_protocol():
    assert protocol.MSG_HUB_STATUS == "hub_status"
    assert protocol.MSG_HUB_STATUS in protocol.SERVER_MESSAGE_TYPES
    assert "MSG_HUB_STATUS" in protocol.__all__
    # Подпись на экране может быть потеряна под нагрузкой — она не ответ.
    assert protocol.is_background_server_frame({"type": protocol.MSG_HUB_STATUS}) is True


# --- хаб: статус --------------------------------------------------------------


class _WS:
    from starlette.websockets import WebSocketState as _State

    client_state = _State.CONNECTED


def _room(monkeypatch, *, delivered: bool = True) -> Any:
    """A Connection stub that records the frames it is asked to queue."""
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.ws = _WS()
    connection._hub_status_sent = None
    frames: list[dict[str, Any]] = []

    async def queue_frame(payload, **kwargs):
        frames.append(payload)
        return delivered

    connection.queue_frame = queue_frame
    connection.frames = frames
    return connection


def test_a_hub_without_a_queue_tells_the_room_it_is_online(monkeypatch):
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    assert hub_app.hub_status_frame() == {
        "type": protocol.MSG_HUB_STATUS, "state": "online", "queue": 0}


def test_a_waiting_job_puts_the_hub_state_into_the_queue(monkeypatch):
    queue = GpuQueue(max_concurrent=1)
    monkeypatch.setattr(hub_app, "_gpu", queue)
    monkeypatch.setattr(hub_app, "_gpu_off", False)
    assert hub_app.hub_status_frame()["state"] == "online"

    async def scenario():
        release = asyncio.Event()
        running = asyncio.ensure_future(queue.submit(
            PRIORITY_UTTERANCE, "livingroom", lambda: _wait(release)))
        waiting = asyncio.ensure_future(queue.submit(
            PRIORITY_UTTERANCE, "bedroom", lambda: _wait(release)))
        await asyncio.sleep(0)
        frame = hub_app.hub_status_frame()
        release.set()
        await asyncio.gather(running, waiting)
        return frame, hub_app.hub_status_frame()

    frame, after = asyncio.run(scenario())
    assert frame == {"type": protocol.MSG_HUB_STATUS, "state": "queue", "queue": 1}
    assert after["state"] == "online" and after["queue"] == 0


async def _wait(event: asyncio.Event) -> str:
    await event.wait()
    return "done"


def test_the_same_status_is_not_sent_to_a_room_twice(monkeypatch):
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    connection = _room(monkeypatch)
    frame = hub_app.hub_status_frame()
    assert asyncio.run(connection.publish_hub_status(frame)) is True
    assert asyncio.run(connection.publish_hub_status(frame)) is False
    assert connection.frames == [frame]
    queued = {"type": protocol.MSG_HUB_STATUS, "state": "queue", "queue": 2}
    assert asyncio.run(connection.publish_hub_status(queued)) is True
    assert connection.frames == [frame, queued]


def test_a_dropped_status_is_sent_again_later(monkeypatch):
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    connection = _room(monkeypatch, delivered=False)
    assert asyncio.run(connection.publish_hub_status()) is False
    assert connection._hub_status_sent is None, "потерянный кадр не считается отправленным"


def test_the_broadcast_reaches_every_connected_room(monkeypatch):
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    first, second = _room(monkeypatch), _room(monkeypatch)
    monkeypatch.setattr(hub_app, "_connections", {first, second})
    delivered = asyncio.run(hub_app.broadcast_hub_status())
    assert delivered == 2
    assert all(room.frames for room in (first, second))


# --- хаб: очередь сама сообщает об изменениях --------------------------------


def test_the_queue_reports_every_change_of_its_shape():
    seen: list[int] = []
    queue = GpuQueue(max_concurrent=1, on_change=lambda: seen.append(len(seen)))

    async def scenario():
        await queue.submit(PRIORITY_UTTERANCE, "livingroom", lambda: _wait_done())

    asyncio.run(scenario())
    # submitted (waiting -> running) and finished (idle again)
    assert len(seen) >= 2


async def _wait_done() -> str:
    return "done"


def test_a_broken_hook_cannot_break_the_gpu_queue():
    def explode() -> None:
        raise RuntimeError("hook")

    queue = GpuQueue(max_concurrent=1, on_change=explode)
    assert asyncio.run(queue.submit(PRIORITY_UTTERANCE, "livingroom", _wait_done)) == "done"


# --- «покажи камеру» ---------------------------------------------------------


@pytest.mark.parametrize("said", [
    "Rowan, покажи камеру", "покажи мне камеру", "покажи комнату",
    "show me the camera", "show the room", "what do you see",
    "muéstrame la cámara", "qué ves",
])
def test_showing_the_camera_is_understood(said):
    assert camera_view.show_camera_requested(said) is True


@pytest.mark.parametrize("said", [
    "Rowan, посмотри, кто там", "кто это", "сколько людей в комнате",
    "who is there", "how many people are in the room", "quién está",
    "включи свет", "привет", "", None,
])
def test_a_question_about_the_frame_is_not_a_request_to_show_it(said):
    assert camera_view.show_camera_requested(said) is False


def test_the_answer_is_in_the_language_of_the_room():
    assert camera_view.language_of("ru-RU") == "ru"
    assert camera_view.language_of("es") == "es"
    assert camera_view.language_of("en") == "en"
    assert camera_view.language_of("") == "en"
    assert "камеру" in camera_view.showing_text("ru")
    assert "Камеры" in camera_view.no_camera_text("ru")
    assert camera_view.showing_text("es").startswith("Mostrando")


def test_labels_come_from_the_frame_and_the_names_from_the_hub():
    tracks = [{"id": "t1", "box": [.1, .2, .3, .6]},
              {"id": "t2", "box": [.5, .2, .7, .6]}]
    labels = camera_view.track_labels(tracks, {"t1": "Макс"})
    assert labels == [{"name": "Макс", "box": [.1, .2, .3, .6]},
                      {"name": "", "box": [.5, .2, .7, .6]}]


def test_labels_read_the_protocol_track_shape_and_skip_broken_boxes():
    rows = {"a": {"track_id": "a", "bbox": [.1, .1, .4, .4], "name": "Макс"},
            "b": {"track_id": "b", "bbox": [.4, .4, .1, .1]},
            "c": {"track_id": "c"},
            "d": None}
    labels = camera_view.track_labels(rows)
    assert labels == [{"name": "Макс", "box": [.1, .1, .4, .4]}]


def test_a_room_name_is_only_a_name_the_hub_really_has():
    tracks = [{"id": "t1", "box": [.1, .1, .2, .2]}]
    assert camera_view.track_labels(tracks, {"t1": {"name": ""}}) == [
        {"name": "", "box": [.1, .1, .2, .2]}]
    assert camera_view.track_labels(tracks, {"t9": "Макс"})[0]["name"] == ""
    assert camera_view.track_labels(None) == []


def _camera_connection(frame: Any, *, tracks: dict[str, Any] | None = None):
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection._reply_language = "ru"
    connection._image_seq = 0
    connection.room = RoomState()
    for key, row in (tracks or {}).items():
        connection.room.tracks[key] = row
    shown: list[dict[str, Any]] = []

    async def request(frame_id):
        return frame

    async def send_image_show(jpeg, w, h, title, ttl_s, tracks=None):
        shown.append({"jpeg": jpeg, "w": w, "h": h, "title": title, "ttl_s": ttl_s,
                      "tracks": tracks})

    connection._request_camera_frame_full = request
    connection._send_image_show = send_image_show
    connection.shown = shown
    return connection


def _frame(**overrides: Any) -> Any:
    values = {"jpeg": b"jpeg", "w": 640, "h": 480,
              "tracks": [{"id": "t1", "box": [.1, .1, .3, .5]}]}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_the_hub_shows_the_room_camera_with_the_names_it_knows():
    frame = _frame(tracks=[{"id": "t1", "box": [.1, .1, .3, .5]},
                           {"id": "t2", "box": [.6, .1, .8, .5]}])
    connection = _camera_connection(frame, tracks={"t1": {"name": "Макс", "box": [.1, .1, .3, .5]},
                                                  "t2": {"name": None, "box": [.6, .1, .8, .5]}})
    spoken = asyncio.run(connection._show_camera_turn("Rowan, покажи камеру"))
    assert spoken == camera_view.showing_text("ru")
    (shown,) = connection.shown
    assert shown["jpeg"] == b"jpeg" and (shown["w"], shown["h"]) == (640, 480)
    assert shown["title"] == camera_view.title_text("ru")
    assert shown["tracks"] == [{"name": "Макс", "box": [.1, .1, .3, .5]},
                               {"name": "", "box": [.6, .1, .8, .5]}]


def test_an_ordinary_request_never_touches_the_camera():
    connection = _camera_connection(_frame())
    assert asyncio.run(connection._show_camera_turn("включи свет")) is None
    assert asyncio.run(connection._show_camera_turn("Rowan, кто там")) is None
    assert connection.shown == []


def test_a_silent_camera_is_admitted_instead_of_faked():
    connection = _camera_connection("the camera is off (privacy mode)")
    spoken = asyncio.run(connection._show_camera_turn("покажи камеру"))
    assert spoken == camera_view.no_camera_text("ru")
    assert connection.shown == []


# --- клиент ------------------------------------------------------------------


def _client_stub() -> Any:
    client = JarvisClient.__new__(JarvisClient)
    client._quiet_turn = False
    client._track_labels_token = 0
    client._local_tasks = set()
    client.overlay = _Recorder()
    return client


class _Recorder:
    def __init__(self) -> None:
        self.hub: list[dict[str, Any]] = []
        self.drawn: list[Any] = []
        self.transcripts: list[dict[str, Any]] = []

    def hub_state(self, payload: dict[str, Any]) -> None:
        self.hub.append(payload)

    def tracks(self, payload: Any) -> None:
        self.drawn.append(payload)

    def transcript(self, payload: dict[str, Any]) -> None:
        self.transcripts.append(payload)


@pytest.mark.parametrize("mode", ["idle", "conversation"])
def test_the_client_shows_the_hub_state_in_both_modes(mode):
    client = _client_stub()
    client._mode = mode
    frame = {"type": protocol.MSG_HUB_STATUS, "state": "queue", "queue": 2}
    asyncio.run(client._route_message(frame))
    assert client.overlay.hub == [frame]


def test_silencing_the_room_takes_the_live_transcript_down():
    client = SimpleNamespace(
        overlay=_Recorder(), audio_out=SimpleNamespace(cancel_pending=lambda: None),
        _recording_live=False, _quiet_turn=False, _barge_preroll=b"", _interrupt_id="",
        _listen_hint_s=0.0, _proactive_listen_s=0.0, _enrollment_until=0.0,
        _selection_until=0.0, _idle_interrupted=False, _idle_playing=False,
        _idle_stream_active=False, _idle_tts_active=False, _tts_active=False,
        _pending_image_show=None, _action_task=None,
        _show_status=lambda *args: None, _clear_inbox=lambda: None)
    client.overlay.cancel_voice_confirmation = lambda: None
    client.overlay.typing = lambda *args: None
    client.overlay.scan_screen = lambda *args: None
    client.overlay.speaker = lambda *args: None
    client.overlay.chat = lambda *args: None
    client.overlay.set_state = lambda *args: None
    JarvisClient._silence_locally(client)
    assert {"clear": True} in client.overlay.transcripts


def test_the_turn_that_is_over_takes_its_transcript_with_it():
    source = (REPO_ROOT / "client" / "main.py").read_text(encoding="utf-8")
    # The turn's own ``finally`` block clears the line the person was reading.
    assert 'self.overlay.transcript({"clear": True})' in source


def test_the_client_draws_the_names_over_the_frame_it_just_received():
    async def scenario():
        client = _client_stub()
        header = {"type": protocol.MSG_IMAGE_SHOW, "w": 640, "h": 480, "ttl_s": 0.05,
                  "tracks": [{"name": "Макс", "box": [.1, .1, .3, .5]}]}
        client._track_labels_token = 0
        client._track_labels_token += 1
        client._show_track_labels(header)
        assert client.overlay.drawn == [
            {"tracks": header["tracks"], "w": 640, "h": 480}]
        await asyncio.gather(*tuple(client._local_tasks))
        return client

    client = asyncio.run(scenario())
    assert client.overlay.drawn[-1] == [], "подписи живут столько же, сколько кадр"


def test_a_photo_without_tracks_takes_the_names_down():
    client = _client_stub()
    client._show_track_labels({"type": protocol.MSG_IMAGE_SHOW, "w": 640, "h": 480})
    assert client.overlay.drawn == [[]]


def test_an_older_photo_may_not_clear_a_newer_photo_labels():
    async def scenario():
        client = _client_stub()
        client._show_track_labels({"w": 1, "h": 1, "ttl_s": 0.05,
                                   "tracks": [{"name": "A", "box": [.1, .1, .2, .2]}]})
        stale = set(client._local_tasks)
        client._show_track_labels({"w": 1, "h": 1, "ttl_s": 5,
                                   "tracks": [{"name": "B", "box": [.1, .1, .2, .2]}]})
        await asyncio.gather(*tuple(stale))
        return client

    client = asyncio.run(scenario())
    assert client.overlay.drawn[-1] != [], "старый таймер не снимает новые подписи"


def test_the_overlay_hands_the_hub_state_and_the_track_names_to_the_page():
    from client.overlay import OverlayHUD

    hud = OverlayHUD({"enabled": True})
    hud._bridge = _FakeBridge()
    hud.hub_state({"state": "offline"})
    hud.tracks({"tracks": [{"name": "Макс", "box": [.1, .1, .2, .2]}], "w": 2, "h": 2})
    assert json.loads(hud._bridge.hub_changed.calls[0][0]) == {"state": "offline"}
    assert json.loads(hud._bridge.tracks_changed.calls[0][0])["w"] == 2


class _FakeSignal:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def emit(self, *args: Any) -> None:
        self.calls.append(args)


class _FakeBridge:
    def __init__(self) -> None:
        self.hub_changed = _FakeSignal()
        self.tracks_changed = _FakeSignal()
