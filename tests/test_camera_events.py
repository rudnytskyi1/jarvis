"""One ``event_id`` per background camera event (ТЗ 4.5)."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common import protocol as proto
from common.config import Config
from common.ids import is_ulid, new_ulid
from hub import app as hub_app
from hub import camera_events as camera_events_mod
from hub.camera_events import KIND_CAMERA, KIND_CLIP, KIND_PRESENCE, KIND_SCREENSHOT, CameraEvents


@pytest.fixture(autouse=True)
def fresh_events(monkeypatch):
    """Each test counts into its own registry instead of the hub-wide one."""
    events = CameraEvents()
    monkeypatch.setattr(camera_events_mod, "camera_events", events)
    return events


def _connection(**attributes):
    conn = hub_app.Connection(SimpleNamespace(client=None), Config())
    conn.send_json = AsyncMock()
    conn.home_id = "livingroom"
    for name, value in attributes.items():
        setattr(conn, name, value)
    return conn


# --- the id the hub mints when it asks for a picture ----------------------


def test_a_frame_request_carries_a_fresh_event_id():
    async def scenario():
        conn = _connection()
        sent = []

        async def send_json(payload):
            sent.append(payload)

        conn.send_json = send_json
        # No client answers: the request times out, which is fine here — the
        # test is about what went out on the wire.
        result = await conn._request_image_locked(
            hub_app.SOURCE_CAMERA, "c1", proto.MSG_CAMERA_REQUEST, 0.05)
        return sent[0], result

    payload, result = asyncio.run(scenario())
    assert payload["type"] == proto.MSG_CAMERA_REQUEST
    assert payload["id"] == "c1"
    assert is_ulid(payload["event_id"])
    assert result == proto.ERR_CLIENT_TIMEOUT


def test_two_requests_never_share_an_event_id():
    async def scenario():
        conn = _connection()
        sent = []

        async def send_json(payload):
            sent.append(payload)

        conn.send_json = send_json
        for _ in range(2):
            loop = asyncio.get_running_loop()
            conn._image_futures[hub_app.SOURCE_CAMERA] = loop.create_future()
            await conn._request_image_locked(hub_app.SOURCE_CAMERA, "c1", proto.MSG_CAMERA_REQUEST, 0.05)
        return sent

    payloads = asyncio.run(scenario())
    assert len(payloads) == 2
    assert payloads[0]["event_id"] != payloads[1]["event_id"]


# --- the hub adopts the id the client echoes ------------------------------


def test_a_requested_frame_keeps_the_event_id_of_its_request(fresh_events):
    from collections import deque

    conn = _connection()
    event_id = new_ulid()
    conn._image_ids[hub_app.SOURCE_CAMERA] = "c1"
    conn._image_event_ids[hub_app.SOURCE_CAMERA] = event_id
    conn._image_incoming[hub_app.SOURCE_CAMERA] = deque()

    conn._on_image_header(hub_app.SOURCE_CAMERA, {"id": "c1", "w": 10, "h": 10,
                                                  "event_id": event_id})
    frame = conn._build_frame(b"jpeg", conn._image_header)

    assert frame.event_id == event_id
    assert frame.reason == hub_app.REASON_REQUEST


def test_a_client_that_sends_no_id_gets_the_one_the_hub_asked_with(fresh_events):
    from collections import deque

    conn = _connection()
    event_id = new_ulid()
    conn._image_ids[hub_app.SOURCE_CAMERA] = "c1"
    conn._image_event_ids[hub_app.SOURCE_CAMERA] = event_id
    conn._image_incoming[hub_app.SOURCE_CAMERA] = deque()

    conn._on_image_header(hub_app.SOURCE_CAMERA, {"id": "c1", "w": 10, "h": 10})

    assert conn._image_header["event_id"] == event_id


def test_a_presence_push_without_an_id_still_gets_one(fresh_events):
    conn = _connection()
    conn._on_image_header(hub_app.SOURCE_CAMERA, {"id": "p7", "reason": "presence", "w": 4, "h": 4})

    event_id = conn._image_header["event_id"]
    assert is_ulid(event_id)
    snapshot = fresh_events.snapshot()
    assert snapshot["total"] == 1
    assert snapshot["by_kind"] == {KIND_PRESENCE: 1}
    assert snapshot["last"]["event_id"] == event_id
    assert snapshot["last"]["home_id"] == "livingroom"


def test_a_presence_burst_is_one_event_however_many_frames_it_sends(fresh_events):
    conn = _connection()
    event_id = new_ulid()
    for seq in (1, 2, 3):
        conn._on_image_header(hub_app.SOURCE_CAMERA, {
            "id": "p7", "reason": "presence", "w": 4, "h": 4, "seq": seq, "of": 3,
            "event_id": event_id,
        })
    snapshot = fresh_events.snapshot()
    assert snapshot["total"] == 1, "a burst is counted once, not per frame"
    assert snapshot["last"]["frames"] == 3


def test_a_received_frame_is_counted_by_its_kind(fresh_events):
    conn = _connection()
    conn.session = None
    event_id = new_ulid()
    conn._expect_image = hub_app.SOURCE_CAMERA
    conn._image_header = {"source": hub_app.SOURCE_CAMERA, "reason": hub_app.REASON_PRESENCE,
                          "id": "p1", "event_id": event_id, "w": 4, "h": 4}
    conn._buffer_presence_frame = Mock()

    conn._deliver_image(b"\xff\xd8\xff\xe0jpeg")

    assert fresh_events.snapshot()["by_kind"] == {KIND_PRESENCE: 1}
    conn._buffer_presence_frame.assert_called_once()
    assert conn._buffer_presence_frame.call_args.args[0].event_id == event_id


def test_a_capture_failure_is_counted_against_its_event(fresh_events):
    conn = _connection()
    event_id = new_ulid()
    conn._image_event_ids[hub_app.SOURCE_CAMERA] = event_id

    conn._on_image_error(hub_app.SOURCE_CAMERA, {"error": "no camera"})

    snapshot = fresh_events.snapshot()
    assert snapshot["failed"] == 1
    assert snapshot["last"]["event_id"] == event_id
    assert snapshot["last"]["kind"] == KIND_CAMERA
    assert snapshot["last"]["ok"] is False


def test_the_screenshot_event_is_counted_as_its_own_kind(fresh_events):
    conn = _connection()
    event_id = new_ulid()
    conn._image_event_ids[hub_app.SOURCE_SCREEN] = event_id

    conn._on_image_error(hub_app.SOURCE_SCREEN, {"error": "no overlay"})

    assert fresh_events.snapshot()["by_kind"] == {KIND_SCREENSHOT: 1}


def test_a_clip_request_and_its_answer_share_one_event(fresh_events):
    conn = _connection()
    conn._can_camera_clip = True
    sent = []

    async def send_json(payload):
        sent.append(payload)

    conn.send_json = send_json

    async def scenario():
        task = asyncio.create_task(conn._request_camera_clip("clip1", 3, 8))
        await asyncio.sleep(0)
        payload = sent[0]
        clip = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
        conn._on_clip_header({"id": "clip1", "format": "mp4", "bytes": len(clip),
                              "event_id": payload["event_id"]})
        conn._on_clip_binary(clip)
        return payload, await task

    payload, result = asyncio.run(scenario())
    assert payload["type"] == proto.MSG_CAMERA_CLIP_REQUEST
    assert is_ulid(payload["event_id"])
    assert result == b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
    snapshot = fresh_events.snapshot()
    assert snapshot["by_kind"] == {KIND_CLIP: 1}
    assert snapshot["last"]["event_id"] == payload["event_id"]


def test_health_publishes_the_camera_events():
    events = camera_events_mod.camera_events
    event_id = new_ulid()
    events.observed(event_id, kind=KIND_PRESENCE, home_id="livingroom", source="camera")

    payload = asyncio.run(hub_app.health())
    assert payload["camera_events"]["last"]["event_id"] == event_id
    assert payload["camera_events"]["by_kind"] == {KIND_PRESENCE: 1}


# --- the client echoes the id --------------------------------------------


def test_the_client_echoes_the_server_event_id_on_every_frame_of_a_burst():
    from client.camera import CameraService

    sent_json: list[dict] = []
    sent_bytes: list[bytes] = []

    async def send_json(payload):
        sent_json.append(payload)

    async def send_bytes(data):
        sent_bytes.append(data)

    camera = CameraService.__new__(CameraService)
    camera._send_json = send_json
    camera._send_bytes = send_bytes
    camera._send_lock = asyncio.Lock()
    camera._sent_state = None
    camera._send_errors = 0
    camera._request_event_id = ""
    event_id = new_ulid()
    pairs = [(b"jpeg1", 4, 4, None), (b"jpeg2", 4, 4, None)]

    asyncio.run(CameraService._send_burst(camera, "c1", proto.CAMERA_REASON_REQUEST, pairs,
                                          event_id=event_id))

    assert [header["event_id"] for header in sent_json] == [event_id, event_id]
    assert sent_bytes == [b"jpeg1", b"jpeg2"]
    assert all(header["id"] == "c1" for header in sent_json)


def test_a_presence_push_mints_its_own_event_id():
    from client.camera import CameraService

    camera = CameraService.__new__(CameraService)
    camera.face_check_interval_s = 0.5
    camera._presence_pending = __import__("threading").Event()
    camera._last_presence_push = 0.0
    camera._send_lock = None
    camera._frame_seq = 0
    camera._tracks = []
    pending = {}

    def _submit(coro):
        # ``_maybe_push_presence`` runs on the YOLO thread and only hands the
        # coroutine to the loop; here we keep it and read its arguments.
        pending["coro"] = coro
        return object()

    camera._submit = _submit

    CameraService._maybe_push_presence(camera)

    coro = pending.get("coro")
    assert coro is not None, "the presence burst was not handed to the loop"
    try:
        assert is_ulid(coro.cr_frame.f_locals["event_id"])
    finally:
        coro.close()


def _presence_camera(**attributes):
    from client.camera import CameraService

    camera = CameraService.__new__(CameraService)
    camera.face_check_interval_s = 0.5
    camera._presence_pending = __import__("threading").Event()
    camera._last_presence_push = 0.0
    camera._send_lock = None
    camera._frame_seq = 0
    camera._tracks = []
    camera._burst_due = False
    camera.privacy = SimpleNamespace(allows_frames=True)
    for name, value in attributes.items():
        setattr(camera, name, value)
    return camera


def test_a_track_that_just_appeared_pushes_its_burst_at_once():
    """Owner's report (2026-09-22): a quick pass-by was never noticed.

    The periodic push waits ``face_check_interval_s`` and carries one frame, so
    somebody who crosses the room in half a second could be gone before the
    hub ever saw a second frame. A new track clears the wait and takes the
    whole burst.
    """
    camera = _presence_camera()
    camera._last_presence_push = time.monotonic()  # inside the periodic interval
    camera._burst_due = True
    submitted = {}

    def _submit(coro):
        submitted["coro"] = coro
        return object()

    camera._submit = _submit
    CameraService = type(camera)
    CameraService._maybe_push_presence(camera, frame=object())

    coro = submitted.get("coro")
    assert coro is not None, "a new track must not wait for the periodic push"
    try:
        assert coro.cr_frame.f_locals["burst"] is True
    finally:
        coro.close()
    assert camera._burst_due is False


def test_the_periodic_push_still_waits_and_stays_one_frame():
    camera = _presence_camera()
    camera._last_presence_push = time.monotonic()
    submitted = {}
    camera._submit = lambda coro: submitted.setdefault("coro", coro)

    type(camera)._maybe_push_presence(camera, frame=object())

    assert "coro" not in submitted


def test_a_presence_burst_takes_the_whole_face_burst_of_frames():
    from client.camera import FACE_BURST, CameraService

    camera = _presence_camera()
    camera._encode = Mock(return_value=(b"one", 4, 4))
    camera._capture_burst_sync = Mock(return_value=[(b"a", 4, 4), (b"b", 4, 4), (b"c", 4, 4)])
    camera._send_burst = AsyncMock()

    asyncio.run(CameraService._encode_and_push_presence(
        camera, object(), "p1", [], event_id=new_ulid(), burst=True))

    camera._capture_burst_sync.assert_called_once_with(FACE_BURST, full=True)
    camera._encode.assert_not_called()
    assert len(camera._send_burst.await_args.args[2]) == 3

    asyncio.run(CameraService._encode_and_push_presence(
        camera, object(), "p2", [], event_id=new_ulid()))
    camera._encode.assert_called_once()
    assert [pair[0] for pair in camera._send_burst.await_args.args[2]] == [b"one"]
