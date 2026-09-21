"""Per-session backpressure: a slow room never holds back the others (ТЗ 4.4/13)."""
from __future__ import annotations

import asyncio

from common import protocol as proto
from hub.outbound import OutboundBuffer


class _Writer:
    """A send target whose writes can be held open on purpose."""

    def __init__(self, *, blocked: bool = False) -> None:
        self.items: list = []
        self.gate = asyncio.Event()
        if not blocked:
            self.gate.set()
        self.started = asyncio.Event()

    async def __call__(self, item):
        self.started.set()
        await self.gate.wait()
        self.items.append(item)

    def release(self) -> None:
        self.gate.set()


def test_background_frames_are_dropped_before_a_reply():
    writer = _Writer(blocked=True)
    buffer = OutboundBuffer(writer, capacity=3)
    payloads = [{"type": "hud", "seq": index} for index in range(3)]

    async def scenario():
        for payload in payloads:
            assert await buffer.enqueue(payload, background=True) is True
        assert await buffer.enqueue({"type": "hud", "seq": 99}, background=True) is False
        # A reply needs room: the queue evicts a background frame instead of
        # dropping the reply or blocking the caller.
        assert await buffer.enqueue({"type": "say", "text": "hi"}) is True
        return buffer.stats(), buffer.pending_labels()

    stats, labels = asyncio.run(scenario())
    assert stats.dropped == 2
    assert stats.dropped_by_type == {"hud": 2}
    assert stats.queued == 3
    assert labels.count("say") == 1


def test_replies_are_never_dropped_even_when_the_queue_is_full():
    writer = _Writer(blocked=True)
    buffer = OutboundBuffer(writer, capacity=2)

    async def scenario():
        for index in range(10):
            assert await buffer.enqueue({"type": "say", "index": index}) is True
        return buffer.stats()

    stats = asyncio.run(scenario())
    assert stats.dropped == 0
    assert stats.queued == 10


def test_order_is_preserved_for_headers_and_their_binary_frames():
    writer = _Writer()
    buffer = OutboundBuffer(writer, capacity=4)

    async def scenario():
        task = asyncio.create_task(buffer.drain())
        await buffer.enqueue({"type": "tts_start"})
        await buffer.enqueue(b"\x01\x02")
        await buffer.enqueue({"type": "tts_end"})
        await buffer.flush()
        await buffer.aclose()
        task.cancel()

    asyncio.run(scenario())
    assert writer.items == [{"type": "tts_start"}, b"\x01\x02", {"type": "tts_end"}]


def test_a_stalled_client_does_not_block_the_others():
    slow_writer = _Writer(blocked=True)
    fast_writer = _Writer()
    slow = OutboundBuffer(slow_writer, capacity=2)
    fast = OutboundBuffer(fast_writer, capacity=2)

    async def scenario():
        fast_task = asyncio.create_task(fast.drain())
        slow_task = asyncio.create_task(slow.drain())
        # Both rooms are announced before the slow socket accepts anything.
        await asyncio.wait_for(slow.enqueue({"type": "config_update"}), timeout=1)
        await asyncio.wait_for(fast.enqueue({"type": "config_update"}), timeout=1)
        for index in range(20):
            await asyncio.wait_for(slow.enqueue({"type": "camera_state", "n": index}, background=True), timeout=1)
            await asyncio.wait_for(fast.enqueue({"type": "camera_state", "n": index}, background=True), timeout=1)
            # A real producer yields between frames; the writers get to run.
            await asyncio.sleep(0)
        await asyncio.wait_for(fast.flush(), timeout=1)
        stats = slow.stats()
        slow_task.cancel()
        fast_task.cancel()
        return stats

    stats = asyncio.run(scenario())
    assert stats.dropped >= 15, "the stalled room must shed its background frames"
    assert len(fast_writer.items) == 21, "the healthy room receives every frame"


def test_background_class_matches_the_protocol_frame_types():
    assert proto.is_background_server_frame({"type": "camera_state"}) is True
    assert proto.is_background_server_frame({"type": "hud"}) is True
    for mtype in (proto.MSG_SAY, proto.MSG_TTS_START, proto.MSG_TTS_END, proto.MSG_TRANSCRIPT,
                  proto.MSG_CONFIG_UPDATE, proto.MSG_ERROR):
        assert proto.is_background_server_frame({"type": mtype}) is False


def test_connection_sends_direct_frames_and_queues_fan_out_frames():
    from starlette.websockets import WebSocketState

    from common.config import Config
    from hub import app as hub_app

    class _WS:
        client_state = WebSocketState.CONNECTED

        def __init__(self):
            self.texts: list[str] = []
            self.binaries: list[bytes] = []

        async def send_text(self, text):
            self.texts.append(text)

        async def send_bytes(self, data):
            self.binaries.append(data)

    connection = hub_app.Connection(_WS(), Config())

    async def scenario():
        # A direct reply is written on the spot: it must never wait behind a
        # queue, and the PCM that follows it must keep its order.
        await connection.send_json({"type": proto.MSG_SAY, "text": "ready"})
        await connection.send_bytes(b"pcm")
        assert connection.ws.texts == ['{"type": "say", "text": "ready"}']
        assert connection.ws.binaries == [b"pcm"]

        # A fan-out frame goes through the session buffer instead.
        assert await connection.queue_frame({"type": proto.MSG_CONFIG_UPDATE, "config_rev": 2}) is True
        await connection.outbox.flush()
        await connection.outbox.aclose()
        if connection._outbox_task is not None:
            connection._outbox_task.cancel()

    asyncio.run(scenario())
    assert len(connection.ws.texts) == 2
    assert connection.outbox.stats().sent == 1


def test_health_reports_dropped_frames(monkeypatch):
    from common.config import Config
    from hub import app as hub_app

    class _Conn:
        outbox = None

    connection = _Conn()
    buffer = OutboundBuffer(_Writer(), capacity=1)
    connection.outbox = buffer

    async def scenario():
        await buffer.enqueue({"type": "hud"})
        await buffer.enqueue({"type": "hud"}, background=True)

    asyncio.run(scenario())
    monkeypatch.setattr(hub_app, "_connections", {connection})
    monkeypatch.setattr(hub_app, "_config", Config())
    stats = asyncio.run(hub_app.health())["outbound"]
    assert stats["clients"] == 1
    assert stats["dropped"] == 1
    assert stats["dropped_by_type"] == {"hud": 1}
