"""v1.7: the room can SEE how enrollment is going.

During voice enrollment the only feedback used to be a sentence in the middle of
a spoken reply, and face enrollment took photos for ten silent seconds - the
owner reported he could not tell what was happening. The server now sends a
HUD caption (say.status, and a status message for background work) and the
client shows it. These tests cover the caption text on the server and the
caption bookkeeping on the client, without a socket or a HUD.
"""
import asyncio

from common import protocol as proto
from server.app import Connection


def test_status_message_is_a_known_server_message():
    assert proto.MSG_STATUS in proto.SERVER_MESSAGE_TYPES
    assert proto.SAY_STATUS_FIELD == "status"


def test_caption_counts_seconds_while_more_speech_is_needed():
    text = Connection._enroll_status_text({"name": "Drew", "samples": 1, "total_speech_s": 4.0})
    assert "Drew" in text
    assert "1 of 3" in text
    assert "6 s more" in text
    assert "keep talking" in text


def test_caption_asks_for_one_sentence_once_the_seconds_are_there():
    text = Connection._enroll_status_text({"name": "Drew", "samples": 2, "total_speech_s": 12.0})
    assert "2 of 3" in text
    assert "one more sentence" in text


def test_caption_says_so_when_somebody_else_spoke():
    note = " [enrollment: sample NOT used - that did not sound like Drew ...]"
    text = Connection._enroll_status_text({"name": "Drew", "samples": 1, "total_speech_s": 4.0}, note)
    assert text.startswith("That was not Drew's voice")


class _FakeOverlay:
    def __init__(self):
        self.states = []
        self.statuses = []

    def set_state(self, state):
        self.states.append(state)

    def set_status(self, text):
        self.statuses.append(text)


def _client():
    from client.main import MODE_IDLE, JarvisClient

    client = JarvisClient.__new__(JarvisClient)
    client.overlay = _FakeOverlay()
    client._status_seq = 0
    client._status_owns_hud = False
    client._mode = MODE_IDLE
    return client


def test_a_background_caption_lights_the_hud_and_then_turns_it_off():
    async def scenario():
        client = _client()
        client._on_status_message({"type": "status", "text": "Taking photos", "ttl_s": 0.05}, in_conversation=False)
        assert client.overlay.statuses[-1] == "Taking photos"
        assert client.overlay.states[-1] == "thinking"
        await asyncio.sleep(0.12)
        assert client.overlay.statuses[-1] == ""
        assert client.overlay.states[-1] == "idle"

    asyncio.run(scenario())


def test_a_newer_caption_is_not_cleared_by_an_older_timer():
    async def scenario():
        client = _client()
        client._show_status("first", 0.05)
        client._show_status("second", 5.0)
        await asyncio.sleep(0.12)
        assert client.overlay.statuses[-1] == "second"

    asyncio.run(scenario())


def test_during_a_conversation_only_the_caption_changes():
    async def scenario():
        from client.main import MODE_CONVERSATION

        client = _client()
        client._mode = MODE_CONVERSATION
        client._on_status_message({"type": "status", "text": "Working", "ttl_s": 5}, in_conversation=True)
        assert client.overlay.statuses[-1] == "Working"
        assert client.overlay.states == []  # the turn owns the HUD state

    asyncio.run(scenario())


def test_a_bogus_ttl_falls_back_to_the_default():
    async def scenario():
        client = _client()
        client._on_status_message({"type": "status", "text": "x", "ttl_s": "soon"}, in_conversation=False)
        assert client.overlay.statuses[-1] == "x"

    asyncio.run(scenario())
