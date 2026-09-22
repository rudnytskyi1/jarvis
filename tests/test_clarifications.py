"""P3-13/P3-14 (F-413, D-12): one clarifying question, and its answer.

"Turn on the light" is a complete request in a room with one lamp and an
ambiguous one in a room with three. The TZ allows exactly ONE question, the
answer to it is read as the next utterance inside the same window as F-113,
and the request is then executed for real - not guessed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common import protocol as proto
from common.config import Config
from hub import app
from hub import clarifications as clarify
from hub.session import Session
from hub.utterances import UtteranceMetrics

LAMPS = [
    {"name": "desk lamp", "type": "magichome", "area": "desk"},
    {"name": "bed lamp", "type": "yeelight", "area": "bed"},
    {"name": "strip", "type": "tuya", "area": "desk"},
]


# --- what counts as an ambiguous request (D-12) ----------------------------


@pytest.mark.parametrize("text, state", [
    ("Rowan, turn on the light", clarify.STATE_ON),
    ("turn off the lights please", clarify.STATE_OFF),
    ("light on", clarify.STATE_ON),
    ("Роуан, включи свет", clarify.STATE_ON),
    ("выключи лампу", clarify.STATE_OFF),
    ("погаси свет", clarify.STATE_OFF),
    ("enciende la luz", clarify.STATE_ON),
    ("apaga las luces", clarify.STATE_OFF),
])
def test_a_bare_light_request_is_recognised(text, state):
    assert clarify.light_request(text) == state


@pytest.mark.parametrize("text", [
    "", "turn on the desk lamp", "what is the light in the kitchen",
    "turn on the light and play music", "Rowan, cinema", "выключи свет и включи фильм",
    "свет", "turn on the kettle", "какой свет тебе нравится",
    "выключи свет на кухне и открой окно",
])
def test_anything_else_is_left_to_the_model(text):
    assert clarify.light_request(text) is None


def test_candidates_are_the_lights_of_the_room():
    devices = [*LAMPS, {"name": "wall switch", "type": "switchbot_bot"},
               {"name": "speaker", "type": "other"}, {"name": ""}, "junk"]
    assert clarify.candidates(devices) == ["desk lamp", "bed lamp", "strip"]
    assert clarify.candidates(devices, kind="switch") == ["wall switch"]


def test_a_named_device_or_its_area_is_not_ambiguous():
    assert clarify.narrowed("turn on the desk lamp", LAMPS) == ["desk lamp"]
    assert clarify.narrowed("turn on the light in the bed", LAMPS) == ["bed lamp"]
    assert clarify.narrowed("turn on the light at the desk", LAMPS) == ["desk lamp", "strip"]
    assert clarify.narrowed("turn on the light", LAMPS) == []


def test_the_question_is_one_line_in_the_language_of_the_turn():
    english = clarify.question(clarify.STATE_ON, ["desk lamp", "strip"], "en")
    russian = clarify.question(clarify.STATE_OFF, ["desk lamp", "strip"], "ru")
    spanish = clarify.question(clarify.STATE_ON, ["desk lamp", "strip"], "es")
    assert english == "Which light should I turn on: desk lamp or strip?"
    assert "или" in russian and "выключить" in russian
    assert spanish.startswith("¿Qué luz")
    assert "?" not in russian[:-1], "one question mark per question"


@pytest.mark.parametrize("answer, expected", [
    ("the desk lamp", "desk lamp"),
    ("Desk lamp please", "desk lamp"),
    ("the strip", "strip"),
    ("the first one", "desk lamp"),
    ("2", "bed lamp"),
    ("второй", "bed lamp"),
    ("el segundo", "bed lamp"),
    ("the last one", None),
    ("never mind", None),
    ("both lamps", None),
    ("desk lamp and strip", None),
    ("", None),
])
def test_an_answer_names_one_candidate_or_nobody(answer, expected):
    assert clarify.resolve(["desk lamp", "bed lamp", "strip"], answer) == expected


def test_the_limit_is_one_question():
    assert clarify.MAX_QUESTIONS == 1
    pending = clarify.Clarification(state=clarify.STATE_ON, options=["a", "b"],
                                    question="Which one?", window_s=5.0)
    assert pending.asked == 1 and pending.exhausted()
    assert not pending.expired()
    assert pending.expired(now=pending.opened_at + 5.1)
    assert pending.remaining_s(now=pending.opened_at + 4.0) == pytest.approx(1.0)


# --- the hub asks exactly one question -------------------------------------


class _Chain:
    """A decision chain that answers what the test wants for one type."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = dict(answers)
        self.calls: list[tuple[str, str]] = []

    async def yes_no(self, question, context, *, decision_type):
        self.calls.append(("yes_no", decision_type))
        value = self.answers.get(decision_type, context.get("heuristic"))
        return SimpleNamespace(value=value)

    async def choose(self, question, options, context, *, decision_type):
        self.calls.append(("choose", decision_type))
        value = self.answers.get(decision_type, context.get("heuristic"))
        return SimpleNamespace(value=value)


class _Turn:
    """One room, one connection and as many utterances as the test needs."""

    def __init__(self, monkeypatch, devices, *, chain=None):
        self.metrics = UtteranceMetrics()
        monkeypatch.setattr(app, "_utterance_metrics", self.metrics)
        monkeypatch.setattr(app, "_decider", None)
        monkeypatch.setattr(app, "_decision_log", False)
        monkeypatch.setattr(app, "_memory", None)
        monkeypatch.setattr(app, "_conversations", None)
        monkeypatch.setattr(app, "_tts", object())
        self.chain = chain
        if chain is not None:
            monkeypatch.setattr(app, "_decision_chain", lambda wake: chain)
        self.brain = SimpleNamespace(
            generate=AsyncMock(return_value=SimpleNamespace(
                text="All right.", history=[], tool_calls=[])),
            verify=AsyncMock())
        monkeypatch.setattr(app, "_llm", self.brain)
        monkeypatch.setattr(app, "_stt", SimpleNamespace(
            transcribe_pcm=lambda *a: (self.text, self.language)))
        monkeypatch.setattr(app, "_voices", SimpleNamespace(
            enabled=True, identify=lambda *a: ("Anton", "admin", 0.9)))
        cfg = Config()
        cfg.server.permissions_enabled = False
        self.conn = app.Connection(SimpleNamespace(client=None), cfg)
        self.conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        self.conn.home_id = "livingroom"
        self.conn.cfg.server.llm.verify_actions = False
        self.conn.cfg.server.diarization.enabled = False
        self.conn.session = Session("room-pc", devices, 25)
        self.conn.send_json = AsyncMock()
        self.conn._announce_speaker = AsyncMock()
        self.conn._log_dialog = AsyncMock()
        self.conn._stream_tts = AsyncMock()
        self.conn._execute_tool = AsyncMock(
            return_value={"ok": True, "reply": "The desk lamp is on."})
        self.conn.presence = SimpleNamespace(present=lambda: {"Anton"}, unknown_count=0)
        self.text = ""
        self.language = "en"

    def said(self) -> list[str]:
        return [str(call.args[0].get("text") or "")
                for call in self.conn.send_json.call_args_list
                if call.args and call.args[0].get("type") == proto.MSG_SAY]

    async def ask(self, text: str) -> str:
        self.text = text
        self.metrics.started(self.conn.utterance_id, home_id="livingroom")
        await self.conn._handle_utterance(b"\0" * 16000)
        trace = self.metrics.last() or {}
        return str(trace.get("note") or "")


def test_the_room_is_asked_once_when_three_lamps_could_be_meant(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)

    note = asyncio.run(turn.ask("Rowan, turn on the light"))

    assert note == "clarification_asked"
    assert turn.said() == ["Which light should I turn on: desk lamp, bed lamp or strip?"]
    assert turn.brain.generate.await_count == 0, "the model does not get to guess"
    pending = turn.conn._clarification
    assert pending is not None and pending.options == ["desk lamp", "bed lamp", "strip"]
    assert pending.state == clarify.STATE_ON and pending.asked == 1


def test_a_room_with_one_lamp_is_never_asked(monkeypatch):
    turn = _Turn(monkeypatch, [LAMPS[0]])

    note = asyncio.run(turn.ask("Rowan, turn on the light"))

    assert note != "clarification_asked"
    assert turn.said() == ["All right."]
    assert turn.conn._clarification is None


def test_a_request_that_names_the_lamp_goes_straight_to_the_model(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)

    note = asyncio.run(turn.ask("Rowan, turn on the desk lamp"))

    assert note != "clarification_asked"
    assert turn.brain.generate.await_count == 1
    assert turn.conn._clarification is None


def test_the_second_ambiguous_request_does_not_get_a_second_question(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)

    first = asyncio.run(turn.ask("Rowan, turn on the light"))
    second = asyncio.run(turn.ask("Rowan, turn on the light"))

    assert first == "clarification_asked"
    assert second != "clarification_asked", "one question, not two"
    assert turn.brain.generate.await_count == 1, "the second turn went to the model"
    assert turn.said()[-1] == "All right."


def test_the_question_comes_out_in_the_speakers_language(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    turn.language = "ru"

    asyncio.run(turn.ask("Роуан, включи свет"))

    assert turn.said() == ["Какой свет включить: desk lamp, bed lamp или strip?"]


def test_a_decider_that_says_the_request_is_clear_silences_the_question(monkeypatch):
    chain = _Chain({"clarification": False})
    turn = _Turn(monkeypatch, LAMPS, chain=chain)

    note = asyncio.run(turn.ask("Rowan, turn on the light"))

    assert ("yes_no", "clarification") in chain.calls
    assert note != "clarification_asked"
    assert turn.brain.generate.await_count == 1
    assert turn.conn._clarification is None


def test_an_expired_question_is_replaced_by_a_new_one(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    asyncio.run(turn.ask("Rowan, turn on the light"))
    turn.conn._clarification.opened_at -= turn.conn._clarification.window_s + 1.0
    turn.conn.send_json.reset_mock()

    note = asyncio.run(turn.ask("Rowan, turn off the light"))

    assert note == "clarification_asked"
    assert turn.said() == ["Which light should I turn off: desk lamp, bed lamp or strip?"]


# --- the answer to the question (P3-14) ------------------------------------


def test_the_answer_runs_the_real_tool(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    asyncio.run(turn.ask("Rowan, turn on the light"))
    turn.conn.send_json.reset_mock()
    turn.conn._execute_tool.reset_mock()

    note = asyncio.run(turn.ask("the bed lamp"))

    assert note == "clarification_ok"
    tool, args = turn.conn._execute_tool.call_args.args
    assert tool == "set_light"
    assert args["device"] == "bed lamp" and args["state"] == "on"
    assert turn.said() == ["The desk lamp is on."], "the room hears the real result"
    assert turn.conn._clarification is None
    assert turn.brain.generate.await_count == 0, "the hub acted; the model did not"


def test_a_numbered_answer_works_too(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    asyncio.run(turn.ask("Rowan, turn on the light"))
    turn.conn._execute_tool.reset_mock()

    asyncio.run(turn.ask("the first one, please"))

    assert turn.conn._execute_tool.call_args.args[1]["device"] == "desk lamp"


def test_an_answer_that_names_nobody_gets_no_second_question(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    asyncio.run(turn.ask("Rowan, turn on the light"))
    turn.conn.send_json.reset_mock()
    turn.conn._execute_tool.reset_mock()

    note = asyncio.run(turn.ask("never mind, what time is it?"))

    assert note != "clarification_asked", "one question, and it is over"
    assert turn.conn._clarification is None
    turn.conn._execute_tool.assert_not_awaited()
    assert turn.brain.generate.await_count == 1, "the ordinary turn continued"


def test_an_answer_after_the_window_changes_nothing(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    asyncio.run(turn.ask("Rowan, turn on the light"))
    turn.conn._clarification.opened_at -= turn.conn._clarification.window_s + 1.0
    turn.conn.send_json.reset_mock()
    turn.conn._execute_tool.reset_mock()

    note = asyncio.run(turn.ask("the bed lamp"))

    assert note == "clarification_expired"
    turn.conn._execute_tool.assert_not_awaited()
    assert turn.said() == ["That question expired, so I did not touch anything."]
    assert turn.conn._clarification is None


def test_the_answer_is_read_inside_the_f113_window(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    turn.conn.cfg.server.confirmations.window_s = 3.0

    asyncio.run(turn.ask("Rowan, turn on the light"))

    assert turn.conn._clarification.window_s == 3.0


def test_a_device_that_left_the_room_is_reported_not_skipped(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    asyncio.run(turn.ask("Rowan, turn on the light"))
    turn.conn.session.devices = [LAMPS[0], LAMPS[1]]
    turn.conn.send_json.reset_mock()
    turn.conn._execute_tool.reset_mock()

    note = asyncio.run(turn.ask("the strip"))

    assert note == "clarification_stale"
    turn.conn._execute_tool.assert_not_awaited()
    assert turn.said() == ["strip is not in the room any more, so I did nothing."]


def test_a_failed_action_is_reported_honestly(monkeypatch):
    turn = _Turn(monkeypatch, LAMPS)
    asyncio.run(turn.ask("Rowan, turn on the light"))
    turn.conn._execute_tool = AsyncMock(
        return_value={"ok": False, "error": "the lamp is offline"})
    turn.conn.send_json.reset_mock()

    note = asyncio.run(turn.ask("the desk lamp"))

    assert note == "clarification_failed"
    assert turn.said() == ["I could not do that. the lamp is offline"]


def test_the_tool_for_helper_picks_by_device_kind():
    assert clarify.tool_for({"name": "bed lamp", "type": "yeelight"}, "off") == (
        "set_light", {"device": "bed lamp", "state": "off"})
    assert clarify.tool_for({"name": "wall switch", "type": "switchbot"}, "on") == (
        "set_switch", {"device": "wall switch", "action": "on"})
