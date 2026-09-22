"""P3-26 (F-419): «когда я приду после 22:00, включи тёплый свет».

The hub recognises a rule-shaped request itself, the LOCAL model composes the
rule with structured output, and the rule only starts working after a spoken
"yes" (F-113) - the same discipline the memory tools follow.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from common.config import Config
from hub import app as hub_app
from hub import automation
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.room_state import RoomState
from hub.storage import Memory

NOW = datetime(2026, 9, 21, 12, 30, tzinfo=UTC)
CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.commit()
    yield conn
    conn.close()


# --- распознавание и составление --------------------------------------------


@pytest.mark.parametrize("text", [
    "когда я приду после 22:00, включи тёплый свет",
    "когда Антон дома, скажи привет",
    "если никого нет дома, выключи свет",
    "when I come home, turn on the warm light",
    "whenever somebody comes in, say hello",
    "cuando llegue a casa, enciende la luz cálida",
])
def test_a_rule_shaped_request_is_recognised(text):
    assert automation.looks_like_rule(text) is True


@pytest.mark.parametrize("text", [
    "включи свет",
    "когда ты придёшь?",
    "when will you come home?",
    "напомни через 20 минут купить молоко",
    "что ты обо мне знаешь?",
    "когда я приду домой",
])
def test_other_phrases_are_left_alone(text):
    assert automation.looks_like_rule(text) is False


class _FakeLlm:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[tuple[list[Any], dict[str, Any], str]] = []

    async def structured_json(self, messages, schema, *, name=""):
        self.calls.append((messages, schema, str(name)))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _draft_answer(**overrides) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "name": "Тёплый свет вечером",
        "trigger": {"kind": "presence", "event": "person_entered", "person_id": "p-anton"},
        "conditions": {"quiet_hours": False},
        "actions": [{"kind": "scene", "scene": "warm"}],
    }
    answer.update(overrides)
    return answer


def test_the_model_composes_a_rule_in_the_hubs_own_shape():
    llm = _FakeLlm(_draft_answer())
    draft = asyncio.run(automation.llm_rule_drafter(llm)(
        "когда я приду после 22:00, включи тёплый свет", language="ru",
        now=NOW, tz=CHICAGO))
    assert draft.trigger.kind is automation.TriggerKind.PRESENCE
    assert draft.actions[0].kind is automation.ActionKind.SCENE
    assert draft.actions[0].scene == "warm"
    messages, schema, name = llm.calls[0]
    assert name == "room_rule" and schema["title"] == "RuleDraft"
    # Часы комнаты едут в промпт: модель не должна считать время по хабу.
    assert "Monday" in messages[0]["content"]
    assert "2026-09-21" in messages[0]["content"]


def test_a_broken_model_answer_never_becomes_a_rule():
    from hub.llm import StructuredUnavailable

    for answer in (_FakeLlm("not an object").answer, {"trigger": {"kind": "presence"}},
                   {"nonsense": True}, StructuredUnavailable("guided JSON refused")):
        llm = _FakeLlm(answer)
        with pytest.raises(automation.RuleDraftUnavailable):
            asyncio.run(automation.llm_rule_drafter(llm)("when I come home, turn on the light"))


def test_the_rule_question_names_the_rule_and_warns():
    rule = automation.Rule(
        home_id="livingroom", name="Тёплый свет",
        trigger=automation.Trigger(kind=automation.TriggerKind.PRESENCE,
                                   event="person_entered"),
        actions=[automation.Action(kind=automation.ActionKind.SCENE, scene="warm")])
    line = automation.rule_question(rule, "ru", window_s=8)
    assert "да" in line and "Тёплый свет" in line and "включить сцену «warm»" in line
    assert "{" not in line


def test_the_answers_read_in_three_languages():
    assert "Готово" in automation.rule_created_answer("Свет", "ru")
    assert "Done" in automation.rule_created_answer("Light", "en")
    assert "Listo" in automation.rule_created_answer("Luz", "es")
    assert "Не смог" in automation.rule_unclear_answer("ru")
    assert "model" in automation.rule_unavailable_answer("en")


# --- настоящий ход ----------------------------------------------------------


def _connection(hub_db, monkeypatch, tmp_path, *, llm: Any = None):
    audit_rows: list[dict[str, Any]] = []

    class _Audit:
        def record(self, **row):
            audit_rows.append(row)

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: _Audit())
    monkeypatch.setattr(hub_app, "_memory", Memory(data_dir=tmp_path))
    monkeypatch.setattr(hub_app, "_llm", llm)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = RoomState()
    connection._speaker_name = "Anton"
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection._pending_confirmation = None
    connection._approved_call = ""
    connection._confirmation_opened = None
    connection._untrusted_reads = []
    connection._utterance_actions = []
    connection._memory_seq = 1
    connection._personal_facts = {}
    connection.cfg = Config()
    connection._reply_lock = asyncio.Lock()
    connection.utterance_id = "utt-1"
    connection.spoken: list[str] = []
    connection.sent: list[dict[str, Any]] = []

    async def _say(voice, say_text, **kwargs):
        connection.spoken.append(str(say_text))

    async def _send_json(payload):
        connection.sent.append(payload)

    connection._stream_tts = _say
    connection.send_json = _send_json
    connection._finish_utterance = lambda **kwargs: None

    async def _log_dialog(*args, **kwargs):
        return None

    connection._log_dialog = _log_dialog
    return connection, audit_rows


def _answer(connection, text: str) -> bool:
    return asyncio.run(connection._resolve_confirmation(
        text, voice=None, session=None, started_at=None, language="ru",
        stt_ms=0, t_start=0.0))


def test_a_rule_is_only_stored_after_the_spoken_yes(hub_db, monkeypatch, tmp_path):
    connection, audit = _connection(hub_db, monkeypatch, tmp_path,
                                    llm=_FakeLlm(_draft_answer()))
    line = asyncio.run(connection._rule_turn(
        "когда я приду после 22:00, включи тёплый свет", "ru"))
    assert line is not None and "да" in line
    assert automation.RuleStore(hub_db).count() == 0
    pending = connection._pending_confirmation
    assert pending is not None and pending.tool == "create_rule"
    assert _answer(connection, "да") is True
    stored = automation.RuleStore(hub_db).all(home_id="livingroom")
    assert len(stored) == 1
    assert stored[0].author_person_id == "p-anton"
    assert stored[0].actions[0].scene == "warm"
    assert sorted(row["action"] for row in audit) == ["confirm.dangerous", "rule.created"]


def test_saying_no_writes_no_rule(hub_db, monkeypatch, tmp_path):
    connection, audit = _connection(hub_db, monkeypatch, tmp_path,
                                    llm=_FakeLlm(_draft_answer()))
    asyncio.run(connection._rule_turn("когда я приду домой, включи свет", "ru"))
    assert _answer(connection, "нет") is True
    assert automation.RuleStore(hub_db).count() == 0
    assert "rule.created" not in [row["action"] for row in audit]


def test_an_unclear_rule_asks_for_other_words_and_stores_nothing(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path,
                                llm=_FakeLlm({"trigger": {"kind": "time", "at": "nonsense"}}))
    line = asyncio.run(connection._rule_turn("когда я приду домой, включи свет", "ru"))
    assert line == automation.rule_unclear_answer("ru")
    assert connection._pending_confirmation is None
    assert automation.RuleStore(hub_db).count() == 0


def test_without_a_local_model_the_hub_says_so(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path, llm=None)
    line = asyncio.run(connection._rule_turn("когда я приду домой, включи свет", "ru"))
    assert line == automation.rule_unavailable_answer("ru")
    assert automation.RuleStore(hub_db).count() == 0


def test_an_unrecognised_speaker_cannot_create_a_rule(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path, llm=_FakeLlm(_draft_answer()))
    connection._speaker_name = "Гость"
    line = asyncio.run(connection._rule_turn("когда я приду домой, включи свет", "ru"))
    assert line == automation.rule_unknown_person_answer("ru")
    assert automation.RuleStore(hub_db).count() == 0


def test_the_feature_can_be_switched_off(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path, llm=_FakeLlm(_draft_answer()))
    connection.cfg = Config(server={"rules": {"voice_creation": False}})
    assert asyncio.run(connection._rule_turn(
        "когда я приду домой, включи свет", "ru")) is None


def test_an_ordinary_turn_is_untouched(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path, llm=_FakeLlm(_draft_answer()))
    assert asyncio.run(connection._rule_turn("включи свет", "ru")) is None
    assert asyncio.run(connection._rule_turn("когда ты придёшь?", "ru")) is None


# --- путь модели (инструмент) -----------------------------------------------


def test_the_model_tool_also_waits_for_its_yes(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path, llm=_FakeLlm(_draft_answer()))
    rule = automation.Rule(
        home_id="livingroom", name="Тест",
        trigger=automation.Trigger(kind=automation.TriggerKind.TIME, at="07:30"),
        actions=[automation.Action(kind=automation.ActionKind.SAY, text="Доброе утро")],
        author_person_id="p-anton")
    asked = connection._confirmation_needed(
        "create_rule", {"rule": rule.model_dump_json(), "spoken": "утреннее правило"})
    assert asked == "утреннее правило"
    # Без "да" вызов не одобрен, поэтому инструмент не выполняется вовсе.
    assert asyncio.run(connection._run_create_rule({"rule": "not json"}))["ok"] is False
    outcome = asyncio.run(connection._execute_tool(
        "create_rule", {"rule": rule.model_dump_json(), "spoken": "утреннее правило"}))
    assert outcome.get("needs_confirmation") is True
    assert automation.RuleStore(hub_db).count() == 0


def test_a_rule_for_another_home_is_refused(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path, llm=_FakeLlm(_draft_answer()))
    other = automation.Rule(
        home_id="kyiv", name="Чужое",
        trigger=automation.Trigger(kind=automation.TriggerKind.TIME, at="07:30"),
        actions=[automation.Action(kind=automation.ActionKind.SAY, text="hi")])
    outcome = asyncio.run(connection._run_create_rule({"rule": other.model_dump_json()}))
    assert outcome["ok"] is False and "belongs" in outcome["error"]
    assert automation.RuleStore(hub_db).count() == 0


def test_a_broken_rule_json_is_refused(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path, llm=_FakeLlm(_draft_answer()))
    outcome = asyncio.run(connection._run_create_rule({"rule": json.dumps({"name": "x"})}))
    assert outcome["ok"] is False
    assert automation.RuleStore(hub_db).count() == 0
