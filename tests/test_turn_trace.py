"""The chain of one request: every step of a turn reaches the panel (F-705)."""
from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hub import app as hub_app
from hub import migrations_runner, turn_trace
from hub.decider import Decision
from hub.decision_log import DecisionLog
from hub.llm import LlmClient
from hub.turn_trace import MAX_TEXT, TurnTraceStore


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    return conn


def test_a_test_run_never_writes_into_the_live_hub_database():
    """The trace goes through ``hub.app``'s gateway: it must not be the real DB."""
    repo_root = Path(__file__).resolve().parents[1]
    live = repo_root / "data" / "hub.db"
    sandbox = Path(os.environ["ROWAN_HUB_DB"])
    assert hub_app._hub_db_path() == sandbox, "пишем в песочницу, а не в живую базу"
    assert hub_app._hub_db_path() != live
    assert sandbox.parent.name == ".pytest-tmp" or str(sandbox).startswith(str(repo_root))


def a_decision(**overrides) -> Decision[str]:
    values = dict(value="fast_command", confidence=0.95, provider="rules", latency_ms=7,
                  decision_id="d-1", input_text="rowan ai turn on the light")
    values.update(overrides)
    return Decision(**values)


# --- the table --------------------------------------------------------------


def test_the_migration_adds_the_event_table_and_links_decisions(tmp_path):
    conn = migrated(tmp_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(turn_events)")}
    assert {"event_id", "turn_id", "home_id", "ts", "kind", "name", "ok",
            "latency_ms", "payload_json"} <= columns
    decisions = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
    assert {"turn_id", "home_id"} <= decisions, "решение привязано к запросу"


def test_the_migration_can_run_twice(tmp_path):
    conn = migrated(tmp_path)
    migrations_runner.migrate(conn)  # idempotent: ALTER is guarded
    assert conn.execute("SELECT COUNT(*) FROM turn_events").fetchone()[0] == 0


# --- what a turn writes -----------------------------------------------------


def test_a_step_is_written_with_its_turn_and_read_back_in_order(tmp_path):
    conn = migrated(tmp_path)
    store = TurnTraceStore(conn)
    store.record("turn", "room", turn_id="u-1", home_id="livingroom",
                 payload={"home": "livingroom"})
    store.record("decision", "rules", turn_id="u-1",
                 payload={"type": "route", "value": "fast_command"}, latency_ms=4)
    store.record("tool", "look_at_camera", turn_id="u-1", ok=False,
                 payload={"args": {"query": "who"}, "result": {"ok": False}}, latency_ms=120)
    steps = store.events("u-1")
    assert [step["kind"] for step in steps] == ["turn", "decision", "tool"]
    assert steps[1]["name"] == "rules" and steps[1]["latency_ms"] == 4
    assert steps[2]["ok"] is False
    assert steps[2]["payload"]["args"] == {"query": "who"}
    assert steps[0]["when"], "у шага есть читаемое время"


def test_the_turn_context_decides_which_request_is_recorded(tmp_path):
    conn = migrated(tmp_path)
    store = turn_trace.configure(conn)
    try:
        with turn_trace.turn("u-9", "livingroom"):
            assert turn_trace.current_turn() == "u-9"
            store.record("llm", "luna", payload={"round": 1})
        assert turn_trace.current_turn() == ""
        assert [step["name"] for step in store.events("u-9")] == ["luna"]
    finally:
        turn_trace.configure(None)


def test_a_step_outside_a_turn_is_not_written(tmp_path):
    conn = migrated(tmp_path)
    store = turn_trace.configure(conn)
    try:
        store.record("tool", "no-turn-here")
        assert store.recent() == [], "фоновая работа не притворяется запросом"
    finally:
        turn_trace.configure(None)


def test_the_module_level_record_is_a_no_op_before_the_hub_opens(tmp_path):
    turn_trace.configure(None)
    turn_trace.record("tool", "nothing-happens")  # must not raise
    assert turn_trace.store() is None


def test_long_text_and_deep_payloads_are_clipped(tmp_path):
    conn = migrated(tmp_path)
    store = TurnTraceStore(conn)
    store.record("say", "reply", turn_id="u-2",
                 payload={"text": "я" * (MAX_TEXT + 500), "nested": {"a": {"b": {"c": 1}}}})
    payload = store.events("u-2")[0]["payload"]
    assert len(payload["text"]) == MAX_TEXT + 3 and payload["text"].endswith("...")
    assert payload["nested"]["a"]["b"] == "{...}", "вложенность тоже обрезана"


def test_bytes_are_counted_never_stored(tmp_path):
    conn = migrated(tmp_path)
    store = TurnTraceStore(conn)
    image = b"\x89PNG and then some"
    store.record("tool", "generate_image", turn_id="u-3", payload={"result": {"png": image}})
    assert store.events("u-3")[0]["payload"]["result"]["png"] == f"<{len(image)} bytes>"


def test_recent_summarizes_each_request(tmp_path):
    conn = migrated(tmp_path)
    store = TurnTraceStore(conn)
    store.record("turn", "room", turn_id="u-1", home_id="livingroom")
    store.record("decision", "jev", turn_id="u-1", latency_ms=475)
    store.record("tool", "telegram_send", turn_id="u-1", ok=False)
    store.record("llm", "luna", turn_id="u-1")
    store.record("turn", "room", turn_id="u-2", home_id="livingroom")
    rows = store.recent()
    assert [row["turn_id"] for row in rows] == ["u-2", "u-1"], "сначала новые"
    newest = rows[1]
    assert (newest["events"], newest["decisions"], newest["tools"], newest["rounds"],
            newest["failed"]) == (4, 1, 1, 1, 1)
    assert set(newest["providers"].split(",")) == {"jev", "luna"}
    assert newest["home_id"] == "livingroom"
    assert newest["seconds"] >= 0.0


def test_a_database_error_is_logged_not_raised(tmp_path, caplog):
    conn = sqlite3.connect(":memory:")  # no turn_events table at all
    store = TurnTraceStore(conn)
    with caplog.at_level("WARNING"):
        store.record("tool", "anything", turn_id="u-1")
    assert "Could not record the trace event" in caplog.text


# --- the decisions table ----------------------------------------------------


def test_a_decision_carries_the_turn_and_lands_in_the_chain(tmp_path):
    conn = migrated(tmp_path)
    store = turn_trace.configure(conn)
    try:
        with turn_trace.turn("u-7", "livingroom"):
            DecisionLog(conn).record(a_decision(), "route", "act")
        row = conn.execute("SELECT turn_id, home_id FROM decisions").fetchone()
        assert row == ("u-7", "livingroom")
        steps = store.events("u-7")
        assert steps[0]["kind"] == "decision" and steps[0]["name"] == "rules"
        assert steps[0]["payload"]["type"] == "route"
        assert steps[0]["payload"]["confidence"] == 0.95
    finally:
        turn_trace.configure(None)


def test_a_decision_outside_a_turn_still_reaches_the_decisions_table(tmp_path):
    """The calibration report (ТЗ 5.4) must not lose a decision to tracing."""
    conn = migrated(tmp_path)
    turn_trace.configure(None)
    DecisionLog(conn).record(a_decision(), "route", "act")
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM turn_events").fetchone()[0] == 0


@pytest.mark.parametrize("kind", ["turn", "decision", "tool", "llm", "say"])
def test_every_kind_the_panel_promises_is_accepted(tmp_path, kind):
    conn = migrated(tmp_path)
    store = TurnTraceStore(conn)
    store.record(kind, "example", turn_id="u-1", payload={"ok": True})
    assert store.events("u-1")[0]["kind"] == kind


def test_the_prompt_that_leaves_the_hub_is_recorded(tmp_path):
    """The owner asked to see the prompts themselves, not only the answers."""
    conn = migrated(tmp_path)
    store = turn_trace.configure(conn)
    settings = SimpleNamespace(
        provider="vllm", model="Qwen3-35B", base_url="http://127.0.0.1:8000/v1",
        api_key="vllm", think=False, temperature=0.2, max_tokens=128,
        max_tool_rounds=4, keep_alive="4h", num_ctx=8192,
    )
    messages = [{"role": "system", "content": "тебе говорят: включи свет"},
                {"role": "user", "content": "включи свет"}]
    client = LlmClient(settings)
    client._chat_openai = lambda given, with_tools: ("готово", [])
    try:
        with turn_trace.turn("u-11", "livingroom"):
            text, calls = asyncio.run(client._chat(messages, with_tools=True))
        assert (text, calls) == ("готово", [])
        steps = store.events("u-11")
        assert [step["kind"] for step in steps] == ["prompt"]
        assert steps[0]["name"] == "Qwen3-35B"
        assert steps[0]["payload"]["tools"] is True
        assert steps[0]["payload"]["messages"] == messages
    finally:
        turn_trace.configure(None)
        client.close()
