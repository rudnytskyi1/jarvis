"""Decisions reach the ``decisions`` table (ТЗ 5.3, 5.4)."""
from __future__ import annotations

import asyncio

from common.config import load_config
from hub import app as hub_app
from hub import migrations_runner
from hub.decider import Decision, DecisionChain, Policy, RulesDecider
from hub.decision_log import DecisionLog


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    return conn


def a_decision(**overrides) -> Decision[str]:
    values = dict(value="fast_command", confidence=0.95, provider="rules", latency_ms=3,
                  decision_id="d-1", input_text="rowan ai turn on the light")
    values.update(overrides)
    return Decision(**values)


def test_a_decision_is_stored_and_read_back(tmp_path):
    conn = migrated(tmp_path)
    log = DecisionLog(conn)
    log.record(a_decision(), "route", "act")
    rows = log.recent()
    assert len(rows) == 1
    assert rows[0]["type"] == "route"
    assert rows[0]["value"] == "fast_command"
    assert rows[0]["outcome"] == "act"
    assert rows[0]["confidence"] == 0.95


def test_the_input_itself_is_not_kept_only_its_fingerprint(tmp_path):
    conn = migrated(tmp_path)
    DecisionLog(conn).record(a_decision(input_text="a private sentence"), "route")
    hash_value, value_json = conn.execute(
        "SELECT input_hash, value_json FROM decisions").fetchone()
    assert "private" not in hash_value and "private" not in value_json


def test_recent_can_be_filtered_by_type_and_is_newest_first(tmp_path):
    conn = migrated(tmp_path)
    log = DecisionLog(conn)
    log.record(a_decision(decision_id="d-1"), "route")
    log.record(a_decision(decision_id="d-2", value="local_fast"), "model_level")
    assert [row["decision_id"] for row in log.recent(limit=5)] == ["d-2", "d-1"]
    assert [row["type"] for row in log.recent(decision_type="route")] == ["route"]


def test_a_policy_turns_a_confidence_into_an_outcome(tmp_path):
    conn = migrated(tmp_path)
    log = DecisionLog(conn)

    class _Provider:
        name = "rules"

        async def choose(self, question, options, context, *, decision_type):
            return a_decision()

    chain = DecisionChain([_Provider()], {"route": ["rules"]},
                          policies={"route": Policy(auto_above=0.9, ask_below=0.6)},
                          recorder=log.record)
    asyncio.run(chain.choose("?", ["fast_command", "llm"], {}, decision_type="route"))
    assert log.recent()[0]["outcome"] == "act"


def test_the_live_chain_knows_the_types_the_pipeline_asks_about():
    assert {"route", "model_level"} <= set(hub_app.DECISION_ORDER)
    assert set(hub_app.DECISION_ORDER) <= set(hub_app.DECISION_POLICIES)


def test_the_app_chain_records_every_routing_decision(tmp_path, monkeypatch):
    conn = migrated(tmp_path)
    log = DecisionLog(conn)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", log)
    config = load_config(__import__("pathlib").Path(__file__).resolve().parents[1] / "config.example.yaml")
    monkeypatch.setattr(hub_app, "_config", config)

    chain = hub_app._decision_chain(["rowan ai"])
    asyncio.run(chain.choose("rowan ai turn on the light", ["fast_command", "llm"],
                             {"text": "rowan ai turn on the light"}, decision_type="route"))
    rows = log.recent(decision_type="route")
    assert len(rows) == 1
    assert rows[0]["provider"] == "rules"
    assert rows[0]["outcome"] in {"act", "log", "ask"}


def test_the_rules_provider_answers_the_model_level_question():
    provider = RulesDecider(wake_phrases=("rowan ai",))
    decision = asyncio.run(provider.choose("?", ["local_fast", "local_strong"],
                                           {"text": "turn on the lamp"},
                                           decision_type="model_level"))
    assert decision.value == "local_fast"
    hard = asyncio.run(provider.choose("?", ["local_fast", "local_strong"],
                                       {"text": "explain why this fails step by step"},
                                       decision_type="model_level"))
    assert hard.value == "local_strong"
