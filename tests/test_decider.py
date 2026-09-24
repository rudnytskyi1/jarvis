"""The decision layer: rules provider, chain and confidence policy (ТЗ section 5)."""
from __future__ import annotations

import asyncio

import pytest

from hub.decider import DecisionUnavailable, Policy, RulesDecider

ROUTE_OPTIONS = ["fast_command", "llm", "banter", "not_addressed"]


def _choose(decider, text, options=ROUTE_OPTIONS, decision_type="route"):
    return asyncio.run(decider.choose(text, options, {"text": text}, decision_type=decision_type))


def test_policy_maps_confidence_to_an_outcome():
    policy = Policy(auto_above=0.8, ask_below=0.5)
    assert policy.outcome(0.9) == "act"
    assert policy.outcome(0.8) == "act"
    assert policy.outcome(0.6) == "log"
    assert policy.outcome(0.2) == "ask"


def test_rules_provider_routes_a_fast_command():
    decision = _choose(RulesDecider(wake_phrases=("rowan",)), "volume 30")
    assert decision.value == "fast_command"
    assert decision.provider == "rules"
    assert decision.confidence > 0.9
    assert decision.latency_ms >= 0
    assert decision.decision_id


def test_rules_provider_routes_free_speech_to_the_model():
    decision = _choose(RulesDecider(wake_phrases=("rowan",)), "какая погода")
    assert decision.value == "llm"


def test_rules_provider_detects_a_wake_address():
    decider = RulesDecider(wake_phrases=("rowan",))
    addressed = asyncio.run(decider.yes_no("addressed?", {"text": "rowan volume 30"}, decision_type="addressed"))
    ignored = asyncio.run(decider.yes_no("addressed?", {"text": "какая погода"}, decision_type="addressed"))
    assert addressed.value is True
    assert ignored.value is False


def test_rules_provider_flags_an_impossible_transcript():
    decider = RulesDecider()
    impossible = asyncio.run(decider.yes_no("noise?", {"text": "x" * 200, "duration_s": 1.0},
                                            decision_type="hallucination"))
    plausible = asyncio.run(decider.yes_no("noise?", {"text": "turn the light off", "duration_s": 2.0},
                                           decision_type="hallucination"))
    assert impossible.value is True
    assert plausible.value is False


def test_rules_provider_refuses_unknown_decisions_instead_of_guessing():
    decider = RulesDecider()
    with pytest.raises(NotImplementedError):
        asyncio.run(decider.yes_no("?", {}, decision_type="taste_in_music"))
    with pytest.raises(NotImplementedError):
        asyncio.run(decider.score("?", {}, scale=(1, 5), decision_type="importance"))


def test_rules_provider_only_answers_with_an_offered_option():
    with pytest.raises(NotImplementedError):
        _choose(RulesDecider(), "volume 30", options=["llm"])


class _Stub:
    def __init__(self, name, *, answer=None, fail=False, delay=0.0, confidence=0.7):
        self.name = name
        self.answer = answer
        self.fail = fail
        self.delay = delay
        self.confidence = confidence
        self.calls = 0

    async def choose(self, question, options, context, *, decision_type):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise NotImplementedError(decision_type)
        from hub.decider import _decision
        return _decision(self.answer, self.confidence, self.name, 0.0)


def test_chain_falls_through_a_provider_that_cannot_answer():
    from hub.decider import DecisionChain

    blind = _Stub("blind", fail=True)
    eyes = _Stub("eyes", answer="fast_command")
    chain = DecisionChain([blind, eyes], {"route": ["blind", "eyes"]}, timeout_s=0.2)
    decision = chain and asyncio.run(chain.choose("?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert decision.provider == "eyes" and decision.value == "fast_command"
    assert blind.calls == 1 and eyes.calls == 1


def test_chain_skips_a_provider_that_times_out():
    from hub.decider import DecisionChain

    slow = _Stub("slow", answer="llm", delay=5.0)
    quick = _Stub("quick", answer="fast_command")
    chain = DecisionChain([slow, quick], {"route": ["slow", "quick"]}, timeout_s=0.01)
    decision = asyncio.run(chain.choose("?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert decision.provider == "quick"


def test_chain_records_every_answer_and_reports_a_missing_chain():
    from hub.decider import DecisionChain

    seen: list = []
    provider = _Stub("eyes", answer="llm")
    chain = DecisionChain([provider], {"route": ["eyes"]},
                          recorder=lambda decision, kind, outcome: seen.append((decision.provider, kind, outcome)))
    asyncio.run(chain.choose("?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert seen == [("eyes", "route", "pending")]
    with pytest.raises(DecisionUnavailable):
        asyncio.run(chain.choose("?", ROUTE_OPTIONS, {}, decision_type="addressee"))


def test_the_route_decision_uses_the_wake_words_of_this_utterance():
    """The chain is a process-wide singleton: the phrase list it was built
    with belongs to one connection, and must not decide for another one.

    Regression: a cached chain built with "rowan ai" routed a connection
    configured with the bare "rowan" to the model instead of the fast path.
    """
    provider = RulesDecider(wake_phrases=("rowan ai",))
    decision = asyncio.run(provider.choose(
        "?", ROUTE_OPTIONS,
        {"text": "rowan volume 30", "wake_words": ["rowan", "roan", "rowen"]},
        decision_type="route"))
    assert decision.value == "fast_command"

    cached = asyncio.run(provider.choose("?", ROUTE_OPTIONS, {"text": "rowan volume 30"},
                                         decision_type="route"))
    assert cached.value == "llm", "without context the configured phrases still apply"


def test_the_addressee_decision_uses_the_wake_words_of_this_utterance():
    provider = RulesDecider(wake_phrases=("rowan ai",))
    decision = asyncio.run(provider.yes_no(
        "?", {"text": "rowan turn the light off", "wake_words": ["rowan"]},
        decision_type="addressed"))
    assert decision.value is True


# --- второй провайдер в цепочке (AU-19) --------------------------------------


def _uncertain_chain(second, *, seen=None):
    """The production shape: rules first, Jev second, one policy for the type."""
    from hub.decider import DecisionChain

    recorder = None
    if seen is not None:
        recorder = lambda decision, kind, outcome: seen.append(
            (decision.provider, kind, round(decision.confidence, 2), outcome))
    return DecisionChain(
        [_Stub("rules", answer="llm", confidence=0.6), second],
        {"route": ["rules", "jev"]}, timeout_s=0.3,
        policies={"route": Policy(auto_above=0.9, ask_below=0.6)},
        recorder=recorder)


def test_the_second_provider_is_asked_when_the_first_answer_is_unsure():
    """AU-19: «правила ответили 0.6» — это догадка, а не решение.

    До этой правки ``rules`` заканчивал цепочку всегда, и в живой базе все 5087
    решений были от ``rules``, а Jev не спрашивали ни разу
    (``scripts/jev_usage_report.py``).
    """
    cloud = _Stub("jev", answer="fast_command", confidence=0.9)
    decision = asyncio.run(_uncertain_chain(cloud).choose(
        "?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert cloud.calls == 1
    assert decision.provider == "jev" and decision.value == "fast_command"


def test_a_quiet_first_answer_stays_when_the_second_is_no_better():
    """Спросили — но своё «0.6» оставили: второй ответ оказался не увереннее."""
    cloud = _Stub("jev", answer="fast_command", confidence=0.5)
    decision = asyncio.run(_uncertain_chain(cloud).choose(
        "?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert cloud.calls == 1
    assert decision.provider == "rules" and decision.value == "llm"


def test_a_confident_first_answer_never_costs_a_cloud_call():
    """0.95 — это «действуй»; платить за второе мнение не за что."""
    from hub.decider import DecisionChain

    cloud = _Stub("jev", answer="fast_command", confidence=0.99)
    chain = DecisionChain([_Stub("rules", answer="fast_command", confidence=0.95), cloud],
                          {"route": ["rules", "jev"]}, timeout_s=0.3,
                          policies={"route": Policy(auto_above=0.9, ask_below=0.6)})
    decision = asyncio.run(chain.choose("?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert cloud.calls == 0
    assert decision.provider == "rules"


def test_a_type_without_a_policy_still_stops_at_the_first_answer():
    """Точки безопасности (admin_rights, injection) идут как раньше: 0.6 решает."""
    from hub.decider import DecisionChain

    cloud = _Stub("jev", answer="fast_command", confidence=0.99)
    chain = DecisionChain([_Stub("rules", answer="llm", confidence=0.6), cloud],
                          {"route": ["rules", "jev"]}, timeout_s=0.3)
    decision = asyncio.run(chain.choose("?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert cloud.calls == 0 and decision.provider == "rules"


def test_both_answers_are_recorded_when_the_second_provider_speaks():
    """Панель должна показывать и переубеждённый ответ, а не только победителя."""
    seen: list = []
    cloud = _Stub("jev", answer="fast_command", confidence=0.9)
    asyncio.run(_uncertain_chain(cloud, seen=seen).choose(
        "?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert seen == [("rules", "route", 0.6, "log"), ("jev", "route", 0.9, "act")]


def test_a_second_provider_that_fails_leaves_the_first_answer():
    """Fail-open (ТЗ раздел 1): облако молчит — работает локальный ответ."""
    cloud = _Stub("jev", fail=True)
    decision = asyncio.run(_uncertain_chain(cloud).choose(
        "?", ROUTE_OPTIONS, {}, decision_type="route"))
    assert cloud.calls == 1
    assert decision.provider == "rules" and decision.value == "llm"
