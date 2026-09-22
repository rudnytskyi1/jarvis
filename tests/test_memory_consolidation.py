"""Ночная консолидация памяти дома (ТЗ F-416)."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from common.config import MemoryConfig
from hub import migrations_runner, vectors
from hub.homes import ensure_home
from hub.memories import Kind, MemoryIndex, Scope, fact_from
from hub.memory_consolidation import (
    DigestFact,
    DigestUnavailable,
    MemoryConsolidationTask,
    MemoryConsolidator,
    llm_summarizer,
    normalize_text,
)

# 09:00 UTC — это 04:00 в Чикаго (CDT), 12:00 в Киеве и 02:00 в Лос-Анджелесе.
NIGHT = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)


def connect(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    ensure_home(conn, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    ensure_home(conn, "la", name="LA", tz="America/Los_Angeles")
    return conn


def home(home_id="livingroom", tz="America/Chicago"):
    return type("Home", (), {"home_id": home_id, "tz": tz})()


def consolidator(conn, *, summarizer=None, embedder=None, audit=None, config=None):
    return MemoryConsolidator(conn, config=config or MemoryConfig(), summarizer=summarizer,
                              embedder=embedder, audit=audit, clock=lambda: NIGHT)


def put(conn, text, *, scope=Scope.HOME, owner="livingroom", kind=Kind.EVENT,
        weight=1.0, created_at=NIGHT, ttl_s=None, vector=None):
    fact = fact_from(scope=scope, owner_id=owner, kind=kind, text=text, weight=weight,
                     ttl_s=ttl_s, vector=vector, now=created_at)
    MemoryIndex(conn).write(fact)
    return fact


# --- когда ---------------------------------------------------------------


def test_a_home_is_consolidated_on_its_own_clock(tmp_path):
    conn = connect(tmp_path)
    try:
        night = consolidator(conn)
        assert night.is_due("livingroom", "America/Chicago", now=NIGHT) is True
        assert night.is_due("kyiv", "Europe/Kyiv", now=NIGHT) is True
        # 02:00 по местному времени Лос-Анджелеса: ночь ещё не наступила.
        assert night.is_due("la", "America/Los_Angeles", now=NIGHT) is False
    finally:
        conn.close()


def test_the_pass_is_not_repeated_on_the_same_local_day(tmp_path):
    conn = connect(tmp_path)
    try:
        night = consolidator(conn)
        night.consolidate_home("livingroom", tz_name="America/Chicago", now=NIGHT)
        later = NIGHT + timedelta(hours=2)
        assert night.is_due("livingroom", "America/Chicago", now=later) is False
        tomorrow = NIGHT + timedelta(days=1)
        assert night.is_due("livingroom", "America/Chicago", now=tomorrow) is True
        assert night.last_pass_day("livingroom") == "2026-09-21"
    finally:
        conn.close()


def test_a_hub_that_was_off_at_four_am_catches_up(tmp_path):
    conn = connect(tmp_path)
    try:
        night = consolidator(conn)
        assert night.is_due("livingroom", "America/Chicago",
                            now=NIGHT + timedelta(hours=6)) is True
    finally:
        conn.close()


def test_an_unknown_time_zone_falls_back_to_utc(tmp_path):
    conn = connect(tmp_path)
    try:
        night = consolidator(conn)
        assert night.is_due("livingroom", "Mars/Olympus", now=NIGHT) is True
    finally:
        conn.close()


def test_the_switch_off_keeps_the_night_quiet(tmp_path):
    conn = connect(tmp_path)
    try:
        night = consolidator(conn, config=MemoryConfig(consolidation_enabled=False))
        assert night.is_due("livingroom", "America/Chicago", now=NIGHT) is False
    finally:
        conn.close()


def test_the_digest_bounds_are_validated_in_the_config():
    with pytest.raises(ValueError, match="consolidation_min_facts"):
        MemoryConfig(consolidation_min_facts=12, consolidation_max_facts=5)


# --- что делает проход ----------------------------------------------------


def test_expired_facts_are_removed(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "milk is over", ttl_s=3600, created_at=NIGHT - timedelta(days=2))
        put(conn, "the lamp is on the desk")
        report = consolidator(conn).consolidate_home("livingroom",
                                                     tz_name="America/Chicago", now=NIGHT)
        assert report.purged == 1
        assert MemoryIndex(conn).count() == 1
    finally:
        conn.close()


def test_old_facts_lose_weight_and_never_reach_the_floor(tmp_path):
    conn = connect(tmp_path)
    try:
        old = put(conn, "old fact", created_at=NIGHT - timedelta(days=10))
        fresh = put(conn, "fresh fact", created_at=NIGHT - timedelta(hours=2))
        ancient = put(conn, "ancient fact", created_at=NIGHT - timedelta(days=40))
        report = consolidator(conn).consolidate_home("livingroom",
                                                     tz_name="America/Chicago", now=NIGHT)
        index = MemoryIndex(conn)
        assert report.decayed == 2
        assert index.read(old.memory_id).weight == pytest.approx(0.9 ** 10, abs=1e-3)
        assert index.read(ancient.memory_id).weight == pytest.approx(0.05, abs=1e-6)
        assert index.read(fresh.memory_id).weight == 1.0
    finally:
        conn.close()


def test_the_same_fact_said_twice_becomes_one_fact_with_the_full_weight(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "Anton likes green tea", weight=0.4,
            created_at=NIGHT - timedelta(hours=5))
        second = put(conn, "  anton   LIKES green tea! ", weight=0.9,
                     created_at=NIGHT - timedelta(hours=1))
        report = consolidator(conn).consolidate_home("livingroom",
                                                     tz_name="America/Chicago", now=NIGHT)
        left = MemoryIndex(conn).active()
        assert report.merged == 1
        assert len(left) == 1
        assert left[0].memory_id == second.memory_id
        assert left[0].weight == pytest.approx(0.9)
    finally:
        conn.close()


def test_two_people_keep_their_own_version_of_a_fact(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "my keys are on the desk", scope=Scope.PERSON, owner="Anton",
            kind=Kind.PERSON_FACT)
        put(conn, "my keys are on the desk", scope=Scope.PERSON, owner="Max",
            kind=Kind.PERSON_FACT)
        report = consolidator(conn).consolidate_home("livingroom",
                                                     tz_name="America/Chicago", now=NIGHT)
        assert report.merged == 0
        assert MemoryIndex(conn).count() == 2
    finally:
        conn.close()


def test_close_embeddings_merge_only_when_both_facts_have_one(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "the light is on", created_at=NIGHT - timedelta(hours=3),
            vector=[1.0, 0.0, 0.0, 0.0])
        put(conn, "the lamp is switched on", created_at=NIGHT - timedelta(hours=1),
            vector=[0.99, 0.01, 0.0, 0.0])
        report = consolidator(conn).consolidate_home("livingroom",
                                                     tz_name="America/Chicago", now=NIGHT)
        assert report.merged == 1
        assert MemoryIndex(conn).count() == 1
    finally:
        conn.close()


def test_a_fact_without_a_vector_is_not_merged_by_meaning(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "the light is on", created_at=NIGHT - timedelta(hours=3),
            vector=[1.0, 0.0, 0.0, 0.0])
        put(conn, "the lamp is switched on", created_at=NIGHT - timedelta(hours=1))
        report = consolidator(conn).consolidate_home("livingroom",
                                                     tz_name="America/Chicago", now=NIGHT)
        assert report.merged == 0
    finally:
        conn.close()


class _Embedder:
    """Считает вектор из длины текста: видно, что он действительно записан."""

    def __init__(self, *, fail=False):
        self.fail = fail
        self.seen: list[str] = []

    def encode(self, texts, *, query=False):  # noqa: ARG002 - как у настоящего эмбеддера
        if self.fail:
            raise RuntimeError("no model on this machine")
        self.seen.extend(texts)
        return [[float(len(text)), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for text in texts]


def test_missing_embeddings_are_caught_up(tmp_path):
    conn = connect(tmp_path)
    try:
        text = "a fact written before the model arrived"
        without = put(conn, text)
        with_vector = put(conn, "a fact with its vector", vector=[1.0, 0.0])
        embedder = _Embedder()
        report = consolidator(conn, embedder=embedder).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        index = MemoryIndex(conn)
        assert report.embedded == 1
        assert embedder.seen == [text]
        stored = index.read(without.memory_id)
        assert stored.vector is not None
        # _Embedder кодирует длину текста: так видно, что вектор записан.
        assert vectors.unpack_vector(stored.vector)[0] == float(len(text))
        assert index.read(with_vector.memory_id).dim == 2
    finally:
        conn.close()


def test_an_embedder_that_is_missing_leaves_the_facts_alone(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "a fact without a vector")
        report = consolidator(conn, embedder=_Embedder(fail=True)).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        assert report.embedded == 0
        assert MemoryIndex(conn).count() == 1
    finally:
        conn.close()


# --- сводка дня -----------------------------------------------------------


def summarizer(facts, *, raises=None):
    """Суммаризатор с заданным ответом; запоминает материал, который получил."""
    seen: dict = {}

    def run(texts, minimum, maximum):
        seen["texts"] = list(texts)
        seen["bounds"] = (minimum, maximum)
        if raises is not None:
            raise raises
        return facts

    return run, seen


def digest_facts(count=5):
    return [{"text": f"fact {index}", "kind": "event"} for index in range(count)]


def test_the_day_becomes_home_facts(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "Anton asked about the physics exam")
        run, seen = summarizer(digest_facts(5))
        report = consolidator(conn, summarizer=run).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        assert report.digest == 5 and report.summarizer == "llm"
        assert seen["bounds"] == (5, 10)
        assert "Anton asked about the physics exam" in seen["texts"]
        written = [fact for fact in MemoryIndex(conn).active() if fact.text.startswith("fact ")]
        assert len(written) == 5
        assert {str(fact.scope) for fact in written} == {"home"}
        assert {fact.owner_id for fact in written} == {"livingroom"}
    finally:
        conn.close()


def test_a_digest_below_five_facts_is_refused(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "something happened")
        run, _ = summarizer(digest_facts(3))
        report = consolidator(conn, summarizer=run).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        assert report.digest == 0
        assert report.summarizer == "none"
        assert "5-10" in report.note
    finally:
        conn.close()


def test_a_summarizer_that_fails_writes_no_summary(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "something happened")
        run, _ = summarizer(None, raises=DigestUnavailable("no model"))
        report = consolidator(conn, summarizer=run).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        assert report.digest == 0 and "DigestUnavailable" in report.note
    finally:
        conn.close()


def test_without_a_summarizer_only_the_mechanics_run(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "something happened")
        report = consolidator(conn).consolidate_home("livingroom",
                                                     tz_name="America/Chicago", now=NIGHT)
        assert report.digest == 0 and report.summarizer == "none"
        assert "summarizer" in report.note
    finally:
        conn.close()


def test_a_day_with_nothing_in_it_is_not_summarised(tmp_path):
    conn = connect(tmp_path)
    try:
        run, seen = summarizer(digest_facts(5))
        report = consolidator(conn, summarizer=run).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        assert report.digest == 0
        assert "nothing" in report.note and "texts" not in seen
    finally:
        conn.close()


def test_the_days_turns_are_part_of_the_material(tmp_path):
    conn = connect(tmp_path)
    try:
        conn.execute(
            "INSERT INTO dialog_turns(turn_id, home_id, role, text, ts) VALUES (?,?,?,?,?)",
            ("t1", "livingroom", "user", "turn the light off", NIGHT.timestamp() - 60))
        conn.commit()
        run, seen = summarizer(digest_facts(5))
        consolidator(conn, summarizer=run).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        assert "turn the light off" in seen["texts"]
    finally:
        conn.close()


def test_the_sources_of_the_digest_are_folded_away(tmp_path):
    conn = connect(tmp_path)
    try:
        source = put(conn, "Anton asked about the physics exam", weight=0.5,
                     created_at=NIGHT - timedelta(hours=4))
        run, _ = summarizer(digest_facts(5))
        report = consolidator(conn, summarizer=run).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        assert report.folded == 1
        assert MemoryIndex(conn).read(source.memory_id) is None
    finally:
        conn.close()


def test_folding_can_be_switched_off(tmp_path):
    conn = connect(tmp_path)
    try:
        source = put(conn, "Anton asked about the physics exam", weight=0.5,
                     created_at=NIGHT - timedelta(hours=4))
        run, _ = summarizer(digest_facts(5))
        config = MemoryConfig(fold_sources=False)
        report = consolidator(conn, summarizer=run, config=config).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        assert report.folded == 0
        assert MemoryIndex(conn).read(source.memory_id) is not None
    finally:
        conn.close()


def test_a_digest_is_a_home_fact_a_guest_never_sees(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "the room watched a film")
        run, _ = summarizer(digest_facts(5))
        consolidator(conn, summarizer=run).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        from hub.memory_search import visible

        digest = [fact for fact in MemoryIndex(conn).active() if fact.text.startswith("fact ")]
        assert digest
        for fact in digest:
            assert visible(fact, person="Anton", home_id="livingroom", member=True) is True
            assert visible(fact, person="Anton", home_id="livingroom", member=False) is False
    finally:
        conn.close()


def test_the_report_reaches_the_audit(tmp_path):
    conn = connect(tmp_path)
    try:
        from hub.audit import AuditLog

        put(conn, "something happened")
        consolidator(conn, audit=AuditLog(conn)).consolidate_home(
            "livingroom", tz_name="America/Chicago", now=NIGHT)
        row = conn.execute("SELECT action, target, result, detail_json FROM audit "
                           "WHERE action='memory.consolidate'").fetchone()
        assert row is not None and row[1] == "livingroom"
        assert json.loads(row[3])["home_id"] == "livingroom"
    finally:
        conn.close()


# --- задача планировщика --------------------------------------------------


def test_the_task_only_visits_the_homes_whose_night_has_come(tmp_path):
    conn = connect(tmp_path)
    try:
        put(conn, "the room watched a film")
        run, _ = summarizer(digest_facts(5))
        task = MemoryConsolidationTask(
            consolidator(conn, summarizer=run),
            homes=[home("livingroom"), home("la", "America/Los_Angeles")],
            interval_s=900,
        )
        assert task.name == "memory.consolidate" and task.interval_s == 900.0
        report = asyncio.run(task.run())
        assert report["homes"] == 1 and report["digest"] == 5
        assert consolidator(conn).last_pass_day("la") == ""
    finally:
        conn.close()


def test_the_task_says_nothing_when_no_home_is_due(tmp_path):
    conn = connect(tmp_path)
    try:
        task = MemoryConsolidationTask(consolidator(conn), homes=[], interval_s=900)
        assert asyncio.run(task.run()) == {"homes": 0}
    finally:
        conn.close()


def test_a_broken_home_does_not_stop_the_others(tmp_path, monkeypatch):
    conn = connect(tmp_path)
    try:
        night = consolidator(conn)
        real = night.consolidate_home_async

        async def explode(home_id, **kwargs):
            if home_id == "livingroom":
                raise RuntimeError("this home is broken")
            return await real(home_id, **kwargs)

        monkeypatch.setattr(night, "consolidate_home_async", explode)
        reports = asyncio.run(night.run_due([home("livingroom"), home("kyiv", "Europe/Kyiv")],
                                            now=NIGHT))
        assert [report.home_id for report in reports] == ["kyiv"]
    finally:
        conn.close()


# --- суммаризатор на модели ----------------------------------------------


class _FakeLlm:
    def __init__(self, answer=None, *, error=None):
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    async def structured_json(self, messages, schema, *, name="result"):
        self.calls.append({"messages": messages, "schema": schema, "name": name})
        if self.error is not None:
            raise self.error
        return self.answer


def test_the_model_summariser_returns_validated_facts():
    llm = _FakeLlm({"facts": [{"text": "one"}, {"text": "two", "kind": "preference"}]})
    result = asyncio.run(llm_summarizer(llm)(["a day of material"], 1, 5))
    assert [fact.text for fact in result] == ["one", "two"]
    assert result[1].kind is Kind.PREFERENCE
    assert llm.calls[0]["schema"]["required"] == ["facts"]


def test_the_model_summariser_refuses_an_answer_without_facts():
    llm = _FakeLlm({"answer": "I do not know"})
    with pytest.raises(DigestUnavailable):
        asyncio.run(llm_summarizer(llm)(["material"], 5, 10))


def test_the_model_summariser_reports_a_missing_guided_json():
    from hub.llm import StructuredUnavailable

    llm = _FakeLlm(error=StructuredUnavailable("no guided json here"))
    with pytest.raises(DigestUnavailable):
        asyncio.run(llm_summarizer(llm)(["material"], 5, 10))


def test_a_digest_fact_is_strict():
    with pytest.raises(ValueError):
        DigestFact(text="   ")
    with pytest.raises(ValueError):
        DigestFact(text="fact", owner_id="Anton")  # у этого контракта такого поля нет
    assert DigestFact(text="  two   words ").text == "two words"


def test_normalisation_makes_two_spellings_one_fact():
    assert normalize_text(" The LIGHT is ON!  ") == "the light is on"
    assert normalize_text(None) == ""
