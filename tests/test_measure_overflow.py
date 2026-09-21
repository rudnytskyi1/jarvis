"""The F-403 overflow threshold comes from a measurement (ТЗ 15.1, F-403)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scripts.measure_overflow import (
    MEASURED_HOMES,
    _write_threshold,
    allowed_extra_wait_s,
    measure,
    overflow_rate,
    percentile,
    summarize,
)


def test_percentiles_use_the_nearest_rank():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(samples, 0.5) == 3.0
    assert percentile(samples, 0.95) == 5.0
    assert percentile([], 0.95) == 0.0


def test_a_summary_reports_median_p95_and_worst():
    summary = summarize([0.1, 0.2, 0.3, 0.4, 9.0])
    assert summary == {"count": 5, "median_s": 0.3, "p95_s": 9.0, "max_s": 9.0}
    assert summarize([])["count"] == 0


def test_the_threshold_is_the_extra_wait_the_latency_table_allows():
    # ТЗ 15.1: transcript 0.7 s + decider 0.4 s, allowed to grow by 1.5x with
    # three active rooms, leaves 0.55 s of extra waiting for the queue.
    assert allowed_extra_wait_s() == 0.55
    assert allowed_extra_wait_s(stt_s=0.7, decide_s=0.4, growth=2.0) == 1.1


def test_the_overflow_rate_counts_only_the_jobs_over_the_threshold():
    assert overflow_rate([0.0, 0.1, 0.6, 2.0], 0.55) == 0.5
    assert overflow_rate([], 0.55) == 0.0
    assert overflow_rate([0.0, 0.0], 0.55) == 0.0


def test_three_active_rooms_are_all_served_and_measured():
    result = asyncio.run(measure(
        homes=MEASURED_HOMES, rate_per_min=4.0, service_s=2.5, utterances_per_home=3,
        max_concurrent=2, fair_share=0.5, max_waiting=64, time_scale=40.0,
    ))
    assert len(result["waits"]) == 3 * len(MEASURED_HOMES)
    assert result["stats"]["running_by_home"] == {}, "every job finished"
    assert result["stats"]["dropped"] == 0


def test_the_measurement_reports_one_admission_wait_per_job():
    """The waits come from the queue's own bookkeeping (queued → started), so
    they are never negative and never include the service time itself."""
    result = asyncio.run(measure(
        homes=MEASURED_HOMES, rate_per_min=600.0, service_s=2.5, utterances_per_home=6,
        max_concurrent=2, fair_share=0.5, max_waiting=64, time_scale=40.0,
    ))
    waits = result["waits"]
    assert len(waits) == 6 * len(MEASURED_HOMES)
    assert all(wait >= 0.0 for wait in waits)
    # Even a saturated run cannot report a wait longer than the whole run:
    # eighteen 2.5 s jobs on two slots are 22 s of work at the very worst.
    assert max(waits) <= 25.0
    assert result["stats"]["last_wait_s"]["utterance"] >= 0.0


def test_the_report_turns_waits_into_an_overflow_decision():
    # The design load mostly stays inside the budget; a queue with a long tail
    # is what the overflow exists for.
    design = [0.0, 0.1, 0.2, 0.3, 0.9]
    threshold = allowed_extra_wait_s()
    assert overflow_rate(design, threshold) == 0.2
    assert overflow_rate([3.0, 4.0], threshold) == 1.0


def test_the_threshold_is_written_into_the_config_without_losing_comments(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "server:\n  port: 1\nmodels:\n  enabled: false\n  routing:\n"
        "    short_chars: 120\n    overflow_wait_s: 1.5    # F-403 threshold\n"
        "    cloud_fallback: false\nclient:\n  client_id: room\n",
        encoding="utf-8",
    )

    _write_threshold(path, 4.46)
    text = path.read_text(encoding="utf-8")
    assert "overflow_wait_s: 4.46    # F-403 threshold" in text
    assert "# F-403 threshold" in text
    assert text.startswith("server:\n  port: 1\n")


def test_writing_the_threshold_needs_the_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("models:\n  routing:\n    cloud_fallback: false\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _write_threshold(path, 2.0)


def test_the_repository_config_carries_a_measured_threshold():
    from common.config import load_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    threshold = cfg.models.routing.overflow_wait_s
    # The value in the config is the calibrated one (DECISIONS.md, P1-24): the
    # extra wait 15.1 allows with three active rooms, not the 1.5 s guess.
    assert threshold == allowed_extra_wait_s()
