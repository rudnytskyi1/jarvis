"""Utterance regression harness (ТЗ section 15.6).

The fixture file is JSONL; every record is validated strictly before use.
Fixtures that need recorded audio or a live model are skipped with a reason
instead of silently passing, and the 95% threshold is enforced over the
fixtures that actually ran.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hub.local_commands import direct_command

FIXTURES = Path(__file__).with_name("fixtures.jsonl")
PASS_THRESHOLD = 0.95
#: The router strips these before matching, exactly like the live pipeline does.
WAKE_WORDS = ("rowan", "rowan ai", "roan ai")


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    language: Literal["en", "ru", "es"]
    transcript: str = Field(min_length=1)
    expected_route: Literal["fast_command", "llm", "banter", "not_addressed"]
    expected_command: dict[str, Any] | None = None
    audio_file: str | None = None
    notes: str = ""


def load_fixtures() -> list[Fixture]:
    records: list[Fixture] = []
    for number, line in enumerate(FIXTURES.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(Fixture.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:  # pragma: no cover - bad file
            raise AssertionError(f"fixtures.jsonl line {number} is invalid: {exc}") from None
    return records


def test_fixture_file_is_well_formed_and_large_enough():
    fixtures = load_fixtures()
    assert len(fixtures) >= 30, "the regression set must keep growing toward 100 recorded utterances"
    assert len({item.id for item in fixtures}) == len(fixtures), "fixture ids must be unique"
    assert {item.language for item in fixtures} >= {"en", "ru"}, "at least two languages must be covered"


def _replayable(item: Fixture) -> bool:
    """Only the text router can be replayed today; audio routes need recordings."""
    return item.expected_route in {"fast_command", "llm"}


@pytest.mark.parametrize("item", [f for f in load_fixtures() if not _replayable(f)],
                         ids=lambda f: f.id)
def test_audio_routes_need_a_recording(item):
    pytest.skip(f"{item.id}: needs a recorded utterance ({item.expected_route}); no audio in the repo yet")


@pytest.mark.parametrize("item", [f for f in load_fixtures() if _replayable(f)], ids=lambda f: f.id)
def test_router_matches_the_expected_route(item):
    result = direct_command(item.transcript, wake_words=WAKE_WORDS)
    if item.expected_route == "fast_command":
        assert result is not None, f"expected the fast path, got None ({item.notes})"
        assert result[0] == item.expected_command, (
            f"expected {item.expected_command}, got {result[0]} ({item.notes})"
        )
    else:
        assert result is None, f"expected the LLM route, but the fast path answered {result[0]}"


def test_routing_pass_rate_meets_the_threshold():
    """The 95% gate from the spec, over the fixtures that actually ran."""
    fixtures = [item for item in load_fixtures() if _replayable(item)]
    failures: list[str] = []
    for item in fixtures:
        result = direct_command(item.transcript, wake_words=WAKE_WORDS)
        if item.expected_route == "fast_command":
            if result is None or result[0] != item.expected_command:
                failures.append(f"{item.id}: expected {item.expected_command}, got {None if result is None else result[0]}")
        elif result is not None:
            failures.append(f"{item.id}: expected the LLM route, got {result[0]}")
    executed = len(fixtures)
    assert executed >= 30, f"only {executed} replayable fixtures; the set must keep growing toward 100"
    rate = (executed - len(failures)) / executed
    assert rate >= PASS_THRESHOLD, (
        f"pass rate {rate:.0%} is below the {PASS_THRESHOLD:.0%} threshold; "
        f"{len(failures)} of {executed} failed:\n" + "\n".join(failures)
    )
