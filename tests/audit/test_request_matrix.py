"""Матрица массового аудита: тысячи проверок договора запросов без сети.

Владелец просил аудит «с несколько тысяч тестов разных сценариев». Живой
стенд (`scripts/live-eval.py` + собранный корпус) стоит денег и времени на
каждый прогон, поэтому рядом с ним идёт эта матрица: те же сценарии, но
проверяются вещи, которые решаются в коде, а не моделью —

* каждый инструмент, которого сценарий ждёт, вообще существует в договоре
  (`hub.tools.TOOL_NAMES`) — опечатка в ожидании ловится сразу;
* сужение набора инструментов по семейству (Jev, U-14) не может спрятать
  инструмент, который сценарию нужен;
* ``pc_control`` кладёт имя приложения туда, куда смотрит клиент;
* сценарии уникальны и не дублируют друг друга.

Корпус берётся из генератора (`scripts/gen-audit-scenarios.py`), поэтому
матрица растёт вместе с ним.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_generator() -> Any:
    path = REPO_ROOT / "scripts" / "gen-audit-scenarios.py"
    spec = importlib.util.spec_from_file_location("gen_audit_scenarios", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_generator = _load_generator()
SCENARIOS: list[dict[str, Any]] = _generator.build_scenarios()


@pytest.fixture(scope="module", autouse=True)
def _config() -> Any:
    """The hub's own config, so the narrowing rules run with real thresholds."""
    from common.config import load_config
    from hub import app as hub_app

    cfg = load_config(str(REPO_ROOT / "config.openai.yaml"))
    hub_app.configure(cfg)
    return cfg


def _ids(rows: list[Any]) -> list[str]:
    return [f"{row[0]}-{'/'.join(row[1:])}" if isinstance(row, tuple) else str(row) for row in rows]


def _expected_names(scenario: dict[str, Any]) -> list[str]:
    names = list(scenario.get("expect_tools") or [])
    names += list(scenario.get("expect_any") or [])
    names += list(scenario.get("expect_first") or [])
    names += list(scenario.get("forbid_tools") or [])
    return names


CORPUS_CASES = [(item["id"], item["said"], name)
                for item in SCENARIOS
                for name in _expected_names(item)]


def test_the_corpus_is_large_enough() -> None:
    """The owner asked for thousands of scenarios; the corpus must stay big."""
    assert len(SCENARIOS) >= 900, f"корпус усох до {len(SCENARIOS)} сценариев"


def test_scenario_ids_are_unique() -> None:
    ids = [item["id"] for item in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_no_scenario_repeats_a_sentence() -> None:
    said = [item["said"] for item in SCENARIOS]
    duplicated = {text for text in said if said.count(text) > 1}
    assert not duplicated, f"дубли реплик: {sorted(duplicated)[:5]}"


@pytest.mark.parametrize("scenario_id,said,name", CORPUS_CASES, ids=_ids(CORPUS_CASES))
def test_expected_tool_exists_in_the_contract(scenario_id: str, said: str, name: str) -> None:
    """An expectation for a tool that does not exist would fail every run."""
    from hub.tools import TOOL_NAMES

    assert name in TOOL_NAMES, f"{scenario_id}: «{said}» ждёт несуществующий {name!r}"


FAMILY_CASES: list[tuple[str, str, str, str]] = []
for _item in SCENARIOS:
    from hub.tools import TOOL_FAMILIES  # noqa: E402 - table built at import time

    _needed = list(_item.get("expect_tools") or []) + list(_item.get("expect_first") or [])
    for _family, _members in TOOL_FAMILIES.items():
        _inside = [name for name in _needed if name in _members]
        if _inside:
            FAMILY_CASES.append((_item["id"], _family, ",".join(_inside), _item["said"]))


@pytest.mark.parametrize("scenario_id,family,needed,said", FAMILY_CASES, ids=_ids(FAMILY_CASES))
def test_family_narrowing_keeps_the_needed_tool(scenario_id: str, family: str,
                                                needed: str, said: str) -> None:
    """Jev's narrowing may only take tools away (U-14) — never the right one.

    ``None`` from ``_narrow_tools_for`` means "every tool stays", which also
    offers the needed one, so both answers are accepted.
    """
    from hub.app import _narrow_tools_for
    from hub.tools import TOOLS

    understanding = {"act": {"value": True, "confidence": 0.99},
                     "family": {"value": family, "confidence": 0.99}}
    offered = _narrow_tools_for(understanding) or TOOLS
    names = {tool["function"]["name"] for tool in offered}
    for name in needed.split(","):
        assert name in names, (
            f"{scenario_id}: «{said}» — сужение на семейство {family} спрятало {name}")


PC_SLOTS = [("open_app", "value", "chrome"), ("open_app", "target", "chrome"),
            ("close_app", "target", "spotify"), ("minimize_app", "value", "discord"),
            ("focus_app", "target", "code"), ("volume_up", "value", "ignored")]


@pytest.mark.parametrize("command,slot,name", PC_SLOTS, ids=_ids(PC_SLOTS))
def test_pc_control_app_name_reaches_the_client(command: str, slot: str, name: str) -> None:
    """The model sends the app name in either slot; the client reads ``value``."""
    from hub.tools import PC_APP_COMMANDS, normalize_pc_control_args

    cleaned = normalize_pc_control_args({"command": command, slot: name})
    if command in PC_APP_COMMANDS:
        assert cleaned.get("value") == name, f"{command}: имя приложения потерялось ({cleaned})"
        assert "target" not in cleaned or cleaned.get("target") == name
    else:
        assert cleaned == {"command": command, slot: name}, "чужая команда не должна меняться"


@pytest.mark.parametrize("value,expected", [
    (None, "left"), ("", "left"), ("right", "right"), ("context", "right"),
    ("double click", "double"), ("middle", "left"), ("RIGHT", "right"),
], ids=lambda value: str(value))
def test_click_button_normalisation(value: Any, expected: str) -> None:
    from hub.tools import normalize_click_button

    assert normalize_click_button(value) == expected
