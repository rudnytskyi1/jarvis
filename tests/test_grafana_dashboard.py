"""Дашборд Grafana собран из настоящих метрик хаба (ТЗ F-707, P4-38).

Дашборд, который ссылается на несуществующую метрику, выглядит рабочим и
показывает пустоту — поэтому каждая метрика из его запросов сверяется с тем,
что хаб действительно экспортирует (`hub.metrics.METRIC_NAMES`), а README
проверяется на то, что он объясняет подключение, а не только лежит рядом.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from hub.metrics import METRIC_NAMES

DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "grafana"
DASHBOARD = DEPLOY / "rowan-dashboard.json"
README = DEPLOY / "README.md"

_METRIC = re.compile(r"\b(rowan_[a-z0-9_]+)\b")
_SUFFIXES = ("_bucket", "_sum", "_count")


def _dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _exprs(panel: dict) -> list[str]:
    return [str(target.get("expr", "")) for target in panel.get("targets", [])]


def test_the_dashboard_file_is_a_real_grafana_dashboard():
    board = _dashboard()
    assert board["title"] == "Rowan hub"
    assert board["uid"] == "rowan-hub"
    assert board["schemaVersion"] >= 30
    assert board["panels"], "дашборд без панелей ничего не показывает"
    assert board["time"] == {"from": "now-6h", "to": "now"}


def test_every_panel_has_a_query_and_a_source():
    board = _dashboard()
    for panel in board["panels"]:
        assert panel["type"] == "timeseries"
        assert panel["title"]
        assert panel["datasource"]["type"] == "prometheus"
        exprs = [expr for expr in _exprs(panel) if expr]
        assert exprs, f"панель {panel['title']!r} без запроса"
        for expr in exprs:
            assert "$__rate_interval" in expr or "rowan_" in expr


def test_every_metric_in_the_dashboard_is_really_exported():
    used: set[str] = set()
    for panel in _dashboard()["panels"]:
        for expr in _exprs(panel):
            for found in _METRIC.findall(expr):
                for suffix in _SUFFIXES:
                    if found.endswith(suffix):
                        found = found[: -len(suffix)]
                        break
                used.add(found)
    # $__rate_interval and the label placeholders are Grafana's, not metrics.
    assert used, "дашборд не спрашивает ни одной метрики Rowan"
    assert used <= METRIC_NAMES, f"панели ждут неизвестные метрики: {used - METRIC_NAMES}"
    # А то, что панели должны показать, покрыто: бюджет, очередь, память, ошибки.
    for name in ("rowan_turn_stage_milliseconds", "rowan_gpu_queue_jobs",
                 "rowan_gpu_vram_bytes", "rowan_turn_errors_total",
                 "rowan_api_budget_usd"):
        assert name in used


def test_the_readme_explains_how_to_connect_the_dashboard():
    readme = README.read_text(encoding="utf-8")
    assert "rowan-dashboard.json" in readme
    assert "/metrics" in readme
    assert "scrape_configs" in readme
    assert "8765" in readme
    # Приватность: README обязан назвать то, чего в метках нет.
    assert "человека" in readme.lower() and "15.5" in readme
