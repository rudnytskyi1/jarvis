"""В метриках нет секретов и людей, и живут они там же, где `/health` (P4-39).

ТЗ 15.5: «секреты — только из окружения; логи и ошибки санитизируются».
Метрики видны Grafana и Prometheus и живут дольше логов, поэтому проверяется
две вещи: набор меток ограничен домом/стадией/классом/видом (`home`, `stage`,
`class`, `kind`, `job`, `level`, `state`, `le`), а сам endpoint не имеет ни
своего порта, ни своей сети — он часть того же приложения, что `/health`.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from common.config import Config, MetricsConfig
from hub import app as hub_app
from hub import metrics as metrics_mod

#: Метки, которые метрике разрешено носить. Здесь нет ни `person`, ни `name`:
#: имени человека в метриках неоткуда взяться.
ALLOWED_LABELS = {"home", "stage", "class", "kind", "job", "level", "state", "le"}

_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="')


def _label_names(text: str) -> set[str]:
    return {name for name in _LABEL.findall(text) if name != "expr"}


@pytest.fixture
def live_metrics(monkeypatch):
    """Real registry, real queue, real budget row — and one person's turn."""
    registry = metrics_mod.MetricsRegistry()
    registry.observe_turn(home_id="livingroom",
                          stages_ms={"stt": 120, "llm": 900, "tts": 300, "total": 1400},
                          degraded=("diarization",), level="local_fast")
    registry.observe_turn(home_id="killfloor", stages_ms={"stt": 90}, ok=False)
    monkeypatch.setattr(hub_app, "_metrics", registry)
    monkeypatch.setattr(hub_app, "_gpu_queue", lambda: None)
    monkeypatch.setattr(hub_app, "_api_budget_status", lambda: {
        "accounted_usd": 1.5, "settled_estimate_usd": 1.0, "reserved_usd": 0.5,
        "limit_usd": 18.0, "unsettled_requests": 2})
    monkeypatch.setattr(hub_app, "_scheduler_snapshot",
                        lambda: [{"name": "digest.daily", "failures": 1}])
    return registry


def test_the_endpoint_writes_only_metrics_labels(live_metrics):
    body = asyncio.run(hub_app.metrics()).body.decode("utf-8")
    assert _label_names(body) <= ALLOWED_LABELS
    assert "livingroom" in body  # метка дома есть: она и есть разрешённая
    # Никаких «человеческих» метрик — ни имени, ни идентификатора человека.
    assert "person" not in body and "speaker" not in body and "utterance_id" not in body


def test_no_secret_or_person_leaks_into_the_metrics(monkeypatch, live_metrics):
    """Секрет в окружении и имя говорящего в ходу — а в метриках их нет."""
    monkeypatch.setenv("ROWAN_TELEGRAM_BOT_TOKEN", "123456:AA-secret-token-value")
    monkeypatch.setenv("ROWAN_ADMIN_PASSWORD", "hunter2-secret")
    turn = SimpleNamespace(utterance_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
                           home_id="livingroom", _degradations=[],
                           _speaker_name="Максим Петров", cfg=Config())
    hub_app.Connection._finish_utterance(turn, stages={"stt": 100, "llm": 200,
                                                       "tts": 50, "total": 400}, ok=True)
    body = asyncio.run(hub_app.metrics()).body.decode("utf-8")
    for secret in ("Максим", "Петров", "AA-secret-token-value", "hunter2-secret",
                   "01ARZ3NDEKTSV4RRFFQ69G5FAV"):
        assert secret not in body


def test_the_metrics_module_never_reads_the_environment():
    """Модуль метрик не имеет доступа к секретам: он их просто не читает."""
    source = (Path(metrics_mod.__file__)).read_text(encoding="utf-8")
    assert "os.environ" not in source
    assert "getenv" not in source


def test_the_endpoint_lives_where_health_lives(monkeypatch):
    """ТЗ F-707: `/metrics` закрыт так же, как `/health` — тот же порт и сеть."""
    paths = {getattr(route, "path", "") for route in hub_app.app.routes}
    assert "/health" in paths and "/metrics" in paths
    # Своего хоста и порта у метрик нет: их неоткуда открыть шире, чем хаб.
    assert set(MetricsConfig.model_fields) == {"enabled"}
    assert not hasattr(MetricsConfig(), "host") and not hasattr(MetricsConfig(), "port")
    # Выключенный endpoint молчит честным кодом, а не пустой страницей.
    monkeypatch.setattr(hub_app, "_config",
                        Config(server={"metrics": {"enabled": False}}))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(hub_app.metrics())
    assert caught.value.status_code == 404
