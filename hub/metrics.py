"""Метрики Prometheus (ТЗ F-707).

`/metrics` отвечает на четыре вопроса владельца железа: сколько ждёт
видеокарта, сколько занимает память, где теряется время хода (по стадиям и по
домам) и сколько денег ушло на облако. Всё это уже считается в хабе —
`hub.gpu_queue`, `hub.utterances`, журнал бюджета — поэтому модуль ничего не
угадывает: он собирает настоящие числа и печатает их в текстовом формате
Prometheus.

**Секретов и персональных данных здесь нет.** Метки — только дом, стадия,
класс очереди и вид ошибки; ни имён людей, ни текста реплик, ни токенов. Это
требование ТЗ 15.5 (логи и метрики санитизируются), поэтому метка человека
не может появиться тут даже случайно: её просто неоткуда взять — в
`MetricsRegistry` дом приходит строкой из конфига, а не именем говорящего.

Формат — текстовая экспозиция Prometheus `text/plain; version=0.0.4`:
`# HELP`/`# TYPE` на каждое семейство, `label="value"` для меток, `+Inf` в
гистограммах. Ничего, кроме печати, модуль не делает.
"""
from __future__ import annotations

import bisect
import logging
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("jarvis.server.metrics")

#: Границы гистограммы в миллисекундах: бюджеты стадий ТЗ 15.1 — 0,4–0,7 с,
#: поэтому интересное лежит ниже первой секунды, а хвост виден до 3 с.
DEFAULT_BUCKETS_MS: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0, 700.0,
                                        1000.0, 1500.0, 3000.0)

STAGE_METRIC = "rowan_turn_stage_milliseconds"
TURNS_METRIC = "rowan_turns_total"
TURN_ERRORS_METRIC = "rowan_turn_errors_total"
DEGRADED_METRIC = "rowan_turn_degraded_total"
LEVEL_METRIC = "rowan_turn_level_total"
QUEUE_METRIC = "rowan_gpu_queue_jobs"
QUEUE_DROPPED_METRIC = "rowan_gpu_queue_dropped_total"
QUEUE_WAIT_METRIC = "rowan_gpu_queue_wait_seconds"
QUEUE_HOME_METRIC = "rowan_gpu_queue_running_by_home"
VRAM_METRIC = "rowan_gpu_vram_bytes"
BUDGET_METRIC = "rowan_api_budget_usd"
BUDGET_PENDING_METRIC = "rowan_api_unsettled_requests"
OUTBOUND_METRIC = "rowan_outbound_dropped_total"
SCHEDULER_METRIC = "rowan_scheduler_failures_total"

_HELP = {
    STAGE_METRIC: "Stage latency of a completed turn, milliseconds",
    TURNS_METRIC: "Completed turns, by home",
    TURN_ERRORS_METRIC: "Turns that produced no reply, by home",
    DEGRADED_METRIC: "Turns that skipped a stage, by home and stage",
    LEVEL_METRIC: "Turns by answering model level and home",
    QUEUE_METRIC: "GPU queue jobs right now",
    QUEUE_DROPPED_METRIC: "GPU jobs refused because the queue was full",
    QUEUE_WAIT_METRIC: "Estimated wait of one GPU job of a class, seconds",
    QUEUE_HOME_METRIC: "GPU jobs running for a home right now",
    VRAM_METRIC: "GPU memory in bytes as the CUDA driver reports it",
    BUDGET_METRIC: "Cloud API accounting in US dollars, this month",
    BUDGET_PENDING_METRIC: "Cloud requests that are still not settled",
    OUTBOUND_METRIC: "Frames dropped because a client was too slow, by kind",
    SCHEDULER_METRIC: "Failed passes of a periodic hub job, by job",
}

#: Every family the hub exports. The Grafana dashboard is checked against this
#: set (``tests/test_grafana_dashboard.py``), so a renamed metric cannot leave a
#: panel that silently shows nothing.
METRIC_NAMES = frozenset(_HELP)


def _escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _escape_help(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("\n", " ")


def _labels(pairs: Sequence[tuple[str, Any]]) -> str:
    if not pairs:
        return ""
    return "{" + ",".join(f'{name}="{_escape_label(value)}"' for name, value in pairs) + "}"


def _number(value: Any) -> str:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return "+Inf" if number > 0 else ("-Inf" if number < 0 else "NaN")
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return repr(number)


@dataclass(frozen=True)
class Sample:
    """One metric family member: name, kind, labels and value."""

    name: str
    kind: str
    value: float
    labels: tuple[tuple[str, Any], ...] = ()
    #: Histograms only: cumulative counts per bucket, in the given order.
    buckets: tuple[tuple[float, int], ...] = ()
    #: Histograms only: how many observations there were.
    count: int = 0

    def lines(self) -> list[str]:
        if self.kind != "histogram":
            return [f"{self.name}{_labels(self.labels)} {_number(self.value)}"]
        lines: list[str] = []
        for bound, seen in self.buckets:
            edge = "+Inf" if bound == float("inf") else _number(bound)
            lines.append(f"{self.name}_bucket{_labels((*self.labels, ('le', edge)))} {seen}")
        lines.append(f"{self.name}_sum{_labels(self.labels)} {_number(self.value)}")
        lines.append(f"{self.name}_count{_labels(self.labels)} {int(self.count)}")
        return lines


def render(samples: Iterable[Sample]) -> str:
    """The Prometheus text exposition: HELP/TYPE once per family, then values."""
    lines: list[str] = []
    declared: set[str] = set()
    for sample in samples:
        if sample.name not in declared:
            declared.add(sample.name)
            lines.append(f"# HELP {sample.name} {_escape_help(_HELP.get(sample.name, sample.name))}")
            lines.append(f"# TYPE {sample.name} {sample.kind}")
        lines.extend(sample.lines())
    return "\n".join(lines) + "\n" if lines else ""


class MetricsRegistry:
    """Counters and stage histograms of the turns the hub processed.

    One registry for the whole process (like ``UtteranceMetrics``), filled from
    the event loop: every finished turn adds its stage durations, and a turn
    that was answered without a stage (a degradation) is counted as such. The
    lock is there because ``/metrics`` may be scraped from another thread.
    """

    def __init__(self, *, buckets: Sequence[float] = DEFAULT_BUCKETS_MS) -> None:
        self._lock = threading.Lock()
        self._buckets = tuple(sorted(float(bound) for bound in buckets))
        self._stage_counts: dict[tuple[str, str], list[int]] = {}
        self._stage_sum: dict[tuple[str, str], float] = {}
        self._stage_total: dict[tuple[str, str], int] = {}
        self._turns: dict[str, int] = {}
        #: ``(home, kind)`` → count; a turn that produced no reply is ``kind="turn"``.
        self._errors: dict[tuple[str, str], int] = {}
        self._degraded: dict[tuple[str, str], int] = {}
        self._levels: dict[tuple[str, str], int] = {}

    # -- writing ------------------------------------------------------------

    def observe_turn(self, *, home_id: str = "", stages_ms: Mapping[str, Any] | None = None,
                     ok: bool = True, degraded: Iterable[str] = (), level: str = "") -> None:
        """Record one finished turn: its stages, its outcome and its model level."""
        home = str(home_id or "")
        with self._lock:
            self._turns[home] = self._turns.get(home, 0) + 1
            if not ok:
                key = (home, "turn")
                self._errors[key] = self._errors.get(key, 0) + 1
            for stage, value in (stages_ms or {}).items():
                key = (home, str(stage))
                counts = self._stage_counts.get(key)
                if counts is None:
                    counts = self._stage_counts[key] = [0] * (len(self._buckets) + 1)
                try:
                    milliseconds = max(0.0, float(value))
                except (TypeError, ValueError):
                    continue
                counts[bisect.bisect_left(self._buckets, milliseconds)] += 1
                self._stage_sum[key] = self._stage_sum.get(key, 0.0) + milliseconds
                self._stage_total[key] = self._stage_total.get(key, 0) + 1
            for stage in degraded or ():
                key = (home, str(stage))
                self._degraded[key] = self._degraded.get(key, 0) + 1
            if level:
                key = (home, str(level))
                self._levels[key] = self._levels.get(key, 0) + 1

    def note_error(self, *, kind: str, home_id: str = "") -> None:
        """Count one error that is not a turn (a refused upload, a broken task)."""
        key = (str(home_id or ""), str(kind))
        with self._lock:
            self._errors[key] = self._errors.get(key, 0) + 1

    # -- reading ------------------------------------------------------------

    def samples(self) -> list[Sample]:
        """The turn metrics, ready for :func:`render`."""
        with self._lock:
            stages = sorted(self._stage_counts)
            turns = sorted(self._turns)
            errors = sorted(self._errors)
            degraded = sorted(self._degraded)
            levels = sorted(self._levels)
            stage_counts = {key: list(self._stage_counts[key]) for key in stages}
            stage_sum = dict(self._stage_sum)
            stage_total = dict(self._stage_total)
            turn_values = dict(self._turns)
            error_values = dict(self._errors)
            degraded_values = dict(self._degraded)
            level_values = dict(self._levels)
        samples: list[Sample] = []
        for key in stages:
            home, stage = key
            counts = stage_counts[key]
            cumulative: list[tuple[float, int]] = []
            running = 0
            for index, bound in enumerate(self._buckets):
                running += counts[index]
                cumulative.append((bound, running))
            running += counts[len(self._buckets)]
            cumulative.append((float("inf"), running))
            samples.append(Sample(STAGE_METRIC, "histogram", stage_sum.get(key, 0.0),
                                  (("home", home), ("stage", stage)), tuple(cumulative),
                                  stage_total.get(key, 0)))
        samples.extend(Sample(TURNS_METRIC, "counter", turn_values[key], (("home", key),))
                       for key in turns)
        samples.extend(Sample(TURN_ERRORS_METRIC, "counter", error_values[key],
                              (("home", key[0]), ("kind", key[1]))) for key in errors)
        samples.extend(Sample(DEGRADED_METRIC, "counter", degraded_values[key],
                              (("home", key[0]), ("stage", key[1]))) for key in degraded)
        samples.extend(Sample(LEVEL_METRIC, "counter", level_values[key],
                              (("home", key[0]), ("level", key[1]))) for key in levels)
        return samples


def queue_samples(stats: Mapping[str, Any] | None) -> list[Sample]:
    """GPU queue length, refusals, per-class wait and per-home occupancy."""
    if not stats:
        return []
    samples = [
        Sample(QUEUE_METRIC, "gauge", float(stats.get("waiting", 0)), (("state", "waiting"),)),
        Sample(QUEUE_METRIC, "gauge", float(stats.get("running", 0)), (("state", "running"),)),
        Sample(QUEUE_DROPPED_METRIC, "counter", float(stats.get("dropped", 0))),
    ]
    for name, seconds in sorted((stats.get("estimated_wait_s") or {}).items()):
        samples.append(Sample(QUEUE_WAIT_METRIC, "gauge", float(seconds), (("class", str(name)),)))
    for home, running in sorted((stats.get("running_by_home") or {}).items()):
        samples.append(Sample(QUEUE_HOME_METRIC, "gauge", float(running), (("home", str(home)),)))
    return samples


def vram_samples(torch_module: Any = None) -> list[Sample]:
    """What the CUDA driver says about video memory, if it can be asked.

    ``torch`` is NOT imported here: if the hub has not loaded it, there is no
    number to report, and importing a GPU stack just to scrape metrics would be
    a heavy side effect of a read-only endpoint.
    """
    module = torch_module if torch_module is not None else sys.modules.get("torch")
    if module is None:
        return []
    try:
        if not module.cuda.is_available():
            return []
        free, total = module.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001 - a broken GPU stack is not a metric
        log.debug("Could not read the VRAM numbers (%s)", exc)
        return []
    free, total = int(free), int(total)
    used = max(0, total - free)
    return [Sample(VRAM_METRIC, "gauge", value, (("kind", kind),))
            for kind, value in (("used", used), ("free", free), ("total", total))]


def budget_samples(status: Mapping[str, Any] | None) -> list[Sample]:
    """What the cloud budget ledger says about this month."""
    if not status:
        return []
    pairs = (("accounted", status.get("accounted_usd", 0.0)),
             ("settled", status.get("settled_estimate_usd", 0.0)),
             ("reserved", status.get("reserved_usd", 0.0)))
    samples = [Sample(BUDGET_METRIC, "gauge", float(value), (("kind", kind),))
               for kind, value in pairs]
    # A missing limit means the owner removed the ceiling (DECISIONS.md
    # API-01); publishing 0.00 would read as "budget nearly used up".
    limit = status.get("limit_usd")
    if isinstance(limit, (int, float)):
        samples.append(Sample(BUDGET_METRIC, "gauge", float(limit), (("kind", "limit"),)))
    samples.append(Sample(BUDGET_PENDING_METRIC, "gauge",
                          float(status.get("unsettled_requests", 0))))
    return samples


def outbound_samples(stats: Mapping[str, Any] | None) -> list[Sample]:
    """Frames the hub dropped because a client was not reading fast enough."""
    if not stats:
        return []
    by_kind = stats.get("dropped_by_type") or {}
    if by_kind:
        return [Sample(OUTBOUND_METRIC, "counter", float(count), (("kind", str(kind)),))
                for kind, count in sorted(by_kind.items())]
    return [Sample(OUTBOUND_METRIC, "counter", float(stats.get("dropped", 0)))]


def scheduler_samples(jobs: Iterable[Mapping[str, Any]] | None) -> list[Sample]:
    """Failed passes of the periodic jobs (ТЗ F-304), one label per job."""
    return [Sample(SCHEDULER_METRIC, "counter", float(job.get("failures", 0)),
                   (("job", str(job.get("name", ""))),))
            for job in (jobs or ())]


__all__ = [
    "DEFAULT_BUCKETS_MS",
    "METRIC_NAMES",
    "MetricsRegistry",
    "Sample",
    "budget_samples",
    "outbound_samples",
    "queue_samples",
    "render",
    "scheduler_samples",
    "vram_samples",
]
