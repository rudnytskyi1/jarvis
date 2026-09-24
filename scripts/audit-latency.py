"""Латентность хода Rowan по реальным источникам (ТЗ 15.1, F-101; AU-11).

Три источника, потому что ни один из них не видит ход целиком:

* **живой хаб** — `turn_events` в `data/hub.db`: настоящие ходы комнаты с
  разбивкой `stt`/`llm`/`tts`/`total`, которую пишет сам `hub/app.py`;
* **первый звук** — строки `First audio N ms after the end of speech` из
  `data/server.log`: это и есть замер F-101 «конец речи → первый звук»;
* **стенд** — отчёты `scripts/live-eval.py` (`.jsonl`): те же реплики через
  настоящий вызов модели, с `understand_ms` (чтение Jev), `rounds` (раунды
  модели) и `model_ms`.

Скрипт ничего не выдумывает: чего в источнике нет, то в отчёте стоит как
`—`, а не как ноль.

Примеры:

    python scripts/audit-latency.py
    python scripts/audit-latency.py --runs data/audit/runs/au-11-before.jsonl \\
        --runs data/audit/runs/au-11-after.jsonl --write docs/AUDIT_LATENCY.md
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Бюджеты ТЗ 15.1, о которых отчитывается скрипт.
FIRST_AUDIO_BUDGET_MS = 1200
FIRST_AUDIO_SLOW_MS = 2500
TURN_SLOW_S = 4.0

FIRST_AUDIO_RE = re.compile(r"First audio (\d+) ms after the end of speech")


def percentile(values: Sequence[float], fraction: float) -> float:
    """Ближайший ранг: без numpy и без интерполяции, чтобы числа совпадали."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


def share(values: Sequence[float], *, at_least: float) -> float:
    """Доля значений не меньше порога (пусто — 0.0, а не деление на ноль)."""
    if not values:
        return 0.0
    return sum(1 for value in values if value >= at_least) / float(len(values))


@dataclass
class Distribution:
    """Одно измерение: сколько ходов и как они распределены."""

    name: str
    values: list[float] = field(default_factory=list)
    unit: str = "мс"

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def median(self) -> float:
        return percentile(self.values, 0.5)

    @property
    def p95(self) -> float:
        return percentile(self.values, 0.95)

    @property
    def worst(self) -> float:
        return max(self.values) if self.values else 0.0

    def row(self) -> str:
        if not self.count:
            return f"| {self.name} | 0 | — | — | — |"
        return (f"| {self.name} | {self.count} | {self.median:.0f} | "
                f"{self.p95:.0f} | {self.worst:.0f} |")


@dataclass
class BenchRun:
    """Один отчёт стенда: реплики, прошедшие настоящий вызов модели."""

    path: Path
    turns: int = 0
    failed: int = 0
    skipped: int = 0
    seconds: list[float] = field(default_factory=list)
    #: ``seconds`` ПЛЮС чтение Jev: то, что человек ждёт от реплики до ответа,
    #: без STT и TTS (их стенд не делает). Есть только у отчётов после AU-11.
    combined: list[float] = field(default_factory=list)
    understand_ms: list[float] = field(default_factory=list)
    model_ms: list[float] = field(default_factory=list)
    rounds: dict[int, int] = field(default_factory=dict)
    first_call_ms: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.path.name

    def lines(self) -> list[str]:
        slow = share(self.seconds, at_least=TURN_SLOW_S)
        out = [f"**{self.label}** — ходов {self.turns}, "
               f"падений {self.failed}, медленнее {TURN_SLOW_S:.0f} с: "
               f"{slow * 100:.1f} % ({sum(1 for s in self.seconds if s >= TURN_SLOW_S)}"
               f"/{self.turns})"]
        if self.seconds:
            out.append(f"  модель+инструменты (без STT, Jev и TTS): медиана "
                       f"{percentile(self.seconds, 0.5):.2f} с, p95 "
                       f"{percentile(self.seconds, 0.95):.2f} с, максимум "
                       f"{max(self.seconds):.2f} с")
        if self.combined:
            slow = share(self.combined, at_least=TURN_SLOW_S)
            over = sum(1 for value in self.combined if value >= TURN_SLOW_S)
            out.append(f"  Jev+модель+инструменты (без STT и TTS): медиана "
                       f"{percentile(self.combined, 0.5):.2f} с, p95 "
                       f"{percentile(self.combined, 0.95):.2f} с, медленнее "
                       f"{TURN_SLOW_S:.0f} с: {slow * 100:.1f} % ({over}/{len(self.combined)})")
        if self.understand_ms:
            out.append(f"  чтение Jev: медиана {percentile(self.understand_ms, 0.5):.0f} мс, "
                       f"p95 {percentile(self.understand_ms, 0.95):.0f} мс")
        if self.model_ms:
            out.append(f"  модель: медиана {percentile(self.model_ms, 0.5):.0f} мс, "
                       f"p95 {percentile(self.model_ms, 0.95):.0f} мс")
        if self.rounds:
            total = sum(self.rounds.values())
            many = sum(count for round_, count in self.rounds.items() if round_ > 1)
            histogram = ", ".join(f"{round_}→{count}"
                                  for round_, count in sorted(self.rounds.items()))
            out.append(f"  раунды модели: {histogram} (больше одного: "
                       f"{many}/{total} = {many / total * 100:.1f} %)")
        if self.first_call_ms:
            out.append(f"  первый вызов инструмента после старта модели: медиана "
                       f"{percentile(self.first_call_ms, 0.5):.0f} мс")
        return out


def read_bench_runs(paths: Iterable[Path]) -> list[BenchRun]:
    """Прочитать отчёты стенда; старые файлы без стадий тоже считаются."""
    runs: list[BenchRun] = []
    for path in paths:
        run = BenchRun(path=path)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or "seconds" not in record:
                continue
            run.turns += 1
            if record.get("ok") is False:
                run.failed += 1
            run.seconds.append(float(record["seconds"]))
            if isinstance(record.get("understand_ms"), int):
                run.understand_ms.append(float(record["understand_ms"]))
                run.combined.append(float(record["seconds"]) + record["understand_ms"] / 1000.0)
            if isinstance(record.get("model_ms"), int):
                run.model_ms.append(float(record["model_ms"]))
            rounds = record.get("rounds")
            if isinstance(rounds, int) and rounds > 0:
                run.rounds[rounds] = run.rounds.get(rounds, 0) + 1
            first = record.get("first_call_ms")
            if isinstance(first, int):
                run.first_call_ms.append(float(first))
        runs.append(run)
    return runs


@dataclass
class HubTurn:
    """Один настоящий ход комнаты, разобранный по событиям трассы."""

    turn_id: str
    stt_ms: int = 0
    llm_ms: int = 0
    tts_ms: int = 0
    total_ms: int = 0
    jev_ms: int = 0
    rounds: int = 0
    model_ms: int = 0
    tool_ms: int = 0
    degraded: bool = False

    @property
    def reached_model(self) -> bool:
        """Был ли у хода настоящий раунд модели (а не локальный ответ)."""
        return self.model_ms > 0


@dataclass
class HubTurns:
    """Ходы живого хаба: стадии из `turn_events`."""

    turns: list[HubTurn] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.turns)

    def stage(self, name: str) -> Distribution:
        return Distribution(name, [float(getattr(turn, name)) for turn in self.turns])

    def with_model(self) -> list[HubTurn]:
        return [turn for turn in self.turns if turn.reached_model]


def read_hub_turns(db: Path) -> HubTurns:
    """Разбивка стадий, которую хаб записал сам (только ходы комнаты)."""
    turns = HubTurns()
    if not db.exists():
        return turns
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        events = conn.execute(
            "SELECT turn_id, ts, kind, name, latency_ms, payload_json"
            " FROM turn_events ORDER BY event_id").fetchall()
    finally:
        conn.close()
    chains: dict[str, list[tuple[float, str, str, int, str]]] = {}
    for turn_id, ts, kind, name, latency, payload in events:
        chains.setdefault(str(turn_id), []).append(
            (float(ts), str(kind), str(name), int(latency or 0), str(payload or "{}")))
    for turn_id, chain in chains.items():
        if turn_id.startswith("telegram:"):
            continue  # ход Telegram — не голосовая реплика комнаты
        finished = [item for item in chain if item[1] == "turn" and item[2] == "finished"]
        if not finished:
            continue
        try:
            data = json.loads(finished[-1][4])
        except ValueError:
            continue
        stages = data.get("durations_ms") or {}
        if not isinstance(stages, dict) or "durations_ms" not in data:
            continue
        turn = HubTurn(
            turn_id=turn_id,
            stt_ms=int(stages.get("stt", 0) or 0),
            llm_ms=int(stages.get("llm", 0) or 0),
            tts_ms=int(stages.get("tts", 0) or 0),
            total_ms=int(stages.get("total", 0) or 0),
            degraded=bool(data.get("degraded")),
        )
        # Внутри стадии ``llm``: чтение Jev, раунды модели и ожидание
        # инструментов. Раунд меряется от события ``prompt`` до ответа
        # ``llm`` следом за ним; событие понимания несёт собственный
        # ``latency_ms`` (AU-11).
        pending_prompt: float | None = None
        for ts, kind, _name, latency, _payload in chain:
            if kind == "understanding":
                turn.jev_ms += latency
            elif kind == "prompt":
                pending_prompt = ts
            elif kind == "llm":
                turn.rounds += 1
                if pending_prompt is not None:
                    turn.model_ms += int(round((ts - pending_prompt) * 1000))
                pending_prompt = None
            elif kind == "tool":
                turn.tool_ms += latency
        turns.turns.append(turn)
    return turns


def read_first_audio(log: Path) -> list[float]:
    """Строки F-101 из лога хаба: «конец речи → первый звук» в миллисекундах."""
    delays: list[float] = []
    if not log.exists():
        return delays
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        found = FIRST_AUDIO_RE.search(line)
        if found:
            delays.append(float(found.group(1)))
    return delays


def find_bench_runs(directory: Path) -> list[Path]:
    """Все отчёты стенда в каталоге, от старых к новым."""
    return sorted(directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)


def report(runs: Sequence[BenchRun], turns: HubTurns, first_audio: Sequence[float],
           *, title: str = "Латентность хода Rowan") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Порог «долгого хода» — {TURN_SLOW_S:.0f} с, бюджет первого звука — "
                 f"{FIRST_AUDIO_BUDGET_MS} мс, «первый звук позже 2,5 с» — "
                 f"{FIRST_AUDIO_SLOW_MS} мс (ТЗ 15.1, F-101).")
    lines.append("")
    lines.append("## Живой хаб: стадии хода комнаты (`turn_events`)")
    lines.append("")
    if turns.count:
        with_model = turns.with_model()
        degraded = sum(1 for turn in turns.turns if turn.degraded)
        lines.append(f"Ходов с разбивкой стадий: **{turns.count}**, из них настоящий раунд "
                     f"модели был у {len(with_model)} (остальные закрылись локальным путём "
                     f"или до модели), с деградацией стадии: {degraded}.")
        lines.append("")
        lines.append("| стадия | ходов | медиана, мс | p95, мс | максимум, мс |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(Distribution(
            "STT (финальный проход)",
            [float(turn.stt_ms) for turn in turns.turns]).row())
        lines.append(Distribution(
            "LLM = Jev + раунды модели + инструменты",
            [float(turn.llm_ms) for turn in with_model]).row())
        lines.append(Distribution(
            "  в нём чтение Jev",
            [float(turn.jev_ms) for turn in with_model]).row())
        lines.append(Distribution(
            "  в нём раунды модели",
            [float(turn.model_ms) for turn in with_model]).row())
        lines.append(Distribution(
            "  в нём ожидание инструментов",
            [float(turn.tool_ms) for turn in with_model]).row())
        lines.append(Distribution(
            "TTS (синтез ответа)", [float(turn.tts_ms) for turn in with_model]).row())
        lines.append(Distribution(
            "весь ход", [float(turn.total_ms) for turn in with_model]).row())
        rounds = [float(turn.rounds) for turn in with_model]
        if rounds:
            many = sum(1 for value in rounds if value > 1)
            lines.append("")
            lines.append(f"Раунды модели: медиана {percentile(rounds, 0.5):.0f}, "
                         f"больше одного — {many}/{len(rounds)} = "
                         f"{many / len(rounds) * 100:.1f} %.")
        if with_model and not any(turn.jev_ms for turn in with_model):
            lines.append("")
            lines.append("Чтение Jev в этих ходах не записано: событие "
                         "`understanding` трассы появилось вместе с AU-11, поэтому "
                         "нулевая строка выше — «нет данных», а не «мгновенно».")
    else:
        lines.append("Ходов комнаты с разбивкой стадий в базе нет.")
    lines.append("")
    lines.append("## Первый звук: конец речи → первый звук (F-101, лог хаба)")
    lines.append("")
    if first_audio:
        slow = share(first_audio, at_least=FIRST_AUDIO_SLOW_MS)
        over = share(first_audio, at_least=FIRST_AUDIO_BUDGET_MS)
        lines.append(f"Замеров: **{len(first_audio)}**; медиана "
                     f"{percentile(first_audio, 0.5):.0f} мс, p95 "
                     f"{percentile(first_audio, 0.95):.0f} мс, максимум "
                     f"{max(first_audio):.0f} мс.")
        lines.append("")
        lines.append(f"- позже {FIRST_AUDIO_BUDGET_MS} мс (бюджет ТЗ 15.1): "
                     f"{sum(1 for value in first_audio if value >= FIRST_AUDIO_BUDGET_MS)}"
                     f"/{len(first_audio)} = {over * 100:.1f} %")
        lines.append(f"- позже {FIRST_AUDIO_SLOW_MS} мс: "
                     f"{sum(1 for value in first_audio if value >= FIRST_AUDIO_SLOW_MS)}"
                     f"/{len(first_audio)} = {slow * 100:.1f} %")
    else:
        lines.append("Замеров первого звука в логе нет.")
    lines.append("")
    lines.append("## Стенд `scripts/live-eval.py`: ход без STT и TTS")
    lines.append("")
    for run in runs:
        lines.extend(run.lines())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", action="append", default=[],
                        help="файл отчёта стенда (.jsonl); можно несколько раз")
    parser.add_argument("--runs-dir", default="data/audit/runs",
                        help="каталог отчётов стенда, если --runs не задан")
    parser.add_argument("--db", default="data/hub.db", help="база живого хаба")
    parser.add_argument("--log", default="data/server.log", help="лог живого хаба")
    parser.add_argument("--write", default="", help="куда записать отчёт (markdown)")
    parser.add_argument("--title", default="Латентность хода Rowan")
    args = parser.parse_args(argv)

    if args.runs:
        paths = [Path(item) if Path(item).is_absolute() else REPO_ROOT / item
                 for item in args.runs]
    else:
        paths = find_bench_runs(REPO_ROOT / args.runs_dir)
    runs = read_bench_runs(paths)
    turns = read_hub_turns(REPO_ROOT / args.db)
    first_audio = read_first_audio(REPO_ROOT / args.log)
    text = report(runs, turns, first_audio, title=args.title)
    print(text)
    if args.write:
        target = Path(args.write)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.write_text(text, encoding="utf-8")
        print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
