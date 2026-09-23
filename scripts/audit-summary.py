"""Сводка массового аудита: что прошло, что сломалось и по каким классам.

Читает JSONL-отчёт живого стенда (``scripts/live-eval.py --jsonl``, по строке
на сценарий) и собирает ``docs/AUDIT_MASS.md``: итоги по семействам, классы
поломок с примерами реплик и список реплик, где первый вызов был не тем.

    python scripts/audit-summary.py --runs data/audit/runs/last.jsonl --write docs/AUDIT_MASS.md
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Классы поломок: узнаваемый кусок текста проблемы -> короткое имя.
CLASSES: list[tuple[str, str]] = [
    ("the family narrowing never offered", "Jev-сужение скрыло нужный инструмент"),
    ("the family narrowing offered none", "Jev-сужение не оставило ни одного из вариантов"),
    ("the first tool called was", "первым вызван не тот инструмент"),
    ("the model never called", "инструмент не вызван вовсе"),
    ("when it should not", "вызван запрещённый инструмент"),
    ("was called without", "в аргументах нет нужного слова"),
    ("the reply never says", "ответ не содержит обещанного"),
    ("no model answered", "модель не ответила (ключ, сеть, бюджет)"),
    ("the turn failed", "ход упал с ошибкой"),
    ("the bench crashed", "стенд упал"),
    ("the hub turn failed", "ход хаба упал"),
    ("the hub answered nothing", "хаб ничего не ответил"),
    ("the hub stored", "хаб сохранил не то"),
]


def classify(problem: str) -> str:
    for marker, name in CLASSES:
        if marker in problem:
            return name
    return problem.split(":")[0][:60]


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="JSONL от live-eval")
    parser.add_argument("--write", default="", help="куда положить markdown-сводку")
    args = parser.parse_args()

    path = Path(args.runs)
    if not path.is_absolute():
        path = REPO_ROOT / path
    runs = load(path)
    if not runs:
        print("в отчёте нет ни одного сценария")
        return 2

    passed = sum(1 for run in runs if run.get("ok"))
    by_family: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    classes: dict[str, list[str]] = collections.defaultdict(list)
    for run in runs:
        family = str(run.get("family") or "?")
        by_family[family][1] += 1
        if run.get("ok"):
            by_family[family][0] += 1
            continue
        for problem in run.get("problems") or []:
            classes[classify(problem)].append(f"{run.get('id')}: {run.get('said')}")

    lines = [
        "# Массовый аудит запросов Rowan",
        "",
        f"Прогон: {time.strftime('%Y-%m-%d %H:%M', time.localtime())} · файл `{path.name}`",
        f"Всего сценариев: **{len(runs)}**, прошло **{passed}**, "
        f"не прошло **{len(runs) - passed}** "
        f"({round(100.0 * passed / len(runs), 1)} %).",
        "",
        "Каждый сценарий идёт через ту же цепочку, что живая комната: Jev читает",
        "реплику один раз (`hub.app.Connection._understand_turn`) и сужает набор",
        "инструментов, затем модель отвечает и зовёт инструменты. Вердикт пишется",
        "по строке на сценарий, поэтому отчёт можно перечитать после каждой правки.",
        "",
        "## Итоги по семействам",
        "",
        "| семейство | прошло | всего | доля |",
        "|---|---:|---:|---:|",
    ]
    for family, (ok, total) in sorted(by_family.items(), key=lambda pair: pair[1][0] / pair[1][1]):
        lines.append(f"| {family} | {ok} | {total} | {round(100.0 * ok / total, 1)} % |")

    lines += ["", "## Классы поломок", "",
              "| класс | сколько | пример реплики |", "|---|---:|---|"]
    for name, examples in sorted(classes.items(), key=lambda pair: -len(pair[1])):
        sample = examples[0].split(": ", 1)[-1][:90].replace("|", "/")
        lines.append(f"| {name} | {len(examples)} | {sample} |")

    lines += ["", "## Что починить в первую очередь", ""]
    for name, examples in sorted(classes.items(), key=lambda pair: -len(pair[1]))[:6]:
        lines.append(f"### {name} — {len(examples)}")
        lines.append("")
        for example in examples[:5]:
            lines.append(f"- {example}")
        lines.append("")

    text = "\n".join(lines) + "\n"
    if args.write:
        target = Path(args.write)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"сводка: {target}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
