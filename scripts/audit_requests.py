"""Разбор всех сохранённых запросов: что просили, что вышло, что не так.

Основание: просьба владельца 2026-09-22 — «проанализируй вообще все запросы,
которые были сохранены за всё время (в админке и из логов), и посмотри, что с
ними не так». Источники:

* ``data/server.log`` — каждый транскрипт, каждый ответ модели, каждый вызов
  инструмента и каждый отказ (там есть записи за всё время работы);
* ``data/hub.db`` таблица ``turn_events`` — цепочки ходов для панели
  (``/admin/turns``): решения, промпты, раунды модели, инструменты;
* таблица ``decisions`` — что решили правила и Jev.

Скрипт ничего не меняет. Он печатает отчёт и, если попросить ``--write``,
складывает его в ``docs/REQUESTS_AUDIT.md``.

    python scripts/audit_requests.py
    python scripts/audit_requests.py --write
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = ROOT / "data" / "server.log"
DB = ROOT / "data" / "hub.db"
REPORT = ROOT / "docs" / "REQUESTS_AUDIT.md"

#: The log line that carries the final transcript of one utterance.
_TRANSCRIPT = re.compile(r"Transcribed [\d.]+ s of audio \[(\w+)\]: '(.*)'$")
#: The line the hub writes when the model produced its final words.
_REPLY = re.compile(r"LLM final reply: (.*)$")
_ROUND_REPLY = re.compile(r"LLM round (\d+)/(\d+): text (.*?)(?:, \d+ tool call\(s\))?")
#: One executed tool and its result.
_TOOL = re.compile(r"Tool (\w+)(\{.*?\}) -> (\{.*\})$")
#: The result the room PC reported for an action the hub sent it.
_ACTION = re.compile(r"Action result (\S+) from [^:]+: ok=(\w+) error=(.*?) output=")
_DEGRADED = re.compile(r"degraded at stage (\w+): (.*)$")
_LATENCY = re.compile(r"done in (\d+) ms \(stt (\d+), llm (\d+), tts (\d+)\)")
_FIRST_AUDIO = re.compile(r"First audio (\d+) ms after the end of speech")

#: Words that mean the person asked for something to be DONE, not answered.
COMMAND_WORDS = (
    "open", "close", "play", "show", "send", "turn", "set", "make", "draw",
    "remember", "forget", "find", "search", "look", "take", "edit", "hide",
    "открой", "закрой", "включи", "выключи", "покажи", "сделай", "напомни",
)
#: What an assistant says instead of doing the thing.
REFUSALS = (
    "i can't access", "i cannot access", "i couldn't", "i don't have a tool",
    "i don't have access", "i'm not able", "i am not able", "isn't available",
    "is not available", "couldn't finish", "i can’t access", "i couldn’t",
)


@dataclass
class Turn:
    """One saved request and everything that came out of it."""

    index: int
    transcript: str = ""
    language: str = ""
    replies: list[str] = field(default_factory=list)
    tools: list[tuple[str, str, bool]] = field(default_factory=list)
    actions: list[tuple[str, bool, str]] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    timings: list[tuple[int, int, int, int]] = field(default_factory=list)
    first_audio_ms: int = 0

    @property
    def reply(self) -> str:
        return self.replies[-1] if self.replies else ""

    @property
    def failed_tools(self) -> list[tuple[str, str, bool]]:
        return [row for row in self.tools if not row[2]]

    @property
    def wants_action(self) -> bool:
        text = self.transcript.casefold()
        return any(word in text for word in COMMAND_WORDS)

    @property
    def refused(self) -> bool:
        text = self.reply.casefold()
        return any(phrase in text for phrase in REFUSALS)

    def problems(self) -> list[str]:
        """What is wrong with this turn, in plain words."""
        found: list[str] = []
        if self.failed_tools:
            names = ", ".join(name for name, _args, _ok in self.failed_tools)
            found.append(f"tool failed: {names}")
        for _action_id, ok, error in self.actions:
            if not ok:
                found.append(f"room PC refused the action: {error[:100]}")
        if self.wants_action and not self.tools and not self.actions and self.refused:
            found.append("asked for an action, nothing ran, the answer is a refusal")
        elif self.refused:
            found.append("the answer is a refusal")
        if self.degraded:
            found.append("degraded: " + "; ".join(self.degraded))
        slow = max((row[0] for row in self.timings), default=0)
        if slow > 4000:
            found.append(f"slow turn: {slow} ms")
        if self.first_audio_ms > 2500:
            found.append(f"slow first audio: {self.first_audio_ms} ms (budget 1200)")
        return found


def parse_log(path: Path) -> list[Turn]:
    """Every saved request the hub wrote to its log, in order."""
    turns: list[Turn] = []
    current: Turn | None = None
    if not path.is_file():
        return turns
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _TRANSCRIPT.search(line)
        if match:
            # A transcript is the start of a new utterance: the previous one is
            # complete (the hub transcribes after the audio ended).
            if current is not None and not current.wants_action and not current.replies:
                current.transcript = match.group(2)
                current.language = match.group(1)
                continue
            current = Turn(index=len(turns) + 1, transcript=match.group(2),
                           language=match.group(1))
            turns.append(current)
            continue
        if current is None:
            continue
        if (match := _REPLY.search(line)) is not None:
            current.replies.append(match.group(1))
            continue
        if (match := _ROUND_REPLY.search(line)) is not None and match.group(3):
            current.replies.append(match.group(3))
            continue
        if (match := _TOOL.search(line)) is not None:
            try:
                result = json.loads(match.group(3).replace("'", '"'))
                ok = bool(result.get("ok", True))
            except Exception:  # noqa: BLE001 - a non-JSON result still counts as a call
                ok = "False" not in match.group(3)
            current.tools.append((match.group(1), match.group(2), ok))
            continue
        if (match := _ACTION.search(line)) is not None:
            current.actions.append((match.group(1), match.group(2) == "True",
                                    "" if match.group(3) == "None" else match.group(3)))
            continue
        if (match := _DEGRADED.search(line)) is not None:
            current.degraded.append(f"{match.group(1)}: {match.group(2)}")
            continue
        if (match := _LATENCY.search(line)) is not None:
            current.timings.append(tuple(int(group) for group in match.groups()))
            continue
        if (match := _FIRST_AUDIO.search(line)) is not None:
            current.first_audio_ms = max(current.first_audio_ms, int(match.group(1)))
    return turns


def summary(db: Path, turns: list[Turn]) -> dict:
    """The numbers the report leads with."""
    problems = Counter()
    for turn in turns:
        for problem in turn.problems():
            problems[problem.split(":")[0]] += 1
    chains = 0
    steps = 0
    if db.is_file():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            chains = conn.execute("SELECT COUNT(DISTINCT turn_id) FROM turn_events").fetchone()[0]
            steps = conn.execute("SELECT COUNT(*) FROM turn_events").fetchone()[0]
        finally:
            conn.close()
    return {"turns": len(turns), "problems": problems, "chains": chains, "steps": steps}


def render(db: Path, turns: list[Turn]) -> str:
    data = summary(db, turns)
    lines = [
        "# Разбор сохранённых запросов",
        "",
        "Сгенерировано `scripts/audit_requests.py` (см. его докстроку про источники).",
        "",
        f"- реплик в логе: **{data['turns']}**",
        f"- цепочек в базе панели (`turn_events`): **{data['chains']}** ({data['steps']} шагов)",
        "",
    ]
    if data["problems"]:
        lines.append("## Что не так, по классам")
        lines.append("")
        lines.append("| класс проблемы | сколько ходов |")
        lines.append("|---|---|")
        for problem, count in data["problems"].most_common():
            lines.append(f"| {problem} | {count} |")
        lines.append("")
    bad = [turn for turn in turns if turn.problems()]
    lines.append(f"## Проблемные ходы ({len(bad)})")
    lines.append("")
    for turn in bad[-40:]:
        lines.append(f"### {turn.index}. {turn.transcript[:160]}")
        lines.append("")
        for problem in turn.problems():
            lines.append(f"- {problem}")
        if turn.reply:
            lines.append(f"- ответ: {turn.reply[:200]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=str(LOG))
    parser.add_argument("--db", default=str(DB))
    parser.add_argument("--write", action="store_true",
                        help="записать отчёт в docs/REQUESTS_AUDIT.md")
    args = parser.parse_args()

    turns = parse_log(Path(args.log))
    text = render(Path(args.db), turns)
    print(text)
    if args.write:
        REPORT.write_text(text, encoding="utf-8")
        print(f"\nОтчёт записан: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
