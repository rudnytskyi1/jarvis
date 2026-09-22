"""Игры между комнатами (ТЗ F-608) на каркасе скилла с состоянием (F-407).

Скилл — это вход в игру, а не сама игра: партия живёт в ``skill_state``
(``ctx.state``), таймер ответа — в ``ctx.scheduler`` (F-407), вопросы пишет
модель (``ctx.quiz_generator``). Скилл возвращает хабу строку для комнаты и
данные партии; говорит и слушает хаб, потому что голос — не дело скилла.

Никаких выдуманных вопросов: нет модели или её ответ непригоден — скилл честно
отказывает (``ok=False``) и называет причину, которую хаб произносит вслух.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hub import games as games_mod
from hub.skills_runtime import SkillResult


class Args(BaseModel):
    """Что человек просит у игры."""

    model_config = ConfigDict(extra="forbid")

    what: Literal["start", "answer", "score", "stop"] = "start"
    #: Для ``start``: тема квиза словами человека.
    topic: str = Field(default="", max_length=80)
    #: Для ``answer``: то, что человек сказал вслух.
    answer: str = Field(default="", max_length=200)
    #: Комнаты-участники; пусто — только комната, из которой позвали.
    homes: list[str] = Field(default_factory=list)


def _language(ctx) -> str:
    return str(getattr(ctx, "language", "") or "ru")


async def run(ctx, args):  # noqa: ANN001, ANN201 - контракт скилла (F-405)
    if not isinstance(args, Args):
        args = Args.model_validate(args or {})
    try:
        engine = games_mod.engine_from_context(ctx)
    except games_mod.QuizError as exc:
        return SkillResult(ok=False, error=str(exc),
                           spoken=games_mod.unavailable_line(str(exc), language=_language(ctx)))
    language = _language(ctx)
    home = str(getattr(ctx, "home_id", "") or "")
    person = str(getattr(ctx, "person_id", "") or "")

    if args.what == "start":
        if not args.topic:
            # Темы нет — хаб спрашивает, а не выбирает за человека.
            return SkillResult(ok=False, error="no topic was named",
                               spoken=games_mod.topic_missing_line(language=language))
        homes = args.homes or [home]
        try:
            round_, line = await engine.start(topic=args.topic, language=language,
                                              home_id=home, person_id=person, homes=homes)
        except games_mod.QuizError as exc:
            return SkillResult(ok=False, error=str(exc), spoken=str(exc))
        except games_mod.QuizUnavailable as exc:
            return SkillResult(ok=False, error=str(exc),
                               spoken=games_mod.unavailable_line(str(exc), language=language))
        return SkillResult(ok=True, spoken=line,
                           data={"round_id": round_.round_id, "topic": round_.topic,
                                 "questions": len(round_.questions),
                                 "scores": dict(round_.scores)})

    if args.what == "answer":
        verdict = engine.answer(home_id=home, text=args.answer, language=language)
        if verdict is None:
            return SkillResult(ok=False, error="no question is open",
                               spoken=games_mod.no_round_line(language=language))
        said = " ".join(part for part in (verdict.line, verdict.next_line) if part)
        return SkillResult(ok=True, spoken=said,
                           data={"correct": verdict.correct, "late": verdict.late,
                                 "expected": verdict.expected, "scores": verdict.scores,
                                 "finished": verdict.finished})

    if args.what == "score":
        line = engine.score(language=language)
        if line is None:
            return SkillResult(ok=False, error="no game is running",
                               spoken=games_mod.no_round_line(language=language))
        return SkillResult(ok=True, spoken=line, data={"scores": engine.snapshot()["scores"]})

    line = engine.finish(language=language, reason="stop")
    if line is None:
        return SkillResult(ok=False, error="no game is running",
                           spoken=games_mod.no_round_line(language=language))
    return SkillResult(ok=True, spoken=line, data={"finished": True})


__all__ = ["Args", "run"]
