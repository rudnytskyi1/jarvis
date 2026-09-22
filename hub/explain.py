"""Объяснимость идентичности: «почему ты решил, что это Макс?» (ТЗ F-215).

ТЗ требует одного: ответить по ``identity_belief.sources`` — то есть теми
числами, которые хаб действительно получил (F-206 их туда положил), и НИЧЕГО
не досочинять. Поэтому здесь нет ни модели, ни догадок: каждая часть ответа —
это либо поле belief, либо честное «этого я не видел».

Пример из ТЗ («голос 0,71, лицо не видно, одежда как утром») собирается из
одного и того же места: сигналы, которые были, называются с их числами,
сигналы, которых не было, называются как отсутствующие, а уверенность и
время решения берутся из строки belief. Если решения нет, ответ говорит и это:
«я этого не решал» честнее выдуманного обоснования, ведь спрашивают именно
чтобы проверить, не выдумал ли хаб.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

#: Порядок ТЗ: голос, лицо, тело («одежда как утром»).
KINDS: tuple[str, ...] = ("voice", "face", "body")

KIND_LABELS: dict[str, dict[str, str]] = {
    "voice": {"ru": "голос", "en": "voice", "es": "voz"},
    "face": {"ru": "лицо", "en": "face", "es": "cara"},
    "body": {"ru": "тело того же дня", "en": "body of the same day",
             "es": "cuerpo del mismo día"},
}

#: Что значит «этого сигнала не было» — именно так, как это звучит человеку.
MISSING: dict[str, dict[str, str]] = {
    "voice": {"ru": "голоса не слышно", "en": "no voice was heard",
              "es": "no se oyó la voz"},
    "face": {"ru": "лица не видно", "en": "the face was not visible",
             "es": "no se vio la cara"},
    "body": {"ru": "тела того же дня нет", "en": "no body of the same day",
             "es": "no había cuerpo del mismo día"},
}

SEEN: dict[str, str] = {
    "ru": "Вот на чём я это решил: {items}.",
    "en": "Here is what I went by: {items}.",
    "es": "Me basé en esto: {items}.",
}
MISSING_INTRO: dict[str, str] = {
    "ru": "Чего я не видел: {items}.",
    "en": "What I did not see: {items}.",
    "es": "Lo que no vi: {items}.",
}
CONFIDENCE: dict[str, str] = {
    "ru": "Общая уверенность {p}.",
    "en": "That is {p} of confidence.",
    "es": "La confianza total es {p}.",
}
AGE: dict[str, str] = {
    "ru": "Решение принято {age}.",
    "en": "That was decided {age}.",
    "es": "Eso se decidió {age}.",
}
AMBIGUOUS: dict[str, str] = {
    "ru": "Сигналы плохо различали этих людей.",
    "en": "The signals did not separate these people well.",
    "es": "Las señales no distinguían bien a estas personas.",
}
CONTEXT: dict[str, str] = {
    "ru": "Сигналы плохо различали этих людей, поэтому решил контекст дома.",
    "en": "The signals did not separate these people, so the house's context decided.",
    "es": "Las señales no los distinguían, así que decidió el contexto de la casa.",
}
REASON_INTRO: dict[str, str] = {
    "ru": "Причина: {reason}.",
    "en": "The reason: {reason}.",
    "es": "El motivo: {reason}.",
}
#: Причины, которые F-206/D-06 пишут в belief, по-человечески.
REASON_TEXT: dict[str, dict[str, str]] = {
    "ru": {"no signal past its threshold": "ни один сигнал не дотянул до своего порога",
           "no candidate": "кандидатов не было",
           "clear lead": "один человек был впереди с запасом",
           "context": "решил контекст дома",
           "ambiguous": "люди не различались"},
    "en": {"no signal past its threshold": "no signal passed its own threshold",
           "no candidate": "there were no candidates",
           "clear lead": "one person was clearly ahead",
           "context": "the context of the house decided",
           "ambiguous": "the people could not be told apart"},
    "es": {"no signal past its threshold": "ninguna señal pasó su umbral",
           "no candidate": "no había candidatos",
           "clear lead": "una persona iba claramente delante",
           "context": "decidió el contexto de la casa",
           "ambiguous": "no se podía distinguir a las personas"},
}
NO_PERSON: dict[str, str] = {
    "ru": "Я никого не назвал.",
    "en": "I did not name anybody.",
    "es": "No nombré a nadie.",
}
NOTHING: dict[str, str] = {
    "ru": "Этого я не решал: у меня нет решения по этому треку.",
    "en": "I did not decide that: I have no belief about this track.",
    "es": "Eso no lo decidí: no tengo ninguna creencia sobre esta pista.",
}
JUST_NOW: dict[str, str] = {
    "ru": "только что", "en": "just now", "es": "ahora mismo",
}
#: Числа с запятой там, где так пишут, и с точкой в английском.
DECIMAL_SEPARATOR: dict[str, str] = {"ru": ",", "en": ".", "es": ","}

_SECONDS: dict[str, tuple[str, str, str]] = {
    "ru": ("секунду", "секунды", "секунд"),
    "en": ("second", "seconds", "seconds"),
    "es": ("segundo", "segundos", "segundos"),
}
_MINUTES: dict[str, tuple[str, str, str]] = {
    "ru": ("минуту", "минуты", "минут"),
    "en": ("minute", "minutes", "minutes"),
    "es": ("minuto", "minutos", "minutos"),
}
_HOURS: dict[str, tuple[str, str, str]] = {
    "ru": ("час", "часа", "часов"),
    "en": ("hour", "hours", "hours"),
    "es": ("hora", "horas", "horas"),
}


def language_of(value: Any, *, default: str = "ru") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in SEEN else default


def _plural(count: int, forms: tuple[str, str, str], language: str) -> str:
    """The right form of «секунда/секунды/секунд» for ``count``."""
    if language == "ru":
        if count % 10 == 1 and count % 100 != 11:
            return forms[0]
        if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
            return forms[1]
        return forms[2]
    return forms[0] if count == 1 else forms[1]


def age_phrase(seconds: float, language: Any = "ru") -> str:
    """How long ago the belief was made, in words (ТЗ F-215)."""
    code = language_of(language)
    if seconds < 2.0:
        return JUST_NOW[code]
    if seconds < 60.0:
        count, forms = int(round(seconds)), _SECONDS[code]
    elif seconds < 3600.0:
        count, forms = int(round(seconds / 60.0)), _MINUTES[code]
    else:
        count, forms = int(round(seconds / 3600.0)), _HOURS[code]
    count = max(1, count)
    spelled = f"{count} {_plural(count, forms, code)}"
    if code == "es":
        return f"hace {spelled}"
    return f"{spelled} назад" if code == "ru" else f"{spelled} ago"


def number(value: Any, language: Any = "ru") -> str:
    """``0.71`` → «0,71» for ru/es and «0.71» for en (как пишет сам ТЗ)."""
    code = language_of(language)
    try:
        text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)
    return text.replace(".", DECIMAL_SEPARATOR[code])


# ---------------------------------------------------------------------------
# вопрос
# ---------------------------------------------------------------------------

ASK_PATTERNS: dict[str, tuple[str, ...]] = {
    "ru": (r"почему\s+ты\s+(?:так\s+)?(?:решил|думаешь|считаешь)",
           r"с\s+чего\s+ты\s+(?:взял|решил)",
           r"откуда\s+ты\s+знаешь",
           r"как\s+ты\s+понял"),
    "en": (r"why\s+do\s+you\s+(?:think|say|believe)",
           r"why\s+did\s+you\s+(?:decide|think)",
           r"how\s+do\s+you\s+know",
           r"how\s+can\s+you\s+tell"),
    "es": (r"por\s+qu[eé]\s+crees",
           r"c[oó]mo\s+sabes",
           r"por\s+qu[eé]\s+(?:dices|piensas)"),
}
#: «...что это Макс» / «...that it's Max» / «...que es Max».
WHO_PATTERNS: dict[str, str] = {
    "ru": r"(?:это|это\s+же)\s+([\w\-]+)",
    "en": r"(?:this\s+is|that\s+is|that's|it's|its)\s+([\w\-]+)",
    "es": r"(?<!\w)(?:que\s+)?es\s+([\w\-]+)",
}
ME_WORDS: dict[str, tuple[str, ...]] = {
    "ru": ("я", "меня", "мне"),
    "en": ("me", "i"),
    "es": ("mi", "yo", "mí"),
}
_COMPILED = {code: re.compile("|".join(patterns), re.IGNORECASE)
             for code, patterns in ASK_PATTERNS.items()}
_WHO = {code: re.compile(pattern, re.IGNORECASE) for code, pattern in WHO_PATTERNS.items()}


@dataclass(frozen=True)
class WhyQuestion:
    """The F-215 question, as far as it was understood."""

    language: str = "ru"
    who: str = ""
    about_me: bool = False


def why_question(text: Any) -> WhyQuestion | None:
    """Whether this utterance asks "why do you think ... ?", and about whom.

    Nothing here is a guess about identity: the answer is about the hub's own
    belief. A question that does not ask this at all returns ``None``, so the
    ordinary turn keeps its ordinary path.
    """
    said = " ".join(str(text or "").split())
    if not said:
        return None
    for code, pattern in _COMPILED.items():
        if not pattern.search(said):
            continue
        who = ""
        found = _WHO[code].search(said)
        if found:
            candidate = found.group(1).strip(".,!?;:")
            if candidate.casefold() not in ME_WORDS[code]:
                who = candidate
        about_me = not who and any(
            re.search(rf"\b{re.escape(word)}\b", said, re.IGNORECASE)
            for word in ME_WORDS[code]
        )
        return WhyQuestion(language=code, who=who, about_me=bool(about_me))
    return None


# ---------------------------------------------------------------------------
# ответ
# ---------------------------------------------------------------------------


def seen_and_missing(belief: Any, language: Any = "ru") -> tuple[list[str], list[str]]:
    """The signals belief really carries, and the ones that are simply absent."""
    code = language_of(language)
    sources = dict(getattr(belief, "sources", None) or {})
    seen: list[str] = []
    missing: list[str] = []
    for kind in KINDS:
        value = sources.get(kind)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            seen.append(f"{KIND_LABELS[kind][code]} {number(value, code)}")
        else:
            missing.append(MISSING[kind][code])
    return seen, missing


def explain(belief: Any = None, *, language: Any = "ru", name: str = "",
            now: float | None = None) -> str:
    """ТЗ F-215: the answer, built only from what the belief actually holds."""
    code = language_of(language)
    parts: list[str] = []
    if name:
        parts.append(f"{name}?")
    if belief is None:
        parts.append(NOTHING[code])
        return " ".join(parts)
    sources = dict(getattr(belief, "sources", None) or {})
    seen, missing = seen_and_missing(belief, code)
    if seen:
        parts.append(SEEN[code].format(items=", ".join(seen)))
    if missing:
        parts.append(MISSING_INTRO[code].format(items=", ".join(missing)))
    if sources.get("context"):
        parts.append(CONTEXT[code])
    elif sources.get("ambiguous"):
        parts.append(AMBIGUOUS[code])
    person_id = getattr(belief, "person_id", None)
    if not person_id:
        parts.append(NO_PERSON[code])
    reason = str(sources.get("reason") or "")
    if reason:
        spelled = REASON_TEXT[code].get(reason, reason)
        parts.append(REASON_INTRO[code].format(reason=spelled))
    if person_id:
        parts.append(CONFIDENCE[code].format(p=number(getattr(belief, "p", 0.0), code)))
    at = float(getattr(belief, "at", 0.0) or 0.0)
    if at:
        seconds = max(0.0, (time.time() if now is None else float(now)) - at)
        parts.append(AGE[code].format(age=age_phrase(seconds, code)))
    return " ".join(parts)


__all__ = [
    "AGE",
    "ASK_PATTERNS",
    "KINDS",
    "KIND_LABELS",
    "MISSING",
    "NOTHING",
    "REASON_TEXT",
    "WhyQuestion",
    "age_phrase",
    "explain",
    "language_of",
    "number",
    "seen_and_missing",
    "why_question",
]
