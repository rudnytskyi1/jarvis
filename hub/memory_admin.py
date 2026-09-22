"""Explicit memory management (ТЗ F-418, 9.4).

Three things a person says to their assistant about memory, and all three are
tools of the hub:

* «запомни, что …» - store a fact (the ``remember`` tool of F-414);
* «забудь, что …» - delete ONE stored fact (``forget_fact``);
* «что ты обо мне знаешь?» - read back the facts that belong to the speaker
  (``list_memory``, read-only, no confirmation).

The first two change stored data, so both go through the spoken "yes" of F-113
(``hub/confirmations.py``): the hub says what it is about to keep or drop and
nothing happens until the room confirms. The knowledge question changes
nothing and is answered from the rows themselves - a model is allowed to
paraphrase a reply, never to invent what the hub remembers.

Finding the ONE fact behind «забудь, что я пью кофе» is a retrieval problem,
so the same tokenisation the hybrid search uses (``hub/memory_search.py``) is
applied here: a fact is a candidate when enough of the words of the request
appear in it, the best candidate wins, and a near-tie is refused with a
question instead of a guess. «Никаких фейков» (ТЗ section 1) applies to
deletion exactly as it does to an answer.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hub.memories import MAX_TEXT_CHARS, Kind, MemoryFact, Scope
from hub.memory_search import tokenize

#: How much of the request has to sit inside a stored fact to call it a match.
MATCH_MIN = 0.5
#: A second fact this close to the best one is a real ambiguity, not noise.
AMBIGUITY_MARGIN = 0.05
#: How many facts one «что ты обо мне знаешь?» answer names.
KNOWLEDGE_LIMIT = 10

#: "запомни, что ..." / "remember that ..." / "recuerda que ...".
_REMEMBER = re.compile(
    r"(?:^|\b)(?:rowan[,\s]+)?(?:please\s+)?"
    r"(?:remember|make a note|note down|note this|save this|keep in mind)"
    r"\s*(?:that|:)?\s*(?P<fact>.+)"
    r"|(?:^|\b)(?:запомни(?:-ка)?|сохрани)\s*(?:,)?\s*(?:что|как)?\s*(?P<fact_ru>.+)"
    r"|(?:^|\b)(?:recuerda|apunta|toma nota de)\s*(?:que|:)?\s*(?P<fact_es>.+)",
    re.IGNORECASE,
)
#: Phrases that merely LOOK like "remember" but belong to other flows.
_NOT_REMEMBER = re.compile(
    r"\bremember (?:my|me)\b|\bupdate my voice\b|\benroll\b"
    r"|\bзапомни меня\b|\bзапомни мой голос\b|\bобнови мой голос\b"
    r"|\brecuerda mi voz\b",
    re.IGNORECASE,
)
#: "забудь, что ..." / "forget that ..." / "olvida que ...".
_FORGET = re.compile(
    r"(?:^|\b)forget\s+(?:that\s+|about\s+)?(?P<fact>.+)"
    r"|(?:^|\b)забудь\s*(?:,)?\s*(?:что\s+|про\s+|о\s+ч[её]м\s+)?(?P<fact_ru>.+)"
    r"|(?:^|\b)olvida\s*(?:que\s+)?(?P<fact_es>.+)",
    re.IGNORECASE,
)
#: «забудь меня» is F-213; «не забудь» is not a deletion at all.
_NOT_FORGET = re.compile(
    r"\bdon'?t forget\b|\bdo not forget\b|\bnever forget\b"
    r"|\bforget (?:me|everything about me|all about me)\b"
    r"|\bdelete (?:my|all my) (?:data|profile|identity|biometrics)\b"
    r"|(?<!не )\bзабудь меня\b|\bзабудь обо мне\b"
    r"|\bне забудь\b|\bне забывай\b"
    r"|\bolv[ií]dame\b|\bno olvides\b",
    re.IGNORECASE,
)
#: "что ты обо мне знаешь?" / "what do you know about me?" / "¿qué sabes de mí?".
_KNOWLEDGE = re.compile(
    r"\bwhat (?:do|did) you (?:know|remember) about me\b"
    r"|\bwhat do you have (?:on|about) me\b"
    r"|\bчто ты (?:обо мне |про меня )?(?:знаешь|помнишь)\b"
    r"|\bчто (?:ты )?(?:знаешь|помнишь) обо мне\b"
    r"|\bqu[eé] sabes (?:de|sobre) m[ií]\b|\bqu[eé] recuerdas de m[ií]\b",
    re.IGNORECASE,
)
#: Trailing punctuation that belongs to the request, not to the fact.
_TRAILING = " .,!?;:…\"'»«"
_MIN_WORDS = 1


def language_of(value: Any, *, default: str = "ru") -> str:
    """The language an answer is spoken in, one of the three of the ТЗ."""
    code = str(value or "").strip().casefold()[:2]
    return code if code in {"ru", "en", "es"} else default


class MemoryRequest(BaseModel):
    """One explicit memory request, as the hub understood it."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["remember", "forget", "knowledge"]
    text: str = Field(default="", max_length=MAX_TEXT_CHARS)
    #: "me" for the speaker, "room" for everybody (the `remember` tool's own
    #: vocabulary); the parsing itself never guesses a name.
    about: str = Field(default="", max_length=100)

    @property
    def shared(self) -> bool:
        return self.about.casefold() in {"room", "home", "everyone", "everybody", "all", "general"}


class Candidate(BaseModel):
    """One stored fact that may be the one a request is about."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Where the fact lives: the ``memories`` row and/or the file store entry.
    memory_id: str = Field(default="", max_length=64)
    file_id: str = Field(default="", max_length=64)
    owner: str = Field(default="", max_length=100)
    scope: str = Field(default="", max_length=16)
    kind: str = Field(default="", max_length=24)


class Selection(BaseModel):
    """What a «забудь …» request matched: one fact, none, or a real tie."""

    model_config = ConfigDict(extra="forbid")

    candidate: Candidate | None = None
    ambiguous: bool = False
    alternatives: list[Candidate] = Field(default_factory=list)


def _clean(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip(_TRAILING).strip()
    return text[:MAX_TEXT_CHARS].rstrip()


def _enough_words(text: str) -> bool:
    return len(tokenize(text)) >= _MIN_WORDS and len(text) >= 3


def parse(text: Any) -> MemoryRequest | None:
    """The explicit request inside an utterance, or ``None``.

    Order matters: «забудь меня» (F-213) is checked before «забудь …», so the
    irreversible deletion keeps its own flow, and «не забудь» never becomes a
    deletion.
    """
    raw = " ".join(str(text or "").split())
    if not raw:
        return None
    if _KNOWLEDGE.search(raw):
        return MemoryRequest(action="knowledge")
    if _NOT_FORGET.search(raw):
        return None
    match = _FORGET.search(raw)
    if match is not None:
        wanted = _clean(next((group for group in match.groups() if group), ""))
        # "forget it" / "забудь" alone is a cancellation, not a deletion.
        if _enough_words(wanted) and tokenize(wanted) not in (["it"], ["это"], ["всё"], [" vše"]):
            return MemoryRequest(action="forget", text=wanted)
        return None
    if _NOT_REMEMBER.search(raw):
        return None
    match = _REMEMBER.search(raw)
    if match is None:
        return None
    fact = _clean(next((group for group in match.groups() if group), ""))
    if not _enough_words(fact):
        return None
    return MemoryRequest(action="remember", text=fact, about="me")


def candidates_from_facts(facts: list[MemoryFact]) -> list[Candidate]:
    """The ``memories`` rows as candidates of a deletion request."""
    return [
        Candidate(
            text=fact.text, memory_id=fact.memory_id, owner=fact.owner_id,
            scope=str(fact.scope), kind=str(fact.kind),
        )
        for fact in facts
    ]


def candidates_from_file(rows: list[dict[str, Any]], *, owner: str = "",
                         home_id: str = "") -> list[Candidate]:
    """The file store's records as candidates (``hub/storage.py::Memory``).

    A record with no person is a fact about the room itself (that is what
    ``Memory.facts()`` means by it), so it is filed against the home - not
    against whichever person happened to be speaking.
    """
    result: list[Candidate] = []
    for row in rows:
        text = _clean(row.get("fact"))
        if not text:
            continue
        person = " ".join(str(row.get("person") or "").split())
        if person:
            result.append(Candidate(
                text=text, file_id=str(row.get("id") or ""), owner=person,
                scope=str(Scope.PERSON), kind=str(Kind.PERSON_FACT),
            ))
        else:
            result.append(Candidate(
                text=text, file_id=str(row.get("id") or ""), owner=str(home_id or owner),
                scope=str(Scope.HOME), kind=str(Kind.HOME_FACT),
            ))
    return result


def score(text: str, query: str) -> float:
    """How much of the request's meaning sits in the fact, 0…1.

    Lexical on purpose: a deletion must be explainable ("these words matched
    this fact"), and the embedding half of the search is allowed to be missing
    on a hub whose CPU model is not installed.
    """
    wanted = tokenize(query)
    if not wanted:
        return 0.0
    have = set(tokenize(text))
    if not have:
        return 0.0
    covered = 0.0
    for word in wanted:
        if word in have:
            covered += 1.0
        elif len(word) >= 4 and any(word in other or other in word for other in have):
            # "ключи" vs "ключей" - a shared stem is a weaker but real match.
            covered += 0.5
    return min(1.0, covered / len(wanted))


def select(candidates: list[Candidate], query: str, *,
           minimum: float = MATCH_MIN) -> Selection:
    """The one fact a deletion request names, or an honest refusal.

    A single winner is returned only when it is both above ``minimum`` and
    clearly ahead of the runner-up; otherwise the answer is a question, which
    the caller words with :func:`ambiguous_question`.
    """
    scored = [item.model_copy(update={"score": score(item.text, query)}) for item in candidates]
    scored = [item for item in scored if item.score >= minimum]
    if not scored:
        return Selection()
    scored.sort(key=lambda item: (-item.score, item.text.casefold()))
    best = scored[0]
    ahead = [item for item in scored[1:] if item.score > best.score - AMBIGUITY_MARGIN]
    if ahead:
        return Selection(candidate=None, ambiguous=True,
                         alternatives=[best, *ahead[:3]])
    return Selection(candidate=best, alternatives=scored[1:4])


def _list(items: list[str]) -> str:
    return "; ".join(f"«{item}»" for item in items)


def remember_question(text: str, language: Any = "ru", *, window_s: float = 8.0) -> str:
    """ТЗ F-113: what the room hears before a fact is kept."""
    seconds = max(1, int(round(window_s)))
    lang = language_of(language)
    if lang == "en":
        return (f"Say yes within {seconds} seconds and I will remember: «{text}». "
                f"Anything else cancels it.")
    if lang == "es":
        return (f"Di sí en {seconds} segundos y recordaré: «{text}». "
                f"Cualquier otra respuesta lo cancela.")
    return (f"Скажи «да» в течение {seconds} секунд, и я запомню: «{text}». "
            f"Любой другой ответ отменяет.")


def forget_question(text: str, language: Any = "ru", *, window_s: float = 8.0) -> str:
    """ТЗ F-113: what the room hears before one fact is dropped."""
    seconds = max(1, int(round(window_s)))
    lang = language_of(language)
    if lang == "en":
        return (f"Say yes within {seconds} seconds and I will forget: «{text}». "
                f"This cannot be undone. Anything else cancels it.")
    if lang == "es":
        return (f"Di sí en {seconds} segundos y olvidaré: «{text}». "
                f"No se puede deshacer. Cualquier otra respuesta lo cancela.")
    return (f"Скажи «да» в течение {seconds} секунд, и я забуду: «{text}». "
            f"Это нельзя вернуть. Любой другой ответ отменяет.")


def ambiguous_question(alternatives: list[Candidate], language: Any = "ru") -> str:
    """One question when several facts match - F-413 allows exactly one."""
    texts = [item.text for item in alternatives[:3]]
    lang = language_of(language)
    if lang == "en":
        return (f"I have more than one fact like that: {_list(texts)}. "
                f"Say which one in one phrase, and I will ask about deleting it.")
    if lang == "es":
        return (f"Tengo más de un dato así: {_list(texts)}. "
                f"Di cuál en una frase y preguntaré antes de borrarlo.")
    return (f"У меня больше одного такого факта: {_list(texts)}. "
            f"Скажи, какой именно, — и я спрошу перед тем, как забыть его.")


def nothing_found(language: Any = "ru") -> str:
    lang = language_of(language)
    if lang == "en":
        return "I could not find such a fact, so I changed nothing."
    if lang == "es":
        return "No encontré ese dato, así que no cambié nada."
    return "Я не нашёл такого факта, поэтому ничего не менял."


def knowledge_answer(texts: list[str], language: Any = "ru", *,
                     limit: int = KNOWLEDGE_LIMIT) -> str:
    """What the hub really knows about the speaker (ТЗ F-418)."""
    lang = language_of(language)
    remembered = [text for text in texts if str(text).strip()][:max(1, int(limit))]
    if not remembered:
        if lang == "en":
            return "I do not know anything about you yet."
        if lang == "es":
            return "Todavía no sé nada de ti."
        return "Я пока ничего о тебе не знаю."
    numbered = " ".join(f"{index}) «{text}»" for index, text in enumerate(remembered, 1))
    if lang == "en":
        return f"Here is what I know about you: {numbered}"
    if lang == "es":
        return f"Esto es lo que sé de ti: {numbered}"
    return f"Вот что я о тебе знаю: {numbered}"


def remembered_answer(text: str, language: Any = "ru") -> str:
    lang = language_of(language)
    if lang == "en":
        return f"Remembered: «{text}»."
    if lang == "es":
        return f"Recordado: «{text}»."
    return f"Запомнил: «{text}»."


def forgotten_answer(text: str, language: Any = "ru") -> str:
    lang = language_of(language)
    if lang == "en":
        return f"Forgotten: «{text}». It cannot be restored."
    if lang == "es":
        return f"Olvidado: «{text}». No se puede recuperar."
    return f"Забыл: «{text}». Вернуть это нельзя."


def cancelled(language: Any = "ru") -> str:
    lang = language_of(language)
    if lang == "en":
        return "Alright, I changed nothing in your memory."
    if lang == "es":
        return "De acuerdo, no cambié nada en tu memoria."
    return "Хорошо, в твоей памяти я ничего не менял."


def expired(language: Any = "ru") -> str:
    lang = language_of(language)
    if lang == "en":
        return "That confirmation expired, so I changed nothing in your memory."
    if lang == "es":
        return "Esa confirmación expiró, así que no cambié nada en tu memoria."
    return "Время подтверждения вышло, поэтому я ничего не менял в твоей памяти."


__all__ = [
    "AMBIGUITY_MARGIN",
    "KNOWLEDGE_LIMIT",
    "MATCH_MIN",
    "Candidate",
    "MemoryRequest",
    "Selection",
    "ambiguous_question",
    "cancelled",
    "candidates_from_facts",
    "candidates_from_file",
    "expired",
    "forget_question",
    "forgotten_answer",
    "knowledge_answer",
    "language_of",
    "nothing_found",
    "parse",
    "remember_question",
    "remembered_answer",
    "score",
    "select",
]
