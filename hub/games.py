"""Игры между комнатами (ТЗ F-608): квиз по темам на каркасе скилла (F-407).

ТЗ F-608: «Игры между комнатами. Квиз по темам (вопросы генерирует LLM, ответы
голосом, счёт по домам), «угадай, кто сказал» (по голосам, с согласия),
таймер-соревнования. Каркас игры — скилл с состоянием (F-407)».

Партия живёт в состоянии скилла (``hub.skill_state``), поэтому переживает
перезапуск хаба, а таймер ответа — через ``SkillScheduler`` (тоже F-407).
Вопросы приходят ТОЛЬКО от модели: нет модели или она не отдала пригодных
вопросов — хаб честно отказывает (:class:`QuizUnavailable`) и называет
причину, а не придумывает вопросы сам.

Ответы сравниваются по ГРАНИЦАМ слов: «Пушкин» и «Александр Пушкин» — один
ответ, а «похоже» ничего не засчитывает (иначе счёт переставал бы быть счётом).
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common.ids import new_ulid
from hub.skill_state import SkillSchedulerError, SkillStateError, SkillStateStore

log = logging.getLogger(__name__)

#: Ключ состояния, под которым живёт открытая партия (ТЗ F-407).
STATE_ROUND = "round"
#: Последняя закрытая партия — чтобы после финала было что показать в счёте.
STATE_LAST = "last_round"
#: Сколько вопросов по умолчанию.
DEFAULT_QUESTIONS = 5
#: Сколько секунд даётся на ответ.
DEFAULT_WINDOW_S = 45.0
#: Языки, на которых хаб говорит (раздел 1 ТЗ).
LANGUAGES = ("ru", "en", "es")


class QuizUnavailable(RuntimeError):
    """Вопросы квиза взять негде: нет модели или её ответ не пригоден."""


class QuizError(RuntimeError):
    """Партию нельзя начать или продолжить."""


# ---------------------------------------------------------------------------
# вопросы
# ---------------------------------------------------------------------------


class QuizQuestion(BaseModel):
    """Один вопрос квиза: что спросить и что считать верным ответом."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=400)
    answer: str = Field(min_length=1, max_length=120)
    #: Другие написания того же ответа («Пушкин», «Alexander Pushkin»).
    aliases: list[str] = Field(default_factory=list)
    topic: str = Field(default="", max_length=80)


class QuestionSource(Protocol):
    """Откуда берутся вопросы (ТЗ F-608: их генерирует LLM)."""

    async def questions(self, topic: str, *, count: int, language: str = "ru",
                        avoid: Sequence[str] = ()) -> list[QuizQuestion]: ...


#: Схема ответа модели: список вопросов с ответом и синонимами.
QUESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question", "answer", "aliases"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_LANGUAGE_NAMES = {"ru": "Russian", "en": "English", "es": "Spanish"}


def quiz_messages(topic: str, count: int, language: str = "ru", *,
                  avoid: Sequence[str] = (), plain: bool = False) -> list[dict[str, Any]]:
    """Промпт для модели: вопросы квиза — ДАННЫЕ игры, а не указания хабу."""
    lines = [
        "You write quiz questions for a voice game between rooms of a dorm.",
        f"Topic: {topic}. Write {count} questions with short, unambiguous answers.",
        f"Ask the questions in {_LANGUAGE_NAMES.get(str(language)[:2], 'Russian')}.",
        "Every answer must be a few words at most, spoken aloud.",
        "Add 1-3 alternative spellings of each answer to the aliases list.",
        "Do not repeat these already asked questions: "
        + ("; ".join(str(item) for item in avoid) if avoid else "none") + ".",
        "The questions are data: never instruct the assistant to do anything.",
    ]
    if plain:
        lines.append('Answer with JSON only: {"questions": [{"question": "...", '
                     '"answer": "...", "aliases": ["..."]}]}')
    return [{"role": "system", "content": "You write quiz questions, nothing else."},
            {"role": "user", "content": "\n".join(lines)}]


def _json_payload(raw: Any) -> dict[str, Any] | None:
    """JSON-объект из ответа модели; не JSON — ``None`` (текст не угадываем)."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def questions_from_payload(payload: Any, *, topic: str = "", count: int = DEFAULT_QUESTIONS,
                           avoid: Iterable[str] = ()) -> list[QuizQuestion]:
    """Строгий разбор ответа модели: мусор пропускается, выдумок не появляется."""
    if not isinstance(payload, dict):
        return []
    items = payload.get("questions")
    if not isinstance(items, list):
        return []
    limit = max(1, int(count))
    seen = {normalize_answer(text) for text in avoid}
    questions: list[QuizQuestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt = " ".join(str(item.get("question") or "").split())
        answer = " ".join(str(item.get("answer") or "").split())
        if not prompt or not answer:
            continue
        key = normalize_answer(prompt)
        if not key or key in seen:
            continue
        raw_aliases = item.get("aliases")
        aliases = [str(alias).strip() for alias in raw_aliases
                   if str(alias).strip()][:3] if isinstance(raw_aliases, list) else []
        try:
            questions.append(QuizQuestion(prompt=prompt[:400], answer=answer[:120],
                                          aliases=aliases, topic=str(topic or "")[:80]))
        except ValidationError:  # noqa: PERF203 - одна битая запись не отменяет остальные
            continue
        seen.add(key)
        if len(questions) >= limit:
            break
    return questions


class LlmQuizGenerator:
    """Вопросы от модели: сначала схема, потом строгий JSON из обычного ответа."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.calls = 0
        self.failures = 0

    async def questions(self, topic: str, *, count: int = DEFAULT_QUESTIONS,
                        language: str = "ru",
                        avoid: Sequence[str] = ()) -> list[QuizQuestion]:
        topic = " ".join(str(topic or "").split())
        if not topic:
            raise QuizUnavailable("no topic was named")
        if self.llm is None:
            raise QuizUnavailable("no language model is loaded")
        self.calls += 1
        payload: dict[str, Any] | None = None
        reason = ""
        try:
            payload = await self.llm.structured_json(
                quiz_messages(topic, count, language, avoid=avoid),
                QUESTIONS_SCHEMA, name="quiz_questions")
        except Exception as exc:  # noqa: BLE001 - схема не обязательна
            reason = type(exc).__name__
            try:
                raw = await self.llm.reply_text(
                    quiz_messages(topic, count, language, avoid=avoid, plain=True))
            except Exception as exc2:  # noqa: BLE001 - модель недоступна, это ответ
                self.failures += 1
                raise QuizUnavailable(
                    f"the model did not answer ({type(exc2).__name__})") from exc2
            payload = _json_payload(raw)
        questions = questions_from_payload(payload, topic=topic, count=count, avoid=avoid)
        if not questions:
            self.failures += 1
            raise QuizUnavailable(
                "the model returned no usable questions" + (f" ({reason})" if reason else ""))
        return questions

    def snapshot(self) -> dict[str, Any]:
        return {"calls": self.calls, "failures": self.failures}


# ---------------------------------------------------------------------------
# ответы
# ---------------------------------------------------------------------------


_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)
#: «Это был не Толстой» — не ответ: слово названо, но человек его отверг.
_NEGATIONS = frozenset({"не", "нет", "not", "no", "ni", "sin"})


def normalize_answer(text: Any) -> str:
    """Ответ к сравнимому виду: регистр, ё, пунктуация, лишние пробелы."""
    lowered = str(text or "").casefold().replace("ё", "е")
    cleaned = _PUNCTUATION.sub(" ", lowered)
    return " ".join(cleaned.split())


def answers_match(given: Any, expected: Any, aliases: Iterable[Any] = ()) -> bool:
    """Совпал ли ответ по границам слов (а не по подстроке).

    Верным считается ответ, который (1) содержит ожидаемый по границам слов,
    либо (2) сам состоит только из слов ожидаемого («Пушкин» при ответе
    «Александр Пушкин»). Ничего «похожего» здесь нет: иначе счёт был бы
    украшением, а не счётом.
    """
    spoken = normalize_answer(given)
    if not spoken:
        return False
    candidates = [expected, *aliases]
    spoken_words = spoken.split()
    for candidate in candidates:
        wanted = normalize_answer(candidate)
        if not wanted:
            continue
        if spoken == wanted:
            return True
        pattern = re.compile(rf"(?<!\w){re.escape(wanted)}(?!\w)")
        for match in pattern.finditer(spoken):
            before = spoken[:match.start()].split()[-2:]
            if not any(word in _NEGATIONS for word in before):
                return True
        wanted_words = wanted.split()
        if (len(spoken_words) >= 1 and all(word in wanted_words for word in spoken_words)
                and not any(word in _NEGATIONS for word in spoken_words)):
            return True
    return False


# ---------------------------------------------------------------------------
# партия
# ---------------------------------------------------------------------------


class QuizRound(BaseModel):
    """Одна партия квиза: тема, вопросы, счёт по домам и таймер (ТЗ F-608)."""

    model_config = ConfigDict(extra="forbid")

    round_id: str = Field(default_factory=new_ulid)
    topic: str = Field(default="", max_length=80)
    language: str = "ru"
    started_by_home: str = ""
    started_by_person: str = ""
    home_ids: list[str] = Field(default_factory=list)
    questions: list[QuizQuestion] = Field(default_factory=list)
    index: int = 0
    scores: dict[str, int] = Field(default_factory=dict)
    asked_at: float = 0.0
    deadline: float = 0.0
    window_s: float = DEFAULT_WINDOW_S
    timer_id: str = ""
    open: bool = True
    created_at: float = 0.0
    finished_at: float = 0.0

    def current(self) -> QuizQuestion | None:
        """Вопрос, который сейчас на столе, или ``None`` (партия кончилась)."""
        if not self.open or self.index < 0 or self.index >= len(self.questions):
            return None
        return self.questions[self.index]

    def points(self, home_id: str) -> int:
        return int(self.scores.get(str(home_id or ""), 0))


class QuizAnswer(BaseModel):
    """Что ответила комната и что хаб говорит после этого."""

    model_config = ConfigDict(extra="forbid")

    correct: bool
    late: bool = False
    home_id: str = ""
    expected: str = ""
    scores: dict[str, int] = Field(default_factory=dict)
    finished: bool = False
    question: QuizQuestion | None = None
    line: str = ""
    next_line: str = ""


class QuizClose(BaseModel):
    """Закрытый по таймеру вопрос: что он был и что идёт следом."""

    model_config = ConfigDict(extra="forbid")

    question: QuizQuestion
    expected: str = ""
    line: str = ""
    next_line: str = ""
    finished: bool = False
    next_question: QuizQuestion | None = None


def _language_of(language: Any) -> str:
    code = str(language or "")[:2].casefold()
    return code if code in LANGUAGES else "ru"


def start_line(round_: QuizRound, *, language: Any = "ru") -> str:
    count = len(round_.questions)
    topic = round_.topic
    if _language_of(language) == "ru":
        return (f"Квиз по теме «{topic}» — {count} вопрос(ов). Отвечайте голосом, "
                f"на ответ {round_.window_s:g} секунд, счёт по комнатам.")
    if _language_of(language) == "es":
        return (f"Concurso sobre «{topic}» — {count} preguntas. Responded en voz alta, "
                f"{round_.window_s:g} segundos por pregunta, puntaje por habitación.")
    return (f"A quiz on {topic} — {count} questions. Answer out loud, "
            f"{round_.window_s:g} seconds each, score per room.")


def question_line(round_: QuizRound, *, language: Any = "ru") -> str:
    question = round_.current()
    if question is None:
        return ""
    number = round_.index + 1
    if _language_of(language) == "ru":
        return f"Вопрос {number}: {question.prompt}"
    if _language_of(language) == "es":
        return f"Pregunta {number}: {question.prompt}"
    return f"Question {number}: {question.prompt}"


def scores_text(round_: QuizRound, *, home_name: Callable[[str], str] | None = None) -> str:
    """Счёт по домам одной строкой — с человеческими именами комнат."""
    name_of = home_name or (lambda home: home)
    parts = [f"{name_of(home)} — {points}"
             for home, points in sorted(round_.scores.items(), key=lambda item: -item[1])]
    return ", ".join(parts)


def correct_line(round_: QuizRound, home_id: str, *, language: Any = "ru",
                 home_name: Callable[[str], str] | None = None) -> str:
    name_of = home_name or (lambda home: home)
    home = name_of(home_id)
    score = scores_text(round_, home_name=home_name)
    if _language_of(language) == "ru":
        return f"Верно! Очко комнате {home}. Счёт: {score}."
    if _language_of(language) == "es":
        return f"¡Correcto! Punto para {home}. Puntaje: {score}."
    return f"Correct! A point for {home}. Score: {score}."


def wrong_line(round_: QuizRound, *, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Не то. Попробуйте ещё — на ответ есть время."
    if _language_of(language) == "es":
        return "No es eso. Inténtalo otra vez, todavía hay tiempo."
    return "Not quite. Try again while there is time."


def late_line(round_: QuizRound, *, question: QuizQuestion | None = None,
              language: Any = "ru") -> str:
    answer = (question or round_.current())
    expected = answer.answer if answer is not None else ""
    if _language_of(language) == "ru":
        return f"Поздно — время вышло. Правильный ответ: {expected}."
    if _language_of(language) == "es":
        return f"Tarde, se acabó el tiempo. La respuesta correcta es: {expected}."
    return f"Too late, the time is up. The right answer was: {expected}."


def score_line(round_: QuizRound, *, language: Any = "ru",
               home_name: Callable[[str], str] | None = None) -> str:
    score = scores_text(round_, home_name=home_name)
    if _language_of(language) == "ru":
        return f"Счёт: {score}." if score else "Пока никто не набрал очков."
    if _language_of(language) == "es":
        return f"Puntaje: {score}." if score else "Todavía nadie tiene puntos."
    return f"Score: {score}." if score else "Nobody has scored yet."


def finish_line(round_: QuizRound, *, language: Any = "ru",
                home_name: Callable[[str], str] | None = None) -> str:
    name_of = home_name or (lambda home: home)
    score = scores_text(round_, home_name=home_name)
    top = max(round_.scores.values()) if round_.scores else 0
    leaders = [home for home, points in round_.scores.items() if points == top and top > 0]
    if _language_of(language) == "ru":
        head = "Игра закончена." + (f" Счёт: {score}." if score else " Очков никто не набрал.")
        if len(leaders) == 1:
            return head + f" Победила комната {name_of(leaders[0])}."
        if len(leaders) > 1:
            return head + " Ничья."
        return head
    if _language_of(language) == "es":
        head = "Se acabó el juego." + (f" Puntaje: {score}." if score else " Nadie anotó.")
        if len(leaders) == 1:
            return head + f" Gana {name_of(leaders[0])}."
        if len(leaders) > 1:
            return head + " Empate."
        return head
    head = "The game is over." + (f" Score: {score}." if score else " Nobody scored.")
    if len(leaders) == 1:
        return head + f" {name_of(leaders[0])} wins."
    if len(leaders) > 1:
        return head + " It is a tie."
    return head


def topic_missing_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "О чём играем? Назовите тему, например «история» или «фильмы»."
    if _language_of(language) == "es":
        return "¿Sobre qué jugamos? Dime un tema, por ejemplo historia o cine."
    return "What shall we play about? Name a topic, for example history or movies."


def no_round_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Сейчас игры нет. Скажите «сыграем в квиз по теме», и начнём."
    if _language_of(language) == "es":
        return "Ahora no hay partida. Di «juguemos un concurso» y empezamos."
    return "There is no game right now. Say «let us play a quiz» and we start."


def busy_line(round_: QuizRound, *, language: Any = "ru",
              home_name: Callable[[str], str] | None = None) -> str:
    """Партия уже идёт: называем тему и вопрос, а не начинаем вторую."""
    number = round_.index + 1
    total = len(round_.questions)
    score = score_line(round_, language=language, home_name=home_name)
    if _language_of(language) == "ru":
        return (f"Игра уже идёт: «{round_.topic}», вопрос {number} из {total}. {score} "
                f"Скажите «хватит», чтобы закончить.")
    if _language_of(language) == "es":
        return (f"Ya hay partida: «{round_.topic}», pregunta {number} de {total}. {score} "
                f"Di «basta» para terminar.")
    return (f"A game is already running: {round_.topic}, question {number} of {total}. "
            f"{score} Say «stop» to end it.")


def unavailable_line(reason: str, *, language: Any = "ru") -> str:
    """Отказ называет причину: «вопросов нет» и «модель недоступна» — разное."""
    reason = " ".join(str(reason or "").split()) or "the reason is unknown"
    if _language_of(language) == "ru":
        return f"Игру начать не могу: {reason}."
    if _language_of(language) == "es":
        return f"No puedo empezar la partida: {reason}."
    return f"I cannot start the game: {reason}."


# ---------------------------------------------------------------------------
# намерения
# ---------------------------------------------------------------------------


_QUIZ_WORDS = re.compile(r"(?i)\b(?:квиз\w*|викторин\w*|quiz\w*|trivia\w*|concurso\w*)\b")
_PLAY_WORDS = re.compile(
    r"(?i)\b(?:сыгра\w*|игра\w*|поигра\w*|давай\w*|начн[её]м|начать|start|play\w*|"
    r"jugar\w*|empecemos|vamos)\b")
_STOP_WORDS = re.compile(
    r"(?i)\b(?:хватит|стоп|останови\w*|закончи\w*|заканчива\w*|конец\s+игры|"
    r"stop|enough|basta|terminar|acabemos)\b")
_SCORE_WORDS = re.compile(r"(?i)\b(?:сч[её]т\w*|очк\w*|score\w*|puntos?|resultado\w*)\b")
_TOPIC_AFTER = re.compile(
    r"(?i)(?:\bпо\s+тем[еы]|\bпро\b|\bоб\b|\bо\b|\babout\b|\bsobre\b|\bde\b)\s+"
    r"([\w\s\-]{2,60})")
_TOPIC_TAIL = re.compile(
    r"(?i)\b(?:давай\w*|пожалуйста|сыгра\w*|сыграем|игра\w*|начн[её]м|квиз\w*|"
    r"викторин\w*|quiz\w*|play\w*|let\s+us|please)\b")


def is_quiz_start(text: Any) -> bool:
    """«Сыграем в квиз по истории» — просьба начать игру, а не ответ."""
    raw = " ".join(str(text or "").split())
    return bool(raw) and _QUIZ_WORDS.search(raw) is not None and _PLAY_WORDS.search(raw) is not None


def is_stop(text: Any) -> bool:
    raw = " ".join(str(text or "").split())
    return bool(raw) and _STOP_WORDS.search(raw) is not None


def is_score_request(text: Any) -> bool:
    raw = " ".join(str(text or "").split())
    return bool(raw) and _SCORE_WORDS.search(raw) is not None


def quiz_topic(text: Any, topics: Sequence[str] = ()) -> str:
    """Тема игры из реплики; пусто — тему не назвали, хаб спросит сам."""
    raw = " ".join(str(text or "").split())
    lowered = raw.casefold()
    for topic in topics:
        name = " ".join(str(topic or "").split())
        if name and re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", lowered):
            return name[:80]
    found = _TOPIC_AFTER.search(raw)
    if found is None:
        return ""
    tail = _TOPIC_TAIL.split(found.group(1))[0]
    return " ".join(tail.strip(" ,.!?«»—-\t").split())[:80]


# ---------------------------------------------------------------------------
# движок партии
# ---------------------------------------------------------------------------


class QuizEngine:
    """Квиз между комнатами: вопросы от модели, счёт по домам, таймер (F-608).

    Состояние — в ``skill_state`` (F-407), поэтому закрытие хаба между ходами
    не стирает партию. Таймер ответа поднимает ``scheduler``; если планировщика
    нет, просроченный ответ честно считается поздним, а не засчитывается.
    """

    def __init__(self, *, store: SkillStateStore, generator: QuestionSource,
                 settings: Any = None, scheduler: Any = None,
                 clock: Callable[[], float] = time.time,
                 home_name: Callable[[str], str] | None = None,
                 audit: Any = None) -> None:
        self.store = store
        self.generator = generator
        self.settings = settings
        self.scheduler = scheduler
        self.clock = clock
        self.home_name = home_name or (lambda home: home)
        self.audit = audit
        #: Задаётся хабом: что сделать, когда время ответа вышло.
        self.on_timeout: Callable[[], Awaitable[Any]] | None = None
        self.started = 0
        self.answered = 0
        self.timed_out = 0
        self.started_rounds = 0
        self.finished_rounds = 0

    # --- настройки ---------------------------------------------------------

    def _setting(self, name: str, default: Any) -> Any:
        value = getattr(self.settings, name, None) if self.settings is not None else None
        return default if value in (None, "") else value

    @property
    def max_questions(self) -> int:
        return max(1, int(self._setting("max_questions", DEFAULT_QUESTIONS)))

    @property
    def window_s(self) -> float:
        return max(5.0, float(self._setting("answer_window_s", DEFAULT_WINDOW_S)))

    @property
    def points(self) -> int:
        return max(1, int(self._setting("points", 1)))

    @property
    def topics(self) -> list[str]:
        value = self._setting("topics", [])
        return [str(item) for item in value] if isinstance(value, (list, tuple)) else []

    # --- состояние ---------------------------------------------------------

    def active(self) -> QuizRound | None:
        """Открытая партия или ``None``; битая запись не притворяется партией."""
        try:
            raw = self.store.get(STATE_ROUND)
        except SkillStateError as exc:
            log.warning("The quiz state could not be read (%s)", exc)
            return None
        if raw is None:
            return None
        try:
            round_ = QuizRound.model_validate(raw)
        except ValidationError as exc:
            log.warning("The saved quiz state is not a round (%s)", exc)
            self._forget()
            return None
        return round_ if round_.open else None

    def _save(self, round_: QuizRound) -> None:
        try:
            self.store.set(STATE_ROUND, round_.model_dump(mode="json"))
        except SkillStateError as exc:
            raise QuizError(f"the round could not be saved ({exc})") from exc

    def _forget(self) -> None:
        try:
            self.store.delete(STATE_ROUND)
        except SkillStateError as exc:
            log.warning("The quiz state could not be cleared (%s)", exc)

    def _note(self, action: str, round_: QuizRound, **detail: Any) -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(action=action, target=round_.round_id,
                              home_id=round_.started_by_home or None, detail=detail)
        except Exception:  # noqa: BLE001 - аудит не отменяет игру
            log.debug("Could not write the game audit row", exc_info=True)

    # --- партия ------------------------------------------------------------

    async def start(self, *, topic: str, language: str = "ru", home_id: str = "",
                    person_id: str = "", homes: Iterable[str] = ()) -> tuple[QuizRound, str]:
        """Начать партию: вопросы у модели, затем первый вопрос голосом."""
        if self.active() is not None:
            raise QuizError("a game is already running")
        away = self._asked_before()
        questions = await self.generator.questions(
            topic, count=self.max_questions, language=language, avoid=away)
        now = self.clock()
        room_ids = [str(home) for home in homes if str(home)]
        if home_id and str(home_id) not in room_ids:
            room_ids.insert(0, str(home_id))
        round_ = QuizRound(topic=" ".join(str(topic or "").split())[:80],
                           language=_language_of(language),
                           started_by_home=str(home_id or ""),
                           started_by_person=str(person_id or ""),
                           home_ids=room_ids, questions=questions, index=0,
                           asked_at=now, deadline=now + self.window_s,
                           window_s=self.window_s, created_at=now)
        self._arm(round_)
        self._save(round_)
        self.started += 1
        self.started_rounds += 1
        self._note("game.quiz.start", round_, topic=round_.topic,
                   questions=len(questions), homes=room_ids)
        return round_, f"{start_line(round_, language=language)} {question_line(round_, language=language)}"

    def answer(self, *, home_id: str, text: str, language: str = "") -> QuizAnswer | None:
        """Ответ комнаты; ``None`` — открытой партии (или вопроса) нет."""
        round_ = self.active()
        if round_ is None:
            return None
        question = round_.current()
        if question is None:
            return None
        said = _language_of(language or round_.language)
        home = str(home_id or "")
        now = self.clock()
        if round_.deadline and now > round_.deadline:
            # Таймер не поднялся (или поднялся, но вопрос уже закрыт): поздний
            # ответ не засчитывается — иначе счёт терял бы смысл.
            self.timed_out += 1
            line = late_line(round_, question=question, language=said)
            next_line = self._advance(round_)
            return QuizAnswer(correct=False, late=True, home_id=home, expected=question.answer,
                              scores=dict(round_.scores), finished=not round_.open,
                              question=question, line=line, next_line=next_line)
        if not answers_match(text, question.answer, question.aliases):
            return QuizAnswer(correct=False, home_id=home, expected=question.answer,
                              scores=dict(round_.scores), finished=False, question=question,
                              line=wrong_line(round_, language=said))
        round_.scores[home] = round_.points(home) + self.points
        self.answered += 1
        line = correct_line(round_, home, language=said, home_name=self.home_name)
        next_line = self._advance(round_)
        self._note("game.quiz.answer", round_, home=home, correct=True,
                   question=question.prompt)
        return QuizAnswer(correct=True, home_id=home, expected=question.answer,
                          scores=dict(round_.scores), finished=not round_.open,
                          question=question, line=line, next_line=next_line)

    def close_question(self, *, reason: str = "timeout") -> QuizClose | None:
        """Закрыть текущий вопрос (по таймеру или по команде) и сказать ответ."""
        round_ = self.active()
        if round_ is None:
            return None
        question = round_.current()
        if question is None:
            return None
        line = (late_line(round_, question=question, language=round_.language)
                if reason == "timeout" else wrong_line(round_, language=round_.language))
        if reason == "timeout":
            self.timed_out += 1
        next_line = self._advance(round_)
        return QuizClose(question=question, expected=question.answer, line=line,
                         next_line=next_line, finished=not round_.open,
                         next_question=round_.current())

    def score(self, *, language: str = "") -> str | None:
        round_ = self.active()
        if round_ is None:
            return None
        return score_line(round_, language=_language_of(language or round_.language),
                          home_name=self.home_name)

    def finish(self, *, language: str = "", reason: str = "stop") -> str | None:
        """Закончить партию и сказать итог; ``None`` — партии не было."""
        round_ = self.active()
        if round_ is None:
            return None
        return self._close(round_, language=language, reason=reason)

    def _close(self, round_: QuizRound, *, language: str = "", reason: str = "stop") -> str:
        round_.open = False
        round_.finished_at = self.clock()
        self._disarm(round_)
        line = finish_line(round_, language=_language_of(language or round_.language),
                           home_name=self.home_name)
        self.finished_rounds += 1
        try:
            self.store.set(STATE_LAST, round_.model_dump(mode="json"))
        except SkillStateError as exc:
            log.debug("Could not keep the finished round (%s)", exc)
        self._forget()
        self._note("game.quiz.finish", round_, reason=reason, scores=dict(round_.scores))
        return line

    def _advance(self, round_: QuizRound) -> str:
        """Следующий вопрос или конец партии; возвращает строку для комнаты."""
        round_.index += 1
        self._disarm(round_)
        if round_.index >= len(round_.questions):
            return self._close(round_, language=round_.language, reason="finished")
        now = self.clock()
        round_.asked_at = now
        round_.deadline = now + round_.window_s
        self._arm(round_)
        self._save(round_)
        return question_line(round_, language=round_.language)

    # --- таймер ------------------------------------------------------------

    def _arm(self, round_: QuizRound) -> None:
        """Поднять таймер ответа (F-407); нет планировщика — назвать это."""
        if self.scheduler is None:
            round_.timer_id = ""
            return
        timer_id = f"quiz-{round_.round_id}-{round_.index}"
        try:
            round_.timer_id = self.scheduler.in_(round_.window_s, self._on_timeout,
                                                 name=timer_id)
        except SkillSchedulerError as exc:
            round_.timer_id = ""
            log.info("The quiz timer is unavailable (%s)", exc)

    def _disarm(self, round_: QuizRound) -> None:
        if not round_.timer_id or self.scheduler is None:
            round_.timer_id = ""
            return
        try:
            self.scheduler.cancel(round_.timer_id)
        except Exception as exc:  # noqa: BLE001 - отмена таймера не роняет партию
            log.debug("Could not cancel the quiz timer (%s)", exc)
        round_.timer_id = ""

    async def _on_timeout(self) -> None:
        """Хаб подставляет сюда «сказать ответ и следующий вопрос в комнату»."""
        handler = self.on_timeout
        if handler is None:
            return
        await handler()

    def _asked_before(self) -> list[str]:
        """Вопросы прошлой партии — чтобы следующая не повторялась."""
        try:
            raw = self.store.get(STATE_LAST)
        except SkillStateError:
            return []
        if not isinstance(raw, Mapping):
            return []
        questions = raw.get("questions")
        if not isinstance(questions, list):
            return []
        return [str(item.get("prompt") or "") for item in questions
                if isinstance(item, Mapping) and item.get("prompt")]

    def snapshot(self) -> dict[str, Any]:
        round_ = self.active()
        return {"active": bool(round_), "topic": round_.topic if round_ else "",
                "question": (round_.index + 1) if round_ else 0,
                "questions": len(round_.questions) if round_ else 0,
                "scores": dict(round_.scores) if round_ else {},
                "started": self.started_rounds, "finished": self.finished_rounds,
                "answers": self.answered, "timeouts": self.timed_out}


def engine_from_context(ctx: Any, *, clock: Callable[[], float] = time.time,
                        audit: Any = None) -> QuizEngine:
    """Собрать движок из контекста скилла (F-407: ``state`` и ``scheduler``).

    Нет состояния — движок тоже поднимается, но первая же запись честно
    откажет: подменять хранилище словарём в памяти значило бы обещать игру,
    которая не переживёт перезапуск.
    """
    store = getattr(ctx, "state", None)
    if store is None:
        raise QuizError("this hub cannot keep skill state")
    generator = getattr(ctx, "quiz_generator", None)
    if generator is None:
        generator = LlmQuizGenerator(getattr(ctx, "model", None))
    return QuizEngine(store=store, generator=generator,
                      settings=getattr(ctx, "game_settings", None),
                      scheduler=getattr(ctx, "scheduler", None), clock=clock,
                      home_name=getattr(ctx, "home_name", None), audit=audit)


__all__ = [
    "DEFAULT_QUESTIONS",
    "DEFAULT_WINDOW_S",
    "LANGUAGES",
    "LlmQuizGenerator",
    "QUESTIONS_SCHEMA",
    "QuizAnswer",
    "QuizClose",
    "QuizEngine",
    "QuizError",
    "QuizQuestion",
    "QuizRound",
    "QuizUnavailable",
    "STATE_LAST",
    "STATE_ROUND",
    "answers_match",
    "busy_line",
    "correct_line",
    "engine_from_context",
    "finish_line",
    "is_quiz_start",
    "is_score_request",
    "is_stop",
    "late_line",
    "no_round_line",
    "normalize_answer",
    "question_line",
    "questions_from_payload",
    "quiz_messages",
    "quiz_topic",
    "score_line",
    "scores_text",
    "start_line",
    "topic_missing_line",
    "unavailable_line",
    "wrong_line",
]
