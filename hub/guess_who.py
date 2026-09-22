"""«Угадай, кто сказал» (ТЗ F-608): игра по голосам, только с согласия.

ТЗ F-608: «Игры между комнатами. Квиз по темам …, «угадай, кто сказал» (по
голосам, с согласия), таймер-соревнования. Каркас игры — скилл с состоянием
(F-407)». Квиз живёт в :mod:`hub.games` (P5-20); здесь — вторая игра и
таймер-соревнование.

Голос — это персональные данные, поэтому партия начинается ТОЛЬКО когда
человек, чью фразу записали, разрешил это (:class:`hub.voice_clone.ConsentStore`
со своим файлом): согласие даёт человек, а не дом, и оно переживает
перезапуск хаба. Сама запись — настоящий WAV в медиах хаба (kind ``audio``,
TTL F-304), а не след в памяти: эту фразу должны услышать другие комнаты.

Партия живёт в состоянии скилла ``games`` (ТЗ F-407, ``hub:games``, ключ
``mystery``) — как и квиз, потому что игра идёт между комнатами. Очко забирает
самая быстрая комната: ценность ответа считается по остатку времени
(:func:`speed_points`), а таймер поднимает ``SkillScheduler`` (F-407). Комната,
которая эту фразу и сказала, очка не получает — она знает ответ.
"""
from __future__ import annotations

import io
import logging
import math
import re
import time
import wave
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common.ids import new_ulid
from hub.games import answers_match
from hub.skill_state import SkillSchedulerError, SkillStateError, SkillStateStore

log = logging.getLogger(__name__)

#: Ключ состояния, под которым живёт открытая партия «кто сказал» (F-407).
STATE_MYSTERY = "mystery"
#: Последняя закрытая партия — чтобы показать, чем кончилось.
STATE_LAST = "last_mystery"
#: Сколько секунд по умолчанию даётся на догадки.
DEFAULT_WINDOW_S = 20.0
#: Сколько очков получает самая быстрая комната.
DEFAULT_POINTS = 5
#: Языки, на которых хаб говорит (раздел 1 ТЗ).
LANGUAGES = ("ru", "en", "es")
#: Короче этого фразу-загадку не расслышать, длиннее — уже разговор.
MIN_PHRASE_S = 0.5
MAX_PHRASE_S = 20.0
#: Сколько секунд ждать саму фразу после «угадай, кто сказал».
ARM_WINDOW_S = 60.0


class GuessUnavailable(RuntimeError):
    """Партию начать нельзя: нет согласия, записи или узнанного человека."""


class GuessError(RuntimeError):
    """Партию нельзя начать или продолжить."""


# ---------------------------------------------------------------------------
# запись
# ---------------------------------------------------------------------------


def pcm_to_wav(pcm: bytes, *, sample_rate: int, channels: int = 1, width: int = 2) -> bytes:
    """PCM16 комнаты → настоящий WAV: его читает и хаб, и человек на стенде."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(max(1, int(channels)))
        handle.setsampwidth(max(1, int(width)))
        handle.setframerate(max(1, int(sample_rate)))
        handle.writeframes(bytes(pcm))
    return buffer.getvalue()


def wav_pcm(data: bytes) -> tuple[bytes, int]:
    """PCM и частота из настоящего WAV; битый файл — названная ошибка."""
    try:
        with wave.open(io.BytesIO(bytes(data)), "rb") as handle:
            rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            frames = handle.readframes(handle.getnframes())
    except Exception as exc:  # noqa: BLE001 - битая запись не «тишина»
        raise ValueError(f"the recording is not a readable wav ({type(exc).__name__})") from exc
    if not frames:
        raise ValueError("the recording holds no audio")
    if channels != 1 or width != 2:
        raise ValueError("the recording is not mono pcm16")
    return frames, rate


def wav_seconds(data: bytes) -> float:
    """Длительность настоящего WAV в секундах."""
    frames, rate = wav_pcm(data)
    return len(frames) / (2.0 * max(1, rate))


class AudioStore(Protocol):
    """Что нужно от медиаха хаба: положить запись и получить её ссылку (F-304)."""

    def save_bytes(self, home_id: str, kind: str, data: bytes, *,
                   filename: str | None = None, ts: float | None = None
                   ) -> tuple[str, Any]: ...


# ---------------------------------------------------------------------------
# ответы и очки
# ---------------------------------------------------------------------------


def speed_points(remaining_s: float, window_s: float, max_points: int) -> int:
    """Ценность ответа по остатку времени: самая быстрая комната забирает максимум.

    Минимум — одно очко: верный ответ всегда что-то значит, а строка «0 очков»
    была бы отказом считать человека.
    """
    top = max(1, int(max_points))
    limit = float(window_s)
    if limit <= 0:
        return top
    share = max(0.0, min(1.0, float(remaining_s) / limit))
    return max(1, int(math.ceil(top * share)))


def name_variants(display_name: Any) -> list[str]:
    """Написания имени человека для сравнения: полное и короткое («Максим», «Макс»?).

    Коротким считается первое слово полного имени; выдумывать уменьшительные
    («Максим» → «Макс») здесь нельзя — такое правило само придумывало бы ответ.
    """
    full = " ".join(str(display_name or "").split())
    if not full:
        return []
    variants = [full]
    first = full.split()[0]
    if first.casefold() != full.casefold():
        variants.append(first)
    return variants


# ---------------------------------------------------------------------------
# намерения
# ---------------------------------------------------------------------------


_START = re.compile(
    r"(?i)\b(?:угада\w*|угадк\w*|сыгра\w*|игра\w*|давай\w*|поигра\w*|"
    r"guess\w*|play\w*|let'?s|adivina\w*|juguemos|jugar)\b"
    r"[^.!?]{0,30}?"
    r"\b(?:кто\s+(?:это\s+)?сказал\w*|кто\s+говори\w*|who\s+(?:said|is|talked)\w*|"
    r"qui[ée]n\s+(?:lo\s+)?(?:dijo|habl\w*))\b")
_START_REVERSED = re.compile(
    r"(?i)\b(?:кто\s+(?:это\s+)?сказал\w*|who\s+(?:said|is)\s+it|qui[ée]n\s+(?:lo\s+)?dijo)\b"
    r"[^.!?]{0,20}?\b(?:угада\w*|guess\w*|adivina\w*)\b")
_CONSENT_WORDS = re.compile(
    r"(?i)\b(?:разреша\w*|разреши\w*|можно|позволя\w*|allow\w*|may\s+use|puedes|permiso)\b")
_CONSENT_VOICE = re.compile(
    r"(?i)\b(?:голос\w*|voice|voz)\b")
_CONSENT_GAME = re.compile(
    r"(?i)\b(?:игр\w*|game\w*|juego\w*|jugar)\b")


def is_guess_start(text: Any) -> bool:
    """«Угадай, кто сказал» — просьба начать игру, а не догадка."""
    raw = " ".join(str(text or "").split())
    if not raw:
        return False
    return bool(_START.search(raw) or _START_REVERSED.search(raw))


def is_voice_consent(text: Any) -> bool:
    """«Разрешаю использовать мой голос в игре» — согласие, сказанное человеком."""
    raw = " ".join(str(text or "").split())
    if not raw:
        return False
    return bool(_CONSENT_WORDS.search(raw) and _CONSENT_VOICE.search(raw)
                and _CONSENT_GAME.search(raw))


# ---------------------------------------------------------------------------
# строки для комнаты
# ---------------------------------------------------------------------------


def _language_of(language: Any) -> str:
    code = str(language or "")[:2].casefold()
    return code if code in LANGUAGES else "ru"


def arm_line(*, language: Any = "ru") -> str:
    """Просьба сказать фразу, которую будут угадывать."""
    if _language_of(language) == "ru":
        return ("Скажи короткую фразу — я дам остальным угадать, кто это сказал. "
                "Только если ты разрешил игру своим голосом.")
    if _language_of(language) == "es":
        return ("Di una frase corta y dejaré que las demás habitaciones adivinen "
                "quién lo dijo. Solo si has permitido usar tu voz en el juego.")
    return ("Say a short phrase and I will let the other rooms guess who said it. "
            "Only if you have allowed your voice in the game.")


def consent_recorded_line(name: Any, *, language: Any = "ru") -> str:
    who = " ".join(str(name or "").split())
    if _language_of(language) == "ru":
        return f"Записала: {who or 'ты'} разрешил(а) игру своим голосом."
    if _language_of(language) == "es":
        return f"Anotado: {who or 'tú'} permite usar su voz en el juego."
    return f"Recorded: {who or 'you'} allows their voice in the game."


def consent_missing_line(name: Any, *, language: Any = "ru") -> str:
    who = " ".join(str(name or "").split())
    if _language_of(language) == "ru":
        return (f"{who or 'Этот человек'} не разрешал(а) игру своим голосом. "
                f"Пусть скажет: «разрешаю использовать мой голос в игре».")
    if _language_of(language) == "es":
        return (f"{who or 'Esa persona'} no ha permitido el juego con su voz. "
                "Que diga: «puedes usar mi voz en los juegos».")
    return (f"{who or 'That person'} has not allowed the game with their voice. "
            "They can say: «you may use my voice in games».")


def unknown_speaker_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Не могу узнать твой голос, а загадка — это голос. Скажи фразу ещё раз."
    if _language_of(language) == "es":
        return "No reconozco tu voz, y el juego va de voces. Dime la frase otra vez."
    return "I cannot recognise your voice, and the game is about voices. Say it again."


def too_short_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Слишком коротко — скажи фразу чуть длиннее, я её записываю."
    if _language_of(language) == "es":
        return "Demasiado corto: di una frase un poco más larga."
    return "That was too short — say a slightly longer phrase."


def prompt_line(round_: GuessRound, *, language: Any = "ru") -> str:
    """Что слышат угадывающие комнаты: вопрос, срок и цена скорости."""
    seconds = f"{round_.window_s:g}"
    if _language_of(language) == "ru":
        return (f"Чья это была фраза? Отвечайте голосом: {seconds} секунд, "
                f"и чем быстрее, тем больше очков (до {round_.points}).")
    if _language_of(language) == "es":
        return (f"¿Quién lo dijo? Responded en voz alta: {seconds} segundos y "
                f"cuanto más rápido, más puntos (hasta {round_.points}).")
    return (f"Whose voice was that? Answer out loud: {seconds} seconds, and the "
            f"faster you are the more points you get (up to {round_.points}).")


def speaker_line(round_: GuessRound, *, language: Any = "ru") -> str:
    """Что слышит комната, которая эту фразу и сказала."""
    if _language_of(language) == "ru":
        return "Фраза записана. Остальные угадывают, кто это сказал — твоя комната очков не получает."
    if _language_of(language) == "es":
        return "Frase grabada. Las demás habitaciones adivinan quién lo dijo; la tuya no puntúa."
    return "The phrase is recorded. The other rooms are guessing; your room does not score."


def correct_line(round_: GuessRound, home_id: str, points: int, *, language: Any = "ru",
                 home_name: Callable[[str], str] | None = None) -> str:
    name_of = home_name or (lambda home: home)
    home = name_of(home_id)
    if _language_of(language) == "ru":
        return f"Верно: это был(а) {round_.speaker_name}. {points} очк(о/а) комнате {home}."
    if _language_of(language) == "es":
        return f"Correcto: era {round_.speaker_name}. {points} puntos para {home}."
    return f"Correct: it was {round_.speaker_name}. {points} points for {home}."


def wrong_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Не он. Пробуйте дальше — время ещё идёт."
    if _language_of(language) == "es":
        return "No es esa persona. Seguid probando, aún hay tiempo."
    return "Not that person. Keep guessing while there is time."


def already_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Эта комната уже угадала."
    if _language_of(language) == "es":
        return "Esta habitación ya acertó."
    return "This room has already guessed it."


def spoiler_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Это же твоя фраза — очко за неё не считается."
    if _language_of(language) == "es":
        return "Esa frase es tuya: no cuenta punto."
    return "That phrase was yours — it does not score."


def answer_line(round_: GuessRound, *, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return f"Время вышло. Это был(а) {round_.speaker_name}."
    if _language_of(language) == "es":
        return f"Se acabó el tiempo. Era {round_.speaker_name}."
    return f"The time is up. It was {round_.speaker_name}."


def scores_text(round_: GuessRound, *, home_name: Callable[[str], str] | None = None) -> str:
    name_of = home_name or (lambda home: home)
    return ", ".join(f"{name_of(home)} — {points}"
                     for home, points in sorted(round_.scores.items(), key=lambda item: -item[1]))


def finish_line(round_: GuessRound, *, language: Any = "ru",
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


def score_line(round_: GuessRound, *, language: Any = "ru",
               home_name: Callable[[str], str] | None = None) -> str:
    score = scores_text(round_, home_name=home_name)
    if _language_of(language) == "ru":
        return f"Счёт: {score}." if score else "Пока никто не угадал."
    if _language_of(language) == "es":
        return f"Puntaje: {score}." if score else "Todavía nadie acertó."
    return f"Score: {score}." if score else "Nobody has guessed yet."


def busy_line(round_: GuessRound, *, language: Any = "ru",
              home_name: Callable[[str], str] | None = None) -> str:
    if _language_of(language) == "ru":
        return (f"Игра «кто сказал» уже идёт: {int(max(0.0, round_.deadline - time.time()))} "
                f"секунд. {score_line(round_, language=language, home_name=home_name)}")
    if _language_of(language) == "es":
        return ("Ya está en marcha «quién lo dijo». "
                + score_line(round_, language=language, home_name=home_name))
    return ("A «who said it» round is already running. "
            + score_line(round_, language=language, home_name=home_name))


def unavailable_line(reason: str, *, language: Any = "ru") -> str:
    reason = " ".join(str(reason or "").split()) or "the reason is unknown"
    if _language_of(language) == "ru":
        return f"Игру начать не могу: {reason}."
    if _language_of(language) == "es":
        return f"No puedo empezar la partida: {reason}."
    return f"I cannot start the game: {reason}."


# ---------------------------------------------------------------------------
# партия
# ---------------------------------------------------------------------------


class GuessAttempt(BaseModel):
    """Одна попытка комнаты. Текст догадки НЕ хранится: это чужие слова."""

    model_config = ConfigDict(extra="forbid")

    home_id: str = ""
    correct: bool = False
    points: int = 0
    at: float = 0.0


class GuessRound(BaseModel):
    """Партия «угадай, кто сказал»: голос, дома-участники, счёт и таймер (F-608)."""

    model_config = ConfigDict(extra="forbid")

    round_id: str = Field(default_factory=new_ulid)
    language: str = "ru"
    home_ids: list[str] = Field(default_factory=list)
    speaker_home: str = ""
    speaker_id: str = ""
    speaker_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    audio_ref: str = ""
    audio_s: float = 0.0
    truncated: bool = False
    started_at: float = 0.0
    deadline: float = 0.0
    window_s: float = DEFAULT_WINDOW_S
    points: int = DEFAULT_POINTS
    scores: dict[str, int] = Field(default_factory=dict)
    guesses: list[GuessAttempt] = Field(default_factory=list)
    timer_id: str = ""
    open: bool = True
    finished_at: float = 0.0

    def points_of(self, home_id: str) -> int:
        return int(self.scores.get(str(home_id or ""), 0))

    def remaining_s(self, now: float) -> float:
        return max(0.0, float(self.deadline) - float(now)) if self.deadline else 0.0


class GuessVerdict(BaseModel):
    """Что ответила комната и что хаб говорит после этого."""

    model_config = ConfigDict(extra="forbid")

    home_id: str = ""
    correct: bool = False
    late: bool = False
    already: bool = False
    spoiler: bool = False
    points: int = 0
    scores: dict[str, int] = Field(default_factory=dict)
    finished: bool = False
    line: str = ""
    next_line: str = ""


class GuessClose(BaseModel):
    """Закрытая партия: кто это был и с каким счётом."""

    model_config = ConfigDict(extra="forbid")

    round: GuessRound
    answer_line: str = ""
    score_line: str = ""
    line: str = ""
    winner: str = ""
    reason: str = ""


class GuessEngine:
    """«Угадай, кто сказал»: запись голоса, согласие, догадки и таймер (F-608/F-407)."""

    def __init__(self, *, store: SkillStateStore, consent: Any, settings: Any = None,
                 scheduler: Any = None, media: AudioStore | None = None,
                 clock: Callable[[], float] = time.time,
                 home_name: Callable[[str], str] | None = None,
                 audit: Any = None) -> None:
        self.store = store
        self.consent = consent
        self.settings = settings
        self.scheduler = scheduler
        self.media = media
        #: ``None`` is "use the real clock": the hub may pass an unset
        #: override, and calling ``None`` broke every round with a TypeError.
        self.clock = clock or time.time
        self.home_name = home_name or (lambda home: home)
        self.audit = audit
        #: Задаётся хабом: что сказать, когда время вышло.
        self.on_timeout: Callable[[], Any] | None = None
        self.started_rounds = 0
        self.finished_rounds = 0
        self.guessed = 0
        self.timed_out = 0
        self.refused = 0

    # --- настройки ---------------------------------------------------------

    def _setting(self, name: str, default: Any) -> Any:
        value = getattr(self.settings, name, None) if self.settings is not None else None
        return default if value in (None, "") else value

    @property
    def window_s(self) -> float:
        return max(5.0, float(self._setting("guess_window_s", DEFAULT_WINDOW_S)))

    @property
    def max_points(self) -> int:
        return max(1, int(self._setting("guess_points", DEFAULT_POINTS)))

    # --- согласие ----------------------------------------------------------

    def consent_granted(self, home_id: str, person_id: str) -> bool:
        if self.consent is None:
            return False
        try:
            return bool(self.consent.granted(str(home_id or ""), str(person_id or "")))
        except Exception as exc:  # noqa: BLE001 - согласие важнее трассировки
            log.warning("Could not read the voice-game consent (%s)", exc)
            return False

    def grant_consent(self, home_id: str, person_id: str) -> bool:
        if self.consent is None:
            return False
        try:
            return bool(self.consent.grant(home_id, person_id, at=self.clock()))
        except Exception as exc:  # noqa: BLE001 - причина важнее трассировки
            raise GuessError(f"the consent could not be saved ({type(exc).__name__})") from exc

    def revoke_consent(self, home_id: str, person_id: str) -> bool:
        if self.consent is None:
            return False
        return bool(self.consent.revoke(home_id, person_id))

    # --- состояние ---------------------------------------------------------

    def active(self) -> GuessRound | None:
        """Открытая партия или ``None``; битая запись партией не притворяется."""
        try:
            raw = self.store.get(STATE_MYSTERY)
        except SkillStateError as exc:
            log.warning("The voice-game state could not be read (%s)", exc)
            return None
        if raw is None:
            return None
        try:
            round_ = GuessRound.model_validate(raw)
        except ValidationError as exc:
            log.warning("The saved mystery state is not a round (%s)", exc)
            self._forget()
            return None
        return round_ if round_.open else None

    def _save(self, round_: GuessRound) -> None:
        try:
            self.store.set(STATE_MYSTERY, round_.model_dump(mode="json"))
        except SkillStateError as exc:
            raise GuessError(f"the round could not be saved ({exc})") from exc

    def _forget(self) -> None:
        try:
            self.store.delete(STATE_MYSTERY)
        except SkillStateError as exc:
            log.warning("The voice-game state could not be cleared (%s)", exc)

    def _note(self, action: str, round_: GuessRound, **detail: Any) -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(action=action, target=round_.round_id,
                              home_id=round_.speaker_home or None, detail=detail)
        except Exception:  # noqa: BLE001 - аудит не отменяет игру
            log.debug("Could not write the voice-game audit row", exc_info=True)

    # --- партия ------------------------------------------------------------

    def start(self, *, speaker_id: str, speaker_name: str, home_id: str,
              home_ids: Iterable[str] = (), audio_pcm: bytes, sample_rate: int,
              aliases: Sequence[str] = (), language: str = "ru") -> tuple[GuessRound, str]:
        """Записать загадку и открыть партию. Без согласия — честный отказ."""
        if self.active() is not None:
            raise GuessError("a round is already running")
        home = str(home_id or "")
        person = str(speaker_id or "")
        who = " ".join(str(speaker_name or "").split())
        if not person or not who:
            raise GuessUnavailable("nobody was recognised in this room")
        if not self.consent_granted(home, person):
            self.refused += 1
            raise GuessUnavailable(f"{who} has not allowed the voice game")
        rate = max(1, int(sample_rate))
        frames = bytes(audio_pcm or b"")
        if len(frames) < int(MIN_PHRASE_S * rate) * 2:
            raise GuessUnavailable("the phrase was too short to keep")
        truncated = False
        limit = int(MAX_PHRASE_S * rate) * 2
        if len(frames) > limit:
            frames, truncated = frames[:limit], True
        if self.media is None:
            raise GuessUnavailable("the hub cannot keep the recording")
        try:
            audio_ref, _ = self.media.save_bytes(
                home, "audio", pcm_to_wav(frames, sample_rate=rate), ts=self.clock())
        except Exception as exc:  # noqa: BLE001 - без записи игра невозможна
            raise GuessUnavailable(
                f"the recording could not be kept ({type(exc).__name__})") from exc
        now = self.clock()
        rooms = [str(room) for room in home_ids if str(room)]
        if home and home not in rooms:
            rooms.insert(0, home)
        round_ = GuessRound(
            language=_language_of(language), home_ids=rooms, speaker_home=home,
            speaker_id=person, speaker_name=who, aliases=list(aliases),
            audio_ref=audio_ref, audio_s=len(frames) / (2.0 * rate), truncated=truncated,
            started_at=now, deadline=now + self.window_s, window_s=self.window_s,
            points=self.max_points)
        self._arm(round_)
        self._save(round_)
        self.started_rounds += 1
        self._note("game.guess.start", round_, speaker=who, home=home,
                   rooms=rooms, seconds=round_.audio_s)
        return round_, prompt_line(round_, language=language)

    def guess(self, *, home_id: str, text: str, language: str = "") -> GuessVerdict | None:
        """Догадка комнаты; ``None`` — открытой партии нет."""
        round_ = self.active()
        if round_ is None:
            return None
        home = str(home_id or "")
        said = _language_of(language or round_.language)
        now = self.clock()
        if round_.deadline and now > round_.deadline:
            self.timed_out += 1
            close = self.close(reason="timeout", language=said)
            return GuessVerdict(home_id=home, late=True, scores=dict(round_.scores),
                                finished=True, line=close.line if close else "")
        if home and home == round_.speaker_home:
            return GuessVerdict(home_id=home, spoiler=True, scores=dict(round_.scores),
                                line=spoiler_line(language=said))
        if home and home in round_.scores:
            return GuessVerdict(home_id=home, already=True, scores=dict(round_.scores),
                                line=already_line(language=said))
        if not answers_match(text, round_.speaker_name, round_.aliases):
            round_.guesses.append(GuessAttempt(home_id=home, correct=False, at=now))
            self._save(round_)
            return GuessVerdict(home_id=home, scores=dict(round_.scores),
                                line=wrong_line(language=said))
        points = speed_points(round_.remaining_s(now), round_.window_s, round_.points)
        round_.scores[home] = round_.points_of(home) + points
        round_.guesses.append(GuessAttempt(home_id=home, correct=True, points=points, at=now))
        self.guessed += 1
        line = correct_line(round_, home, points, language=said, home_name=self.home_name)
        # Партия кончается, когда угадали ВСЕ, кто может: комната говорящего
        # очка не получает, поэтому её ждать нечего.
        everyone = bool(round_.home_ids) and all(
            room == round_.speaker_home or room in round_.scores for room in round_.home_ids)
        self._note("game.guess.answer", round_, home=home, points=points)
        if everyone:
            close = self.close(reason="guessed", language=said)
            return GuessVerdict(home_id=home, correct=True, points=points,
                                scores=dict(round_.scores), finished=True, line=line,
                                next_line=close.line if close else "")
        self._save(round_)
        return GuessVerdict(home_id=home, correct=True, points=points,
                            scores=dict(round_.scores), line=line)

    def close(self, *, reason: str = "timeout", language: str = "") -> GuessClose | None:
        """Закрыть партию: назвать, кто это был, и счёт. ``None`` — партии нет."""
        round_ = self.active()
        if round_ is None:
            return None
        said = _language_of(language or round_.language)
        round_.open = False
        round_.finished_at = self.clock()
        self._disarm(round_)
        answer = answer_line(round_, language=said)
        score = finish_line(round_, language=said, home_name=self.home_name)
        winner = self._winner(round_)
        self.finished_rounds += 1
        try:
            self.store.set(STATE_LAST, round_.model_dump(mode="json"))
        except SkillStateError as exc:
            log.debug("Could not keep the finished mystery round (%s)", exc)
        self._forget()
        self._note("game.guess.finish", round_, reason=reason, scores=dict(round_.scores))
        return GuessClose(round=round_, answer_line=answer, score_line=score,
                          line=" ".join(part for part in (answer, score) if part),
                          winner=winner, reason=reason)

    def finish(self, *, language: str = "", reason: str = "stop") -> str | None:
        """Закончить партию голосом; ``None`` — партии не было."""
        close = self.close(reason=reason, language=language)
        return None if close is None else close.line

    def score(self, *, language: str = "") -> str | None:
        round_ = self.active()
        if round_ is None:
            return None
        return score_line(round_, language=_language_of(language or round_.language),
                          home_name=self.home_name)

    @staticmethod
    def _winner(round_: GuessRound) -> str:
        top = max(round_.scores.values()) if round_.scores else 0
        leaders = [home for home, points in round_.scores.items() if points == top and top > 0]
        return leaders[0] if len(leaders) == 1 else ""

    # --- таймер ------------------------------------------------------------

    def _arm(self, round_: GuessRound) -> None:
        """Поднять таймер партии (F-407); нет планировщика — назвать это."""
        if self.scheduler is None:
            round_.timer_id = ""
            return
        timer_id = f"guess-{round_.round_id}"
        try:
            round_.timer_id = self.scheduler.in_(round_.window_s, self._on_timeout,
                                                 name=timer_id)
        except SkillSchedulerError as exc:
            round_.timer_id = ""
            log.info("The voice-game timer is unavailable (%s)", exc)

    def _disarm(self, round_: GuessRound) -> None:
        if not round_.timer_id or self.scheduler is None:
            round_.timer_id = ""
            return
        try:
            self.scheduler.cancel(round_.timer_id)
        except Exception as exc:  # noqa: BLE001 - отмена таймера не роняет партию
            log.debug("Could not cancel the voice-game timer (%s)", exc)
        round_.timer_id = ""

    async def _on_timeout(self) -> None:
        handler = self.on_timeout
        if handler is None:
            return
        result = handler()
        if hasattr(result, "__await__"):
            await result

    # --- снимок ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        round_ = self.active()
        return {
            "active": bool(round_),
            "speaker": round_.speaker_name if round_ else "",
            "rooms": list(round_.home_ids) if round_ else [],
            "scores": dict(round_.scores) if round_ else {},
            "started": self.started_rounds, "finished": self.finished_rounds,
            "guessed": self.guessed, "timeouts": self.timed_out, "refused": self.refused,
            "consents": self._consent_count(),
        }

    def _consent_count(self) -> int:
        if self.consent is None:
            return 0
        snapshot = getattr(self.consent, "snapshot", None)
        if not callable(snapshot):
            return 0
        try:
            return int(snapshot().get("consents", 0))
        except Exception:  # noqa: BLE001 - счёт согласий не стоит игры
            return 0


__all__ = [
    "ARM_WINDOW_S",
    "DEFAULT_POINTS",
    "DEFAULT_WINDOW_S",
    "GuessAttempt",
    "GuessClose",
    "GuessEngine",
    "GuessError",
    "GuessRound",
    "GuessUnavailable",
    "GuessVerdict",
    "LANGUAGES",
    "MAX_PHRASE_S",
    "MIN_PHRASE_S",
    "STATE_LAST",
    "STATE_MYSTERY",
    "already_line",
    "answer_line",
    "arm_line",
    "busy_line",
    "consent_missing_line",
    "consent_recorded_line",
    "correct_line",
    "finish_line",
    "is_guess_start",
    "is_voice_consent",
    "name_variants",
    "pcm_to_wav",
    "prompt_line",
    "score_line",
    "scores_text",
    "speaker_line",
    "speed_points",
    "spoiler_line",
    "too_short_line",
    "unavailable_line",
    "unknown_speaker_line",
    "wav_pcm",
    "wav_seconds",
    "wrong_line",
]
