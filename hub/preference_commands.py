"""Изменение настроек человека голосом (ТЗ F-607).

Персонализация F-607 — это четыре настройки (язык, голос ответа, wake-фраза,
стиль) и любимые сцены. Этот модуль отвечает на один вопрос: «просит ли эта
фраза что-то изменить, и на что именно». Он ничего не пишет: слова человека
превращаются в строгие :class:`PreferenceChange`, а применяет их хаб
(``hub/app.py::Connection._preference_turn``), записывая каждое реальное
изменение в аудит (F-706).

Правила разбора намеренно узкие: команда начинается с глагола («отвечай /
говори / speak / responde») или с явного имени настройки, поэтому обычная
болтовня («расскажи шутку») настройку не трогает. Незнакомое слово не
угадывается: если фраза не разобрана целиком, настройка не меняется.

Темп речи («говори медленнее») в ТЗ F-607 не назван: это настройка движка
дома, а не человека, и модуль честно говорит об этом, а не делает вид, что
запомнил. Выбор «по умолчанию» — в ``DECISIONS.md`` (P4-28).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Поля, которые правда живут в ``person_preferences`` (F-607).
FIELDS = ("language", "style", "voice", "wake_word")

#: Слово человека → код языка (whisper-код, как в ``hub.languages``).
_LANG_WORDS: tuple[tuple[str, str], ...] = (
    ("en", r"англ\w*|english|ingl[eé]s"),
    ("ru", r"русск\w*|russian|ruso"),
    ("es", r"испанск\w*|spanish|espa[nñ]ol"),
)

#: Слово человека → имя стиля (словарь ``person_preferences.STYLES``).
_STYLE_WORDS: tuple[tuple[str, str], ...] = (
    ("brief", r"кратко|коротко|покороче|brief(?:ly)?|breve|keep (?:it|your answers) short"),
    ("formal", r"формально|вежливо|formal(?:mente)?|amable"),
    ("playful", r"игриво|шутливо|с юмором|подшучивай|playful(?:ly)?|juguet[oó]n"
                r"|con humor|joke around"),
    ("default", r"(?:обычный|нормальный|стандартный|default|normal|usual) стиль"
                r"|без стиля|сбрось стиль|estilo (?:normal|por defecto)"),
)


def _lang_word(pattern: str) -> str | None:
    """The code for one language word, or ``None``."""
    for code, words in _LANG_WORDS:
        if re.fullmatch(words, pattern.strip(), re.IGNORECASE):
            return code
    return None


def _style_word(pattern: str) -> str | None:
    """The style name for one style word, or ``None``."""
    for style, words in _STYLE_WORDS:
        if re.fullmatch(words, pattern.strip(), re.IGNORECASE):
            return style
    return None


_LANG_ALTERNATION = "|".join(f"(?:{words})" for _code, words in _LANG_WORDS)
_STYLE_ALTERNATION = "|".join(f"(?:{words})" for _style, words in _STYLE_WORDS)

#: «отвечай по-английски», «speak English», «responde en español».
_LANGUAGE = re.compile(
    r"(?:отвечай|говори|общайся|разговаривай|пиши)\s+(?:мне\s+)?(?:на|по)[- ]?"
    r"(?P<ru>" + _LANG_ALTERNATION + r")"
    r"|(?:answer|reply|speak|talk|respond|write)\s+(?:to me\s+)?in\s+(?P<en>"
    + _LANG_ALTERNATION + r")"
    r"|(?:answer|reply|speak|talk|respond)\s+(?P<en2>english|russian|spanish)"
    r"|(?:responde|contesta|habla|escr[ií]beme)\s+en\s+(?P<es>" + _LANG_ALTERNATION + r")",
    re.IGNORECASE)

#: «говори кратко», «be formal», «estilo normal».
_STYLE = re.compile(
    r"(?:отвечай|говори|пиши|будь|be|answer|reply|speak|say it|s[eé]|habla)"
    r"\s+(?:мне\s+)?(?P<verb>" + _STYLE_ALTERNATION + r")"
    r"|^(?P<bare>" + _STYLE_ALTERNATION + r")$",
    re.IGNORECASE)

#: «говори голосом X», «use voice X», «habla con la voz X».
_VOICE = re.compile(
    r"(?:говори|отвечай|разговаривай)\s+(?:мне\s+)?голосом\s+(?P<ru>.+)"
    r"|смени\s+(?:свой\s+)?голос\s+на\s+(?P<ru2>.+)"
    r"|(?:use|speak|talk|reply|answer)\s+(?:with\s+)?(?:the\s+)?voice\s+(?P<en>.+)"
    r"|change\s+(?:your\s+)?voice\s+to\s+(?P<en2>.+)"
    r"|habla\s+con\s+la\s+voz\s+(?P<es>.+)",
    re.IGNORECASE)

#: «просыпайся на слово X», «wake word X», «palabra de activación X».
_WAKE = re.compile(
    r"просыпайся\s+на\s+(?:слово\s+)?(?P<ru>.+)"
    r"|реагируй\s+на\s+(?:слово\s+)?(?P<ru2>.+)"
    r"|wake[\s-]?word[:\s]+(?P<en>.+)"
    r"|palabra\s+de\s+activaci[oó]n[:\s]+(?P<es>.+)",
    re.IGNORECASE)


@dataclass(frozen=True)
class PreferenceChange:
    """One setting the person asked to change, in the hub's own vocabulary."""

    field: str
    value: str


def _clean_fragment(value: Any, *, cut: bool = False) -> str:
    """A spoken value with quotes and trailing punctuation taken off.

    ``cut`` also stops at the end of the first clause ("говори голосом ru_1 и
    говори кратко" names the voice ``ru_1``), because a voice name or a wake
    phrase is a short fragment, not the rest of the sentence.
    """
    text = " ".join(str(value or "").split())
    if cut:
        text = re.split(r"[.,!?;]|\s+(?:и|and|y)\s+", text, maxsplit=1)[0]
    return text.strip().strip("«»\"'„“”").strip().strip(" .!?,").strip()


def _first_group(match: re.Match[str]) -> str:
    for value in match.groupdict().values():
        if value:
            return value
    return ""


def preference_changes(text: Any) -> list[PreferenceChange]:
    """Every setting this utterance asks to change (empty list: none).

    A phrase can ask for more than one thing in one breath
    («отвечай по-английски и кратко»): each rule is tried, and the same field
    is only reported once.
    """
    said = " ".join(str(text or "").split())
    if not said:
        return []
    found: dict[str, PreferenceChange] = {}

    match = _LANGUAGE.search(said)
    if match:
        code = _lang_word(_first_group(match))
        if code:
            found.setdefault("language", PreferenceChange("language", code))

    match = _STYLE.search(said)
    if match:
        style = _style_word(_first_group(match))
        if style:
            found.setdefault("style", PreferenceChange("style", style))

    match = _VOICE.search(said)
    if match:
        name = _clean_fragment(_first_group(match), cut=True)
        if name:
            found.setdefault("voice", PreferenceChange("voice", name))

    match = _WAKE.search(said)
    if match:
        phrase = _clean_fragment(_first_group(match), cut=True)
        if phrase:
            found.setdefault("wake_word", PreferenceChange("wake_word", phrase))

    return [found[field] for field in FIELDS if field in found]


#: «говори медленнее» — темпа речи среди настроек человека в ТЗ нет.
_SPEED = re.compile(
    r"\b(?:говори|отвечай)\s+(?:мне\s+)?(?:медленнее|быстрее|помедленнее|потише|погромче)\b"
    r"|\b(?:speak|talk|answer)\s+(?:more\s+)?(?:slowly|faster)\b"
    r"|\bhabla\s+m[aá]s\s+(?:despacio|r[aá]pido)\b",
    re.IGNORECASE)


def speed_request(text: Any) -> bool:
    """True when the person asked for a speech RATE, which F-607 does not set."""
    return bool(_SPEED.search(" ".join(str(text or "").split())))


# --- what the hub says back -------------------------------------------------

#: How one language sounds in another ("по-английски" / "in English").
LANGUAGE_IN: dict[str, dict[str, str]] = {
    "ru": {"ru": "по-русски", "en": "по-английски", "es": "по-испански"},
    "en": {"ru": "in Russian", "en": "in English", "es": "in Spanish"},
    "es": {"ru": "en ruso", "en": "en inglés", "es": "en español"},
}

_LINES: dict[str, dict[str, str]] = {
    "ru": {
        "language": "Хорошо, буду отвечать {language}.",
        "style": "Хорошо, {style}.",
        "voice": "Хорошо, буду говорить голосом «{voice}».",
        "wake_word": "Хорошо, буду просыпаться на «{wake_word}».",
        "style_brief": "буду отвечать кратко",
        "style_formal": "буду отвечать формально и вежливо",
        "style_playful": "буду отвечать с юмором",
        "style_default": "вернусь к обычному стилю",
        "unknown_voice": "Сначала мне нужно узнать ваш голос: настройки принадлежат человеку.",
        "foreign": "В этом доме я не читаю ваш профиль: вы не разрешили делиться им. "
                   "Скажите «разреши узнавать меня в других домах», и настройки поедут с вами.",
        "not_saved": "Не получилось сохранить настройку, поэтому ничего не изменилось.",
        "speed": "Темп речи я на человека пока не настраиваю — это настройка дома. "
                 "Зато могу отвечать короче («говори кратко»), сменить язык, голос "
                 "или фразу пробуждения.",
    },
    "en": {
        "language": "Okay, I will answer {language}.",
        "style": "Okay, {style}.",
        "voice": "Okay, I will use the voice {voice}.",
        "wake_word": "Okay, I will wake on {wake_word}.",
        "style_brief": "I will keep my answers brief",
        "style_formal": "I will be formal and polite",
        "style_playful": "I will keep it playful",
        "style_default": "I will go back to the usual style",
        "unknown_voice": "I need to recognize your voice first: these settings belong to a person.",
        "foreign": "I do not read your profile in this room - you have not allowed it to be "
                   "shared. Say “share my identity with other homes”, and your settings travel.",
        "not_saved": "I could not save that setting, so nothing changed.",
        "speed": "I do not set the speech rate per person yet - that is the room's setting. "
                 "I can answer briefly (“be brief”), or change your language, voice or "
                 "wake phrase.",
    },
    "es": {
        "language": "De acuerdo, responderé {language}.",
        "style": "De acuerdo, {style}.",
        "voice": "De acuerdo, usaré la voz {voice}.",
        "wake_word": "De acuerdo, me despertaré con {wake_word}.",
        "style_brief": "responderé breve",
        "style_formal": "responderé formal y amable",
        "style_playful": "responderé con humor",
        "style_default": "volveré al estilo normal",
        "unknown_voice": "Primero necesito reconocer tu voz: los ajustes son de una persona.",
        "foreign": "En esta habitación no leo tu perfil: no has permitido compartirlo. "
                   "Di «comparte mi identidad», y tus ajustes viajarán contigo.",
        "not_saved": "No pude guardar el ajuste, así que nada cambió.",
        "speed": "Todavía no ajusto la velocidad por persona: es un ajuste de la habitación. "
                 "Puedo responder breve («sé breve») o cambiar tu idioma, voz o palabra "
                 "de activación.",
    },
}


def _table(language: Any) -> dict[str, str]:
    return _LINES.get(str(language or "").casefold(), _LINES["ru"])


def _field_line(change: PreferenceChange, language: str) -> str:
    table = _table(language)
    if change.field == "language":
        code = change.value if change.value in LANGUAGE_IN.get(language, {}) else change.value
        return table["language"].format(language=LANGUAGE_IN.get(language, {}).get(code, code))
    if change.field == "style":
        return table["style"].format(style=table.get(f"style_{change.value}", change.value))
    return table[change.field].format(**{change.field: change.value})


def change_lines(changes: list[PreferenceChange], language: Any = "ru") -> str:
    """The confirmation, in the person's language (their NEW one if it changed)."""
    chosen = str(language or "ru").casefold()
    for change in changes:
        if change.field == "language" and change.value in _LINES:
            chosen = change.value
            break
    return " ".join(_field_line(change, chosen) for change in changes)


def unknown_voice_line(language: Any = "ru") -> str:
    return _table(language)["unknown_voice"]


def foreign_line(language: Any = "ru") -> str:
    return _table(language)["foreign"]


def not_saved_line(language: Any = "ru") -> str:
    return _table(language)["not_saved"]


def speed_line(language: Any = "ru") -> str:
    return _table(language)["speed"]


__all__ = [
    "FIELDS",
    "LANGUAGE_IN",
    "PreferenceChange",
    "change_lines",
    "foreign_line",
    "not_saved_line",
    "preference_changes",
    "speed_line",
    "speed_request",
    "unknown_voice_line",
]
