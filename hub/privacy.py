"""Приватность камеры: слова и подтверждения (ТЗ F-303).

«Rowan, перестань смотреть» — это просьба о приватности, а не просьба о
тишине: микрофон остаётся включённым, иначе камеру нельзя было бы вернуть тем
же голосом («смотри снова»). Поэтому здесь только разбор фразы и текст ответа
на трёх языках; сам режим живёт на клиенте (`client/privacy.py`), потому что
кадры уходят именно с него.

Понимание фразы осторожное: «не смотри на меня» — это тоже выключение, а
«посмотри, кто там» — обычная просьба посмотреть кадр, и она приватность не
трогает.
"""
from __future__ import annotations

import re
from typing import Any

ON_TEXT: dict[str, str] = {
    "ru": "Хорошо, камера выключена. Кадры больше не уходят; скажи «смотри снова», когда захочешь вернуть.",
    "en": "Alright, the camera is off. No more frames are sent; say \"watch again\" to bring it back.",
    "es": "De acuerdo, la cámara está apagada. No se envían más imágenes; di \"mira otra vez\" para volver.",
}
OFF_TEXT: dict[str, str] = {
    "ru": "Хорошо, камера снова смотрит.",
    "en": "Alright, the camera is watching again.",
    "es": "De acuerdo, la cámara vuelve a mirar.",
}
ALREADY_OFF: dict[str, str] = {
    "ru": "Камера уже выключена — кадры и так не уходят.",
    "en": "The camera is already off - no frames are being sent.",
    "es": "La cámara ya está apagada: no se envían imágenes.",
}
ALREADY_ON: dict[str, str] = {
    "ru": "Камера и так включена.",
    "en": "The camera is already on.",
    "es": "La cámara ya está encendida.",
}

#: Просьба перестать смотреть (privacy on).
_OFF = re.compile(
    r"\b(?:перестань|хватит|прекрати)\s+(?:меня\s+)?(?:смотреть|наблюдать|подглядывать)"
    r"|\bне\s+(?:смотри|подглядывай|наблюдай)\b"
    r"|\bвыключи\s+(?:камеру|видео|наблюдение)\b"
    r"|\bотключи\s+(?:камеру|видео|наблюдение)\b"
    r"|\b(?:stop|quit)\s+(?:watching|looking|spying)\b"
    r"|\bdon'?t\s+(?:watch|look at)\s+me\b"
    r"|\bturn\s+(?:the\s+)?camera\s+off\b"
    r"|\bdeja\s+de\s+(?:mirar|observar|espiar)\b"
    r"|\bno\s+me\s+(?:mires|observes)\b"
    r"|\bapaga\s+(?:la\s+)?c[áa]mara\b", re.IGNORECASE)

#: Просьба смотреть снова (privacy off).
_ON = re.compile(
    r"\bсмотри\s+(?:снова|опять)\b"
    r"|\bвключи\s+(?:камеру|видео|наблюдение)\b"
    r"|\bможешь\s+(?:снова\s+)?смотреть\b"
    r"|\b(?:watch|look)\s+again\b"
    r"|\bturn\s+(?:the\s+)?camera\s+(?:on|back on)\b"
    r"|\bmira\s+(?:otra\s+vez|de\s+nuevo)\b"
    r"|\benciende\s+(?:la\s+)?c[áa]mara\b", re.IGNORECASE)

#: Обычная просьба посмотреть кадр — это НЕ возврат камеры.
_LOOK_REQUEST = re.compile(
    r"\b(?:посмотри|взгляни|глянь)\b|\blook\s+at\b|\bwhat\s+do\s+you\s+see\b"
    r"|\bmira\s+(?:la|el|qu[ée]|a\s+)", re.IGNORECASE)


def language_of(value: Any, *, default: str = "ru") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in ON_TEXT else default


def privacy_command(text: Any) -> bool | None:
    """``True`` = stop watching, ``False`` = watch again, ``None`` = not this."""
    said = " ".join(str(text or "").split())
    if not said:
        return None
    if _OFF.search(said):
        return True
    if _ON.search(said):
        return False
    return None


def looks_like_look_request(text: Any) -> bool:
    """A request for one picture ("look at the room"), not a privacy command."""
    return bool(_LOOK_REQUEST.search(" ".join(str(text or "").split())))


def confirmation(wanted: bool, language: Any = "ru", *, changed: bool = True) -> str:
    """What the room hears after the request (never a silent switch)."""
    code = language_of(language)
    if not changed:
        return ALREADY_OFF[code] if wanted else ALREADY_ON[code]
    return ON_TEXT[code] if wanted else OFF_TEXT[code]


__all__ = [
    "ALREADY_OFF",
    "ALREADY_ON",
    "OFF_TEXT",
    "ON_TEXT",
    "confirmation",
    "language_of",
    "looks_like_look_request",
    "privacy_command",
]
