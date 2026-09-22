"""«Покажи камеру»: кадр комнаты с именами над треками (ТЗ F-708).

ТЗ F-708: "имена над треками по запросу «покажи камеру»". Значит нужно ровно
две вещи, и обе — данные, а не догадки:

* понять, что человек ПОПРОСИЛ посмотреть камеру (ru/en/es), и не спутать это с
  просьбой посмотреть кадр для ответа («посмотри, кто там» — это
  ``look_at_camera``, F-314);
* собрать подписи над треками из того, что хаб ЗНАЕТ: имя — из ``tracks``
  комнаты (F-201/F-207), прямоугольник — оттуда же. Незнакомец получает
  подпись без имени, а не выдуманное «человек 2».
"""
from __future__ import annotations

from typing import Any

#: Фразы «покажи камеру» на трёх языках пользователя (ТЗ 1: en/ru/es).
_PHRASES = (
    "покажи камеру", "покажи мне камеру", "включи камеру на экране",
    "что видно в комнате", "покажи комнату", "покажи, что видно",
    "show me the camera", "show the room", "show camera", "what do you see",
    "show me what you see", "muéstrame la cámara", "muestra la cámara",
    "enséñame la cámara", "qué ves",
)
#: Что человека НЕ надо понимать как «покажи камеру»: это вопрос о содержимом
#: кадра, и на него отвечает зрение (F-314), а не экран.
_NOT_A_REQUEST = (
    "кто там", "кто это", "сколько людей", "что за", "прочитай",
    "who is", "who's there", "how many people", "read the", "quién está",
    "cuántas personas",
)
#: Ответы хаба на трёх языках пользователя (ТЗ 1). Фраза короткая: экран уже
#: показывает кадр, слова только подтверждают, что просьба выполнена.
_SHOWING = {'ru': "Показываю камеру.", 'en': "Showing the camera.", 'es': "Mostrando la cámara."}
_NO_CAMERA = {'ru': "Камеры нет или она не отвечает.",
              'en': "There is no camera, or it is not answering.",
              'es': "No hay cámara, o no responde."}
_TITLE = {'ru': "Камера комнаты", 'en': "Room camera", 'es': "Cámara de la habitación"}


def language_of(value: Any) -> str:
    """Map a language name/prefix to ru/en/es (Spanish and English otherwise)."""
    text = str(value or "").strip().lower()
    if text.startswith(("ru", "рус")):
        return "ru"
    if text.startswith(("es", "spa", "esp")):
        return "es"
    return "en"


def showing_text(language: Any = "") -> str:
    return _SHOWING[language_of(language)]


def no_camera_text(language: Any = "") -> str:
    return _NO_CAMERA[language_of(language)]


def title_text(language: Any = "") -> str:
    return _TITLE[language_of(language)]


def show_camera_requested(text: Any, language: str = "") -> bool:
    """Did the person ask to SEE the room camera on the screen (ТЗ F-708)?"""
    phrase = " ".join(str(text or "").lower().replace("ё", "е").split())
    if not phrase or len(phrase) > 200:
        return False
    if any(blocked in phrase for blocked in _NOT_A_REQUEST):
        return False
    return any(asked in phrase for asked in _PHRASES)


def _box_of(row: Any) -> list[float] | None:
    """The normalized ``[x1, y1, x2, y2]`` of one track row, or ``None``."""
    if not isinstance(row, dict):
        return None
    box = row.get("box", row.get("bbox"))
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        values = [float(value) for value in box[:4]]
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _track_id_of(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("id", row.get("track_id")) or "")


def track_labels(tracks: Any, names: Any = None) -> list[dict[str, Any]]:
    """Names over the boxes of ONE image, for the room HUD (ТЗ F-708).

    Geometry and names come from different places on purpose:

    * ``tracks`` are the tracks that arrived WITH the image being shown
      (``{"id"/"track_id", "box"/"bbox"}``) — labels on an older frame would
      point at people who have since walked away;
    * ``names`` is what the hub already knows about the room's tracks
      (``{"<track id>": "Макс"}``, or rows carrying ``name``); a track nobody
      was identified in gets an EMPTY label, because the screen must not
      invent a person to fill the gap.
    """
    if isinstance(tracks, dict):
        rows = list(tracks.values()) if tracks and "box" not in tracks else [tracks]
    elif isinstance(tracks, (list, tuple)):
        rows = list(tracks)
    else:
        rows = []
    known: dict[str, str] = {}
    if isinstance(names, dict):
        for key, value in names.items():
            text = value.get("name") if isinstance(value, dict) else value
            if text:
                known[str(key)] = str(text)
    labels: list[dict[str, Any]] = []
    for row in rows:
        box = _box_of(row)
        if box is None:
            continue
        track_id = _track_id_of(row)
        name = str(row.get("name") or "") if isinstance(row, dict) else ""
        labels.append({"name": name or known.get(track_id, ""), "box": box})
    return labels


__all__ = ["language_of", "no_camera_text", "show_camera_requested", "showing_text",
           "title_text", "track_labels"]
