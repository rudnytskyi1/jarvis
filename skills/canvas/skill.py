"""Canvas LMS по токену человека — сценарий 3 ТЗ (F-421, P3-30).

У Canvas токен выдают на пользователя, поэтому скилл берёт токен ТОГО, кто
спросил (``ctx.token``), а не общий: чужой учётной записью Rowan не
пользуется. Токена нет — скилл честно говорит об этом, а не отвечает пустым
списком, который выглядел бы как «ничего не задано».

Что скилл умеет (всё — чтение):

* ``week`` — что сдавать в ближайшие дни: ближайшие события Canvas, из них
  задания с настоящим сроком;
* ``todo`` — список «сделать» (Canvas ``users/self/todo``);
* ``courses`` — активные курсы человека;
* ``grades`` — текущие оценки по курсам (``include[]=total_scores``).

Сеть и чужой ответ остаются ошибками скилла: таймаут, обрыв, ``401`` и «не
данные» превращаются в ``SkillResult(ok=False)`` с причиной. Ни одной
выдуманной работы, срока или оценки (ТЗ раздел 1).
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hub.skills_runtime import SkillResult

log = logging.getLogger("jarvis.skill.canvas")

DEFAULT_TIMEOUT_S = 8.0
DEFAULT_DAYS = 7
DEFAULT_LANGUAGE = "ru"
PAGE_SIZE = 50


class Args(BaseModel):
    """Что человек просит у Canvas."""

    model_config = ConfigDict(extra="forbid")

    what: Literal["week", "todo", "courses", "grades"] = "week"
    #: Сколько дней вперёд считается «на этой неделе»; 0 — из настройки хаба.
    days: int = Field(default=0, ge=0, le=60)
    #: Курс словами или числом — сузить ответ одним курсом.
    course: str = Field(default="", max_length=120)


class CanvasError(RuntimeError):
    """Canvas не ответил так, чтобы ответу можно было верить."""


_MONTHS = {
    "ru": ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
           "августа", "сентября", "октября", "ноября", "декабря"),
    "en": ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"),
    "es": ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
           "agosto", "septiembre", "octubre", "noviembre", "diciembre"),
}
_AT = {"ru": "в", "en": "at", "es": "a las"}
_DUE = {"ru": "срок", "en": "due", "es": "entrega"}
_NO_DEADLINES = {
    "ru": "На ближайшие дни в Canvas ничего не задано.",
    "en": "Canvas shows nothing due in the next few days.",
    "es": "Canvas no muestra nada para entregar en los próximos días.",
}
_NO_TODO = {"ru": "Список «сделать» в Canvas пуст.",
            "en": "Your Canvas to-do list is empty.",
            "es": "Tu lista de tareas de Canvas está vacía."}
_NO_COURSES = {"ru": "Canvas не показал ни одного активного курса.",
               "en": "Canvas shows no active courses.",
               "es": "Canvas no muestra ningún curso activo."}
_NO_GRADES = {"ru": "Canvas не отдал оценки.",
              "en": "Canvas did not return any grades.",
              "es": "Canvas no devolvió ninguna nota."}
_NO_TOKEN = {
    "ru": "Нет токена Canvas для этого человека: попроси владельца хаба его положить.",
    "en": "There is no Canvas token for this person: ask the hub owner to add one.",
    "es": "No hay token de Canvas para esta persona: pide al dueño del hub que lo añada.",
}
_NO_BASE = {
    "ru": "Не настроен адрес Canvas: нужен адрес школы.",
    "en": "The Canvas address is not configured: the school's URL is needed.",
    "es": "La dirección de Canvas no está configurada: falta la URL de la escuela.",
}
_REJECTED = {
    "ru": "Canvas не принял токен этого человека.",
    "en": "Canvas refused this person's token.",
    "es": "Canvas rechazó el token de esta persona.",
}
_NETWORK = {"ru": "Canvas не ответил: {reason}.",
            "en": "Canvas did not answer: {reason}.",
            "es": "Canvas no respondió: {reason}."}
_WEEK_HEAD = {"ru": "На этой неделе сдать: ", "en": "Due soon: ",
              "es": "Para entregar pronto: "}
_TODO_HEAD = {"ru": "В списке «сделать»: ", "en": "On your to-do list: ",
              "es": "En tu lista de tareas: "}
_COURSES_HEAD = {"ru": "Твои курсы: ", "en": "Your courses: ",
                 "es": "Tus cursos: "}
_GRADES_HEAD = {"ru": "Оценки: ", "en": "Grades: ", "es": "Notas: "}
_POINTS = {"ru": "{value} баллов", "en": "{value} points", "es": "{value} puntos"}
_NO_SCORE = {"ru": "оценки пока нет", "en": "no grade yet", "es": "aún sin nota"}


def language_of(value: Any, *, default: str = DEFAULT_LANGUAGE) -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in {"ru", "en", "es"} else default


def timezone_of(name: Any, *, default: str = "UTC") -> ZoneInfo:
    try:
        return ZoneInfo(str(name or default))
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return ZoneInfo(default)


def spoken_due(value: Any, *, language: str = DEFAULT_LANGUAGE, tz: Any = "UTC") -> str:
    """Срок словами: «25 сентября в 23:59» — а не отметка времени из API."""
    moment = _moment(value)
    if moment is None:
        return ""
    local = moment.astimezone(timezone_of(tz))
    name = _MONTHS[language][local.month - 1]
    return f"{local.day} {name} {_AT[language]} {local.hour:02d}:{local.minute:02d}"


def _moment(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _client(ctx: Any) -> httpx.AsyncClient:
    timeout = float(getattr(ctx, "timeout_s", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S)
    return httpx.AsyncClient(
        timeout=timeout, transport=getattr(ctx, "transport", None),
        base_url=str(getattr(ctx, "base_url", "") or "").rstrip("/") + "/",
        headers={"Authorization": f"Bearer {getattr(ctx, 'token', '') or ''}",
                 "Accept": "application/json",
                 "User-Agent": "Rowan/1.0 (dorm assistant)"})


async def _get(client: httpx.AsyncClient, path: str, params: dict[str, Any] | None = None) -> Any:
    """Один запрос к Canvas: всё, что не настоящие данные, — ошибка скилла."""
    try:
        response = await client.get(path, params=params)
    except httpx.TimeoutException as exc:
        raise CanvasError("the request timed out") from exc
    except httpx.HTTPError as exc:
        raise CanvasError(f"the network is unavailable ({type(exc).__name__})") from exc
    if response.status_code in (401, 403):
        raise CanvasError("__rejected__")
    if response.status_code != 200:
        raise CanvasError(f"Canvas answered {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise CanvasError("Canvas answered something that is not data") from exc
    return payload


def _assignment_rows(payload: Any) -> list[dict[str, Any]]:
    """Собрать задания из ответа Canvas, ничего не додумывая."""
    if not isinstance(payload, list):
        raise CanvasError("Canvas answered something that is not a list")
    rows: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        assignment = entry.get("assignment") if isinstance(entry.get("assignment"), dict) else entry
        title = str(assignment.get("name") or entry.get("title") or "").strip()
        due = _moment(assignment.get("due_at") or entry.get("end_at") or assignment.get("todo_date"))
        if not title:
            continue
        rows.append({
            "title": title[:200],
            "due_at": due.isoformat() if due else "",
            "course_id": assignment.get("course_id") or entry.get("course_id") or "",
            "points": assignment.get("points_possible"),
            "url": str(assignment.get("html_url") or entry.get("html_url") or "")[:300],
            "_due": due,
        })
    return rows


def _course_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise CanvasError("Canvas answered something that is not a list")
    rows: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("course_code") or "").strip()
        if not name:
            continue
        enrollment = next((item for item in (entry.get("enrollments") or [])
                           if isinstance(item, dict)), {})
        score = enrollment.get("computed_current_score")
        grade = enrollment.get("computed_current_grade")
        rows.append({"course_id": entry.get("id") or "", "name": name[:200],
                     "code": str(entry.get("course_code") or "")[:60],
                     "score": score, "grade": str(grade or "")[:20]})
    return rows


async def run(ctx: Any, args: Args) -> SkillResult:
    """Ответить по данным Canvas — или честно сказать, чего не хватает."""
    if not isinstance(args, Args):
        try:
            args = Args.model_validate(args or {})
        except ValidationError as exc:
            return SkillResult(ok=False, error=f"bad canvas arguments: {exc.error_count()}")
    language = language_of(getattr(ctx, "language", "") or DEFAULT_LANGUAGE)
    if not str(getattr(ctx, "base_url", "") or "").strip():
        return SkillResult(ok=False, error=_NO_BASE[language])
    if not str(getattr(ctx, "token", "") or "").strip():
        return SkillResult(ok=False, error=_NO_TOKEN[language])
    days = int(args.days or getattr(ctx, "days", DEFAULT_DAYS) or DEFAULT_DAYS)
    tz = getattr(ctx, "timezone", "") or "UTC"
    try:
        async with _client(ctx) as client:
            if args.what in ("courses", "grades"):
                payload = await _get(client, "api/v1/courses",
                                     {"enrollment_state": "active", "per_page": PAGE_SIZE,
                                      "include[]": "total_scores"})
            elif args.what == "todo":
                payload = await _get(client, "api/v1/users/self/todo",
                                     {"per_page": PAGE_SIZE})
            else:
                payload = await _get(client, "api/v1/users/self/upcoming_events",
                                     {"per_page": PAGE_SIZE})
        # Разбор чужого ответа — там же, где сеть: «не список» тоже честный отказ.
        return _answer(payload, args=args, language=language, days=days, tz=tz)
    except CanvasError as exc:
        reason = str(exc)
        if reason == "__rejected__":
            return SkillResult(ok=False, error=_REJECTED[language])
        return SkillResult(ok=False, error=_NETWORK[language].format(reason=reason))
    except Exception as exc:  # noqa: BLE001 - скилл не роняет хаб
        log.warning("The Canvas skill failed (%s)", exc)
        return SkillResult(ok=False, error=_NETWORK[language].format(reason=type(exc).__name__))


def _answer(payload: Any, *, args: Args, language: str, days: int, tz: Any) -> SkillResult:
    """Превратить ответ Canvas в слова и данные — без единой выдумки."""
    if args.what in ("courses", "grades"):
        courses = _filter_courses(_course_rows(payload), args.course)
        if args.what == "courses":
            lines = [f"{row['name']}"
                     + (f" ({row['code']})" if row["code"] else "")
                     for row in courses]
            head = _COURSES_HEAD[language]
            empty = _NO_COURSES[language]
            data = {"courses": courses}
        else:
            lines = [f"{row['name']}: " + (_score_words(row, language))
                     for row in courses]
            head = _GRADES_HEAD[language]
            empty = _NO_GRADES[language]
            data = {"grades": [{"course": row["name"], "score": row["score"],
                                "grade": row["grade"]} for row in courses]}
    else:
        rows = _assignment_rows(payload)
        now = datetime.now(UTC)
        if args.what == "week":
            limit = now + timedelta(days=days)
            rows = [row for row in rows
                    if row["_due"] is not None and now <= row["_due"] <= limit]
            rows.sort(key=lambda row: row["_due"])
            head, empty = _WEEK_HEAD[language], _NO_DEADLINES[language]
            lines = [_assignment_words(row, language, tz) for row in rows]
            data = {"deadlines": [_public(row) for row in rows], "days": days}
        else:
            rows.sort(key=lambda row: (row["_due"] is None, row["_due"] or now))
            head, empty = _TODO_HEAD[language], _NO_TODO[language]
            lines = [_assignment_words(row, language, tz) for row in rows]
            data = {"todo": [_public(row) for row in rows]}
    if not lines:
        return SkillResult(ok=True, spoken=empty, data=data)
    spoken = head + "; ".join(line for line in lines if line) + "."
    return SkillResult(ok=True, spoken=spoken, data=data)


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _filter_courses(rows: list[dict[str, Any]], wanted: str) -> list[dict[str, Any]]:
    needle = " ".join(str(wanted or "").split()).casefold()
    if not needle:
        return rows
    return [row for row in rows
            if needle in str(row["name"]).casefold() or needle == str(row["course_id"])]


def _score_words(row: dict[str, Any], language: str) -> str:
    if row.get("score") is not None:
        return f"{row['score']}%"
    if row.get("grade"):
        return str(row["grade"])
    return _NO_SCORE[language]


def _assignment_words(row: dict[str, Any], language: str, tz: Any) -> str:
    text = f"«{row['title']}»"
    due = row["_due"]
    if due is not None:
        text += f" — {_DUE[language]} {spoken_due(due, language=language, tz=tz)}"
    if row.get("points") is not None:
        text += ", " + _POINTS[language].format(value=_number(row["points"]))
    return text


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number == int(number) else f"{number:g}"
