"""Google Calendar по OAuth человека (ТЗ F-421, P3-31).

Скилл читает ближайшие события и умеет записать НОВОЕ событие в календарь
того, кто попросил: у каждого человека свой refresh-токен (``ctx.refresh_token``),
и Rowan не ходит в чужой календарь. Всё, что нужно для доступа, лежит в
окружении хаба (правило раздела 1 ТЗ): client id, client secret и
refresh-токен; скилл меняет refresh-токен на короткоживущий access-токен
Google и никогда его не печатает.

Без доступа скилл честно отказывает («нет доступа к календарю»), а не отвечает
пустым списком, который выглядел бы как «встреч нет». Так же честно он
говорит про отозванное согласие, отказ Google и таймаут.

Удаления здесь нет намеренно: скилл создаёт и читает, но ничего не убирает из
чужого календаря.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from datetime import time as clock
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hub.skills_runtime import SkillResult

log = logging.getLogger("jarvis.skill.calendar")

TOKEN_URL = "https://oauth2.googleapis.com/token"
API_URL = "https://www.googleapis.com/calendar/v3"
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_DAYS = 1
DEFAULT_LANGUAGE = "ru"
MAX_EVENTS = 10


class Args(BaseModel):
    """Что человек просит у календаря."""

    model_config = ConfigDict(extra="forbid")

    what: Literal["next", "today", "week", "create"] = "today"
    #: Для ``week``: сколько дней вперёд; 0 — из настройки хаба.
    days: int = Field(default=0, ge=0, le=60)
    #: Для ``create``: название события словами человека.
    title: str = Field(default="", max_length=200)
    #: Для ``create``: местное время начала, ``YYYY-MM-DD HH:MM``.
    start: str = Field(default="", max_length=40)
    #: Для ``create``: сколько минут длится событие.
    duration_min: int = Field(default=60, ge=5, le=720)


class CalendarError(RuntimeError):
    """Календарь не ответил так, чтобы ответу можно было верить."""


_MONTHS = {
    "ru": ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
           "августа", "сентября", "октября", "ноября", "декабря"),
    "en": ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"),
    "es": ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
           "agosto", "septiembre", "octubre", "noviembre", "diciembre"),
}
_AT = {"ru": "в", "en": "at", "es": "a las"}
_TODAY = {"ru": "Сегодня", "en": "Today", "es": "Hoy"}
_TOMORROW = {"ru": "Завтра", "en": "Tomorrow", "es": "Mañana"}
_NEXT = {"ru": "Дальше", "en": "Next", "es": "Después"}
_ALL_DAY = {"ru": "весь день", "en": "all day", "es": "todo el día"}
_NOTHING_TODAY = {"ru": "На сегодня встреч нет.",
                  "en": "There is nothing in the calendar today.",
                  "es": "No hay nada en el calendario hoy."}
_NOTHING_SOON = {"ru": "В ближайшие дни встреч нет.",
                 "en": "There is nothing in the calendar in the next few days.",
                 "es": "No hay nada en el calendario en los próximos días."}
_CREATED = {"ru": "Записала «{title}» на {when}.",
            "en": "I put “{title}” in the calendar for {when}.",
            "es": "Anoté «{title}» en el calendario para {when}."}
_NEED_TITLE = {"ru": "Не поняла, что записать: скажи название встречи.",
               "en": "I did not catch what to add: say the title.",
               "es": "No entendí qué anotar: dime el título."}
_NEED_START = {
    "ru": "Не поняла, на когда записать: скажи день и время.",
    "en": "I did not catch when to add it: say the day and the time.",
    "es": "No entendí cuándo anotarlo: dime el día y la hora.",
}
_NO_ACCESS = {
    "ru": "Нет доступа к Google Calendar: его не выдавали этому человеку.",
    "en": "There is no Google Calendar access for this person.",
    "es": "No hay acceso a Google Calendar para esta persona.",
}
_REJECTED = {"ru": "Google не принял доступ к календарю: нужно разрешить его заново.",
             "en": "Google refused the calendar access: it has to be granted again.",
             "es": "Google rechazó el acceso al calendario: hay que concederlo otra vez."}
_NETWORK = {"ru": "Календарь не ответил: {reason}.",
            "en": "The calendar did not answer: {reason}.",
            "es": "El calendario no respondió: {reason}."}


def language_of(value: Any, *, default: str = DEFAULT_LANGUAGE) -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in {"ru", "en", "es"} else default


def timezone_of(name: Any, *, default: str = "UTC") -> ZoneInfo:
    try:
        return ZoneInfo(str(name or default))
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return ZoneInfo(default)


def _moment(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Дата без времени (событие «весь день») считается полночью UTC: без
    # пояса её нельзя сравнивать с моментом, а гадать пояс нельзя.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_local_start(value: Any, *, tz: Any = "UTC",
                      now: datetime | None = None) -> datetime:
    """``'2026-09-25 15:00'`` (и пара простых слов) → момент в часах комнаты."""
    text = " ".join(str(value or "").split())
    if not text:
        raise CalendarError("__need_start__")
    zone = timezone_of(tz)
    day = (now or datetime.now(UTC)).astimezone(zone).date()
    lowered = text.casefold()
    if lowered.startswith(("завтра", "tomorrow", "mañana")):
        day = day + timedelta(days=1)
        text = text[len(text.split()[0]):].strip()
    elif lowered.startswith(("сегодня", "today", "hoy")):
        text = text[len(text.split()[0]):].strip()
    for shape in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%H:%M", "%H.%M", "%d.%m %H:%M"):
        try:
            parsed = datetime.strptime(text, shape)
        except ValueError:
            continue
        if shape in ("%H:%M", "%H.%M"):
            return datetime.combine(day, parsed.time(), tzinfo=zone)
        if shape == "%d.%m %H:%M":
            return datetime(day.year, parsed.month, parsed.day, parsed.hour, parsed.minute,
                            tzinfo=zone)
        return datetime(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute,
                        tzinfo=zone)
    raise CalendarError("__need_start__")


def spoken_when(moment: datetime, *, language: str = DEFAULT_LANGUAGE, tz: Any = "UTC",
                now: datetime | None = None) -> str:
    """«25 сентября в 15:00», «сегодня в 15:00» — словами, а не ISO."""
    zone = timezone_of(tz)
    local = moment.astimezone(zone)
    today = (now or datetime.now(UTC)).astimezone(zone).date()
    days = (local.date() - today).days
    day = (f"{local.day} {_MONTHS[language][local.month - 1]}" if days not in (0, 1)
           else _TODAY[language].casefold() if days == 0 else _TOMORROW[language].casefold())
    return f"{day} {_AT[language]} {local.hour:02d}:{local.minute:02d}"


def _client(ctx: Any) -> httpx.AsyncClient:
    timeout = float(getattr(ctx, "timeout_s", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S)
    return httpx.AsyncClient(timeout=timeout, transport=getattr(ctx, "transport", None),
                             headers={"Accept": "application/json",
                                      "User-Agent": "Rowan/1.0 (dorm assistant)"})


async def access_token(client: httpx.AsyncClient, ctx: Any) -> str:
    """Refresh-токен человека → короткоживущий access-токен Google."""
    for name in ("client_id", "client_secret", "refresh_token"):
        if not str(getattr(ctx, name, "") or "").strip():
            raise CalendarError("__no_access__")
    try:
        response = await client.post(TOKEN_URL, data={
            "client_id": str(getattr(ctx, "client_id", "")),
            "client_secret": str(getattr(ctx, "client_secret", "")),
            "refresh_token": str(getattr(ctx, "refresh_token", "")),
            "grant_type": "refresh_token"})
    except httpx.TimeoutException as exc:
        raise CalendarError("the request timed out") from exc
    except httpx.HTTPError as exc:
        raise CalendarError(f"the network is unavailable ({type(exc).__name__})") from exc
    if response.status_code in (400, 401, 403):
        raise CalendarError("__rejected__")
    if response.status_code != 200:
        raise CalendarError(f"Google answered {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise CalendarError("Google answered something that is not data") from exc
    token = str((payload or {}).get("access_token") or "")
    if not token:
        raise CalendarError("__rejected__")
    return token


async def list_events(client: httpx.AsyncClient, *, token: str, calendar_id: str,
                      start: datetime, end: datetime) -> list[dict[str, Any]]:
    """События между двумя моментами; всё, что не список, — ошибка."""
    try:
        response = await client.get(
            f"{API_URL}/calendars/{calendar_id}/events",
            params={"timeMin": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    "timeMax": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    "singleEvents": "true", "orderBy": "startTime",
                    "maxResults": MAX_EVENTS},
            headers={"Authorization": f"Bearer {token}"})
    except httpx.TimeoutException as exc:
        raise CalendarError("the request timed out") from exc
    except httpx.HTTPError as exc:
        raise CalendarError(f"the network is unavailable ({type(exc).__name__})") from exc
    if response.status_code in (401, 403):
        raise CalendarError("__rejected__")
    if response.status_code == 404:
        raise CalendarError("this calendar does not exist")
    if response.status_code != 200:
        raise CalendarError(f"Google answered {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise CalendarError("Google answered something that is not data") from exc
    items = (payload or {}).get("items")
    if items is None:
        raise CalendarError("the calendar answer has no events list")
    if not isinstance(items, list):
        raise CalendarError("the calendar answer has no events list")
    rows: list[dict[str, Any]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        start_value = entry.get("start") or {}
        title = str(entry.get("summary") or "").strip()
        all_day = bool(isinstance(start_value, dict) and start_value.get("date"))
        begin = _moment((start_value or {}).get("dateTime") or (start_value or {}).get("date"))
        if not title:
            continue
        rows.append({"title": title[:200], "start_at": begin.isoformat() if begin else "",
                     "all_day": all_day, "location": str(entry.get("location") or "")[:120],
                     "url": str(entry.get("htmlLink") or "")[:300],
                     "_start": begin})
    rows = [row for row in rows if row["_start"] is not None]
    rows.sort(key=lambda row: row["_start"])
    return rows


async def create_event(client: httpx.AsyncClient, *, token: str, calendar_id: str,
                       title: str, start: datetime, minutes: int, tz: Any) -> dict[str, Any]:
    """Записать новое событие; ответ Google — единственное доказательство."""
    end = start + timedelta(minutes=max(5, int(minutes)))
    body = {"summary": title[:200],
            "start": {"dateTime": start.isoformat(), "timeZone": str(tz or "UTC")},
            "end": {"dateTime": end.isoformat(), "timeZone": str(tz or "UTC")}}
    try:
        response = await client.post(f"{API_URL}/calendars/{calendar_id}/events",
                                     json=body,
                                     headers={"Authorization": f"Bearer {token}"})
    except httpx.TimeoutException as exc:
        raise CalendarError("the request timed out") from exc
    except httpx.HTTPError as exc:
        raise CalendarError(f"the network is unavailable ({type(exc).__name__})") from exc
    if response.status_code in (401, 403):
        raise CalendarError("__rejected__")
    if response.status_code not in (200, 201):
        raise CalendarError(f"Google answered {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise CalendarError("Google answered something that is not data") from exc
    if not isinstance(payload, dict) or not str(payload.get("id") or ""):
        raise CalendarError("Google did not confirm the new event")
    return {"event_id": str(payload["id"])[:120], "title": title[:200],
            "start_at": start.isoformat(), "end_at": end.isoformat(),
            "url": str(payload.get("htmlLink") or "")[:300]}


def _event_words(row: dict[str, Any], language: str, tz: Any, now: datetime) -> str:
    text = f"«{row['title']}»"
    if row["all_day"]:
        return f"{text} — {_ALL_DAY[language]}"
    return f"{text} — {spoken_when(row['_start'], language=language, tz=tz, now=now)}"


async def run(ctx: Any, args: Args) -> SkillResult:
    """Ответить по календарю человека — или честно сказать, чего не хватает."""
    if not isinstance(args, Args):
        try:
            args = Args.model_validate(args or {})
        except ValidationError as exc:
            return SkillResult(ok=False, error=f"bad calendar arguments: {exc.error_count()}")
    language = language_of(getattr(ctx, "language", "") or DEFAULT_LANGUAGE)
    tz = getattr(ctx, "timezone", "") or "UTC"
    zone = timezone_of(tz)
    now = datetime.now(UTC)
    calendar_id = str(getattr(ctx, "calendar_id", "") or "primary")
    if args.what == "create":
        title = " ".join(str(args.title or "").split())
        if not title:
            return SkillResult(ok=False, error=_NEED_TITLE[language])
        try:
            start = parse_local_start(args.start, tz=zone, now=now)
        except CalendarError:
            return SkillResult(ok=False, error=_NEED_START[language])
    days = int(args.days or getattr(ctx, "days", DEFAULT_DAYS) or DEFAULT_DAYS)
    if args.what == "today":
        end = datetime.combine(now.astimezone(zone).date(), clock(23, 59, 59), tzinfo=zone)
    else:
        end = now + timedelta(days=max(1, days))
    try:
        async with _client(ctx) as client:
            token = await access_token(client, ctx)
            if args.what == "create":
                created = await create_event(client, token=token, calendar_id=calendar_id,
                                             title=title, start=start,
                                             minutes=args.duration_min, tz=zone)
                spoken = _CREATED[language].format(
                    title=created["title"],
                    when=spoken_when(start, language=language, tz=zone, now=now))
                return SkillResult(ok=True, spoken=spoken, data={"event": created})
            rows = await list_events(client, token=token, calendar_id=calendar_id,
                                     start=now, end=end)
    except CalendarError as exc:
        reason = str(exc)
        if reason == "__no_access__":
            return SkillResult(ok=False, error=_NO_ACCESS[language])
        if reason == "__rejected__":
            return SkillResult(ok=False, error=_REJECTED[language])
        return SkillResult(ok=False, error=_NETWORK[language].format(reason=reason))
    except Exception as exc:  # noqa: BLE001 - скилл не роняет хаб
        log.warning("The calendar skill failed (%s)", exc)
        return SkillResult(ok=False, error=_NETWORK[language].format(reason=type(exc).__name__))
    # Окно проверяется и здесь: если сервис вернул больше, чем просили,
    # «на этой неделе» всё равно остаётся неделей. Событие «весь день» живёт
    # целым днём комнаты, поэтому у него сравниваются даты, а не полуночь UTC.
    first_day = now.astimezone(zone).date()
    last_day = end.astimezone(zone).date()

    def _in_window(row: dict[str, Any]) -> bool:
        if row["all_day"]:
            return first_day <= row["_start"].date() <= last_day
        return now - timedelta(minutes=1) <= row["_start"] <= end

    rows = [row for row in rows if _in_window(row)]
    if args.what == "next":
        rows = rows[:1]
    if not rows:
        empty = _NOTHING_TODAY[language] if args.what == "today" else _NOTHING_SOON[language]
        return SkillResult(ok=True, spoken=empty, data={"events": []})
    head = _NEXT[language] if args.what == "next" else ""
    lines = [_event_words(row, language, zone, now) for row in rows]
    spoken = (head + ": " if head else "") + "; ".join(lines) + "."
    return SkillResult(ok=True, spoken=spoken,
                       data={"events": [{key: value for key, value in row.items()
                                         if not key.startswith("_")} for row in rows]})
