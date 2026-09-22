"""Погода по Open-Meteo — без ключа и без выдумок (ТЗ F-421, P3-29).

Open-Meteo открыт, поэтому у скилла нет ни ключа, ни секрета: единственная
зависимость — сеть. Сеть здесь и есть самая честная часть: таймаут, отказ
соединения, незнакомое место и чужой ответ становятся ``SkillResult(ok=False)``
с причиной, а не правдоподобной температурой. Раздел погоды в утреннем
брифинге (F-420) читает тот же ``data``, что отдаётся голосом.

Числа WMO (0..99) переводятся в слова здесь, а не моделью: «+7, дождь» — это
факт из ответа API, и модель не должна догадываться, что значит код 61.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hub.skills_runtime import SkillResult

log = logging.getLogger("jarvis.skill.weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
#: ТЗ F-421: у скилла есть таймаут, а не «ждать вечно».
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_LANGUAGE = "ru"


class Args(BaseModel):
    """Что модель может попросить у скилла погоды."""

    model_config = ConfigDict(extra="forbid")

    #: Город или место словами («Chicago», «Киев»). Пусто — место комнаты.
    location: str = Field(default="", max_length=120)
    #: ``now`` — только текущая погода, ``today``/``tomorrow`` — с прогнозом.
    when: Literal["now", "today", "tomorrow"] = "today"
    #: Система единиц; по умолчанию метрическая (в ТЗ других нет).
    units: Literal["metric", "imperial"] = "metric"


class WeatherError(RuntimeError):
    """Погоду получить не удалось — и скилл говорит, почему."""


# WMO weather codes, grouped the way people talk about weather.
_CODE_WORDS: dict[str, dict[int, str]] = {
    "ru": {0: "ясно", 1: "почти ясно", 2: "переменная облачность", 3: "пасмурно",
           45: "туман", 48: "изморозь", 51: "слабая морось", 53: "морось",
           55: "сильная морось", 56: "ледяная морось", 57: "сильная ледяная морось",
           61: "небольшой дождь", 63: "дождь", 65: "сильный дождь",
           66: "ледяной дождь", 67: "сильный ледяной дождь",
           71: "небольшой снег", 73: "снег", 75: "сильный снег", 77: "снежная крупа",
           80: "небольшой ливень", 81: "ливень", 82: "сильный ливень",
           85: "снежные заряды", 86: "сильные снежные заряды",
           95: "гроза", 96: "гроза с градом", 99: "сильная гроза с градом"},
    "en": {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
           45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
           55: "heavy drizzle", 56: "freezing drizzle", 57: "heavy freezing drizzle",
           61: "light rain", 63: "rain", 65: "heavy rain",
           66: "freezing rain", 67: "heavy freezing rain",
           71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
           80: "light showers", 81: "showers", 82: "violent showers",
           85: "snow showers", 86: "heavy snow showers",
           95: "a thunderstorm", 96: "a thunderstorm with hail",
           99: "a heavy thunderstorm with hail"},
    "es": {0: "despejado", 1: "casi despejado", 2: "parcialmente nublado", 3: "cubierto",
           45: "niebla", 48: "niebla helada", 51: "llovizna débil", 53: "llovizna",
           55: "llovizna fuerte", 56: "llovizna helada", 57: "llovizna helada fuerte",
           61: "lluvia débil", 63: "lluvia", 65: "lluvia fuerte",
           66: "lluvia helada", 67: "lluvia helada fuerte",
           71: "nieve débil", 73: "nieve", 75: "nieve fuerte", 77: "granos de nieve",
           80: "chubascos débiles", 81: "chubascos", 82: "chubascos fuertes",
           85: "chubascos de nieve", 86: "chubascos de nieve fuertes",
           95: "tormenta", 96: "tormenta con granizo", 99: "tormenta fuerte con granizo"},
}

_NO_RESULT = {
    "ru": "Не нашла такое место: {place}. Скажи город иначе.",
    "en": "I could not find {place}. Say the place another way.",
    "es": "No encontré {place}. Dime el lugar de otra forma.",
}
_NO_LOCATION = {
    "ru": "Не знаю, для какого места смотреть погоду: скажи город.",
    "en": "I do not know which place to check: say the city.",
    "es": "No sé de qué lugar mirar el tiempo: dime la ciudad.",
}
_NO_NETWORK = {
    "ru": "Не смогла узнать погоду: {reason}.",
    "en": "I could not get the weather: {reason}.",
    "es": "No pude consultar el tiempo: {reason}.",
}
_NOW_WORDS = {"ru": "сейчас", "en": "now", "es": "ahora"}
_TODAY_WORDS = {"ru": "днём", "en": "today", "es": "hoy"}
_TOMORROW_WORDS = {"ru": "завтра", "en": "tomorrow", "es": "mañana"}
_PRECIP_WORDS = {"ru": "вероятность осадков {value}%",
                 "en": "a {value}% chance of precipitation",
                 "es": "probabilidad de precipitación del {value}%"}
_FROM_TO = {"ru": "от {low} до {high}", "en": "from {low} to {high}",
            "es": "de {low} a {high}"}


def language_of(value: Any, *, default: str = DEFAULT_LANGUAGE) -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in {"ru", "en", "es"} else default


def describe_code(code: Any, language: Any = DEFAULT_LANGUAGE) -> str:
    """Код WMO словами; незнакомый код честно называется неизвестной погодой."""
    lang = language_of(language)
    try:
        number = int(code)
    except (TypeError, ValueError):
        return {"ru": "погода без описания", "en": "unreported weather",
                "es": "tiempo sin descripción"}[lang]
    words = _CODE_WORDS[lang]
    if number in words:
        return words[number]
    closest = min(words, key=lambda key: abs(key - number))
    return words[closest]


def _degrees(value: Any, units: str) -> str:
    if value is None:
        return ""
    number = float(value)
    sign = "+" if (units == "metric" and number > 0) else ""
    return f"{sign}{round(number)}"


def spoken_lines(payload: dict[str, Any], *, language: str = DEFAULT_LANGUAGE) -> str:
    """Факты словами: сейчас и (если есть) день целиком."""
    lang = language_of(language)
    units = str(payload.get("units") or "metric")
    place = str(payload.get("location") or "")
    parts: list[str] = []
    current = payload.get("current") or {}
    if current:
        line = _NOW_WORDS[lang]
        if place:
            line = {"ru": f"В {place} {line}", "en": f"In {place} {line}",
                    "es": f"En {place} {line}"}[lang]
        if current.get("temperature") is not None:
            line += " " + _degrees(current["temperature"], units)
        condition = describe_code(current.get("code"), lang)
        line = f"{line}, {condition}"
        parts.append(line)
    day = payload.get("day") or {}
    if day and str(payload.get("when") or "today") != "now":
        when = (_TOMORROW_WORDS if payload.get("when") == "tomorrow"
                else _TODAY_WORDS)[lang]
        line = when
        if day.get("min") is not None and day.get("max") is not None:
            line += " " + _FROM_TO[lang].format(low=_degrees(day["min"], units),
                                                high=_degrees(day["max"], units))
        line += ", " + describe_code(day.get("code"), lang)
        if day.get("precipitation_probability") is not None:
            line += ", " + _PRECIP_WORDS[lang].format(
                value=int(day["precipitation_probability"]))
        parts.append(line)
    if not parts:
        raise WeatherError("the forecast has no numbers")
    clean = [part for part in parts if part]
    # Каждое предложение начинается с большой буквы: брифинг читает это вслух.
    clean = [part[:1].upper() + part[1:] for part in clean]
    return ". ".join(clean) + "."


def _client(ctx: Any) -> httpx.AsyncClient:
    timeout = float(getattr(ctx, "timeout_s", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S)
    transport = getattr(ctx, "transport", None)
    return httpx.AsyncClient(timeout=timeout, transport=transport,
                             headers={"User-Agent": "Rowan/1.0 (dorm assistant)"})


async def _json(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Один запрос: сеть и чужой ответ — это ошибки скилла, а не хаба."""
    try:
        response = await client.get(url, params=params)
    except httpx.TimeoutException as exc:
        raise WeatherError("the weather service did not answer in time") from exc
    except httpx.HTTPError as exc:
        raise WeatherError(f"the weather service is unreachable ({type(exc).__name__})") from exc
    if response.status_code != 200:
        raise WeatherError(f"the weather service answered {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise WeatherError("the weather service answered something that is not data") from exc
    if not isinstance(payload, dict):
        raise WeatherError("the weather service answered something that is not data")
    return payload


async def lookup(client: httpx.AsyncClient, place: str, *,
                 language: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    """Город словами → широта, долгота и настоящее название."""
    query = " ".join(str(place or "").split())[:120]
    if not query:
        raise WeatherError("no place was named")
    payload = await _json(client, GEOCODE_URL, {"name": query, "count": 1,
                                                "language": language, "format": "json"})
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise WeatherError(f"no place matches {query}")
    first = results[0]
    if not isinstance(first, dict) or first.get("latitude") is None:
        raise WeatherError(f"no place matches {query}")
    name = str(first.get("name") or query)
    country = str(first.get("country") or "")
    return {"name": name, "country": country,
            "latitude": float(first["latitude"]), "longitude": float(first["longitude"])}


async def forecast(client: httpx.AsyncClient, place: dict[str, Any], *,
                   when: str = "today", units: str = "metric",
                   timezone: str = "auto") -> dict[str, Any]:
    """Погода и (когда просят) прогноз на сегодня или завтра."""
    params: dict[str, Any] = {
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                 "precipitation_probability_max",
        "forecast_days": 2, "timezone": timezone or "auto",
    }
    if units == "imperial":
        params.update({"temperature_unit": "fahrenheit", "wind_speed_unit": "mph"})
    payload = await _json(client, FORECAST_URL, params)
    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    index = 1 if when == "tomorrow" else 0
    days = [str(item) for item in (daily.get("time") or [])]
    if index >= len(days) or not isinstance(current, dict) or not isinstance(daily, dict):
        raise WeatherError("the forecast is missing the requested day")

    def pick(key: str) -> Any:
        values = daily.get(key) or []
        return values[index] if index < len(values) else None

    data: dict[str, Any] = {
        "location": place["name"], "country": place.get("country", ""),
        "when": when, "units": units, "date": days[index] if days else "",
        "current": ({} if when == "tomorrow" else {
            "temperature": current.get("temperature_2m"),
            "code": current.get("weather_code"),
            "wind": current.get("wind_speed_10m"),
        }),
        "day": {"min": pick("temperature_2m_min"), "max": pick("temperature_2m_max"),
                "code": pick("weather_code"),
                "precipitation_probability": pick("precipitation_probability_max")},
    }
    return data


async def run(ctx: Any, args: Args) -> SkillResult:
    """Погода для названного места или для места комнаты (ТЗ F-421)."""
    if not isinstance(args, Args):
        try:
            args = Args.model_validate(args or {})
        except ValidationError as exc:
            return SkillResult(ok=False, error=f"bad weather arguments: {exc.error_count()}")
    language = language_of(getattr(ctx, "language", "") or DEFAULT_LANGUAGE)
    place = args.location or str(getattr(ctx, "location", "") or "")
    if not place:
        return SkillResult(ok=False, error=_NO_LOCATION[language])
    timezone = str(getattr(ctx, "timezone", "") or "auto")
    async with _client(ctx) as client:
        try:
            found = await lookup(client, place, language=language)
            payload = await forecast(client, found, when=args.when, units=args.units,
                                     timezone=timezone)
        except WeatherError as exc:
            message = str(exc)
            if message.startswith("no place matches "):
                error = _NO_RESULT[language].format(place=place)
            else:
                error = _NO_NETWORK[language].format(reason=message)
            log.info("The weather skill could not answer: %s", message)
            return SkillResult(ok=False, error=error)
        except Exception as exc:  # noqa: BLE001 - скилл не роняет хаб
            log.warning("The weather skill failed (%s)", exc)
            return SkillResult(ok=False,
                               error=_NO_NETWORK[language].format(reason=type(exc).__name__))
    try:
        spoken = spoken_lines(payload, language=language)
    except WeatherError as exc:
        return SkillResult(ok=False, error=_NO_NETWORK[language].format(reason=str(exc)))
    return SkillResult(ok=True, spoken=spoken, data=payload)
