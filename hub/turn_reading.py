"""Одно чтение одного запроса: Jev понимает реплику, набор инструментов сужается.

Чтение реплики (``JevDecider.understand``, U-10…U-14) и сужение набора
инструментов по семейству жили только в голосовом ходе
(``hub/app.py::Connection._understand_turn``). Владелец 2026-09-23: «в Telegram
должны быть те же возможности, что и у голосового ассистента» — поэтому и
голосовой ход, и запрос из Telegram-чата
(``hub/telegram_control.py``) зовут ОДНУ эту функцию. Иначе «в Telegram
работает иначе» может случиться молча (ПРОГРЕСС аудита, AU-19).

Правило то же, что было у голосового хода, — fail-open (ТЗ раздел 1, «не
ломать работающее»): нет клиента Jev, нет ключа, комната не имеет права
отправлять текст в облако, таймаут, непонятный ответ или низкая уверенность
оставляют запрос ровно таким, каким он был до этого чтения, — со всеми
инструментами.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from hub import turn_trace
from hub.tools import (
    CORE_TOOLS,
    TOOL_FAMILY_MEANINGS,
    TOOL_FAMILY_NAMES,
    TOOLS,
    tools_for_family,
)

log = logging.getLogger("jarvis.server.understanding")

#: Сколько символов реплики уходит Jev: вопрос о смысле короткого запроса, а не
#: стенограмма. Копия того же предела, что стояла в голосовом ходе.
MAX_TEXT = 1000


def narrow_tools_for(understanding: Mapping[str, Any] | Any, *,
                     threshold: float = 0.65) -> list[dict[str, Any]] | None:
    """The tool list one utterance is answered with (U-14), or ``None`` for all.

    Three answers are used, and only above the configured confidence: ``act``
    (is anything to be done at all), ``family`` (which family of tools) and
    ``single`` (is this one request or several in one sentence). Everything else
    - no Jev, a timeout, an unusable answer, a family that is not one of ours, a
    low confidence - returns ``None``, and the model gets every tool exactly as
    it did before this change.
    """
    if not isinstance(understanding, Mapping):
        return None
    # UG-08: «открой ютуб и сделай громче» просит две разные семьи одним
    # дыханием. Сужение по той, которую Jev назвал первой, прячет вторую
    # половину просьбы, поэтому при «просьб несколько» набор не сужается вовсе.
    single = understanding.get("single")
    if (isinstance(single, Mapping) and single.get("value") is False
            and float(single.get("confidence") or 0.0) >= threshold):
        log.info("Jev read the turn as several requests (confidence %.2f); "
                 "every tool stays available",
                 float(single.get("confidence") or 0.0))
        return None
    family = understanding.get("family")
    named = ""
    if isinstance(family, Mapping) and float(family.get("confidence") or 0.0) >= threshold:
        named = str(family.get("value") or "")
        if named == "none":
            named = ""
    act = understanding.get("act")
    # A turn Jev is sure is only a question needs no device tool at all; the
    # core set (remember, recall, speak, message) stays, so the model can still
    # save or recall the very thing it is answering about. The bar is higher
    # than for a family, because being wrong here costs an action.
    if (isinstance(act, Mapping) and act.get("value") is False
            and float(act.get("confidence") or 0.0) >= max(0.9, threshold)):
        # «Ничего не делать» и семейство могут прийти одним чтением: «who is in
        # the room?» — это вопрос, но ответить на него можно только взглядом.
        # The core-only set used to win here and hid ``look_at_camera`` /
        # ``find_object`` / ``list_people`` (AU-03, mass audit 2026-09-23).
        # Every family set already contains the core tools, so trusting the
        # named family costs nothing on a real conversation.
        if named:
            tools = tools_for_family(named)
            if tools is not None:
                log.info("Jev read the turn as %r (confidence %.2f, %d tools offered); "
                         "the act answer was 'nothing to do', the family still rules",
                         named, float(family.get("confidence") or 0.0), len(tools))
                return tools
        allowed = set(CORE_TOOLS)
        return [tool for tool in TOOLS if tool["function"]["name"] in allowed]
    if not named:
        return None
    tools = tools_for_family(named)
    if tools is None:
        return None
    log.info("Jev read the turn as %r (confidence %.2f, %d tools offered)",
             named, float(family.get("confidence") or 0.0), len(tools))
    return tools


async def read_turn(client: Any, text: Any, *, home_id: str = "", room: str = "",
                    timeout_s: float = 2.0,
                    utterance_id: str = "") -> Mapping[str, Any] | None:
    """The whole reading of one request in ONE batched call (U-10…U-14).

    Returns ``{name: {"value": ..., "confidence": ...}}`` or ``None`` when
    nothing could be read. The call is fail-open on purpose: a reading never
    breaks the request it was made for. Its own time is written to the turn
    trace as a ``understanding`` event, so the owner panel shows that Jev read
    this request - including the requests that came from Telegram (AU-19).
    """
    if client is None:
        return None
    context = {
        'home_id': str(home_id or ''),
        'room': str(room or ''),
        'text': ' '.join(str(text or '').split())[:MAX_TEXT],
    }
    started = time.perf_counter()
    try:
        found = await client.understand(
            context, families=list(TOOL_FAMILY_NAMES), meanings=TOOL_FAMILY_MEANINGS,
            timeout_s=max(0.05, float(timeout_s)))
    except Exception as exc:  # noqa: BLE001 - a reading never breaks a request
        turn_trace.record("understanding", str(getattr(client, 'model', '') or 'jev'),
                          payload={'text': context['text'],
                                   'error': f'{type(exc).__name__}: {exc}'},
                          ok=False,
                          latency_ms=int((time.perf_counter() - started) * 1000))
        log.info("The batched Jev reading is unavailable (%s); every tool stays available",
                 exc, extra={'utterance_id': utterance_id} if utterance_id else {})
        return None
    turn_trace.record("understanding", str(getattr(client, 'model', '') or 'jev'), payload={
        'text': context['text'],
        'answers': {name: {'value': item['value'],
                           'confidence': round(float(item['confidence']), 2)}
                    for name, item in found.items()},
    }, latency_ms=int((time.perf_counter() - started) * 1000))
    return found


async def turn_tools(client: Any, text: Any, *, home_id: str = "", room: str = "",
                     threshold: float = 0.65, timeout_s: float = 2.0,
                     utterance_id: str = "") -> list[dict[str, Any]] | None:
    """The narrowed tool list for one request, or ``None`` for "every tool".

    Both callers (the voice turn and the Telegram request) go through here, so
    the two paths cannot drift apart: the reading, the trace event and the
    narrowing are the same code.
    """
    found = await read_turn(client, text, home_id=home_id, room=room,
                            timeout_s=timeout_s, utterance_id=utterance_id)
    if not found:
        return None
    return narrow_tools_for(found, threshold=threshold)


__all__ = ["MAX_TEXT", "narrow_tools_for", "read_turn", "turn_tools"]
