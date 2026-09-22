"""JevDecider — провайдер решений TypeSafe AI Jev (ТЗ 5.2, 5.5).

ТЗ описывает три операции интерфейса ``Decider``, и у Jev им соответствуют три
режима: ``yes_no`` с вероятностью, ``choose`` из вариантов и ``score`` по
шкале. Провайдер тонкий: он собирает запрос, зовёт HTTP-ручку и превращает
ответ в ``Decision``; логики решений здесь нет — она в цепочке
(``DecisionChain``), которая даёт каждому провайдеру свой таймаут и уходит к
следующему при отказе.

Честность важнее вида:

* без ключа провайдера просто нет (его не собирает ``hub/app.py``), поэтому
  хаб решает на локальных провайдерах, а не выдумывает вердикт;
* флаг дома ``cloud_decisions`` (по умолчанию false, ТЗ 5.5) выключает Jev для
  комнаты: запрос не уходит вовсе, цепочка идёт дальше;
* в облако уходит только текст и метаданные: ``_privacy_context`` пропускает
  скаляры и выбрасывает кадры, звук и эмбеддинги (ТЗ 5.5);
* любой отказ (нет сети, таймаут, не 200, битый JSON, значение вне вариантов,
  нет уверенности) — это ``DecisionUnavailable`` с причиной словами, а не
  решение «на глазок».

Форма запроса собрана в ОДНОМ месте (:meth:`JevDecider._payload`) и правится
там же, когда появится доступ: точные имена полей раннего доступа TypeSafe
исполнителю неизвестны (открытый вопрос раздела 17).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from hub.decider import Decision, DecisionUnavailable, _decision

log = logging.getLogger("jarvis.server.jev")

#: Ключи, которые в облако не уходят никогда (ТЗ 5.5: только текст и метаданные).
_PRIVATE_KEYS = ("jpeg", "image", "frame", "photo", "audio", "pcm", "vector",
                 "embedding", "face", "crop", "clip")


def _privacy_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Только текст и метаданные: скаляры и короткие списки скаляров."""
    clean: dict[str, Any] = {}
    for key, value in dict(context or {}).items():
        name = str(key).casefold()
        if any(marker in name for marker in _PRIVATE_KEYS):
            continue
        if isinstance(value, bool | int | float):
            clean[str(key)] = value
        elif isinstance(value, str):
            clean[str(key)] = value[:500]
        elif isinstance(value, list | tuple) and len(value) <= 20 and all(
                isinstance(item, bool | int | float | str) for item in value):
            clean[str(key)] = [item[:100] if isinstance(item, str) else item
                               for item in value]
    return clean


class JevDecider:
    """The three Decider operations over one HTTP endpoint (ТЗ 5.2)."""

    name = "jev"

    def __init__(self, *, base_url: str, api_key: str, path: str = "/v1/decide",
                 timeout_s: float = 0.4, transport: Any = None,
                 allowed_for: Callable[[str], bool] | None = None) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.path = str(path or "/v1/decide")
        self._api_key = str(api_key or "")
        self.timeout_s = max(0.05, float(timeout_s))
        self._transport = transport
        #: ТЗ 5.5: ``cloud_decisions`` дома. Без явного разрешения не идём.
        self.allowed_for = allowed_for or (lambda _home: False)

    # --- the three interfaces ----------------------------------------------

    async def yes_no(self, question: str, context: dict[str, Any], *,
                     decision_type: str) -> Decision[bool]:
        started = time.perf_counter()
        value, confidence = await self._ask("yes_no", question, context,
                                            decision_type=decision_type)
        if not isinstance(value, bool):
            raise DecisionUnavailable(f"jev answered {value!r} to a yes/no question")
        return _decision(value, confidence, self.name, started,
                         str(context.get("text") or question))

    async def choose(self, question: str, options: Sequence[str], context: dict[str, Any], *,
                     decision_type: str) -> Decision[str]:
        started = time.perf_counter()
        value, confidence = await self._ask("choose", question, context,
                                            options=list(options),
                                            decision_type=decision_type)
        chosen = str(value)
        if chosen not in {str(item) for item in options}:
            raise DecisionUnavailable(f"jev chose {chosen!r}, which was not offered")
        return _decision(chosen, confidence, self.name, started,
                         str(context.get("text") or question))

    async def score(self, question: str, context: dict[str, Any], *, scale: tuple[int, int],
                    decision_type: str) -> Decision[int]:
        started = time.perf_counter()
        value, confidence = await self._ask("score", question, context,
                                            scale=list(scale), decision_type=decision_type)
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise DecisionUnavailable(f"jev scored {value!r}, which is not a number") from exc
        low, high = int(scale[0]), int(scale[1])
        if not low <= number <= high:
            raise DecisionUnavailable(f"jev scored {number}, outside {low}..{high}")
        return _decision(number, confidence, self.name, started,
                         str(context.get("text") or question))

    # --- one call ----------------------------------------------------------

    def _payload(self, mode: str, question: str, context: Mapping[str, Any], *,
                 options: list[str] | None = None, scale: list[int] | None = None,
                 decision_type: str = "") -> dict[str, Any]:
        """The one place the early-access request shape lives (ТЗ 5.2)."""
        payload: dict[str, Any] = {
            "mode": str(mode),
            "question": str(question or ""),
            "context": _privacy_context(context),
            "decision_type": str(decision_type or ""),
        }
        if options is not None:
            payload["options"] = [str(option) for option in options]
        if scale is not None:
            payload["scale"] = [int(scale[0]), int(scale[1])]
        return payload

    async def _ask(self, mode: str, question: str, context: Mapping[str, Any], *,
                   options: list[str] | None = None, scale: list[int] | None = None,
                   decision_type: str = "") -> tuple[Any, float]:
        home = str(dict(context or {}).get("home_id") or "")
        if not home or not self.allowed_for(home):
            raise DecisionUnavailable(
                f"cloud decisions are switched off for {home or 'this room'}")
        if not self.base_url:
            raise DecisionUnavailable("no Jev base URL is configured")
        if not self._api_key:
            raise DecisionUnavailable("no Jev API key is available in the environment")
        url = f"{self.base_url}{self.path if self.path.startswith('/') else '/' + self.path}"
        payload = self._payload(mode, question, context, options=options, scale=scale,
                                decision_type=decision_type)
        try:
            async with httpx.AsyncClient(transport=self._transport,
                                         timeout=self.timeout_s) as client:
                response = await client.post(
                    url, json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}",
                             "Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise DecisionUnavailable(f"jev is unreachable: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise DecisionUnavailable(f"jev answered HTTP {response.status_code}")
        try:
            body = response.json()
            value = body["value"]
            confidence = float(body["confidence"])
        except (ValueError, KeyError, TypeError) as exc:
            raise DecisionUnavailable("jev answered something that is not a decision") from exc
        if not 0.0 <= confidence <= 1.0:
            raise DecisionUnavailable(f"jev reported confidence {confidence}, outside 0..1")
        return value, confidence


__all__ = ["JevDecider"]
