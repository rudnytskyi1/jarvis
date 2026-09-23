"""JevDecider — провайдер решений TypeSafe AI Jev (ТЗ 5.2, 5.5).

ТЗ описывает три операции интерфейса ``Decider``, и у Jev им соответствуют три
примитива System One: ``noul`` (вероятность «да»), ``choice`` (выбор из
вариантов) и ``score`` (оценка по рубрике). Провайдер тонкий: он собирает
запрос, зовёт HTTP-ручку и превращает ответ в ``Decision``; логики решений
здесь нет — она в цепочке (``DecisionChain``), которая даёт каждому
провайдеру свой таймаут и уходит к следующему при отказе.

Ручка и форма тела взяты из официального SDK ``typesafe-sdk`` 0.7.1
(``SYSTEM_ONE_PATH = /v1/systemone``, тело ``{state, model, questions}``,
ответ ``{"answers": {name: {"type": ..., ...}}}``); сам SDK сюда не тянуть не
нужно — запрос тот же, а транспорт свой и подменяемый в тестах. Ключ может
быть от TypeSafe напрямую (``base_url: https://api.typesafe.ai``) или от
OpenRouter (``base_url: https://openrouter.ai/api``): модель всё равно
называет сервер, а имя провайдера остаётся ``jev``.

Честность важнее вида:

* без ключа провайдера просто нет (его не собирает ``hub/app.py``), поэтому
  хаб решает на локальных провайдерах, а не выдумывает вердикт;
* флаг дома ``cloud_decisions`` (по умолчанию false, ТЗ 5.5) выключает Jev для
  комнаты: запрос не уходит вовсе, цепочка идёт дальше;
* в облако уходит только текст и метаданные: ``_privacy_context`` пропускает
  скаляры и выбрасывает кадры, звук и эмбеддинги (ТЗ 5.5);
* любой отказ (нет сети, таймаут, не 200, битый JSON, значение вне вариантов,
  тип ответа не тот, что спросили) — это ``DecisionUnavailable`` с причиной
  словами, а не решение «на глазок».
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from hub.decider import Decision, DecisionUnavailable, _decision

log = logging.getLogger("jarvis.server.jev")

#: Путь System One в API TypeSafe (typesafe-sdk 0.7.1: ``/v1/systemone``).
DEFAULT_PATH = "/v1/systemone"
#: Модель по умолчанию: сервер сам решает, какая версия Jev за ней стоит.
DEFAULT_MODEL = "jev-latest"
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

    def __init__(self, *, base_url: str, api_key: str, path: str = DEFAULT_PATH,
                 model: str = DEFAULT_MODEL,
                 timeout_s: float = 1.5, transport: Any = None,
                 allowed_for: Callable[[str], bool] | None = None) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.path = str(path or DEFAULT_PATH)
        self.model = str(model or DEFAULT_MODEL)
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
        """The one place the request shape lives (ТЗ 5.2, System One).

        ``state`` carries the text to judge; the question itself is typed:
        ``noul`` for yes/no, ``choice`` with named criteria, ``score`` with an
        ordered rubric. ``decision_type`` is not sent — the server answers the
        question it is given, and our own calibration lives in the chain.
        """
        return {
            "state": self._state(question, context, decision_type=decision_type),
            "model": self.model,
            "questions": {"answer": self._question(mode, question, options, scale)},
        }

    def _state(self, question: str, context: Mapping[str, Any], *, decision_type: str = "") -> Any:
        """Текст и метаданные: то, из чего Jev делает вывод (ТЗ 5.5)."""
        clean = _privacy_context(context)
        text = str(clean.pop("text", "") or question or "")
        if decision_type and "decision" not in clean:
            clean["decision"] = str(decision_type)
        return {**clean, "text": text} if clean else text

    @staticmethod
    def _question(mode: str, question: str, options: list[str] | None,
                  scale: list[int] | None,
                  meanings: Mapping[str, str] | None = None) -> dict[str, Any]:
        """The typed question of one Decider operation.

        ``meanings`` are the ``criteria`` descriptions of a choice question -
        the SDK's own example pairs every option with one ("billing": "Charges
        and refunds"), and without them the server answers an obvious question
        with a confidence too low to act on (see TOOL_FAMILY_MEANINGS).
        """
        if mode == "yes_no":
            return {"type": "noul", "instructions": str(question or "")}
        if mode == "choose":
            known = dict(meanings or {})
            offered = {str(option): known.get(str(option)) or str(option)
                       for option in (options or [])}
            if not offered:
                raise DecisionUnavailable("a choice question needs options")
            return {"type": "choice", "instructions": str(question or ""),
                    "criteria": offered}
        # score: the rubric runs from the low end of the scale to the high end,
        # one label per level; the answer comes back as a rubric index, so the
        # label carries the caller's own level number.
        low, high = int((scale or [0, 0])[0]), int((scale or [0, 0])[1])
        if high < low:
            raise DecisionUnavailable(f"a score question needs a scale, got {scale!r}")
        return {"type": "score", "instructions": str(question or ""),
                "criteria": [f"score {level}" for level in range(low, high + 1)]}

    def _answer_of(self, mode: str, body: Mapping[str, Any], *,
                   options: list[str] | None, scale: list[int] | None) -> tuple[Any, float]:
        """Turn the typed answer into ``(value, confidence)``, or refuse."""
        answers = body.get("answers")
        if not isinstance(answers, Mapping):
            raise DecisionUnavailable("jev answered without answers")
        answer = answers.get("answer")
        if not isinstance(answer, Mapping):
            raise DecisionUnavailable("jev did not answer the question")
        return self._answer_item(mode, answer, options=options, scale=scale)

    def _answer_item(self, mode: str, answer: Mapping[str, Any], *,
                     options: list[str] | None, scale: list[int] | None) -> tuple[Any, float]:
        """One answer of one typed question, validated against what was asked.

        Split out of :meth:`_answer_of` so the batched call can check each
        question on its own: an unusable answer to one question must not throw
        away the answers to the others (U-10).
        """
        kind = str(answer.get("type") or "")
        if mode == "yes_no":
            if kind != "noul":
                raise DecisionUnavailable(f"jev answered {kind or 'nothing'} to a yes/no question")
            try:
                probability = float(answer["noul"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DecisionUnavailable("jev sent a noul answer without a probability") from exc
            if not 0.0 <= probability <= 1.0:
                raise DecisionUnavailable(f"jev reported a probability of {probability}")
            # The answer is yes above one half; the confidence is how far the
            # probability is from a coin toss, so 0.51 is honestly unsure.
            return probability >= 0.5, max(probability, 1.0 - probability)
        if mode == "choose":
            if kind != "choice":
                raise DecisionUnavailable(f"jev answered {kind or 'nothing'} to a choice question")
            chosen = str(answer.get("choice") or "")
            if chosen not in {str(option) for option in (options or [])}:
                raise DecisionUnavailable(f"jev chose {chosen!r}, which was not offered")
            probabilities = answer.get("probabilities")
            fallback = probabilities.get(chosen, 0.0) if isinstance(probabilities, Mapping) else 0.0
            try:
                confidence = float(answer.get("confidence", fallback))
            except (TypeError, ValueError):
                confidence = float(fallback)
            return chosen, self._checked_confidence(confidence)
        if kind != "score":
            raise DecisionUnavailable(f"jev answered {kind or 'nothing'} to a score question")
        try:
            scored = float(answer["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DecisionUnavailable("jev sent a score answer without a score") from exc
        low, high = int((scale or [0, 0])[0]), int((scale or [0, 0])[1])
        # The API reports the rubric level, and the rubric starts at ``low``.
        offset = int(round(scored))
        if not 0 <= offset <= high - low:
            raise DecisionUnavailable(f"jev scored {scored}, outside the rubric {low}..{high}")
        try:
            confidence = float(answer.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return low + offset, self._checked_confidence(confidence)

    @staticmethod
    def _checked_confidence(confidence: float) -> float:
        if not 0.0 <= confidence <= 1.0:
            raise DecisionUnavailable(f"jev reported confidence {confidence}, outside 0..1")
        return confidence

    # --- one batched reading of a whole utterance (U-10) --------------------

    #: The three questions of one "understanding" call. They are asked together
    #: because System One answers every question from the same reading of
    #: ``state``: three separate calls cost three times the latency for the
    #: same judgement, and the turn's budget is 1.2 s (ТЗ 15.1).
    ACT_QUESTION = (
        "Is the person asking Rowan to DO something - to use a device, a computer, "
        "a camera, a picture, the browser, or to send a message - rather than just "
        "asking a question or making a remark?"
    )
    FAMILY_QUESTION = (
        "Which kind of action does this request need? Choose browser for a web page "
        "or a search, pc for programs, windows, files or typing on the computer, "
        "vision for looking at the screen or the camera, media for showing, saving, "
        "drawing or editing a picture, memory for remembering or recalling, people "
        "for faces, voices and roles, devices for lights and switches, notify for "
        "sending a message or making a rule, none when no action is asked for."
    )
    FOLLOWUP_QUESTION = (
        "Is this request a continuation of what Rowan was just doing or talking "
        "about (\"it\", \"that\", \"again\", \"the same\", \"close it\"), rather "
        "than a new request with no reference to the previous turn?"
    )

    async def understand(self, context: Mapping[str, Any], *, families: Sequence[str],
                         meanings: Mapping[str, str] | None = None,
                         timeout_s: float | None = None,
                         decision_type: str = "understanding") -> dict[str, Any]:
        """The whole reading of one utterance in ONE typed request (U-10).

        Returns ``{name: {"value": ..., "confidence": ...}}`` for the questions
        Jev answered. A question that is missing, of the wrong type, or outside
        the offered options is simply absent: the caller keeps its own
        behaviour for it, which is what makes this safe to put in front of
        every turn (docs/PLAN_UNDERSTANDING.md, section 3).
        """
        options = [str(name) for name in families] or ["none"]
        payload = {
            "model": self.model,
            "state": self._state("", context, decision_type=decision_type),
            "questions": {
                "act": self._question("yes_no", self.ACT_QUESTION, None, None),
                "family": self._question("choose", self.FAMILY_QUESTION, options, None,
                                         meanings=meanings),
                "followup": self._question("yes_no", self.FOLLOWUP_QUESTION, None, None),
            },
        }
        body = await self._post(payload, context, timeout_s=timeout_s)
        answers = body.get("answers")
        if not isinstance(answers, Mapping):
            raise DecisionUnavailable("jev answered without answers")
        found: dict[str, Any] = {}
        for name, mode, offered in (("act", "yes_no", None),
                                    ("family", "choose", options),
                                    ("followup", "yes_no", None)):
            item = answers.get(name)
            if not isinstance(item, Mapping):
                continue
            try:
                value, confidence = self._answer_item(mode, item, options=offered, scale=None)
            except DecisionUnavailable as exc:
                log.info("Jev did not answer %s (%s); that part keeps the old behaviour",
                         name, exc)
                continue
            found[name] = {"value": value, "confidence": confidence}
        if not found:
            raise DecisionUnavailable("jev answered none of the understanding questions")
        return found

    async def _ask(self, mode: str, question: str, context: Mapping[str, Any], *,
                   options: list[str] | None = None, scale: list[int] | None = None,
                   decision_type: str = "") -> tuple[Any, float]:
        payload = self._payload(mode, question, context, options=options, scale=scale,
                                decision_type=decision_type)
        body = await self._post(payload, context, timeout_s=None)
        return self._answer_of(mode, body, options=options, scale=scale)

    async def _post(self, payload: Mapping[str, Any], context: Mapping[str, Any], *,
                    timeout_s: float | None = None) -> Mapping[str, Any]:
        """One System One request, with the room's permission checked first."""
        home = str(dict(context or {}).get("home_id") or "")
        if not home or not self.allowed_for(home):
            raise DecisionUnavailable(
                f"cloud decisions are switched off for {home or 'this room'}")
        if not self.base_url:
            raise DecisionUnavailable("no Jev base URL is configured")
        if not self._api_key:
            raise DecisionUnavailable("no Jev API key is available in the environment")
        url = f"{self.base_url}{self.path if self.path.startswith('/') else '/' + self.path}"
        budget = self.timeout_s if timeout_s is None else max(0.05, float(timeout_s))
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=budget) as client:
                response = await client.post(
                    url, json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}",
                             "Accept": "application/json",
                             "Content-Type": "application/json"})
        except httpx.HTTPError as exc:
            raise DecisionUnavailable(f"jev is unreachable: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise DecisionUnavailable(f"jev answered HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise DecisionUnavailable("jev answered something that is not a decision") from exc
        if not isinstance(body, Mapping):
            raise DecisionUnavailable("jev answered something that is not a decision")
        return body


__all__ = ["DEFAULT_MODEL", "DEFAULT_PATH", "JevDecider"]
