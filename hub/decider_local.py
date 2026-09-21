"""The local model as a decision provider (ТЗ 5.2).

``RulesDecider`` answers the questions heuristics can answer and nothing else;
this provider answers the ones that need reading the utterance. It asks the
same local model the reply comes from (vLLM by default, see F-402) under a
JSON schema, so the answer is always one of the options the caller offered —
and for a yes/no question it first asks for ``logprobs``: a probability can be
compared with the confidence policy of ТЗ 5.4, while the word alone cannot.

Nothing here decides anything on its own: it returns a :class:`Decision` (or
raises :class:`DecisionUnavailable` so the chain moves on to the next
provider). The chain owns the timeout, the order and the recording.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from hub.decider import Decision, DecisionUnavailable, _decision  # noqa: PLC2701 - same package
from hub.llm import StructuredUnavailable

log = logging.getLogger("jarvis.server.decider")

T = TypeVar("T")

#: How sure a schema-constrained answer is treated as. The model picked one of
#: the offered options, but it gave no probability, so this stays below the
#: "act on it" threshold of the policies: a decision without a number behind it
#: is logged and, where the policy asks, confirmed rather than trusted blindly.
SCHEMA_CONFIDENCE = 0.7
#: How sure a scaled score (1..5) is treated as.
SCORE_CONFIDENCE = 0.6

SYSTEM_PROMPT = (
    "You are the decision layer of a voice assistant. You are given one "
    "question about one utterance and must answer it, never chat. "
    "Answer only with the requested JSON object."
)


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def _word(token: str) -> str:
    """``" yes"``/``"Yes"``/``"YES"`` → ``yes``: the vocabulary token's word."""
    return re.sub(r"[^0-9a-z]+", "", (token or "").strip().lower())


class LocalLLMDecider(Generic[T]):
    """A :class:`hub.decider.Decider` backed by the local model (ТЗ 5.2)."""

    name = "local_llm"

    def __init__(self, client: Any, *, name: str = "local_llm", max_tokens: int = 24) -> None:
        #: The chat client (``hub.llm.LlmClient``): anything with
        #: ``structured_json`` and (optionally) ``first_token_probabilities``.
        self._client = client
        self.name = name
        self._max_tokens = int(max_tokens)

    # --- the three questions of the interface ------------------------------

    async def yes_no(self, question: str, context: dict[str, Any], *,
                     decision_type: str) -> Decision[bool]:
        messages = self._messages(question, context, "Answer with one word: yes or no.")
        started = time.perf_counter()
        probabilities = await self._probabilities(messages)
        value, confidence = self._from_probabilities(probabilities, ("yes", "no"))
        if value is None:
            answer = await self._ask(messages, _schema({"answer": {"type": "string",
                                                                  "enum": ["yes", "no"]}},
                                                      ["answer"]))
            word = str(answer.get("answer") or "").strip().lower()
            if word not in {"yes", "no"}:
                raise DecisionUnavailable(f"the model answered {word!r} to a yes/no question")
            value, confidence = word == "yes", SCHEMA_CONFIDENCE
        return _decision(bool(value), confidence, self.name, started,
                         self._input(question, context))

    async def choose(self, question: str, options: Sequence[str], context: dict[str, Any], *,
                     decision_type: str) -> Decision[str]:
        choices = [str(option) for option in options]
        if not choices:
            raise DecisionUnavailable("there is nothing to choose from")
        messages = self._messages(
            question, context,
            "Pick exactly one of these options and answer with the option text: "
            + ", ".join(choices) + ".",
        )
        started = time.perf_counter()
        answer = await self._ask(messages, _schema(
            {"answer": {"type": "string", "enum": choices}}, ["answer"]))
        value = str(answer.get("answer") or "")
        if value not in choices:
            raise DecisionUnavailable(f"the model picked {value!r}, which was not offered")
        return _decision(value, SCHEMA_CONFIDENCE, self.name, started,
                         self._input(question, context))

    async def score(self, question: str, context: dict[str, Any], *, scale: tuple[int, int],
                    decision_type: str) -> Decision[int]:
        low, high = int(scale[0]), int(scale[1])
        messages = self._messages(
            question, context,
            f"Answer with a whole number from {low} to {high}.",
        )
        started = time.perf_counter()
        answer = await self._ask(messages, _schema(
            {"score": {"type": "integer", "minimum": low, "maximum": high}}, ["score"]))
        raw = answer.get("score")
        if isinstance(raw, bool) or not isinstance(raw, int) or not low <= raw <= high:
            raise DecisionUnavailable(f"the model scored {raw!r} outside {scale}")
        return _decision(int(raw), SCORE_CONFIDENCE, self.name, started,
                         self._input(question, context))

    # --- talking to the model ---------------------------------------------

    def _messages(self, question: str, context: dict[str, Any], instruction: str) -> list[dict[str, str]]:
        facts = json.dumps(context, ensure_ascii=False, default=str)[:4000]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nContext: {facts}\n{instruction}"},
        ]

    @staticmethod
    def _input(question: str, context: dict[str, Any]) -> str:
        text = context.get("text")
        return f"{question}: {text}" if text else question

    async def _probabilities(self, messages: list[dict[str, str]]) -> dict[str, float]:
        """First-token probabilities of a yes/no answer, or ``{}`` without logprobs."""
        reader = getattr(self._client, "first_token_probabilities", None)
        if not callable(reader):
            return {}
        try:
            probabilities = await reader(messages, max_tokens=1)
        except Exception as exc:  # noqa: BLE001 - the schema path is the fallback
            log.debug("first-token probabilities unavailable (%s)", exc)
            return {}
        if not isinstance(probabilities, dict):
            return {}
        normalized: dict[str, float] = {}
        for key, value in probabilities.items():
            if not isinstance(value, (int, float)):
                continue
            word = _word(str(key))
            if word:
                normalized[word] = normalized.get(word, 0.0) + float(value)
        return normalized

    @staticmethod
    def _from_probabilities(probabilities: dict[str, float],
                            words: tuple[str, str]) -> tuple[bool | None, float]:
        """``(value, confidence)`` from a first-token distribution, or ``(None, 0)``.

        The two words are normally where almost all of the mass sits; anything
        else (a chatty model, a token split differently) leaves the choice to
        the schema path instead of guessing.
        """
        yes, no = (probabilities.get(word, 0.0) for word in words)
        total = yes + no
        if total <= 0.0:
            return None, 0.0
        return yes >= no, (yes if yes >= no else no) / total

    async def _ask(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        asker = getattr(self._client, "structured_json", None)
        if not callable(asker):
            raise DecisionUnavailable("the client cannot answer under a JSON schema")
        try:
            answer = await asker(messages, schema, name="decision")
        except StructuredUnavailable as exc:
            raise DecisionUnavailable(str(exc)) from None
        except Exception as exc:  # noqa: BLE001 - a broken endpoint is not an answer
            raise DecisionUnavailable(f"the local model could not be asked ({exc})") from None
        if not isinstance(answer, dict):
            raise DecisionUnavailable("the local model returned no decision object")
        return answer


__all__ = ["LocalLLMDecider", "SCHEMA_CONFIDENCE", "SCORE_CONFIDENCE", "SYSTEM_PROMPT"]
