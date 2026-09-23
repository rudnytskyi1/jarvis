"""Budgeted text-only OpenAI transport; no automatic retries or hosted tools.

Only models with reviewed pricing are accepted. Tool execution/permissions stay in
Connection. Each function round is reserved separately before network I/O.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from common.openai_models import OPENAI_TEXT_RATES
from hub.api_budget import ApiBudget, CloudUnavailable

log = logging.getLogger(__name__)

#: Where the OpenAI-shaped Responses API lives when the config names no base.
DEFAULT_RESPONSES_BASE = "https://api.openai.com/v1"


def responses_url(base_url: str | None) -> str:
    """The one request URL of this transport, from the configured base URL.

    The owner moved the hub from gpt-5.6-luna to DeepSeek on 2026-09-22, and
    DeepSeek speaks the same Responses API at its own host
    (``https://api.deepseek.com/v1/responses``, verified live). The transport
    therefore follows ``server.llm.base_url`` (and each model level's own
    ``base_url``) instead of a hard-coded host, so no code change is needed to
    move a provider again.
    """
    base = str(base_url or "").strip() or DEFAULT_RESPONSES_BASE
    base = base.rstrip("/")
    if base.endswith("/responses"):
        return base
    return base + "/responses"


#: Where a shortened text stops being a text and starts being a fragment.
TRUNCATION_MARKER = "\n[... middle of a very long turn omitted ...]\n"


def shorten_text(text: str, limit: int) -> str:
    """Keep the head and the tail of ``text`` inside ``limit`` UTF-8 bytes."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    keep = max(64, limit // 2)
    tail = max(0, limit - keep - len(TRUNCATION_MARKER.encode("utf-8")))
    return (raw[:keep] + TRUNCATION_MARKER.encode("utf-8") + (raw[-tail:] if tail else b"")).decode(
        "utf-8", "ignore")


def response_input(messages: list[dict]) -> list[dict]:
    items = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("budgeted provider accepts text only")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": content})
        else:
            if content:
                items.append({"role": role, "content": content})
            for call in message.get("tool_calls", []):
                fn = call["function"]
                items.append({"type": "function_call", "call_id": call["id"],
                              "name": fn["name"], "arguments": fn["arguments"]})
    return items


class ResponsesClient:
    def __init__(self, cfg: Any, *, ledger_path: Path | None = None, transport=None):
        if cfg.model not in OPENAI_TEXT_RATES:
            raise ValueError("Budgeted OpenAI requires reviewed model pricing")
        self.model = cfg.model
        self.key_env = getattr(cfg, "api_key_env", "OPENAI_API_KEY")
        self.url = responses_url(getattr(cfg, "base_url", ""))
        #: The name the person hears when this provider cannot be reached. It
        #: follows the configured host, so a DeepSeek hub does not blame OpenAI.
        self.service = "DeepSeek" if "deepseek" in urlsplit(self.url).netloc else "OpenAI"
        self.max_output = min(int(cfg.max_tokens), 2048)
        self.max_input_bytes = int(getattr(cfg, "max_input_bytes", 64000))
        self.budget = ApiBudget(ledger_path or Path(__file__).resolve().parents[1] / "data" / "api_usage.sqlite3",
                                getattr(cfg, "monthly_budget_usd", 18.0), model=self.model)
        self.http = httpx.Client(timeout=httpx.Timeout(30, connect=5), transport=transport,
                                 follow_redirects=False)

    def complete(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        key = os.environ.get(self.key_env, "").strip()
        if not key:
            raise CloudUnavailable(f"Set {self.key_env} on the server to enable "
                                   f"{self.service}. Local commands still work.")
        # strict=False preserves optional arguments in the existing schemas.
        definitions = [{"type": "function", **t["function"], "strict": False} for t in tools]
        payload = {"model": self.model, "input": response_input(messages), "tools": definitions,
                   "max_output_tokens": self.max_output, "reasoning": {"effort": "none"},
                   "parallel_tool_calls": False, "store": False, "service_tier": "default"}
        data = self._request(payload, key)
        texts, calls = [], []
        for item in data.get("output", []):
            if item.get("type") == "function_call":
                calls.append({"id": item["call_id"], "type": "function",
                              "function": {"name": item["name"], "arguments": item["arguments"]}})
            elif item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        texts.append(part["text"])
                    elif part.get("type") == "refusal":
                        texts.append(part["refusal"])
        return "\n".join(texts), calls

    # --- one budgeted look at an image (ТЗ F-404) ---------------------------

    def describe_image(self, jpeg: bytes, prompt: str, *, max_output: int | None = None) -> str:
        """Answer ``prompt`` about one JPEG, charged to the same ledger (F-404).

        The call is deliberately shaped like the text one: the reservation is
        taken BEFORE the network, the usage is reconciled after, and a failure
        keeps the reservation because a timeout may still have been billed.
        """
        key = os.environ.get(self.key_env, "").strip()
        if not key:
            raise CloudUnavailable(f"Set {self.key_env} on the server to enable {self.service}. "
                                   "Local commands still work.")
        if not jpeg:
            raise CloudUnavailable("There is no image to look at.")
        question = " ".join(str(prompt or "").split()) or "Describe what is in this image."
        image_url = "data:image/jpeg;base64," + base64.b64encode(bytes(jpeg)).decode("ascii")
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": question},
                {"type": "input_image", "image_url": image_url},
            ]}],
            "max_output_tokens": min(int(max_output or self.max_output), 2048),
            "reasoning": {"effort": "none"},
            "parallel_tool_calls": False,
            "store": False,
            "service_tier": "default",
        }
        data = self._request(payload, key,
                             too_long="This image is too large for the configured API allowance.")
        texts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        texts.append(part["text"])
        return "\n".join(texts).strip()

    def _request(self, payload: dict, key: str, *, too_long: str | None = None) -> dict:
        """One charged round trip: reserve, post, settle, hand back the body.

        A body over ``max_input_bytes`` is TRIMMED, never refused (the owner
        asked on 2026-09-22 — DECISIONS.md API-02): the oldest turns are
        dropped and the newest request always survives, so a long conversation
        keeps answering instead of saying "conversation is too long".
        """
        encoded = self._fit(payload)
        if len(encoded) > self.max_input_bytes and too_long is not None:
            # Only a single picture ends up here: a data URL cannot be
            # truncated without sending broken bytes, so it is reported.
            raise CloudUnavailable(too_long)
        if len(encoded) > self.max_input_bytes:
            log.warning(
                "Request body is %d bytes over the configured %d; sending the trimmed body anyway",
                len(encoded), self.max_input_bytes)
        # Conservative byte estimate plus framing/schema headroom. This is not
        # an exact tokenizer, and an image is charged by the API's own rules, so
        # the reservation stays generous on purpose.
        try:
            reservation = self.budget.reserve(len(encoded) * 2 + 4096, int(payload["max_output_tokens"]))
        except CloudUnavailable:
            raise
        except Exception as exc:
            raise CloudUnavailable("API accounting is unavailable; no request was sent.") from exc
        try:
            response = self.http.post(self.url,
                                      headers={"Authorization": f"Bearer {key}"}, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            # Keep the reservation even on timeout/disconnect: billing is unknown.
            log.warning("OpenAI request failed (%s); reservation retained", type(exc).__name__)
            raise CloudUnavailable(f"{self.service} is unavailable. "
                                   "I couldn't finish this request.") from exc
        usage = data.get("usage") or {}
        try:
            self.budget.settle(reservation, usage["input_tokens"], usage["output_tokens"],
                               input_tokens_details=usage.get('input_tokens_details'))
        except Exception as exc:
            # A valid response may be used, but no money is refunded on missing usage.
            log.warning("Could not reconcile OpenAI usage (%s); reservation retained", type(exc).__name__)
        if data.get("status") != "completed":
            raise CloudUnavailable("The model did not finish its response; no partial tool calls were executed.")
        return data

    # --- fitting one request into the owner's ceiling ----------------------

    #: Text fields of an input item that may be shortened as a last resort.
    _TEXT_KEYS = ("content", "output", "text")

    def _fit(self, payload: dict) -> bytes:
        """Make ``payload`` fit, in place, and return its encoded bytes."""
        limit = self.max_input_bytes
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(encoded) <= limit:
            return encoded
        items = payload.get("input")
        dropped = 0
        if isinstance(items, list):
            while len(items) > 2:
                self._drop_oldest(items)
                dropped += 1
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if len(encoded) <= limit:
                    break
        if dropped:
            log.info("Trimmed %d older input item(s) so the request fits %d bytes",
                     dropped, limit)
        if len(encoded) > limit:
            encoded = self._cap_texts(payload, limit)
        return encoded

    @staticmethod
    def _drop_oldest(items: list) -> None:
        """Forget the oldest turn, keeping function calls paired with results."""
        victim = items[1]
        call_id = victim.get("call_id") if isinstance(victim, dict) else None
        doomed = [1]
        if call_id is not None:
            doomed += [index for index, item in enumerate(items)
                       if index > 1 and isinstance(item, dict) and item.get("call_id") == call_id]
        for index in sorted(doomed, reverse=True):
            del items[index]

    def _cap_texts(self, payload: dict, limit: int) -> bytes:
        """Shorten the text of a body that has no whole turns left to drop."""
        floor = 512
        budget = max(floor, limit // 4)
        while True:
            self._truncate_texts(payload.get("input"), budget)
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if len(encoded) <= limit or budget <= floor:
                return encoded
            budget = max(floor, budget // 2)

    @classmethod
    def _truncate_texts(cls, node: Any, budget: int) -> None:
        """Cap every text field of ``node`` at ``budget`` bytes, in place."""
        if isinstance(node, dict):
            for key, value in node.items():
                if key in cls._TEXT_KEYS and isinstance(value, str):
                    node[key] = shorten_text(value, budget)
                elif key != "image_url":
                    cls._truncate_texts(value, budget)
        elif isinstance(node, list):
            for item in node:
                cls._truncate_texts(item, budget)

    def close(self):
        self.http.close()
