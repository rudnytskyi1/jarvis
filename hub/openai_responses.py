"""Budgeted text-only OpenAI transport; no automatic retries or hosted tools.

Only models with reviewed pricing are accepted. Tool execution/permissions stay in
Connection. Each function round is reserved separately before network I/O.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from common.openai_models import OPENAI_TEXT_RATES
from hub.api_budget import ApiBudget, CloudUnavailable

log = logging.getLogger(__name__)


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
        self.max_output = min(int(cfg.max_tokens), 2048)
        self.max_input_bytes = int(getattr(cfg, "max_input_bytes", 64000))
        self.budget = ApiBudget(ledger_path or Path(__file__).resolve().parents[1] / "data" / "api_usage.sqlite3",
                                getattr(cfg, "monthly_budget_usd", 18.0), model=self.model)
        self.http = httpx.Client(timeout=httpx.Timeout(30, connect=5), transport=transport,
                                 follow_redirects=False)

    def complete(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        key = os.environ.get(self.key_env, "").strip()
        if not key:
            raise CloudUnavailable(f"Set {self.key_env} on the server to enable OpenAI. Local commands still work.")
        # strict=False preserves optional arguments in the existing schemas.
        definitions = [{"type": "function", **t["function"], "strict": False} for t in tools]
        payload = {"model": self.model, "input": response_input(messages), "tools": definitions,
                   "max_output_tokens": self.max_output, "reasoning": {"effort": "none"},
                   "parallel_tool_calls": False, "store": False, "service_tier": "default"}
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(encoded) > self.max_input_bytes:
            raise CloudUnavailable("Conversation is too long for the configured API allowance. Start a new conversation.")
        # Conservative text byte estimate plus framing/schema headroom. This is
        # not an exact tokenizer. Usage is reconciled without assuming caching.
        try:
            reservation = self.budget.reserve(len(encoded) * 2 + 4096, self.max_output)
        except CloudUnavailable:
            raise
        except Exception as exc:
            raise CloudUnavailable("API accounting is unavailable; no request was sent.") from exc
        try:
            response = self.http.post("https://api.openai.com/v1/responses",
                                      headers={"Authorization": f"Bearer {key}"}, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            # Keep the reservation even on timeout/disconnect: billing is unknown.
            log.warning("OpenAI request failed (%s); reservation retained", type(exc).__name__)
            raise CloudUnavailable("OpenAI is unavailable. I couldn't finish this request.") from exc
        usage = data.get("usage") or {}
        try:
            self.budget.settle(reservation, usage["input_tokens"], usage["output_tokens"],
                               input_tokens_details=usage.get('input_tokens_details'))
        except Exception as exc:
            # A valid response may be used, but no money is refunded on missing usage.
            log.warning("Could not reconcile OpenAI usage (%s); reservation retained", type(exc).__name__)
        if data.get("status") != "completed":
            raise CloudUnavailable("The model did not finish its response; no partial tool calls were executed.")
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

    def close(self):
        self.http.close()
