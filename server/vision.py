"""Screen vision for the ``look_at_screen`` tool (SPEC §3, §5).

The client sends a JPEG screenshot of the room PC; this module asks a local
Ollama vision model about it and returns the answer as plain text. The answer
goes straight back to the LLM as a tool result, so failures are reported as text
and never raised.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

from server.llm import native_base_url

log = logging.getLogger("jarvis.server.vision")

#: Vision models are slow; SPEC §4 allows up to 120 s for the whole step.
REQUEST_TIMEOUT_S = 120.0

#: Room left for the answer — it is read aloud, so it must stay short.
MAX_ANSWER_TOKENS = 512

#: Answers longer than this are truncated before they reach the LLM.
MAX_ANSWER_CHARS = 2000

DEFAULT_QUERY = "Describe what is currently on the screen."

#: Wrapper around the LLM's question; keeps the vision model terse and factual.
PROMPT_TEMPLATE = (
    "You are looking at a screenshot of a Windows PC screen.\n"
    "Answer the question below from what you can actually see. Be concrete: name "
    "the application, window title, page or game, and read out any text that "
    "matters (error messages, headings, values). If the screen does not show the "
    "answer, say so plainly. Answer in English, in at most three short sentences, "
    "with no markdown.\n\n"
    "Question: {query}"
)


def _extract_answer(data: Any) -> str:
    """Pull the assistant text out of an Ollama ``/api/chat`` response."""
    if not isinstance(data, dict):
        return ""
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    response = data.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    return ""


class VisionClient:
    """Ollama vision model wrapper used by the ``look_at_screen`` tool."""

    def __init__(self, cfg_llm: Any) -> None:
        self.base_url = native_base_url(str(cfg_llm.base_url))
        self.model = str(getattr(cfg_llm, "vision_model", "") or "qwen3-vl:30b")
        try:
            self.temperature = float(getattr(cfg_llm, "temperature", 0.6))
        except (TypeError, ValueError):
            self.temperature = 0.6
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT_S)
        log.info("Vision model: %s via %s", self.model, self.base_url)

    # ------------------------------------------------------------------ request

    def _request(self, jpeg_bytes: bytes, query: str) -> str:
        """Blocking POST to ``/api/chat`` with the screenshot attached."""
        payload = {
            "model": self.model,
            "stream": False,
            # Keep the (small) vision model resident next to the chat model so
            # repeated screen questions do not reload it from disk.
            "keep_alive": "4h",
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(query=query),
                    "images": [base64.b64encode(jpeg_bytes).decode("ascii")],
                }
            ],
            "options": {
                "num_predict": MAX_ANSWER_TOKENS,
                "temperature": self.temperature,
            },
        }
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        return _extract_answer(response.json())

    async def describe_screenshot(self, jpeg_bytes: bytes, query: str | None = None) -> str:
        """Answer ``query`` about ``jpeg_bytes``; returns an error string on failure."""
        question = " ".join(str(query or "").split()) or DEFAULT_QUERY
        if not jpeg_bytes:
            return "Screen check failed: the screenshot is empty."

        log.info(
            "Asking %s about the screen (%d KB): %r",
            self.model,
            len(jpeg_bytes) // 1024,
            question,
        )
        try:
            answer = await asyncio.to_thread(self._request, jpeg_bytes, question)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()[:300] if exc.response is not None else ""
            log.error("Vision model returned HTTP %s: %s", exc.response.status_code, detail)
            return (
                f"Screen check failed: the vision model returned HTTP "
                f"{exc.response.status_code}. Is {self.model!r} pulled in Ollama?"
            )
        except httpx.TimeoutException:
            log.error("Vision model timed out after %.0f s", REQUEST_TIMEOUT_S)
            return "Screen check failed: the vision model did not answer in time."
        except Exception as exc:
            log.exception("Vision request failed")
            return f"Screen check failed: {exc}"

        if not answer:
            log.warning("Vision model returned an empty answer")
            return "Screen check failed: the vision model returned an empty answer."
        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS].rstrip() + "…"
        log.info("Vision answer: %r", answer)
        return answer

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            log.debug("Could not close the vision HTTP client", exc_info=True)


async def describe_screenshot(
    jpeg_bytes: bytes, query: str | None, cfg_llm: Any
) -> str:
    """One-shot helper matching SPEC §3 for callers without a :class:`VisionClient`."""
    client = VisionClient(cfg_llm)
    try:
        return await client.describe_screenshot(jpeg_bytes, query)
    finally:
        client.close()


__all__ = ["VisionClient", "describe_screenshot", "REQUEST_TIMEOUT_S", "DEFAULT_QUERY"]
