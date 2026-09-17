"""Screen vision for the ``look_at_screen`` and ``click_screen`` tools (SPEC §3, §5).

The client sends a JPEG screenshot of the room PC; this module asks a local
Ollama vision model about it. There are two entry points, both going to the
native ``/api/chat`` endpoint with ``stream: false``, ``think: false``, a long
``keep_alive`` and a 120 s timeout:

* :meth:`VisionClient.describe_screenshot` — answers a question about the screen
  in specific detail (window titles, video and list titles with their channels,
  button labels, visible text). The answer goes straight back to the LLM as a
  tool result.
* :meth:`VisionClient.locate_on_screen` — Qwen-VL grounding: returns the
  normalized click point of a described element, or ``None`` when the model did
  not name one. The answer is mapped onto the screenshot's own pixels before it
  is normalized, see :data:`GROUNDING_GRID`.

Neither call ever raises: a failure becomes an error string (describe) or
``None`` (locate), because the caller has to keep the tool loop running.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any

import httpx

from server.llm import native_base_url

log = logging.getLogger("jarvis.server.vision")

#: Vision models are slow; SPEC §4 allows up to 120 s for the whole step.
REQUEST_TIMEOUT_S = 120.0

#: Room left for a screen description. The answer only feeds the chat model,
#: which re-phrases it for speech, so it must be specific but not an essay:
#: generation time scales with this, and it was the main cost of a screen
#: question (295 tokens ~= 4 s). ~180 tokens keeps every concrete detail.
MAX_ANSWER_TOKENS = 200

#: A click point is a handful of tokens; a tight budget keeps grounding fast.
MAX_POINT_TOKENS = 64

#: Answers longer than this are truncated before they reach the LLM.
MAX_ANSWER_CHARS = 2000

DEFAULT_QUERY = "Describe what is currently on the screen."

#: Wrapper around the LLM's question. The model must name what it sees: a
#: one-word answer ("YouTube") is useless to a voice assistant that has to read
#: the result out loud.
DESCRIBE_PROMPT_TEMPLATE = (
    "You are looking at a screenshot of a Windows PC screen.\n"
    "Answer the question below from what is actually visible, and be SPECIFIC. "
    "Name the application and the exact window or browser tab title. Quote the "
    "titles of videos, tracks, files or list entries together with their channel, "
    "author or artist. Name the labels written on the relevant buttons, and read "
    "out the text that matters: headings, error messages, values, the text in "
    "search boxes.\n"
    "Never answer with a single word or a bare category such as 'YouTube', 'a "
    "browser', 'a video' or 'a game' — always give the concrete names and text "
    "you can read. When several items are listed and the question is about them, "
    "list the first few by title, in order. If the screen genuinely does not "
    "contain the answer, say so and describe what is on it instead.\n"
    "Be concise: answer in at most TWO or THREE short sentences of plain "
    "English, packed with the concrete names and text, with no preamble, no "
    "markdown, no bullet points, no code. Do not describe the whole screen when "
    "the question is about one thing — answer the question.\n\n"
    "Question: {query}"
)

#: Qwen-VL grounding answers on a normalized 0-1000 grid, and it does so
#: whatever the prompt asks for: the configured model returns the very same
#: numbers for the same element in a 1600x900 and in a 1280x800 screenshot, and
#: claims they are "pixels" when asked to declare the scale. Asking for the grid
#: it actually uses is therefore the only way to aim the cursor correctly;
#: coordinates ABOVE this value cannot come from that grid and are read as raw
#: image pixels instead, so a vision model that really answers in pixels still
#: works.
GROUNDING_GRID = 1000

#: Grounding prompt. The model is told the real size of the image it is looking
#: at (SPEC §3) and answers on the normalized grid above.
LOCATE_PROMPT_TEMPLATE = (
    "You are looking at a screenshot of a Windows PC screen. The image is "
    "{width} pixels wide and {height} pixels high.\n"
    "Find this element: {target}\n"
    "Give the single best point to click on it — the centre of the element — on "
    "the normalized grid: x = 0 at the left edge and x = {grid} at the right "
    "edge, y = 0 at the top edge and y = {grid} at the bottom edge.\n"
    'Reply with strict JSON and nothing else: {{"x": <integer 0-{grid}>, '
    '"y": <integer 0-{grid}>}}. '
    "No explanation, no units, no markdown, no code fences.\n"
    "If the element is not visible anywhere on this screen, reply with exactly: "
    "not found"
)

#: Braces-delimited candidates for the JSON object the grounding prompt asks for.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")

#: Last-resort coordinate extraction: the first two integers in the reply.
_INT_RE = re.compile(r"-?\d+")


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


def _coerce_number(value: Any) -> float | None:
    """Read one coordinate from a JSON value (number, or a string like ``"640px"``)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            match = _INT_RE.search(text)
            if match is not None:
                return float(match.group())
    return None


def _point_from_mapping(data: Any) -> tuple[float, float] | None:
    """Return ``(x, y)`` from a parsed ``{"x": …, "y": …}`` object."""
    if not isinstance(data, dict):
        return None
    x = _coerce_number(data.get("x"))
    y = _coerce_number(data.get("y"))
    if x is None or y is None:
        return None
    return x, y


def parse_point(reply: str) -> tuple[float, float] | None:
    """Parse the model's answer into raw pixel coordinates (SPEC §3).

    JSON is tried first — the whole reply, then every ``{…}`` block inside it,
    which survives code fences and a chatty sentence around the object. Only if
    that finds nothing do the first two integers of the reply win.
    """
    text = str(reply or "").strip()
    if not text:
        return None

    try:
        point = _point_from_mapping(json.loads(text))
    except (ValueError, TypeError):
        point = None
    if point is not None:
        return point

    for match in _JSON_OBJECT_RE.finditer(text):
        try:
            candidate = json.loads(match.group())
        except (ValueError, TypeError):
            continue
        point = _point_from_mapping(candidate)
        if point is not None:
            return point

    numbers = _INT_RE.findall(text)
    if len(numbers) >= 2:
        try:
            return float(numbers[0]), float(numbers[1])
        except ValueError:  # pragma: no cover - the regex only matches integers
            return None
    return None


class VisionClient:
    """Ollama vision model wrapper used by ``look_at_screen`` and ``click_screen``."""

    def __init__(self, cfg_llm: Any) -> None:
        self.base_url = native_base_url(str(cfg_llm.base_url))
        self.model = str(getattr(cfg_llm, "vision_model", "") or "qwen3-vl:30b")
        try:
            self.temperature = float(getattr(cfg_llm, "temperature", 0.6))
        except (TypeError, ValueError):
            self.temperature = 0.6
        #: Keep the vision model resident between screen questions, exactly like
        #: the chat model — with one model serving both there is nothing to swap.
        self.keep_alive = str(getattr(cfg_llm, "keep_alive", "4h") or "4h")
        #: Small context on purpose. The vision model (qwen3-vl:8b) is a SEPARATE
        #: model from the chat model now, so it must fit in VRAM ALONGSIDE millard
        #: (22 GB) with OLLAMA_MAX_LOADED_MODELS>=2. A vision prompt + one image is
        #: well under 4k tokens; 4096 keeps the model at ~6 GB (vs ~10 GB at its
        #: 32k default) so both stay resident and there is nothing to swap.
        self.num_ctx = 4096
        #: Cleared when the server rejects the "think" field (older Ollama builds).
        self._send_think = True
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT_S)
        log.info("Vision model: %s via %s", self.model, self.base_url)

    # ------------------------------------------------------------------ request

    def _request(
        self,
        jpeg_bytes: bytes,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Blocking POST to ``/api/chat`` with the screenshot attached."""
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(jpeg_bytes).decode("ascii")],
                }
            ],
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if self._send_think:
            # Reasoning off: the answer must be the description, not a monologue.
            payload["think"] = False

        url = f"{self.base_url}/api/chat"
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            if self._send_think and "think" in body.lower():
                # Older Ollama builds reject the field for non-reasoning models.
                log.warning("Ollama rejected the 'think' field — retrying without it")
                self._send_think = False
                payload.pop("think", None)
                response = self._client.post(url, json=payload)
                response.raise_for_status()
            else:
                raise
        return _extract_answer(response.json())

    async def _ask(
        self,
        jpeg_bytes: bytes,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, str]:
        """Run one vision request off the event loop.

        Returns ``(answer, error)`` — exactly one of them is filled in, so no
        caller of this module ever has to handle an exception.
        """
        try:
            answer = await asyncio.to_thread(
                self._request, jpeg_bytes, prompt, max_tokens, temperature
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            detail = exc.response.text.strip()[:300] if exc.response is not None else ""
            log.error("Vision model returned HTTP %s: %s", status, detail)
            return "", (
                f"the vision model returned HTTP {status}. "
                f"Is {self.model!r} pulled in Ollama?"
            )
        except httpx.TimeoutException:
            log.error("Vision model timed out after %.0f s", REQUEST_TIMEOUT_S)
            return "", "the vision model did not answer in time."
        except Exception as exc:  # noqa: BLE001 - the tool result must survive anything
            log.exception("Vision request failed")
            return "", f"{exc}"
        return answer, ""

    # ------------------------------------------------------------------ describe

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
        answer, error = await self._ask(
            jpeg_bytes,
            DESCRIBE_PROMPT_TEMPLATE.format(query=question),
            MAX_ANSWER_TOKENS,
            self.temperature,
        )
        if error:
            return f"Screen check failed: {error}"
        if not answer:
            log.warning("Vision model returned an empty answer")
            return "Screen check failed: the vision model returned an empty answer."
        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS].rstrip() + "…"
        log.info("Vision answer: %r", answer)
        return answer

    # ------------------------------------------------------------------ grounding

    async def locate_on_screen(
        self, jpeg_bytes: bytes, target: str, img_w: int, img_h: int
    ) -> tuple[float, float] | None:
        """Locate ``target`` in the screenshot and return a normalized click point.

        :param jpeg_bytes: the screenshot as sent by the client.
        :param target: the visual description the LLM gave to ``click_screen``.
        :param img_w: width of ``jpeg_bytes`` in pixels (from the screenshot header).
        :param img_h: height of ``jpeg_bytes`` in pixels.
        :returns: ``(x_norm, y_norm)`` in ``0..1``, or ``None`` when the model did
            not answer with usable coordinates. Never raises.
        """
        description = " ".join(str(target or "").split())
        if not jpeg_bytes:
            log.warning("Cannot locate %r: the screenshot is empty", description)
            return None
        if not description:
            log.warning("Cannot locate an element without a description")
            return None
        try:
            width = int(img_w)
            height = int(img_h)
        except (TypeError, ValueError):
            log.warning("Cannot locate %r: image size %r x %r is not numeric", description, img_w, img_h)
            return None
        if width <= 0 or height <= 0:
            log.warning("Cannot locate %r: image size %dx%d is invalid", description, width, height)
            return None

        log.info("Asking %s to locate %r in a %dx%d screenshot", self.model, description, width, height)
        answer, error = await self._ask(
            jpeg_bytes,
            LOCATE_PROMPT_TEMPLATE.format(
                width=width, height=height, target=description, grid=GROUNDING_GRID
            ),
            MAX_POINT_TOKENS,
            # Grounding is a lookup, not a creative task: keep it deterministic.
            0.0,
        )
        if error:
            log.warning("Could not locate %r: %s", description, error)
            return None
        if not answer:
            log.warning("The vision model returned nothing for %r", description)
            return None

        point = parse_point(answer)
        if point is None:
            log.warning("No coordinates in the vision reply for %r: %r", description, answer)
            return None

        raw_x, raw_y = point
        if raw_x <= GROUNDING_GRID and raw_y <= GROUNDING_GRID:
            # The normalized grounding grid the prompt asked for.
            space = f"0-{GROUNDING_GRID}"
            x = raw_x / GROUNDING_GRID * width
            y = raw_y / GROUNDING_GRID * height
        else:
            # Off the grid: the model answered in raw pixels of the image.
            space = "pixels"
            x, y = raw_x, raw_y
        x = min(max(x, 0.0), float(width))
        y = min(max(y, 0.0), float(height))
        x_norm = x / float(width)
        y_norm = y / float(height)
        log.info(
            "Located %r at %.0f,%.0f px of %dx%d -> %.3f,%.3f (%s reply %r)",
            description, x, y, width, height, x_norm, y_norm, space, answer,
        )
        return x_norm, y_norm

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            log.debug("Could not close the vision HTTP client", exc_info=True)


async def describe_screenshot(jpeg_bytes: bytes, query: str | None, cfg_llm: Any) -> str:
    """One-shot helper matching SPEC §3 for callers without a :class:`VisionClient`."""
    client = VisionClient(cfg_llm)
    try:
        return await client.describe_screenshot(jpeg_bytes, query)
    finally:
        client.close()


async def locate_on_screen(
    jpeg_bytes: bytes, target: str, img_w: int, img_h: int, cfg_llm: Any
) -> tuple[float, float] | None:
    """One-shot grounding helper for callers without a :class:`VisionClient`."""
    client = VisionClient(cfg_llm)
    try:
        return await client.locate_on_screen(jpeg_bytes, target, img_w, img_h)
    finally:
        client.close()


__all__ = [
    "VisionClient",
    "describe_screenshot",
    "locate_on_screen",
    "parse_point",
    "GROUNDING_GRID",
    "REQUEST_TIMEOUT_S",
    "DEFAULT_QUERY",
]
