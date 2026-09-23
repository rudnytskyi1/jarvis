"""The cloud fallback for hard images (ТЗ F-404).

The ТЗ allows ONE cloud vision model, and only where the home says so
(``homes[].cloud_vision: true``) - a picture of somebody's room leaving the
house is the owner's decision. This module is that door, and it is deliberately
narrow: the call goes through the same budgeted transport the text levels use,
so the reservation is taken before the network, the usage is reconciled after,
and a picture never goes anywhere when the ledger says there is no money.

No key, no model, no money - ``CloudVision`` refuses with a reason instead of
pretending to have looked.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hub.api_budget import CloudUnavailable
from hub.openai_responses import ResponsesClient

log = logging.getLogger(__name__)

#: The prompt a cloud look starts from. Specific on purpose: the answer is
#: spoken to the person, so a one-word reply is useless.
DEFAULT_PROMPT = (
    "Look at this image and answer the question about it. Name what you actually "
    "see - the application, window or object, and quote any visible titles or "
    "labels. Never guess: if something is not visible, say so."
)

#: A screenshot of a real desktop is far bigger than the hub's API allowance for
#: one request (``server.llm.max_input_bytes``, 128 KB here), and base64 makes it
#: a third bigger again. The picture is therefore re-encoded until it fits: the
#: model needs readable windows, not the original pixels.
_SHRINK_STEPS = ((0.75, 80), (0.6, 75), (0.5, 70), (0.4, 70), (0.3, 65), (0.25, 60))


def fit_for_api(jpeg: bytes, limit: int, *, log_shrunk: bool = True) -> bytes:
    """Shrink a JPEG until it fits ``limit`` bytes of the API request, or give up.

    Returns the original bytes when it already fits or when Pillow cannot decode
    them: a caller that cannot shrink must not silently drop the picture.
    """
    # Base64 costs a third more, and the prompt plus framing need room too.
    budget = max(4096, int(int(limit) * 0.7) - 2048)
    if len(jpeg) <= budget:
        return jpeg
    try:
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - an unreadable frame is not ours to fix
        log.warning("Could not shrink the frame for the cloud (%s)", exc)
        return jpeg
    for scale, quality in _SHRINK_STEPS:
        size = (max(64, int(image.width * scale)), max(64, int(image.height * scale)))
        out = io.BytesIO()
        image.resize(size, Image.LANCZOS).save(out, "JPEG", quality=quality)
        if out.tell() <= budget:
            if log_shrunk:
                log.info("Shrank the frame for the cloud: %d KB -> %d KB at %dx%d",
                         len(jpeg) // 1024, out.tell() // 1024, size[0], size[1])
            return out.getvalue()
    return jpeg


def _as_cfg(entry: Any, *, monthly_budget_usd: float, max_tokens: int | None = None) -> Any:
    """A level in the shape the budgeted transport reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model=str(getattr(entry, "model", "") or ""),
        api_key_env=str(getattr(entry, "api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY"),
        max_tokens=int(max_tokens or getattr(entry, "max_tokens", 1024) or 1024),
        monthly_budget_usd=float(monthly_budget_usd),
    )


class CloudVision:
    """One opt-in cloud vision model, charged to the hub's own ledger."""

    def __init__(self, entry: Any, *, monthly_budget_usd: float = 18.0,
                 ledger_path: Path | None = None, transport: Any = None,
                 max_tokens: int | None = None) -> None:
        if not str(getattr(entry, "model", "") or "").strip():
            raise CloudUnavailable("The cloud vision level names no model.")
        self.level_model = str(entry.model)
        try:
            self.client = ResponsesClient(
                _as_cfg(entry, monthly_budget_usd=monthly_budget_usd, max_tokens=max_tokens),
                ledger_path=ledger_path, transport=transport)
        except CloudUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a model without reviewed pricing
            raise CloudUnavailable(
                f"The cloud vision model {self.level_model!r} has no reviewed pricing; "
                "it is not used.") from exc

    def describe(self, jpeg: bytes, query: str | None = None) -> str:
        """Answer ``query`` about one JPEG; the text, or a sentence saying why not.

        Never raises: the tool loop has to keep running, and a cloud problem
        must read as an answer, exactly like the local vision client's errors.
        """
        question = " ".join(str(query or "").split())
        prompt = f"{DEFAULT_PROMPT}\n\nQuestion: {question}" if question else DEFAULT_PROMPT
        # The request carries the picture as base64 inside one JSON body, and the
        # hub's own allowance for that body is small (128 KB here): a real
        # screenshot has to be shrunk first or the look never leaves the house.
        jpeg = fit_for_api(jpeg, int(getattr(self.client, "max_input_bytes", 128000)))
        try:
            answer = self.client.describe_image(jpeg, prompt)
        except CloudUnavailable as exc:
            log.warning("Cloud vision is unavailable: %s", exc)
            return f"Screen check failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - a look at an image is never fatal
            log.warning("Cloud vision failed (%s)", type(exc).__name__)
            return "Screen check failed: the cloud vision request did not complete."
        if not answer:
            log.warning("Cloud vision returned an empty answer")
            return "Screen check failed: the cloud vision model returned an empty answer."
        log.info("Cloud vision (%s) answered in %d characters", self.level_model, len(answer))
        return answer

    def close(self) -> None:
        self.client.close()


__all__ = ["DEFAULT_PROMPT", "CloudVision", "fit_for_api"]
