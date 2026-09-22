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


__all__ = ["DEFAULT_PROMPT", "CloudVision"]
