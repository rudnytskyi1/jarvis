"""What release the hub wants its room clients to run (ТЗ 4.9, OTA).

The hub owns the version: one tag in ``config.yaml``, told to every client when
it connects. The client decides how to get there (``client/ota.py``) — fetch,
check out, migrate its config, restart, and roll back if it cannot stay up.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Message type the client listens for (proto v2 extension of ТЗ 4.9).
MSG_RELEASE = "release"


def desired_tag(cfg: Any) -> str:
    """The tag the hub asks for, or ``""`` when OTA is not configured."""
    return str(getattr(getattr(cfg, "server", None), "client_release", "") or "").strip()


def release_frame(cfg: Any) -> dict[str, Any] | None:
    """The frame sent to a room when it connects, or ``None`` when unset."""
    tag = desired_tag(cfg)
    if not tag:
        return None
    return {"type": MSG_RELEASE, "tag": tag, "utf8": True}


__all__ = ["MSG_RELEASE", "desired_tag", "release_frame"]
