"""Which model looks at images (ТЗ F-404).

The ТЗ asks for ONE local multimodal level for screenshots and camera frames,
plus a cloud fallback for hard images behind the home's own flag. This module
is the single place that answers "who looks at this picture", so the answer
cannot drift between the router, the tool loop and the log:

``local_vision``
    The level from ``models.levels`` - the same shape every text level has, so
    its endpoint, model and budget live with the rest of the model settings.
    Used whenever ``models.enabled`` and the level names a model.

``server.llm``
    The classic single-model hub's vision model (``server.llm.vision_model``).
    It stays the answer while the level scheme is switched off, so an existing
    hub keeps seeing exactly what it saw before F-404 landed.

Nothing configured means ``None``: the tools then answer honestly that no
vision model is loaded instead of asking a text model to describe a picture it
cannot see.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from common.config import LLMConfig
from hub.vision import VisionClient

#: ultralytics replaces ``PIL.Image.open`` with its own version that, when the
#: picture cannot be read, tries to ``pip install pi-heif`` - a blocking network
#: call inside a hub turn. Rowan never installs packages while it runs (the
#: package is installed, not fetched), so the automatic install stays off. The
#: variable is read by ultralytics when it is imported, and this module is one of
#: the places a picture is opened.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

log = logging.getLogger(__name__)

#: The level of the local multimodal model (ТЗ F-404).
LEVEL_VISION = "local_vision"

#: A screenshot bigger than this is a busy one: windows, tables, small text.
HARD_IMAGE_BYTES = 300_000
#: Words that ask for detail the small model usually gets wrong.
HARD_QUERY_WORDS = ("таблиц", "список", "текст", "надпис", "код", "ошибк", "цифр",
                    "table", "list", "text", "code", "error", "detail", "read")
#: Share of hard horizontal edges (measured on the picture) from which an image
#: is read as "full of small text". Calibrated on synthetic pictures: a page of
#: text scores 0.16, a smooth photo and a simple UI 0.01, a gradient 0.00.
TEXT_SCORE_THRESHOLD = 0.10
#: The width a picture is measured at, so the score does not depend on the camera.
TEXT_MEASURE_WIDTH = 400


def text_score(jpeg: bytes) -> float:
    """How much small alternating detail the picture has (a crude "is there text").

    Text, tables and code are rows full of hard horizontal edges; a photo of a
    face, a gradient or a simple window is not. The picture is scaled to a fixed
    width first so the same scene scores the same whatever the camera sends, and
    the busiest 15% of rows are averaged - text lives in lines, not everywhere.
    A blank or unreadable JPEG scores 0.0.

    Honest limits: a photo of a printed page scores high - and that IS a hard
    image for a small model; pure white noise scores high too, but a camera
    frame is not white noise.
    """
    try:
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(bytes(jpeg))).convert("L")
    except Exception:  # noqa: BLE001 - an unreadable frame is not "hard", it is nothing
        return 0.0
    width, height = image.size
    if width <= 1 or height <= 1:
        return 0.0
    if width > TEXT_MEASURE_WIDTH:
        image = image.resize((TEXT_MEASURE_WIDTH, max(1, height * TEXT_MEASURE_WIDTH // width)))
        width, height = image.size
    pixels = list(image.getdata())
    rows: list[float] = []
    for y in range(height):
        row = pixels[y * width:(y + 1) * width]
        transitions = sum(1 for x in range(width - 1) if abs(row[x + 1] - row[x]) >= 48)
        rows.append(transitions / (width - 1))
    rows.sort()
    busiest = rows[max(0, int(len(rows) * 0.85)):]
    return sum(busiest) / len(busiest) if busiest else 0.0


def image_is_hard(jpeg: bytes, query: str = "", *, people: int = 0) -> bool:
    """An honest reading of "this picture is hard" (ТЗ F-404).

    Four signals, all of them facts about THIS picture or THIS question:
    several faces in the frame, a busy screenshot (it is big), small text
    measured on the picture itself, and a question that asks for details. The
    rules live here as the fallback; :meth:`hub.model_router.ModelRouter.vision_choose`
    lets the Decider (D-10) overrule them, and both answers end up in the log.
    """
    if int(people or 0) >= 2:
        return True
    if len(bytes(jpeg or b"")) >= HARD_IMAGE_BYTES:
        return True
    if text_score(jpeg) >= TEXT_SCORE_THRESHOLD:
        return True
    words = str(query or "").casefold()
    return any(word in words for word in HARD_QUERY_WORDS)


def vision_level_name(models_cfg: Any) -> str:
    """The configured name of the vision level (defaults to ``local_vision``)."""
    routing = getattr(models_cfg, "routing", None)
    name = str(getattr(routing, "vision_level", "") or "").strip()
    return name or LEVEL_VISION


def vision_entry(models_cfg: Any) -> Any:
    """The provisioned local vision level, or ``None`` when there is none."""
    if models_cfg is None or not bool(getattr(models_cfg, "enabled", False)):
        return None
    entry = (getattr(models_cfg, "levels", None) or {}).get(vision_level_name(models_cfg))
    return entry if entry is not None and entry.ready else None


def vision_source(cfg: Any) -> str:
    """Where the vision model would come from (for logs and ``/health``)."""
    models_cfg = getattr(cfg, "models", None)
    entry = vision_entry(models_cfg)
    if entry is not None:
        return f"level:{vision_level_name(models_cfg)}"
    classic = getattr(getattr(cfg, "server", None), "llm", None)
    if str(getattr(classic, "vision_model", "") or "").strip():
        return "server.llm.vision_model"
    return ""


def build_vision(cfg: Any) -> VisionClient | None:
    """The client that looks at images, or ``None`` when nothing can.

    The classic model is deliberately still used when the level scheme is on
    but its vision level has no model yet: half-filled model levels must not
    take the hub's sight away. That fallback is logged, never silent.
    """
    models_cfg = getattr(cfg, "models", None)
    entry = vision_entry(models_cfg)
    if entry is not None:
        name = vision_level_name(models_cfg)
        log.info("Vision comes from model level %s (%s)", name, entry.model)
        return VisionClient(_level_as_llm(entry))
    if bool(getattr(models_cfg, "enabled", False)):
        log.info("Model levels are on but %s has no model; "
                 "server.llm.vision_model answers instead (ТЗ F-404)",
                 vision_level_name(models_cfg))
    classic = getattr(getattr(cfg, "server", None), "llm", None)
    if classic is None or not str(getattr(classic, "vision_model", "") or "").strip():
        return None
    return VisionClient(classic)


def _level_as_llm(entry: Any) -> LLMConfig:
    """A level in the shape :class:`hub.vision.VisionClient` reads.

    The level's own ``model`` IS the vision model here, so it is passed as
    ``vision_model``: the client then uses the level's endpoint, temperature,
    keep-alive and context exactly like the text levels are used.
    """
    return LLMConfig(
        provider=entry.provider,
        base_url=entry.base_url,
        model=entry.model,
        api_key=entry.api_key,
        api_key_env=entry.api_key_env,
        temperature=entry.temperature,
        max_tokens=entry.max_tokens,
        think=entry.think,
        extra_body=dict(entry.extra_body),
        vision_model=entry.model,
    )


def cloud_vision_entry(cfg: Any, *, allowed_cloud_levels: tuple[str, ...] = ("cloud_cheap",
                                                                           "cloud_strong")) -> Any:
    """The provisioned cloud level a hard image could go to, or ``None``."""
    models_cfg = getattr(cfg, "models", None)
    if models_cfg is None or not bool(getattr(models_cfg, "enabled", False)):
        return None
    routing = getattr(models_cfg, "routing", None)
    if not bool(getattr(routing, "cloud_vision", False)):
        return None
    name = str(getattr(routing, "vision_cloud_level", "") or "")
    if name not in allowed_cloud_levels:
        return None
    entry = (getattr(models_cfg, "levels", None) or {}).get(name)
    return entry if entry is not None and entry.ready else None


def build_cloud_vision(cfg: Any, *, ledger_path: Any = None) -> Any:
    """The cloud vision client of this hub, or ``None`` (ТЗ F-404)."""
    entry = cloud_vision_entry(cfg)
    if entry is None:
        return None
    from hub.vision_cloud import CloudVision

    llm_cfg = getattr(getattr(cfg, "server", None), "llm", None)
    # 0 is the owner's "no ceiling" and must not become the $18 default here.
    try:
        monthly = float(getattr(llm_cfg, "monthly_budget_usd", 18.0))
    except (TypeError, ValueError):
        monthly = 18.0
    try:
        client = CloudVision(entry, monthly_budget_usd=monthly, ledger_path=ledger_path)
    except Exception as exc:  # noqa: BLE001 - the hub keeps running without the cloud
        log.warning("Cloud vision is not available (%s); images stay local", exc)
        return None
    log.info("Cloud vision is available: level %s (%s)",
             getattr(getattr(cfg, "models", None).routing, "vision_cloud_level", ""), entry.model)
    return client


__all__ = ["HARD_IMAGE_BYTES", "HARD_QUERY_WORDS", "LEVEL_VISION", "TEXT_MEASURE_WIDTH",
           "TEXT_SCORE_THRESHOLD", "build_cloud_vision", "build_vision",
           "cloud_vision_entry", "image_is_hard", "text_score", "vision_entry",
           "vision_level_name", "vision_source"]
