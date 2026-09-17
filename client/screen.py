"""Screen capture for the room client (SPEC §7, v1.1).

The server's ``look_at_screen`` tool asks this machine for a picture of its
screen: it sends ``screenshot_request``, and the client answers with a
``screenshot`` header followed by exactly ONE binary JPEG frame (SPEC §4).

Capture goes through Pillow's ``ImageGrab`` (Windows GDI) and is therefore
blocking, so callers run it off the event loop::

    from client.screen import capture_jpeg
    jpeg = await asyncio.to_thread(capture_jpeg)

The image is downscaled to at most :data:`MAX_WIDTH_PX` pixels wide and encoded
as JPEG with quality :data:`JPEG_QUALITY` — enough detail for a vision model
without pushing megabytes over the WebSocket. Any failure raises
:class:`ScreenCaptureError` with a message the caller forwards verbatim in
``screenshot_error``.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Tuple

log = logging.getLogger(__name__)

#: Longest edge of the sent image along X; taller screens keep their aspect ratio.
MAX_WIDTH_PX = 1600
#: JPEG quality used for the encoded screenshot.
JPEG_QUALITY = 80
#: Value of the ``format`` field in the ``screenshot`` header (SPEC §4, C->S #6).
SCREENSHOT_FORMAT = "jpeg"

__all__ = [
    "MAX_WIDTH_PX",
    "JPEG_QUALITY",
    "SCREENSHOT_FORMAT",
    "ScreenCaptureError",
    "capture_jpeg",
]


class ScreenCaptureError(RuntimeError):
    """The screen could not be grabbed or the image could not be encoded."""


def _load_pillow() -> Tuple[Any, Any]:
    """Import Pillow lazily so a missing dependency cannot break client startup."""
    try:
        from PIL import Image, ImageGrab  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ScreenCaptureError(
            "Pillow is not installed on the room PC. Install the client "
            "dependencies: pip install -r client/requirements.txt"
        ) from exc
    return Image, ImageGrab


def _resample_filter(image_module: Any) -> Any:
    """LANCZOS constant across Pillow versions (moved to ``Image.Resampling`` in 9.1)."""
    resampling = getattr(image_module, "Resampling", None)
    if resampling is not None:
        return resampling.LANCZOS
    return image_module.LANCZOS  # pragma: no cover - Pillow < 9.1


def capture_jpeg(max_width: int = MAX_WIDTH_PX, quality: int = JPEG_QUALITY) -> bytes:
    """Grab the primary screen and return it as JPEG bytes.

    :param max_width: downscale the grab so it is at most this many pixels wide.
    :param quality: JPEG quality (1..95).
    :raises ScreenCaptureError: capture or encoding failed — the message is
        meant to be sent to the server as ``screenshot_error.error``.
    """
    image_module, grab_module = _load_pillow()

    try:
        image = grab_module.grab()
    except Exception as exc:  # noqa: BLE001 - any GDI failure becomes one error
        raise ScreenCaptureError(f"screen capture failed: {exc}") from exc
    if image is None:  # pragma: no cover - depends on the display driver
        raise ScreenCaptureError("screen capture returned no image")

    try:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ScreenCaptureError("screen capture returned an empty image")
        if image.mode != "RGB":
            # JPEG cannot store an alpha channel (ImageGrab may return RGBA).
            image = image.convert("RGB")

        limit = max(1, int(max_width))
        if width > limit:
            scaled_height = max(1, int(round(height * limit / float(width))))
            image = image.resize((limit, scaled_height), _resample_filter(image_module))

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=max(1, min(95, int(quality))),
            optimize=True,
        )
        encoded_size = image.size
    except ScreenCaptureError:
        raise
    except Exception as exc:  # noqa: BLE001 - encoding must not leak raw errors
        raise ScreenCaptureError(f"screenshot encoding failed: {exc}") from exc
    finally:
        close = getattr(image, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - defensive
                pass

    data = buffer.getvalue()
    if not data:
        raise ScreenCaptureError("screenshot encoding produced no data")
    log.debug(
        "Screenshot captured: %dx%d source, %dx%d sent, %d bytes JPEG",
        width,
        height,
        encoded_size[0],
        encoded_size[1],
        len(data),
    )
    return data
