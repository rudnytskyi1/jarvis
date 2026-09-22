"""Часть экрана для зренческих инструментов (ТЗ F-511: «скриншот области»).

ТЗ просит не только «прочитай экран», но и «прочитай область экрана»: окно
мессенджера, угол с часами, правую половину. Клиент всегда отдаёт целый кадр
(протокол v1/v2 знает один ``screenshot``), поэтому область выбирается на
хабе — ДО того, как картинка уедет модели зрения: меньше картинки в промпте,
меньше шансов, что модель примется читать чужие окна вокруг.

Область задаётся словами, которые люди действительно говорят, или числами в
долях экрана (``x, y, width, height`` от 0 до 1). Ничего не угадывается:
непонятная строка — это отказ с объяснением, а не «наверное, левая половина».
"""
from __future__ import annotations

import io
import logging
import re
from typing import Any

log = logging.getLogger("jarvis.server.screen_regions")

#: Box is normalized ``(x, y, width, height)`` of the screen, 0…1.
Box = tuple[float, float, float, float]

#: The regions people ask for by name (en/ru/es).
NAMED: dict[str, Box] = {
    "left": (0.0, 0.0, 0.5, 1.0),
    "left half": (0.0, 0.0, 0.5, 1.0),
    "right": (0.5, 0.0, 0.5, 1.0),
    "right half": (0.5, 0.0, 0.5, 1.0),
    "top": (0.0, 0.0, 1.0, 0.5),
    "top half": (0.0, 0.0, 1.0, 0.5),
    "bottom": (0.0, 0.5, 1.0, 0.5),
    "bottom half": (0.0, 0.5, 1.0, 0.5),
    "center": (0.25, 0.25, 0.5, 0.5),
    "centre": (0.25, 0.25, 0.5, 0.5),
    "middle": (0.25, 0.25, 0.5, 0.5),
    "top left": (0.0, 0.0, 0.5, 0.5),
    "top right": (0.5, 0.0, 0.5, 0.5),
    "bottom left": (0.0, 0.5, 0.5, 0.5),
    "bottom right": (0.5, 0.5, 0.5, 0.5),
    "левую половину": (0.0, 0.0, 0.5, 1.0),
    "левая половина": (0.0, 0.0, 0.5, 1.0),
    "правую половину": (0.5, 0.0, 0.5, 1.0),
    "правая половина": (0.5, 0.0, 0.5, 1.0),
    "верх": (0.0, 0.0, 1.0, 0.5),
    "низ": (0.0, 0.5, 1.0, 0.5),
    "центр": (0.25, 0.25, 0.5, 0.5),
    "левый верхний угол": (0.0, 0.0, 0.5, 0.5),
    "правый верхний угол": (0.5, 0.0, 0.5, 0.5),
    "левый нижний угол": (0.0, 0.5, 0.5, 0.5),
    "правый нижний угол": (0.5, 0.5, 0.5, 0.5),
    "mitad izquierda": (0.0, 0.0, 0.5, 1.0),
    "mitad derecha": (0.5, 0.0, 0.5, 1.0),
    "arriba": (0.0, 0.0, 1.0, 0.5),
    "abajo": (0.0, 0.5, 1.0, 0.5),
    "centro": (0.25, 0.25, 0.5, 0.5),
}

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def parse_region(value: Any) -> Box | None:
    """«bottom right» / ``0.5,0,0.5,1`` → normalized box, или ``None``."""
    if value is None:
        return None
    text = " ".join(str(value).replace("_", " ").replace("-", " ").split())
    if not text:
        return None
    folded = text.casefold().strip(" .,;:")
    if folded in NAMED:
        return NAMED[folded]
    numbers = [float(match.replace(",", ".")) for match in _NUMBER.findall(folded)]
    if len(numbers) != 4:
        return None
    x, y, width, height = numbers
    if width <= 0.0 or height <= 0.0 or width > 1.0 or height > 1.0:
        return None
    x = min(max(0.0, x), 1.0 - min(width, 1.0))
    y = min(max(0.0, y), 1.0 - min(height, 1.0))
    x = max(0.0, x)
    y = max(0.0, y)
    if width < 0.02 or height < 0.02:
        return None
    return (x, y, min(width, 1.0 - x), min(height, 1.0 - y))


def region_words(box: Box) -> str:
    """A short human name for a box, for the turn trace and the tool result."""
    x, y, width, height = box
    for name, known in NAMED.items():
        if all(abs(a - b) < 1e-6 for a, b in zip(box, known, strict=True)):
            return name
    return (f"x={x:.2f}, y={y:.2f}, width={width:.2f}, height={height:.2f} "
            "(as a fraction of the screen)")


def crop_jpeg(jpeg: bytes, box: Box) -> bytes:
    """The JPEG of just that part of the screen (unchanged when it is everything)."""
    x, y, width, height = box
    if (x, y, width, height) == (0.0, 0.0, 1.0, 1.0):
        return jpeg
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a hub dependency
        log.warning("Pillow is unavailable; the whole screen is described")
        return jpeg
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            px0 = int(round(x * image.width))
            py0 = int(round(y * image.height))
            px1 = int(round((x + width) * image.width))
            py1 = int(round((y + height) * image.height))
            px0 = max(0, min(px0, image.width - 1))
            py0 = max(0, min(py0, image.height - 1))
            px1 = max(px0 + 1, min(px1, image.width))
            py1 = max(py0 + 1, min(py1, image.height))
            crop = image.crop((px0, py0, px1, py1)).convert("RGB")
            output = io.BytesIO()
            crop.save(output, format="JPEG", quality=90)
            return output.getvalue()
    except Exception as exc:  # noqa: BLE001 - a bad crop must not lose the picture
        log.warning("Could not crop the screenshot to %s (%s)", box, exc)
        return jpeg


__all__ = ["Box", "NAMED", "crop_jpeg", "parse_region", "region_words"]
