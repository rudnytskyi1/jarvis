"""CLIP-эмбеддинги регионов кадра (ТЗ F-305).

ТЗ F-305 индексирует не только «что видно», но и «как это выглядит»: рядом с
подписью объекта (`keys`) лежит вектор его региона, чтобы поиск мог сравнивать
не только слова. Здесь живёт этот второй кусок — `open_clip` ViT-B/32 поверх
вырезок кадра.

Правила те же, что у детектора: пакет и веса грузятся ЛЕНИВО (на первом
кадре), а незнакомая или сломанная установка даёт `EmbedderUnavailable` с
названием причины. **Никогда не выдумывается вектор**: без модели строка
объекта всё равно пишется (её видно глазами), просто без `vector`, и это
честно называется в отчёте.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from typing import Any

from hub.vectors import pack_vector

log = logging.getLogger("jarvis.server.object_embed")

#: Пакет, который даёт CLIP (ТЗ F-305: open_clip ViT-B/32). Импорт ленивый.
CLIP_PACKAGE = "open_clip"
DEFAULT_MODEL = "ViT-B/32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class EmbedderUnavailable(RuntimeError):
    """CLIP на этом хабе не поднялся: пакета нет, весов нет, GPU не ответил."""


def crop_box(width: int, height: int, bbox: Sequence[float]) -> tuple[int, int, int, int] | None:
    """``bbox`` в пикселях кадра → целый прямоугольник внутри картинки.

    Детектор может отдать координаты за краем кадра или (после сжатия) слегка
    вверх ногами; вырезка, которой нет, честно возвращает ``None``, а не
    «вектор пустой картинки».
    """
    if width <= 0 or height <= 0 or len(bbox) < 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox[:4])
    except (TypeError, ValueError):
        return None
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    x1 = max(0, min(int(width), int(left)))
    y1 = max(0, min(int(height), int(top)))
    x2 = max(0, min(int(width), int(right)))
    y2 = max(0, min(int(height), int(bottom)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


class ClipEmbedder:
    """Regions of a frame as CLIP vectors (float32, little-endian) via sqlite-vec."""

    def __init__(self, *, model: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED,
                 device: str = "") -> None:
        self.model_name = str(model or DEFAULT_MODEL)
        self.pretrained = str(pretrained or DEFAULT_PRETRAINED)
        self.device_name = str(device or "")
        self._model: Any = None
        self._preprocess: Any = None
        self._torch: Any = None
        self._error: str = ""

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._model, self._preprocess, self._torch
        if self._error:
            raise EmbedderUnavailable(self._error)
        try:
            import open_clip  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001 - пакета нет: это ответ, а не падение
            self._error = f"{CLIP_PACKAGE} is not installed ({type(exc).__name__})"
            raise EmbedderUnavailable(self._error) from exc
        device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model, _train_preprocess, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, device=device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - битые веса тоже честная причина
            self._error = (f"{CLIP_PACKAGE} could not load {self.model_name}/"
                           f"{self.pretrained} ({type(exc).__name__})")
            raise EmbedderUnavailable(self._error) from exc
        self._model, self._preprocess, self._torch = model, preprocess, torch
        log.info("Object embedder ready: %s/%s on %s", self.model_name, self.pretrained, device)
        return self._model, self._preprocess, self._torch

    def encode_regions(self, frame: bytes, boxes: Sequence[Sequence[float]]) -> list[bytes | None]:
        """One vector per box, aligned with ``boxes``; ``None`` where a crop is empty.

        Raises :class:`EmbedderUnavailable` when the model itself is missing —
        the caller decides whether to keep the sighting without a vector.
        """
        model, preprocess, torch = self._load()
        from PIL import Image

        with Image.open(io.BytesIO(frame)) as picture:
            image = picture.convert("RGB")
            width, height = image.size
            crops: list[Any] = []
            for box in boxes:
                found = crop_box(width, height, box)
                crops.append(None if found is None else preprocess(image.crop(found)))
        ready = [(index, crop) for index, crop in enumerate(crops) if crop is not None]
        packed: list[bytes | None] = [None] * len(crops)
        if not ready:
            return packed
        with torch.no_grad():
            batch = torch.stack([crop for _index, crop in ready])
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            rows = features.detach().cpu().numpy().astype("float32")
        for (index, _crop), row in zip(ready, rows, strict=True):
            packed[index] = pack_vector([float(value) for value in row])
        return packed


def vector_dim(vector: bytes | None) -> int:
    """How many float32 numbers a packed vector holds (0 for "no vector")."""
    return len(vector) // 4 if vector else 0


__all__ = [
    "CLIP_PACKAGE",
    "DEFAULT_MODEL",
    "DEFAULT_PRETRAINED",
    "ClipEmbedder",
    "EmbedderUnavailable",
    "crop_box",
    "vector_dim",
]
