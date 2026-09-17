"""SAM3 open-vocabulary object detection for the ``find_object`` tool (v1.5).

The client sends a JPEG (room camera or screen, reusing the same pull machinery
as ``look_at_camera``/``look_at_screen``); this module runs it through a local
SAM3 image model with a free-text prompt and returns the boxes it found.

Everything about SAM3 is LAZY and guarded by a lock:

* the third-party ``sam3`` package lives outside the normal Python path
  (``third_party/sam3/server``) and is only added to ``sys.path`` here;
* ``torch``, ``sam3.model_builder`` and ``sam3.model.sam3_image_processor`` are
  only imported on the first :meth:`Sam3Engine.segment` call;
* the ~3.45 GB checkpoint is only loaded onto the GPU on that same first call.

This means the server starts instantly and keeps working even when SAM3, its
checkpoint or its extra dependencies (timm, ftfy, regex, iopath,
huggingface_hub) are missing or broken: every failure becomes a clear
``{"ok": False, "error": str}`` instead of an exception, and a load failure is
cached so a broken install is not retried on every call.

:meth:`Sam3Engine.segment` is fully blocking (SAM3 inference is synchronous
CUDA work) — callers run it with ``asyncio.to_thread``, exactly like
``server/face.py``'s ``detect_and_embed``.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.server.segment")

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Directory holding the importable ``sam3`` package (scouted, not re-explored).
SAM3_PACKAGE_DIR = REPO_ROOT / "third_party" / "sam3" / "server"

#: v1.6: box color for the annotated detections photo pushed to the room
#: screen (``#FF3355``, as PIL RGB).
BOX_COLOR = (0xFF, 0x33, 0x55)
BOX_WIDTH_PX = 3


def draw_boxes(
    jpeg_bytes: bytes,
    boxes: list[Any] | None,
    scores: list[Any] | None = None,
    quality: int = 85,
) -> bytes:
    """Draw ``find_object``'s boxes on ``jpeg_bytes`` (v1.6, PIL, pure-python).

    ``boxes`` are normalized ``[x1, y1, x2, y2]`` (0..1, exactly what
    :meth:`Sam3Engine.segment` returns) — each drawn as a 3px rectangle in
    :data:`BOX_COLOR` with its score as a small label above it. Used to put
    the detections on the room TV so the owner can see what was found.

    Never raises: any failure (bad bytes, an unreadable box row, no PIL) logs
    and returns ``jpeg_bytes`` UNCHANGED, so a broken annotation never blocks
    the photo push. Returns the input unchanged (no-op) when there is nothing
    to draw.
    """
    if not jpeg_bytes or not boxes:
        return jpeg_bytes
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415 - lazy, matches the rest of the module

        with Image.open(io.BytesIO(jpeg_bytes)) as handle:
            handle.load()
            image = handle.convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        scores = scores or []
        for index, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                log.warning("Skipping a malformed detection box: %r", box)
                continue
            rect = (x1 * width, y1 * height, x2 * width, y2 * height)
            draw.rectangle(rect, outline=BOX_COLOR, width=BOX_WIDTH_PX)
            if index < len(scores):
                try:
                    label = f"{float(scores[index]):.2f}"
                except (TypeError, ValueError):
                    label = ""
                if label:
                    label_y = max(0.0, rect[1] - 14)
                    draw.text((rect[0] + 2, label_y), label, fill=BOX_COLOR)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=max(1, min(95, int(quality))))
        return buffer.getvalue()
    except Exception:
        log.exception("Could not draw detection boxes on the frame - using the plain photo")
        return jpeg_bytes


class Sam3Engine:
    """Lazy SAM3 wrapper backing the ``find_object`` tool.

    ``__init__`` does no importing, no ``sys.path`` surgery and no GPU work —
    just remembers the config — so building one at server startup (SPEC-style
    "construct, don't load") is free. The first :meth:`segment` call does all
    the heavy lifting once, behind :attr:`_lock`; later calls reuse the loaded
    processor (or the cached failure reason).
    """

    def __init__(self, cfg_segment: Any = None) -> None:
        self.enabled = bool(getattr(cfg_segment, "enabled", True))
        self.checkpoint = str(getattr(cfg_segment, "checkpoint", "") or "")
        try:
            self.confidence = float(getattr(cfg_segment, "confidence", 0.5))
        except (TypeError, ValueError):
            self.confidence = 0.5
        #: Guards both the lazy load and each inference call: Sam3Processor
        #: keeps per-call state on the instance, so two segment() calls must
        #: never run concurrently against the same processor.
        self._lock = threading.Lock()
        self._processor: Any = None
        self._torch: Any = None
        #: Set once a load attempt fails, so a broken install is not retried
        #: on every call (same pattern as server/face.py's FaceEngine).
        self._failed_reason: str | None = None
        log.info(
            "SAM3 object finder %s (checkpoint %s, confidence %.2f)",
            "enabled" if self.enabled else "disabled",
            self.checkpoint or "<unset>",
            self.confidence,
        )

    # ------------------------------------------------------------------ status

    @property
    def loaded(self) -> bool:
        """True once the model is actually in GPU memory."""
        return self._processor is not None

    @property
    def available(self) -> bool:
        """True when a call could plausibly succeed (``/health``'s ``sam`` flag).

        Loaded already, or simply enabled in the config and not yet known to be
        broken — this is a cheap, non-blocking check: it never triggers the
        load itself, and never touches the filesystem or the GPU.
        """
        if self._processor is not None:
            return True
        return self.enabled and self._failed_reason is None

    # ------------------------------------------------------------------ loading

    def _load(self) -> str | None:
        """Load SAM3 once. Returns an error string, or ``None`` on success."""
        with self._lock:
            if self._processor is not None:
                return None
            if self._failed_reason is not None:
                return self._failed_reason
            if not self.enabled:
                self._failed_reason = "the object finder is disabled in the config"
                return self._failed_reason
            if not self.checkpoint or not Path(self.checkpoint).is_file():
                self._failed_reason = (
                    f"SAM3 checkpoint not found: {self.checkpoint or '<unset>'}"
                )
                return self._failed_reason

            package_dir = str(SAM3_PACKAGE_DIR)
            if package_dir not in sys.path:
                sys.path.insert(0, package_dir)

            try:
                import torch  # noqa: PLC0415 - lazy, heavy
            except Exception as exc:
                log.warning("PyTorch could not be imported for SAM3", exc_info=True)
                self._failed_reason = f"PyTorch is not available: {exc}"
                return self._failed_reason

            if not torch.cuda.is_available():
                self._failed_reason = (
                    "SAM3 needs a CUDA GPU, but none is available on this machine"
                )
                return self._failed_reason

            try:
                from sam3.model_builder import (  # noqa: PLC0415 - lazy, heavy
                    build_sam3_image_model,
                )
                from sam3.model.sam3_image_processor import (  # noqa: PLC0415
                    Sam3Processor,
                )
            except Exception as exc:
                log.warning(
                    "SAM3 could not be imported - find_object stays off "
                    "(third_party/sam3 or one of its dependencies is missing)",
                    exc_info=True,
                )
                self._failed_reason = f"SAM3 is not available: {exc}"
                return self._failed_reason

            log.info("Loading SAM3 from %s (this can take a while)...", self.checkpoint)
            try:
                # The BPE vocab ships inside the package (server/sam3/assets/),
                # but the builder's default resolves one level higher.
                bpe = (
                    Path(self.checkpoint).resolve().parents[1]
                    / "sam3"
                    / "assets"
                    / "bpe_simple_vocab_16e6.txt.gz"
                )
                model = build_sam3_image_model(
                    checkpoint_path=self.checkpoint,
                    bpe_path=str(bpe) if bpe.is_file() else None,
                    device="cuda",
                    eval_mode=True,
                    load_from_HF=False,
                )
                processor = Sam3Processor(
                    model, device="cuda", confidence_threshold=self.confidence
                )
            except torch.cuda.OutOfMemoryError:
                log.warning("Not enough GPU memory to load SAM3")
                torch.cuda.empty_cache()
                self._failed_reason = "not enough GPU memory to load SAM3"
                return self._failed_reason
            except Exception as exc:
                log.exception("Could not build the SAM3 model")
                self._failed_reason = f"could not load SAM3: {exc}"
                return self._failed_reason

            self._torch = torch
            self._processor = processor
            log.info("SAM3 is ready (checkpoint %s)", self.checkpoint)
            return None

    # ------------------------------------------------------------------ inference

    def segment(self, jpeg_bytes: bytes, prompt: str) -> dict[str, Any]:
        """Find every instance of ``prompt`` in ``jpeg_bytes``.

        Blocking, runs on CUDA. Returns
        ``{"ok": True, "count": int, "boxes": [[x1, y1, x2, y2], ...], "scores": [float, ...]}``
        with boxes normalized to ``0..1`` of the image, or
        ``{"ok": False, "error": str}``. Never raises.
        """
        text = " ".join(str(prompt or "").split())
        if not text:
            return {"ok": False, "error": "find_object needs a non-empty target"}
        if not jpeg_bytes:
            return {"ok": False, "error": "no image to search"}

        error = self._load()
        if error is not None:
            return {"ok": False, "error": error}

        torch = self._torch
        processor = self._processor
        if torch is None or processor is None:  # pragma: no cover - _load() sets error otherwise
            return {"ok": False, "error": "SAM3 is not available"}

        try:
            from PIL import Image  # noqa: PLC0415 - lazy, matches the rest of the module

            with Image.open(io.BytesIO(jpeg_bytes)) as handle:
                handle.load()
                image = handle.convert("RGB")
        except Exception as exc:
            log.warning("Could not decode the image for SAM3: %s", exc)
            return {"ok": False, "error": f"could not decode the image: {exc}"}

        width, height = image.size
        if width <= 0 or height <= 0:
            return {"ok": False, "error": "the image has no usable size"}

        log.info("SAM3: looking for %r in a %dx%d image", text, width, height)
        try:
            with self._lock:
                state = processor.set_image(image)
                state = processor.set_text_prompt(text, state)
        except torch.cuda.OutOfMemoryError:
            log.warning("SAM3 ran out of GPU memory for %r", text)
            torch.cuda.empty_cache()
            return {"ok": False, "error": "not enough GPU memory for SAM3 right now"}
        except Exception as exc:
            log.exception("SAM3 segmentation failed for %r", text)
            return {"ok": False, "error": f"SAM3 segmentation failed: {exc}"}

        boxes = state.get("boxes") if isinstance(state, dict) else None
        scores = state.get("scores") if isinstance(state, dict) else None
        if boxes is None:
            return {"ok": True, "count": 0, "boxes": [], "scores": []}

        try:
            boxes_cpu = boxes.detach().to("cpu")
            scores_cpu = scores.detach().to("cpu") if scores is not None else None
            boxes_out: list[list[float]] = []
            scores_out: list[float] = []
            for i in range(boxes_cpu.shape[0]):
                x1, y1, x2, y2 = (float(v) for v in boxes_cpu[i].tolist())
                boxes_out.append(
                    [
                        min(max(x1 / width, 0.0), 1.0),
                        min(max(y1 / height, 0.0), 1.0),
                        min(max(x2 / width, 0.0), 1.0),
                        min(max(y2 / height, 0.0), 1.0),
                    ]
                )
                scores_out.append(float(scores_cpu[i]) if scores_cpu is not None else 0.0)
        except Exception as exc:
            log.exception("Could not read SAM3's output for %r", text)
            return {"ok": False, "error": f"could not read SAM3 output: {exc}"}

        log.info("SAM3: %d match(es) for %r", len(boxes_out), text)
        return {"ok": True, "count": len(boxes_out), "boxes": boxes_out, "scores": scores_out}


__all__ = ["Sam3Engine", "SAM3_PACKAGE_DIR", "draw_boxes", "BOX_COLOR", "BOX_WIDTH_PX"]
