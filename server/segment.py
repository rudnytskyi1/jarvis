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

_GIB = float(1024 ** 3)
#: Free VRAM required before SAM3 is loaded at all. The checkpoint alone is
#: ~3.45 GB and the grounding pass needs room on top of it, on a GPU shared
#: with Ollama (chat + vision models held resident for hours) and
#: faster-whisper. Checking first is not just politeness: a CUDA out-of-memory
#: raised INSIDE a kernel arrives as ``torch.AcceleratorError``, not
#: ``torch.cuda.OutOfMemoryError``, and can leave this process's CUDA context
#: unusable - it once hung the very next Whisper transcription forever and left
#: the assistant deaf to its wake word until the server was restarted by hand.
LOAD_FREE_VRAM_BYTES = int(6 * _GIB)
#: Free VRAM required for one inference once the model is already resident.
RUN_FREE_VRAM_BYTES = int(2 * _GIB)
#: Below this much free VRAM, SAM3 is unloaded before an Ollama vision call.
#: Measured on the 5090: qwen2.5vl:7b at num_ctx 4096 takes 8.4 GB resident,
#: not the 5.5 GB "ollama ps" reports for the weights alone - the KV cache and
#: the compute buffers are the rest, and they are what makes it and SAM3
#: mutually exclusive on a card this size.
VISION_FREE_VRAM_BYTES = int(9 * _GIB)


def free_vram_bytes() -> int | None:
    """Bytes free on the GPU, or ``None`` when that cannot be read.

    Module level so the connection layer can decide whether it needs to free
    something up before asking for a segmentation, without importing torch
    itself or reaching into :class:`Sam3Engine`.
    """
    try:
        import torch  # noqa: PLC0415 - lazy, heavy

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:  # noqa: BLE001 - a driver that will not answer is not fatal
        log.debug("Could not read free GPU memory", exc_info=True)
        return None


def _is_cuda_oom(exc: BaseException) -> bool:
    """True for any flavour of CUDA out-of-memory, whatever class it arrives as.

    ``torch.cuda.OutOfMemoryError`` is only raised when the caching allocator
    itself refuses; an allocation failing inside a kernel surfaces as a plain
    ``torch.AcceleratorError``/``RuntimeError`` whose message carries the real
    reason, so the message is the only reliable signal.
    """
    return "out of memory" in str(exc).lower()


def _label_font(image_height: int) -> Any:
    """A readable TrueType font scaled to the image, or PIL's default.

    The default bitmap font is ~11 px: on a 1080p camera frame the score label
    was there but invisible, which read as "no labels at all".
    """
    from PIL import ImageFont  # noqa: PLC0415 - lazy, like the rest of the module

    size = max(18, int(image_height / 28))
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001 - try the next font
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 can scale it
    except Exception:  # noqa: BLE001 - ancient Pillow
        return ImageFont.load_default()


def draw_boxes(
    jpeg_bytes: bytes,
    boxes: list[Any] | None,
    scores: list[Any] | None = None,
    quality: int = 85,
    label: str = "",
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
        font = _label_font(height)
        # Thicker boxes on a big frame: 3 px is a hairline on 1080p.
        stroke = max(BOX_WIDTH_PX, int(height / 250))
        name = " ".join(str(label or "").split())
        for index, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                log.warning("Skipping a malformed detection box: %r", box)
                continue
            rect = (x1 * width, y1 * height, x2 * width, y2 * height)
            draw.rectangle(rect, outline=BOX_COLOR, width=stroke)

            caption = name
            if index < len(scores):
                try:
                    caption = f"{name} {float(scores[index]):.0%}".strip()
                except (TypeError, ValueError):
                    caption = name
            if not caption:
                continue
            # A filled chip behind the text so it stays readable on any photo.
            try:
                left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
                text_w, text_h = right - left, bottom - top
            except Exception:  # noqa: BLE001 - very old Pillow
                text_w, text_h = len(caption) * 10, 20
            pad = max(4, text_h // 4)
            chip_h = text_h + pad * 2
            chip_x = rect[0]
            chip_y = rect[1] - chip_h
            if chip_y < 0:  # no room above the box: put the chip inside it
                chip_y = rect[1]
            draw.rectangle(
                (chip_x, chip_y, chip_x + text_w + pad * 2, chip_y + chip_h),
                fill=BOX_COLOR,
            )
            draw.text((chip_x + pad, chip_y + pad), caption, fill=(255, 255, 255), font=font)
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

    # ------------------------------------------------------------------ memory

    def unload(self) -> bool:
        """Drop SAM3 off the GPU, freeing its checkpoint and workspace.

        SAM3 and the Ollama vision model cannot both be resident on this 32 GB
        card, and the vision model is asked for on almost every turn while
        find_object is occasional. So whichever one is needed evicts the other,
        and this is the SAM3 side of that bargain: the next find_object pays a
        reload, which is the right place for the cost to land.

        Returns True when something was actually released.
        """
        with self._lock:
            if self._processor is None:
                return False
            self._processor = None
            # A previous hard failure is not a reason to refuse a fresh start.
            self._failed_reason = None
            torch = self._torch
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - freeing memory is best-effort
                log.debug("Could not empty the CUDA cache after unloading SAM3", exc_info=True)
        log.info("Unloaded SAM3 from the GPU")
        return True

    @staticmethod
    def _free_vram(torch: Any) -> int | None:
        """Bytes free on the GPU right now, or ``None`` if it cannot be read."""
        try:
            free, _total = torch.cuda.mem_get_info()
            return int(free)
        except Exception:  # noqa: BLE001 - a driver that will not answer is not fatal
            log.debug("Could not read free GPU memory", exc_info=True)
            return None

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

            # Having a CUDA device says nothing about having room on it: this
            # GPU also holds Ollama's chat and vision models (resident for
            # hours) and faster-whisper. Refuse up front rather than find out
            # inside a kernel - see LOAD_FREE_VRAM_BYTES for why that matters.
            # Deliberately NOT latched into _failed_reason: memory frees up.
            free = self._free_vram(torch)
            if free is not None and free < LOAD_FREE_VRAM_BYTES:
                log.warning(
                    "Not loading SAM3: only %.1f GB free on the GPU, it needs about %.0f GB",
                    free / _GIB, LOAD_FREE_VRAM_BYTES / _GIB,
                )
                return (
                    f"not enough free GPU memory to load SAM3 right now "
                    f"({free / _GIB:.1f} GB free, it needs about "
                    f"{LOAD_FREE_VRAM_BYTES / _GIB:.0f} GB)"
                )

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

        free = self._free_vram(torch)
        if free is not None and free < RUN_FREE_VRAM_BYTES:
            log.warning(
                "Skipping SAM3 for %r: only %.1f GB free on the GPU", text, free / _GIB
            )
            return {
                "ok": False,
                "error": (
                    f"not enough free GPU memory to look for that right now "
                    f"({free / _GIB:.1f} GB free)"
                ),
            }

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
            if _is_cuda_oom(exc):
                # A raw CUDA out-of-memory from inside a kernel, NOT a
                # torch.cuda.OutOfMemoryError. The context may be unusable
                # from here on, and the next thing to touch it was Whisper,
                # which hung forever and left the assistant deaf. Take SAM3
                # out of service rather than risk a second one.
                log.error(
                    "SAM3 hit a raw CUDA out-of-memory for %r - turning it off "
                    "until the server restarts, so it cannot take the GPU down with it",
                    text,
                )
                with self._lock:
                    self._processor = None
                    self._failed_reason = (
                        "SAM3 ran out of GPU memory and is off until the server restarts"
                    )
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 - the context may already be gone
                    pass
                return {"ok": False, "error": "not enough GPU memory for SAM3 right now"}
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
