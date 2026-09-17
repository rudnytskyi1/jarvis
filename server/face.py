"""Face detection, embedding and matching for the room camera (SPEC v1.4).

The client pushes JPEG frames of the C920 (one every
``client.camera.face_check_interval_s`` while somebody is visible, plus the
frames the server pulls with ``camera_request``). This module turns such a
frame into 512-d embeddings with insightface's ``buffalo_l`` pack and matches
them by cosine similarity against the ``face_embeddings`` stored in
``data/people.json`` by :mod:`server.speaker`.

Camera requests and presence pushes now arrive as multi-frame bursts (SPEC
v1.4's burst extension). :meth:`FaceEngine.best_face` picks the single best
detection across such a burst (used by staged face enrollment) by ranking
every detection with the pure, easily tested :func:`select_best_face`.

Two hard rules, because the voice pipeline must keep working without a camera:

* insightface, onnxruntime and the model pack are imported and loaded LAZILY,
  on the first frame — the server starts (and answers ``/health``) even when
  none of them is installed.
* nothing here raises outward: a failure logs and becomes ``[]``
  (:meth:`FaceEngine.detect_and_embed`) or ``(None, 0.0)``
  (:meth:`FaceEngine.match`).

onnxruntime runs on CUDA when the GPU wheel and the CUDA libraries are there
and falls back to the CPU provider otherwise; the chosen provider is logged
once.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

log = logging.getLogger("jarvis.server.face")

#: insightface model pack (detection + 512-d ArcFace recognition).
MODEL_NAME = "buffalo_l"

#: Detector input size; 640x640 is the pack's default and is plenty for a 720p
#: webcam frame of a room.
DET_SIZE = (640, 640)

#: Length of a buffalo_l recognition embedding.
EMBEDDING_DIM = 512

#: onnxruntime providers tried in order: GPU first, CPU as the fallback.
CUDA_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
CPU_PROVIDERS = ("CPUExecutionProvider",)

#: Default cosine threshold; ``server.face.threshold`` overrides it.
DEFAULT_THRESHOLD = 0.45

#: Minimum detector confidence for a detection to count as a real face. The
#: buffalo_l detector emits weak boxes on posters, reflections and textures
#: (they showed up as a phantom "unknown person" next to the seated owner);
#: 0.60 keeps genuine faces and drops those.
MIN_FACE_DET_SCORE = 0.60
#: A real face is also not tiny: a box smaller than this fraction of the frame
#: side is almost always a false positive far in the background.
MIN_FACE_SIDE_FRAC = 0.06

_spec_cache: dict[str, bool] = {}


def insightface_installed() -> bool:
    """True when ``import insightface`` can succeed (checked once, cached)."""
    cached = _spec_cache.get("insightface")
    if cached is not None:
        return cached
    try:
        found = importlib.util.find_spec("insightface") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installation
        found = False
    _spec_cache["insightface"] = found
    if not found:
        log.info(
            "insightface is not installed - face recognition stays off "
            "(pip install -r server/requirements.txt)"
        )
    return found


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes into the BGR uint8 array insightface expects.

    Uses OpenCV when it is available (it ships with insightface) and falls back
    to Pillow. Returns ``None`` when the bytes are not a decodable image.
    """
    if not jpeg_bytes:
        return None
    buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    try:
        import cv2  # noqa: PLC0415 - optional, imported on the first frame only

        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is not None and image.size:
            return image
        log.warning("OpenCV could not decode a %d byte camera frame", len(jpeg_bytes))
    except ImportError:
        pass
    except Exception:
        log.exception("OpenCV failed to decode a camera frame")

    try:
        from io import BytesIO  # noqa: PLC0415 - only needed on the Pillow path

        from PIL import Image  # noqa: PLC0415 - optional dependency

        with Image.open(BytesIO(jpeg_bytes)) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    except ImportError:
        log.warning("Neither OpenCV nor Pillow is installed - cannot decode camera frames")
        return None
    except Exception as exc:
        # A truncated or non-JPEG frame is a client problem, not a crash here.
        log.warning("Could not decode a %d byte camera frame: %s", len(jpeg_bytes), exc)
        return None
    if rgb.size == 0:
        return None
    # insightface works in BGR, like the rest of the OpenCV world.
    return np.ascontiguousarray(rgb[:, :, ::-1])


def provider_candidates() -> tuple[tuple[tuple[str, ...], int, str], ...]:
    """The ``(providers, ctx_id, label)`` combinations to try, best first.

    onnxruntime accepts a provider list containing names it does not have and
    silently runs on the next one, which would make the log claim CUDA on a
    CPU-only wheel — so the GPU attempt is only made when the runtime really
    offers ``CUDAExecutionProvider``.
    """
    cpu = (CPU_PROVIDERS, -1, "CPU")
    # CPU on purpose: the onnxruntime CUDA session was strangling Ollama on
    # the same GPU - chat generation dropped from ~150 to ~10 tok/s whenever
    # presence matching was active (measured 2026-09-17). A face match takes
    # ~300 ms on the CPU once per presence burst, which nobody notices.
    log.info("insightface runs on the CPU (GPU is reserved for the LLM and Whisper)")
    return (cpu,)


def _unused_gpu_candidates() -> tuple[tuple[tuple[str, ...], int, str], ...]:
    """The old GPU-first order, kept for reference/manual experiments."""
    gpu = (CUDA_PROVIDERS, 0, "CUDA")
    cpu = (CPU_PROVIDERS, -1, "CPU")
    try:
        import onnxruntime  # noqa: PLC0415 - lazy, like the rest of the stack

        available = set(onnxruntime.get_available_providers())
    except Exception:
        log.warning("Could not ask onnxruntime for its providers - trying CUDA first")
        return (gpu, cpu)
    if "CUDAExecutionProvider" in available:
        return (gpu, cpu)
    log.info(
        "onnxruntime has no CUDA provider (%s) - faces are matched on the CPU. "
        "Install onnxruntime-gpu for GPU inference.",
        ", ".join(sorted(available)) or "none",
    )
    return (cpu,)


def _det_score(face: Any) -> float:
    """insightface's own detection confidence for one face (0.0 when unusable)."""
    try:
        value = float(getattr(face, "det_score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _bbox_area(bbox: Any) -> float:
    """Pixel area of an insightface ``[x1, y1, x2, y2]`` box (0.0 when unusable)."""
    try:
        box = np.asarray(bbox, dtype=np.float32).ravel()
        if box.size < 4:
            return 0.0
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
    except (TypeError, ValueError):
        return 0.0
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors; 0.0 when either has no length."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def select_best_face(
    per_frame_detections: Iterable[Iterable[tuple[float, float, Any]]] | None,
) -> tuple[Any, float] | None:
    """Pick the best ``(det_score, bbox_area, embedding)`` across several frames.

    SPEC v1.4's burst extension: a person's better angle (mid head-turn during
    enrollment, or simply a frame where they were not moving) should win over a
    worse one from another frame of the same burst, instead of the pipeline
    always keeping whatever frame #1 happened to show. The ranking is
    ``det_score * sqrt(bbox_area)`` -- a large, confidently-detected face beats
    a small or blurry one.

    ``per_frame_detections`` is one iterable of detections per frame (as
    :meth:`FaceEngine.best_face` builds from a JPEG burst), so this function
    itself never decodes an image or touches insightface -- it is plain
    arithmetic over whatever tuples are handed to it, which is what makes it
    testable with fake detections. A malformed row is skipped, never raised.

    :returns: ``(embedding, score)`` of the winning detection, or ``None`` when
        nothing usable was found in any frame.
    """
    best_embedding: Any = None
    best_score = -1.0
    for frame_detections in per_frame_detections or []:
        for row in frame_detections or []:
            try:
                det_score, bbox_area, embedding = row
                score = float(det_score) * math.sqrt(max(float(bbox_area), 0.0))
            except (TypeError, ValueError):
                log.warning("Skipping a malformed detection row: %r", row)
                continue
            if score > best_score:
                best_score = score
                best_embedding = embedding
    if best_embedding is None:
        return None
    return best_embedding, best_score


class FaceEngine:
    """insightface ``buffalo_l`` wrapper: detect, embed and match faces.

    One instance is shared by every connection (the model is a few hundred MB
    of ONNX weights). :meth:`detect_and_embed` is blocking — call it through
    ``asyncio.to_thread`` — while :meth:`match` is plain numpy and cheap.
    """

    def __init__(self, cfg_face: Any = None) -> None:
        self.enabled = bool(getattr(cfg_face, "enabled", True))
        try:
            self.threshold = float(getattr(cfg_face, "threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            self.threshold = DEFAULT_THRESHOLD
        #: Set once a load attempt failed, so it is not retried per frame.
        self._failed = False
        self._app: Any = None
        self._provider = "?"
        self._lock = threading.Lock()
        log.info(
            "Face recognition %s (threshold %.2f)",
            "enabled" if self.enabled else "disabled",
            self.threshold,
        )

    # ------------------------------------------------------------------ status

    @property
    def loaded(self) -> bool:
        """True once the model pack is in memory."""
        return self._app is not None

    @property
    def provider(self) -> str:
        """Which onnxruntime provider the model runs on (``?`` before loading)."""
        return self._provider

    @property
    def available(self) -> bool:
        """True when a frame could plausibly be processed (``/health``'s ``face``).

        Enabled in the config, insightface importable, and no earlier load
        failure. It never triggers the load itself.
        """
        if not self.enabled or self._failed:
            return False
        if self._app is not None:
            return True
        return insightface_installed()

    # ------------------------------------------------------------------ loading

    def _get_app(self) -> Any:
        """Load the model pack once; ``None`` when it cannot be used."""
        with self._lock:
            if self._app is not None:
                return self._app
            if self._failed or not self.enabled:
                return None
            try:
                from insightface.app import FaceAnalysis  # noqa: PLC0415 - lazy
            except Exception:
                log.warning(
                    "insightface could not be imported - face recognition is off "
                    "(pip install -r server/requirements.txt)",
                    exc_info=True,
                )
                self._failed = True
                return None

            for providers, ctx_id, label in provider_candidates():
                try:
                    app = FaceAnalysis(name=MODEL_NAME, providers=list(providers))
                    app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
                except Exception:
                    log.warning(
                        "insightface %s could not start on %s", MODEL_NAME, label,
                        exc_info=True,
                    )
                    continue
                self._app = app
                self._provider = label
                log.info("insightface %s ready on %s", MODEL_NAME, label)
                return app

            log.error(
                "insightface %s could not be loaded on any provider - "
                "face recognition is off for this run",
                MODEL_NAME,
            )
            self._failed = True
            return None

    # ------------------------------------------------------------------ detection

    def _face_detections(self, jpeg_bytes: bytes) -> list[tuple[float, float, np.ndarray]]:
        """Every face of one frame as ``(det_score, bbox_area_px, embedding)``.

        The shared decode + insightface call behind both :meth:`detect_and_embed`
        (keeps only the bbox area, for the "largest face" single-frame path) and
        :meth:`best_face` (needs the raw detection score too, for the burst
        ranking). Never raises.
        """
        if not self.enabled or not jpeg_bytes:
            return []
        app = self._get_app()
        if app is None:
            return []
        image = decode_jpeg(jpeg_bytes)
        if image is None:
            return []
        try:
            faces = app.get(image)
        except Exception:
            log.exception("insightface failed on a %d byte frame", len(jpeg_bytes))
            return []

        frame_side = float(min(image.shape[0], image.shape[1])) if image.ndim >= 2 else 0.0
        min_area = (MIN_FACE_SIDE_FRAC * frame_side) ** 2 if frame_side else 0.0
        found: list[tuple[float, float, np.ndarray]] = []
        for face in faces or []:
            score = _det_score(face)
            area = _bbox_area(getattr(face, "bbox", None))
            # Drop weak/tiny detections: phantom faces on posters/reflections.
            if score < MIN_FACE_DET_SCORE:
                log.debug("Dropping a low-confidence face (det_score %.2f)", score)
                continue
            if min_area and area < min_area:
                log.debug("Dropping a tiny background face (area %.0f < %.0f)", area, min_area)
                continue
            raw = getattr(face, "normed_embedding", None)
            if raw is None:
                raw = getattr(face, "embedding", None)
            if raw is None:
                continue
            try:
                vector = np.asarray(raw, dtype=np.float32).ravel()
            except (TypeError, ValueError):
                log.warning("Skipping a face whose embedding is not numeric")
                continue
            if vector.size == 0 or not np.isfinite(vector).all():
                continue
            found.append((score, area, vector))
        return found

    def detect_and_embed(self, jpeg_bytes: bytes) -> list[tuple[float, np.ndarray]]:
        """Embed every face in one JPEG frame.

        :returns: ``[(bbox_area_px, embedding), …]`` sorted by area, largest
            face first — the person closest to the camera. Empty when face
            recognition is off, the frame is undecodable, nobody is in it or
            anything at all went wrong. Never raises.
        """
        found = [
            (area, vector) for _score, area, vector in self._face_detections(jpeg_bytes)
        ]
        found.sort(key=lambda item: item[0], reverse=True)
        log.debug("Detected %d face(s) in a %d byte frame", len(found), len(jpeg_bytes))
        return found

    def best_face(self, frames_jpegs: Iterable[bytes]) -> tuple[np.ndarray, float] | None:
        """Pick the single best face across a burst of JPEG frames (SPEC v1.4).

        Runs detection on every frame and ranks every detection found with
        :func:`select_best_face` (``det_score * sqrt(bbox_area)``), so a
        person's best angle across the burst wins the sample instead of always
        frame #1 — used by staged face enrollment.

        :returns: ``(embedding, score)`` of the winning detection, or ``None``
            when no frame had a usable face. Never raises.
        """
        per_frame = [self._face_detections(jpeg) for jpeg in (frames_jpegs or [])]
        return select_best_face(per_frame)

    # ------------------------------------------------------------------ matching

    def match(
        self,
        embedding: Any,
        face_profiles: Mapping[str, Sequence[Iterable[float]]] | None,
        threshold: float | None = None,
    ) -> tuple[str | None, float]:
        """Match one embedding against the enrolled face profiles.

        :param face_profiles: ``{name: [embedding, …]}`` from
            :meth:`server.speaker.VoiceRegistry.face_profiles`.
        :param threshold: cosine threshold; ``server.face.threshold`` by default.
        :returns: ``(name, score)`` for the best profile at or above the
            threshold, otherwise ``(None, best_score)``. Never raises.
        """
        try:
            vector = np.asarray(embedding, dtype=np.float32).ravel()
        except (TypeError, ValueError):
            log.warning("Cannot match a face: the embedding is not numeric")
            return None, 0.0
        if vector.size == 0:
            return None, 0.0
        limit = self.threshold if threshold is None else float(threshold)

        best_name: str | None = None
        best_score = -1.0
        try:
            for name, vectors in (face_profiles or {}).items():
                for raw in vectors or []:
                    other = np.asarray(raw, dtype=np.float32).ravel()
                    if other.size != vector.size:
                        log.warning(
                            "Skipping a %d-d profile vector for %s (expected %d)",
                            other.size, name, vector.size,
                        )
                        continue
                    score = cosine(other, vector)
                    if score > best_score:
                        best_name, best_score = str(name), score
        except Exception:
            log.exception("Face matching failed")
            return None, 0.0

        if best_name is not None and best_score >= limit:
            log.info("Face: %s (score %.2f)", best_name, best_score)
            return best_name, best_score
        return None, max(best_score, 0.0)


__all__ = [
    "FaceEngine",
    "MODEL_NAME",
    "DET_SIZE",
    "EMBEDDING_DIM",
    "DEFAULT_THRESHOLD",
    "CUDA_PROVIDERS",
    "CPU_PROVIDERS",
    "cosine",
    "decode_jpeg",
    "insightface_installed",
    "provider_candidates",
    "select_best_face",
]
