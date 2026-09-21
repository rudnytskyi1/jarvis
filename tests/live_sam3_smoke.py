"""Live smoke test for server/segment.py — no server, no config.yaml needed.

Builds a :class:`server.segment.Sam3Engine` directly against the local SAM3
checkpoint, draws a small synthetic image and runs one real segmentation call
on the GPU. This is NOT part of the pytest suite (SAM3 needs a real CUDA GPU
and loads a ~3.45 GB checkpoint) — run it directly with the jarvis conda env's
python from the repo root::

    C:\\Users\\Anton\\anaconda3\\envs\\jarvis\\python.exe tests\\live_sam3_smoke.py

SAM3 may or may not actually recognize a crude drawing of two filled circles
as "red circle" — the only thing asserted is that the whole pipeline (lazy
sys.path/import, model load, inference, tensor -> dict conversion) runs
without crashing and comes back as a well-formed result, VRAM permitting.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from common.config import SegmentConfig  # noqa: E402
from hub.segment import Sam3Engine  # noqa: E402


def _make_test_jpeg() -> bytes:
    """640x480 white image with two filled red circles."""
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((80, 150, 220, 290), fill=(220, 20, 20))
    draw.ellipse((400, 180, 540, 320), fill=(220, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def main() -> None:
    cfg = SegmentConfig()
    print(f"SAM3 checkpoint: {cfg.checkpoint}")
    print(f"Checkpoint exists: {Path(cfg.checkpoint).is_file()}")

    engine = Sam3Engine(cfg)
    jpeg_bytes = _make_test_jpeg()
    print(f"Test image: {len(jpeg_bytes)} bytes JPEG, 640x480, 2 red circles")

    print("Calling segment(jpeg, 'red circle') ...")
    result = engine.segment(jpeg_bytes, "red circle")
    print("segment() result:", result)

    assert result.get("ok") is True, f"segment() failed: {result.get('error')}"
    assert result.get("count", -1) >= 0, "count must be a non-negative number"
    assert isinstance(result.get("boxes"), list)
    assert isinstance(result.get("scores"), list)
    assert len(result["boxes"]) == len(result["scores"]) == result["count"]
    for box in result["boxes"]:
        assert len(box) == 4
        assert all(0.0 <= v <= 1.0 for v in box), f"box not normalized: {box}"

    print(f"OK: SAM3 pipeline ran end to end, {result['count']} match(es).")


if __name__ == "__main__":
    main()
