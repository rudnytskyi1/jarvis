"""Export the client's YOLO11x detector to a TensorRT FP16 engine (ТЗ F-312).

"Ускорение клиента. Экспорт YOLO11x в TensorRT (FP16) для 3060 Ti."

Run it ON the room PC that has the NVIDIA card, once (and again after a driver or
TensorRT upgrade):

    python scripts/export_tensorrt.py --output client/yolo11x.engine

The client then finds the engine through its ``client.camera.profiles`` section
(the ``tensorrt`` profile is tried first) and picks it only if the measured
latency actually fits - see :mod:`client.vision_profile`.

Nothing here pretends: without Ultralytics, without CUDA or without TensorRT the
script says exactly what is missing and exits non-zero. ``--check`` prints the
same answer as JSON and always exits 0, so a deployment script can ask "can this
machine build the engine?" without triggering a build.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = "yolo11x.pt"
DEFAULT_OUTPUT = "yolo11x.engine"
#: Inferences timed after the export (warm-up excluded), like the client does.
MEASURE_FRAMES = 3


def environment() -> dict[str, Any]:
    """What this machine can actually do - checked, not assumed."""
    report: dict[str, Any] = {"ultralytics": importlib.util.find_spec("ultralytics") is not None}
    try:
        import torch  # noqa: PLC0415 - optional heavy import, only when asked

        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
        report["device"] = torch.cuda.get_device_name(0) if report["cuda"] else None
    except Exception as exc:  # noqa: BLE001 - torch is optional here
        report["torch"] = None
        report["cuda"] = False
        report["device"] = None
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    report["tensorrt"] = importlib.util.find_spec("tensorrt") is not None
    return report


def blockers(report: dict[str, Any]) -> list[str]:
    """The honest reasons an export cannot be done on this machine."""
    missing: list[str] = []
    if not report.get("ultralytics"):
        missing.append("нет ultralytics (pip install ultralytics)")
    if not report.get("cuda"):
        missing.append("CUDA-устройство недоступно — TensorRT-движок собирается только на GPU")
    if not report.get("tensorrt"):
        missing.append("нет TensorRT (pip install tensorrt, версия под драйвер)")
    return missing


def solve(path: str) -> Path:
    """Resolve a path against the repository root, so the CLI works anywhere."""
    target = Path(path).expanduser()
    return target if target.is_absolute() else (REPO_ROOT / target)


def measure(engine_path: Path, *, frames: int = MEASURE_FRAMES) -> float:
    """Median milliseconds per inference of the exported engine (real runs)."""
    import numpy as np  # noqa: PLC0415 - optional, only when measuring
    from ultralytics import YOLO  # noqa: PLC0415 - optional, only when measuring

    model = YOLO(str(engine_path))
    blank = np.zeros((640, 640, 3), dtype="uint8")
    model.predict(source=blank, imgsz=640, device=0, half=True, verbose=False)
    costs: list[float] = []
    for _ in range(max(1, frames)):
        started = time.perf_counter()
        model.predict(source=blank, imgsz=640, device=0, half=True, verbose=False)
        costs.append((time.perf_counter() - started) * 1000.0)
    return float(sorted(costs)[len(costs) // 2])


def export(weights: Path, output: Path, *, imgsz: int = 640, half: bool = True) -> Path:
    """Build the TensorRT engine with Ultralytics and return the engine path."""
    from ultralytics import YOLO  # noqa: PLC0415 - optional, only when exporting

    if not weights.exists():
        # Ultralytics downloads the official weights itself; say so, then let it.
        print(f"Веса {weights} не найдены локально — Ultralytics скачает их сама.")
    model = YOLO(str(weights))
    produced = model.export(format="engine", imgsz=imgsz, half=half, device=0)
    produced_path = Path(str(produced))
    if produced_path.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        produced_path.replace(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Экспорт YOLO11x в TensorRT FP16 (ТЗ F-312)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="исходные веса Ultralytics")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="куда положить .engine")
    parser.add_argument("--imgsz", type=int, default=640, help="размер входа движка")
    parser.add_argument("--half", dest="half", action="store_true", default=True,
                        help="FP16 (по умолчанию включён — этого требует ТЗ)")
    parser.add_argument("--no-half", dest="half", action="store_false", help="FP32")
    parser.add_argument("--check", action="store_true",
                        help="только отчёт о возможностях машины (JSON, код выхода 0)")
    args = parser.parse_args(argv)

    report = environment()
    missing = blockers(report)
    if args.check:
        print(json.dumps({**report, "blockers": missing}, ensure_ascii=False, indent=2))
        return 0
    if missing:
        print("Экспорт невозможен:", file=sys.stderr)
        for reason in missing:
            print(f"  - {reason}", file=sys.stderr)
        print("Ничего не собрано и ничего не выдумано: запустите скрипт на машине с GPU.",
              file=sys.stderr)
        return 2
    output = solve(args.output)
    try:
        engine = export(solve(args.weights), output, imgsz=args.imgsz, half=args.half)
        latency_ms = measure(engine)
    except Exception as exc:  # noqa: BLE001 - the export failed, and we say why
        print(f"Экспорт не удался: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"engine": str(engine), "imgsz": args.imgsz, "half": args.half,
                      "latency_ms": round(latency_ms, 2), "device": report.get("device")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
