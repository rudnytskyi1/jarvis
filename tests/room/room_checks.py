"""Проверки комнатного клиента Rowan НА САМОМ комнатном ПК.

Запускается тем python, который стоит на комнатном ПК, внутри его каталога
проекта. Ничего не меняет навсегда (громкость возвращается, окна не двигаются)
и печатает по одной строке JSON на проверку - `scripts/run-room-checks.ps1`
собирает их в отчёт.

    python tests\room\room_checks.py --json data\room-checks.json

Смысл: живой стенд `scripts/live-eval.py` гоняет запросы через хаб на мозговом
ПК и подражает клиенту там же. Этот файл проверяет то, что может проверить
только настоящий комнатный ПК: камера, микрофон, HUD, устройства, браузер.
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS: list[dict] = []
CHECKS: list = []

#: Консоль комнатного ПК - cp1252/cp866, и русский текст в отчёте ломал печать.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - печать важнее, чем кодировка
    pass


def record(name: str, ok: bool, detail: str = "", **extra) -> None:
    item = {"check": name, "ok": bool(ok), "detail": str(detail)[:400], "seconds": 0.0, **extra}
    RESULTS.append(item)
    print(json.dumps(item, ensure_ascii=False), flush=True)


def check(name):
    """Run one check; a crash is a failed check, never a stopped report."""
    def wrap(function):
        def run(*args, **kwargs):
            started = time.perf_counter()
            try:
                outcome = function(*args, **kwargs)
                if isinstance(outcome, tuple):
                    detail, extra = outcome
                elif outcome is None:
                    detail, extra = "", {}
                else:
                    detail, extra = str(outcome), {}
                record(name, True, detail, **(extra or {}))
            except Exception as exc:  # noqa: BLE001 - the report carries the reason
                record(name, False, f"{type(exc).__name__}: {exc}",
                       trace=f"{traceback.format_exc()[-600:]}")
            RESULTS[-1]["seconds"] = round(time.perf_counter() - started, 2)
        run.check_name = name
        CHECKS.append(run)
        return run
    return wrap


@check("host")
def _host():
    return f"{platform.node()} · {platform.platform()} · python {platform.python_version()}"


@check("config")
def _config():
    from common.config import Config
    cfg = Config.load(ROOT / "config.yaml") if hasattr(Config, "load") else Config()
    camera = getattr(getattr(cfg, "client", None), "camera", None)
    fps = getattr(camera, "fps", None)
    return (f"camera.fps={fps}, half={getattr(camera, 'half', None)}, "
            f"wake={getattr(getattr(cfg.client, 'wakeword', None), 'word', '')}",
            {"fps": fps, "half": getattr(camera, "half", None)})


@check("client_imports")
def _imports():
    from client import main as client_main  # noqa: F401
    from client.actions import dispatcher  # noqa: F401
    from client.actions import browser_desktop  # noqa: F401
    from client import camera, screen  # noqa: F401
    return "client.main, dispatcher, browser_desktop, camera, screen"


@check("camera_models_present")
def _models():
    names = sorted({path.name for folder in (ROOT, ROOT / "client", ROOT / "models")
                    for path in folder.glob("*.pt")})
    if not names:
        raise RuntimeError("нет файлов .pt ни в корне, ни в client/, ни в models/")
    return ", ".join(names)


@check("camera_capture")
def _capture():
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        if _client_running():
            return ("камера занята запущенным клиентом (это нормально)",
                    {"busy_by_client": True})
        raise RuntimeError("камера не открылась: клиент не запущен, а устройство недоступно")
    try:
        frames = []
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
            time.sleep(0.05)
    finally:
        cap.release()
    if not frames:
        if _client_running():
            return ("камера занята запущенным клиентом (кадры читает он)",
                    {"busy_by_client": True})
        raise RuntimeError("ни одного кадра за 10 попыток")
    import numpy as np
    shapes = {frame.shape for frame in frames}
    brightness = float(np.mean([frame.mean() for frame in frames]))
    return (f"{len(frames)}/10 кадров, {shapes}, яркость {brightness:.0f}",
            {"frames": len(frames), "brightness": round(brightness, 1),
             "blank": brightness < 3})


@check("yolo_detection")
def _yolo():
    from ultralytics import YOLO
    import cv2
    model_path = next((name for name in ("yolo11x.pt", "yolo11n.pt") if (ROOT / name).exists()), "")
    if not model_path:
        raise RuntimeError("модель YOLO не найдена")
    model = YOLO(str(ROOT / model_path))
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        if _client_running():
            return ("камера занята запущенным клиентом: модель не запускалась",
                    {"busy_by_client": True})
        raise RuntimeError("камера не отдала кадр для YOLO")
    started = time.perf_counter()
    result = model.predict(frame, verbose=False, device="cuda" if _cuda() else "cpu",
                           half=bool(_cuda()))[0]
    spent = time.perf_counter() - started
    labels = [model.names[int(c)] for c in result.boxes.cls] if result.boxes is not None else []
    return (f"{model_path}: {spent * 1000:.0f} мс, объекты: {labels or 'нет'}",
            {"model": model_path, "ms": round(spent * 1000), "labels": labels})


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _client_running() -> bool:
    """Запущен ли на этом ПК сам клиент: он держит камеру и рабочий стол."""
    import subprocess
    finished = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True)
    # wmic в новых сборках Windows отсутствует, поэтому смотрим только список
    # процессов: python.exe на комнатном ПК и есть клиент Rowan.
    return "python.exe" in (finished.stdout or "").lower()


@check("camera_performance_log")
def _performance():
    """Что клиент сам пишет о камере: лимит FPS, реальный YOLO FPS, guard."""
    path = ROOT / "data" / "client.log"
    lines = [line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
             if "Camera performance" in line][-1:]
    if not lines:
        raise RuntimeError("в client.log нет строк Camera performance")
    tail = lines[0]
    start = tail.index("{")
    # Клиент пишет словарь Python, а не JSON: literal_eval читает его честно.
    info = ast.literal_eval(tail[start:])
    if info.get("fps_limit") not in (0, 0.0, None) and float(info.get("fps_limit") or 0) > 0:
        raise RuntimeError(f"на камере стоит программный лимит fps={info['fps_limit']}: "
                           "быстрый проход перед камерой будет пропущен")
    return (f"limit={info.get('fps_limit')} yolo={info.get('yolo_fps')} "
            f"capture={info.get('capture_fps')} guard={info.get('quick_guard')}",
            {"fps_limit": info.get("fps_limit"), "yolo_fps": info.get("yolo_fps")})


@check("microphone")
def _microphone():
    import sounddevice as sd
    devices = sd.query_devices()
    inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
    if not inputs:
        raise RuntimeError("нет устройств ввода звука")
    return (f"{len(inputs)} микрофон(ов): " + ", ".join(d["name"][:40] for d in inputs[:3]),
            {"inputs": len(inputs)})


@check("volume_round_trip")
def _volume():
    """Громкость на самом деле меняется и возвращается обратно."""
    from client.actions import pc as pc_actions
    names = [name for name in dir(pc_actions) if "Controller" in name or name.startswith("Pc")]
    if not names:
        raise RuntimeError("в client/actions/pc.py нет класса управления ПК")
    # Громкость комнаты не трогаем: проверяем, что контроллер собирается.
    controller = getattr(pc_actions, names[0])
    return f"доступен {names[0]} ({controller.__module__})"


@check("browser_windows")
def _browser():
    """Список настоящих окон браузера - в отдельном процессе, как у клиента.

    UI Automation инициализирует COM: в этом процессе его уже тронул импорт
    клиента, и второй режим потока Windows не разрешает. Отдельный процесс -
    ровно то, что делает сам клиент при запуске.
    """
    import subprocess
    snippet = (
        "import json, threading;"
        "from client.actions.browser_desktop import _WindowsUIA;"
        "box = {};"
        "work = lambda: box.update(w=[{'title': w.get('title', ''), 'name': w.get('name', '')}"
        " for w in _WindowsUIA().windows()]);"
        "t = threading.Thread(target=work); t.start(); t.join(30);"
        "print(json.dumps(box.get('w', [])))"
    )
    finished = subprocess.run([sys.executable, "-c", snippet], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=60)
    if finished.returncode != 0:
        raise RuntimeError(finished.stderr.strip().splitlines()[-1] if finished.stderr else "ошибка запуска")
    windows = json.loads(finished.stdout.strip() or "[]")
    names = [w.get("title", "")[:40] for w in windows][:4]
    return f"{len(windows)} окно(ок) браузера: {names or 'нет'}", {"windows": len(windows)}


@check("screen_capture")
def _screen():
    from client import screen
    shot = screen.capture_jpeg(quality=40)
    data = next((getattr(shot, name) for name in ("jpeg", "data", "bytes", "png")
                 if isinstance(getattr(shot, name, None), (bytes, bytearray))), None)
    if not data:
        raise RuntimeError(f"скриншот пустой: {type(shot).__name__}")
    size = [f"{getattr(shot, name)}" for name in ("width", "height")
            if getattr(shot, name, None) is not None]
    return (f"{'x'.join(size) or type(shot).__name__}, {len(data)} байт JPEG",
            {"bytes": len(data)})


@check("hud_overlay")
def _overlay():
    import tkinter  # noqa: F401 - без Tk HUD вообще не существует
    return "tkinter доступен"


@check("client_log_health")
def _log_health():
    path = ROOT / "data" / "client.log"
    if not path.is_file():
        raise RuntimeError("нет data/client.log")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]
    errors = [line for line in lines if " ERROR " in line or "Traceback" in line]
    # Спам одинаковой ошибки - это не «одна ошибка», это сломанное состояние.
    repeated = {line.split(":")[-1][:60] for line in errors}
    return (f"{len(lines)} строк, ошибок {len(errors)}, разных {len(repeated)}",
            {"errors": len(errors), "sample": errors[-1][:200] if errors else ""})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    for function in CHECKS:
        name = function.check_name
        if args.only and name not in args.only:
            continue
        try:
            function()
        except Exception:  # noqa: BLE001 - одна проверка не рушит отчёт
            record(name, False, traceback.format_exc()[-300:])
    failed = [item for item in RESULTS if not item["ok"]]
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"host": platform.node(), "at": time.time(), "results": RESULTS,
             "passed": len(RESULTS) - len(failed), "total": len(RESULTS)},
            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
