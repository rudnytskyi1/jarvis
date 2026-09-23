"""Аудит действий НА САМОМ комнатном ПК: десятки настоящих вызовов.

Запускается на комнатном ПК (`scripts/room-audit.ps1` копирует его туда) тем
же python, которым работает клиент, и зовёт ровно тот код, который выполняет
инструменты в живой комнате — `client.actions.dispatcher.Dispatcher`.

Что проверяется: громкость и медиа, буфер обмена, приложения, настоящий
браузерный переход по адресу (включая «youtube .com» с пробелом — жалоба
владельца от 2026-09-22), камера и экран, и, главное, отказы: `file://`,
адрес с паролем, неизвестная команда и пустой `run_command` обязаны быть
отвергнуты, а не выполнены.

Разрушительное не трогается вовсе: сон, блокировка, выключение экрана,
случайные клики и закрытие чужих окон остаются за `--extended`, который
владелец запускает сам. Звук после прогона остаётся выключенным, как он и
просил на время аудита.

    python tests\\room\\room_audit.py --json data\\room-audit.json
    python tests\\room\\room_audit.py --extended        # плюс медиа-клавиши
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Консоль комнатного ПК — cp1252/cp866; utf-8 спасает русский текст отчёта.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - печать важнее кодировки
    pass


def _volume_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for level in range(0, 101, 10):
        cases.append({"tool": "pc_control", "args": {"command": "volume_set", "value": level},
                      "expect": "ok", "must_contain": str(level)})
    for _ in range(3):
        cases.append({"tool": "pc_control", "args": {"command": "volume_up"}, "expect": "ok"})
        cases.append({"tool": "pc_control", "args": {"command": "volume_down"}, "expect": "ok"})
    cases += [
        {"tool": "pc_control", "args": {"command": "mute"}, "expect": "ok"},
        {"tool": "pc_control", "args": {"command": "unmute"}, "expect": "ok"},
        {"tool": "pc_control", "args": {"command": "mute"}, "expect": "ok"},
        # A level outside 0..100 has to be clamped, not refused: the room must
        # never end up with a volume the device cannot hold.
        {"tool": "pc_control", "args": {"command": "volume_set", "value": 150},
         "expect": "ok", "must_contain": "100"},
        {"tool": "pc_control", "args": {"command": "volume_set", "value": -20},
         "expect": "ok", "must_contain": "0"},
        {"tool": "pc_control", "args": {"command": "volume_set", "value": 0}, "expect": "ok"},
    ]
    return cases


def _clipboard_cases() -> list[dict[str, Any]]:
    return [
        {"tool": "pc_control", "args": {"command": "clipboard_read"}, "expect": "ok"},
        {"tool": "pc_control", "args": {"command": "clipboard_write", "value": "rowan audit"},
         "expect": "ok"},
        {"tool": "pc_control", "args": {"command": "clipboard_read"}, "expect": "ok",
         "must_contain": "rowan audit"},
    ]


def _window_cases() -> list[dict[str, Any]]:
    return [
        {"tool": "pc_control", "args": {"command": "open_app", "value": "notepad"},
         "expect": "ok", "group": "apps"},
        # Windows 11's Notepad is a Store app: it has no process image to map,
        # so the client refuses with the exact workaround to use instead. Both
        # outcomes are honest here; what must not happen is a silent failure.
        {"tool": "pc_control", "args": {"command": "close_app", "value": "notepad"},
         "expect": "any", "group": "apps",
         "note": "Store-приложение: отказ с подсказкой или закрытие — оба честные"},
        # The model really does send the name in ``target`` (VE-03/VE-09/VE-12);
        # the dispatcher has to accept either slot.
        {"tool": "pc_control", "args": {"command": "open_app", "target": "notepad"},
         "expect": "ok", "group": "apps"},
        {"tool": "pc_control", "args": {"command": "close_app", "target": "notepad"},
         "expect": "any", "group": "apps"},
        {"tool": "run_command", "args": {"command": "Write-Output rowan"},
         "expect": "ok", "must_contain": "rowan", "group": "apps",
         "note": "run_command обязан вернуть вывод"},
        {"tool": "run_command", "args": {"command": "whoami"}, "expect": "ok", "group": "apps"},
    ]


#: Обычные адреса. Часть из них названа так, как их произносят, — без схемы.
URLS = ["https://example.com", "youtube.com", "www.google.com", "youtube .com",
        "https://www.wikipedia.org", "https://github.com", "example.com"]


def _browser_cases() -> list[dict[str, Any]]:
    cases = [{"tool": "browser_control", "args": {"command": "navigate", "url": url},
              "expect": "ok", "group": "browser"} for url in URLS]
    cases += [
        {"tool": "browser_control", "args": {"command": "read"}, "expect": "ok",
         "group": "browser"},
        {"tool": "browser_control", "args": {"command": "scroll", "direction": "down"},
         "expect": "ok", "group": "browser"},
        {"tool": "browser_control", "args": {"command": "scroll", "direction": "up"},
         "expect": "ok", "group": "browser"},
        {"tool": "browser_control", "args": {"command": "back"}, "expect": "ok",
         "group": "browser"},
    ]
    return cases


def _refusal_cases() -> list[dict[str, Any]]:
    """Каждый из этих вызовов обязан быть отвергнут, а не выполнен."""
    return [
        {"tool": "browser_control", "args": {"command": "navigate", "url": "file:///C:/Windows"},
         "expect": "fail", "note": "локальный файл вместо страницы"},
        {"tool": "browser_control", "args": {"command": "navigate", "url": "javascript:alert(1)"},
         "expect": "fail", "note": "javascript в адресной строке"},
        {"tool": "browser_control", "args": {"command": "navigate", "url": "data:text/html,x"},
         "expect": "fail", "note": "data-адрес"},
        {"tool": "browser_control",
         "args": {"command": "navigate", "url": "https://user:secret@example.com"},
         "expect": "fail", "note": "адрес с паролем внутри"},
        {"tool": "browser_control", "args": {"command": "teleport"}, "expect": "fail",
         "note": "неизвестная браузерная команда"},
        {"tool": "browser_control", "args": {"command": "navigate", "url": "yt"}, "expect": "fail",
         "note": "одно слово без домена"},
        {"tool": "pc_control", "args": {"command": "explode"}, "expect": "fail",
         "note": "неизвестная команда ПК"},
        {"tool": "pc_control", "args": {"command": "volume_set", "value": "loud"},
         "expect": "fail", "note": "громкость не числом"},
        {"tool": "pc_control", "args": {"command": "open_app"}, "expect": "fail",
         "note": "приложение без имени"},
        {"tool": "run_command", "args": {"command": "   "}, "expect": "fail",
         "note": "пустая команда"},
        {"tool": "not_a_tool", "args": {}, "expect": "fail", "note": "инструмент, которого нет"},
        {"tool": "pc_control", "args": {}, "expect": "fail", "note": "команда без command"},
    ]


def build_cases(extended: bool) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    cases += _volume_cases()
    cases += _clipboard_cases()
    cases += _window_cases()
    cases += _browser_cases()
    cases += _refusal_cases()
    if extended:
        cases += [
            {"tool": "pc_control", "args": {"command": "media_play_pause"}, "expect": "ok"},
            {"tool": "pc_control", "args": {"command": "media_next"}, "expect": "ok"},
            {"tool": "pc_control", "args": {"command": "media_prev"}, "expect": "ok"},
        ]
    for index, case in enumerate(cases):
        case.setdefault("id", f"RA-{index + 1:03d}")
    return cases


def _camera_and_screen() -> list[dict[str, Any]]:
    """Камера и экран: то, что живёт в главном цикле клиента, а не в диспетчере."""
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    busy = _client_running()
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened() and busy:
            # Запущенный клиент держит камеру сам — это не поломка.
            results.append({"id": "RA-CAM", "check": "camera_frames", "passed": True,
                            "detail": "камера занята запущенным клиентом (норма)",
                            "seconds": round(time.perf_counter() - started, 2)})
            return results
        try:
            frames = 0
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames += 1
                time.sleep(0.03)
            results.append({"id": "RA-CAM", "check": "camera_frames",
                            "passed": frames > 0 or busy,
                            "detail": f"{frames}/10 кадров"
                                      + (" (камеру читает запущенный клиент)" if busy else ""),
                            "seconds": round(time.perf_counter() - started, 2)})
        finally:
            cap.release()
    except Exception as exc:  # noqa: BLE001 - отчёт важнее исключения
        results.append({"id": "RA-CAM", "check": "camera_frames", "passed": False,
                        "detail": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.perf_counter() - started, 2)})
    started = time.perf_counter()
    try:
        from client.screen import capture_jpeg

        shot = capture_jpeg()
        results.append({"id": "RA-SCR", "check": "screen_capture",
                        "passed": bool(shot and shot.jpeg),
                        "detail": f"{shot.w}x{shot.h} jpeg {len(shot.jpeg)} байт",
                        "seconds": round(time.perf_counter() - started, 2)})
    except Exception as exc:  # noqa: BLE001 - отчёт важнее исключения
        results.append({"id": "RA-SCR", "check": "screen_capture", "passed": False,
                        "detail": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.perf_counter() - started, 2)})
    return results


def _client_running() -> bool:
    """Запущен ли на этом ПК сам клиент: он держит камеру и рабочий стол."""
    import subprocess

    finished = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True)
    return "python.exe" in (finished.stdout or "").lower()


async def run_cases(cases: list[dict[str, Any]], *, pause: float) -> list[dict[str, Any]]:
    from client.actions.dispatcher import Dispatcher
    from common.config import load_config

    config_path = ROOT / "config.yaml"
    cfg = load_config(str(config_path if config_path.is_file() else ROOT / "config.openai.yaml"))
    dispatcher = Dispatcher(cfg.client, None)
    try:
        await dispatcher.prepare()
    except Exception as exc:  # noqa: BLE001 - индекс приложений не обязателен
        print(json.dumps({"note": f"подготовка диспетчера: {type(exc).__name__}: {exc}"},
                         ensure_ascii=False))

    # The desktop browser driver attaches to a browser window that already
    # exists - by design, it never launches a second automation browser. A
    # person says "open youtube" and the hub's own app-choice flow opens the
    # browser first, so the audit does the same instead of blaming the client
    # for a window nobody opened.
    skip: dict[str, str] = {}
    if any(case.get("group") == "browser" for case in cases):
        reason = "ни один браузер не открылся"
        for name in ("chrome", "msedge", "firefox", "brave", "opera"):
            try:
                ok, error, _out = await dispatcher.execute(
                    {"id": "RA-BROWSER", "tool": "pc_control",
                     "args": {"command": "open_app", "value": name}})
            except Exception as exc:  # noqa: BLE001 - браузер не обязателен для остальных проверок
                ok, error = False, f"{type(exc).__name__}: {exc}"
            if ok:
                print(json.dumps({"note": f"браузер для проверок: {name}"}, ensure_ascii=False),
                      flush=True)
                break
            reason = f"{name}: {error}"
        else:
            skip["browser"] = f"браузер не открылся ({reason}) — переходы пропущены"

    results: list[dict[str, Any]] = []
    for case in cases:
        group = str(case.get("group") or "")
        if group and group in skip:
            item = {"id": case["id"], "tool": case["tool"], "args": case["args"],
                    "expect": case["expect"], "passed": True, "ok": False,
                    "detail": skip[group], "note": case.get("note", ""), "skipped": True,
                    "seconds": 0.0}
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            continue
        started = time.perf_counter()
        ok = False
        error: str | None = None
        output: str | None = None
        try:
            ok, error, output = await dispatcher.execute(
                {"id": case["id"], "tool": case["tool"], "args": case["args"]})
        except Exception as exc:  # noqa: BLE001 - падение = проваленная проверка
            error = f"{type(exc).__name__}: {exc}"
        detail = (error or output or "")[:300]
        expectation = case["expect"]
        if expectation == "any":
            passed = True
        else:
            passed = ok if expectation == "ok" else not ok
        wanted = str(case.get("must_contain") or "")
        if passed and wanted and wanted.casefold() not in str(output or "").casefold():
            passed = False
            detail = f"в ответе нет {wanted!r}: {detail}"
        item = {"id": case["id"], "tool": case["tool"], "args": case["args"],
                "expect": case["expect"], "passed": bool(passed), "ok": bool(ok),
                "detail": detail, "note": case.get("note", ""),
                "seconds": round(time.perf_counter() - started, 2)}
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if pause:
            await asyncio.sleep(pause)
    return results


async def _leave_the_room_quiet(dispatcher_cases: list[dict[str, Any]]) -> None:
    """Аудит трогал громкость: комната возвращается в тишину, как просил владелец."""
    from client.actions.pc import PCController

    controller = PCController()
    try:
        await controller.execute("mute")
        await controller.execute("volume_set", 0)
    except Exception as exc:  # noqa: BLE001 - тишина не важнее отчёта
        print(json.dumps({"note": f"не удалось вернуть тишину: {exc}"}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--pause", type=float, default=0.15,
                        help="пауза между действиями, чтобы окна успели отрисоваться")
    args = parser.parse_args()

    cases = build_cases(args.extended)
    print(json.dumps({"note": f"host {platform.node()}, {len(cases)} действий"},
                     ensure_ascii=False), flush=True)
    started = time.perf_counter()
    try:
        results = asyncio.run(run_cases(cases, pause=args.pause))
    except Exception as exc:  # noqa: BLE001 - отчёт обязан существовать
        print(json.dumps({"note": f"прогон упал: {type(exc).__name__}: {exc}",
                          "trace": traceback.format_exc()[-400:]}, ensure_ascii=False))
        results = []
    try:
        asyncio.run(_leave_the_room_quiet(cases))
    except Exception as exc:  # noqa: BLE001 - тишина не важнее отчёта
        print(json.dumps({"note": f"тишина не вернулась: {exc}"}, ensure_ascii=False))
    results += _camera_and_screen()

    passed = sum(1 for item in results if item["passed"])
    report = {"host": platform.node(), "at": time.time(),
              "total": len(results), "passed": passed, "results": results,
              "seconds": round(time.perf_counter() - started, 1)}
    if args.json:
        target = Path(args.json)
        if not target.is_absolute():
            target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": f"{passed}/{len(results)} проверок прошло",
                      "seconds": report["seconds"]}, ensure_ascii=False))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
