"""Звук комнатного ПК — тем же путём, которым им управляет ассистент.

Запускается НА комнатном ПК (его копирует `scripts/room-audio.ps1`). Владелец
просил выключить звук на ПК, где идут проверки, чтобы он не мешал: скрипт
зовёт `client.actions.pc.PCController` — ровно тот код, который выполняет
`pc_control` в живом клиенте, — и печатает одну строку JSON.

    python room_audio.py --command mute
    python room_audio.py --command unmute
    python room_audio.py --command volume_set --value 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="mute",
                        choices=["mute", "unmute", "volume_set"])
    parser.add_argument("--value", type=int, default=0)
    args = parser.parse_args()

    from client.actions.pc import PCController

    controller = PCController()

    async def run():
        await controller.prepare()
        if args.command == "volume_set":
            return await controller.execute("volume_set", args.value)
        return await controller.execute(args.command)

    try:
        result = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001 - the caller only needs the truth
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "command": args.command,
                      "detail": getattr(result, "detail", str(result))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
