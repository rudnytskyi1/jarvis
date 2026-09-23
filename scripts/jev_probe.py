"""Один настоящий вопрос к TypeSafe Jev — проверка, что он отвечает живьём.

Владелец 2026-09-23: «надеюсь, что TypeSafe Jev активно используется». Числа
берутся из ``scripts/jev_usage_report.py``, а этот скрипт показывает, что ключ и
путь рабочие: он делает ровно тот batched-вызов, который хаб делает на каждой
обычной реплике (``hub.app.Connection._understand_turn``), и печатает ответ
Jev — семейство инструментов, «есть ли что делать» и «это продолжение?».

    python scripts/jev_probe.py                      # фраза-пример
    python scripts/jev_probe.py "включи свет на кухне"

Ключ берётся из ``.env`` (``JEV_API_KEY``), как у хаба; в лог и на экран он не
печатается — только признак «есть/нет».
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = "открой ютуб и включи видео про мистера биса"


def _load_env(path: Path, environ: dict[str, str]) -> list[str]:
    """Те же строки ``KEY=value``, что читает ``scripts/openai-key-store.ps1``."""
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in environ:
            continue
        environ[key] = value.strip().strip('"').strip("'")
        loaded.append(key)
    return loaded


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    loaded = _load_env(REPO / ".env", os.environ)

    from common.config import load_config
    from hub.jev_decider import JevDecider
    from hub.tools import TOOL_FAMILY_MEANINGS, TOOL_FAMILY_NAMES

    settings = load_config(REPO / "config.openai.yaml").server.decider.providers.jev
    key = os.environ.get(settings.api_key_env, "")
    print(f"env: {', '.join(loaded) if loaded else 'нет .env'}")
    print(f"jev: enabled={settings.enabled} model={settings.model} "
          f"url={settings.base_url}{settings.path} key={'есть' if key else 'НЕТ'}")
    if not key:
        print(f"{settings.api_key_env} не задан — живого ответа не будет")
        return 1
    client = JevDecider(base_url=settings.base_url, api_key=key, path=settings.path,
                        model=settings.model, timeout_s=settings.timeout_ms / 1000.0,
                        allowed_for=lambda _home: True)
    context = {"home_id": "livingroom", "room": "livingroom", "text": args.text}
    print(f"вопрос: {args.text!r}")
    started = time.perf_counter()
    try:
        found = await client.understand(
            context, families=list(TOOL_FAMILY_NAMES), meanings=TOOL_FAMILY_MEANINGS,
            timeout_s=settings.timeout_ms / 1000.0)
    except Exception as exc:  # noqa: BLE001 - отчёт, а не падение
        print(f"Jev не ответил за {time.perf_counter() - started:.2f} с: {exc}")
        return 1
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(f"ответ за {elapsed_ms} мс (бюджет конфига {settings.timeout_ms} мс)")
    for name in ("act", "family", "followup"):
        item = found.get(name)
        if item is None:
            print(f"  {name}: (не ответил)")
            continue
        print(f"  {name}: {item['value']!r} (уверенность {float(item['confidence']):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
