"""Один настоящий ответ облачной модели — проверка, что чатбот жив.

Владелец 2026-09-23: «monthly api allowance убери нахер у меня чатбот не
работает». Скрипт отвечает на два вопроса сразу, без Telegram и без комнаты:

1. что стоит в живом конфиге про предел расходов (``0`` = предела нет);
2. отвечает ли облачная модель прямо сейчас — тем же клиентом, которым
   пользуется хаб (``hub.llm.LlmClient``, тот же ключ из ``.env``).

    python scripts/llm_probe.py                  # короткая проверка
    python scripts/llm_probe.py "привет, ты тут?"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = "Say one short sentence: are you working?"


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
    from hub.api_budget import ApiBudget
    from hub.llm import LlmClient

    settings = load_config(REPO / "config.openai.yaml").server.llm
    key = os.environ.get(settings.api_key_env, "")
    limit = float(settings.monthly_budget_usd or 0.0)
    print(f"env: {', '.join(loaded) if loaded else 'нет .env'}")
    print(f"llm: provider={settings.provider} model={settings.model} "
          f"url={settings.base_url} key={'есть' if key else 'НЕТ'}")
    print(f"предел расходов: {'нет (0)' if limit == 0 else f'{limit} USD в месяц'}")
    ledger = ApiBudget(REPO / "data" / "api_usage.sqlite3", monthly_usd=limit,
                       model=settings.model)
    status = ledger.status()
    print(f"потрачено в этом месяце: {status.get('used_usd', 0):.4f} USD, "
          f"предел: {status.get('limit_usd')}")

    client = LlmClient(settings)
    print(f"вопрос: {args.text!r}")
    started = time.perf_counter()
    try:
        text = await client.reply_text([{"role": "user", "content": args.text}])
    except Exception as exc:  # noqa: BLE001 - это и есть предмет проверки
        print(f"ответа нет за {time.perf_counter() - started:.2f} с: "
              f"{type(exc).__name__}: {exc}")
        return 1
    print(f"ответ за {int((time.perf_counter() - started) * 1000)} мс: {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
