"""Один настоящий вопрос к TypeSafe Jev — проверка, что он отвечает живьём.

Владелец 2026-09-23: «надеюсь, что TypeSafe Jev активно используется». Числа
берутся из ``scripts/jev_usage_report.py``, а этот скрипт показывает, что ключ и
путь рабочие: он делает ровно тот batched-вызов, который хаб делает на каждой
обычной реплике (``hub.app.Connection._understand_turn``), и печатает ответ
Jev — семейство инструментов, «есть ли что делать» и «это продолжение?».

    python scripts/jev_probe.py                      # фраза-пример
    python scripts/jev_probe.py "включи свет на кухне"

Вторая половина скрипта — цепочка решений хаба целиком
(``hub.app._decision_chain``): она спрашивает тип ``route`` у настоящей цепочки
``[rules, jev]`` и печатает, КТО ответил. До AU-19 здесь всегда отвечал
``rules``: правила закрывали вопрос своей догадкой 0.6, и Jev не спрашивали ни
разу (5087 решений, все ``rules``).

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
    for name in ("act", "family", "single", "followup"):
        item = found.get(name)
        if item is None:
            print(f"  {name}: (не ответил)")
            continue
        print(f"  {name}: {item['value']!r} (уверенность {float(item['confidence']):.2f})")
    await _probe_chain(args.text)
    return 0


async def _probe_chain(text: str) -> None:
    """The hub's own decision chain, asked live: who answers ``route``?"""
    from common.config import load_config
    from hub import app as hub_app

    # Решения этой пробы — настоящие, но это не ходы комнаты: они не должны
    # попадать в калибровку владельца (DECISIONS.md TEST-DB-01). Первые прогоны
    # AU-19 писали их в живую `data/hub.db`; теперь у пробы своя база, как у
    # стенда (переменная ``ROWAN_HUB_DB`` — тот же выключатель, что у
    # ``scripts/live-eval.py``).
    probe_db = REPO / "data" / "audit" / "jev-probe-hub.db"
    probe_db.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(hub_app.HUB_DB_ENV, str(probe_db))
    cfg = load_config(REPO / "config.openai.yaml")
    hub_app.configure(cfg)
    # Свежая цепочка на каждый запуск: проба не должна пользоваться кэшем
    # решений предыдущего прогона.
    hub_app._decider = None
    hub_app._jev_client = None
    chain = hub_app._decision_chain(["rowan ai"])
    if chain is None:
        print("цепочка решений недоступна: провайдеров нет")
        return
    home = str((cfg.homes[0].home_id if cfg.homes else "") or "")
    print(f"\nцепочка решений (home={home or 'нет дома'}):")
    for question in (text, "rowan ai volume 20"):
        started = time.perf_counter()
        try:
            decision = await chain.choose(
                "fast command or model request?",
                ["fast_command", "llm"],
                {"text": question, "wake_words": ["rowan ai"], "home_id": home},
                decision_type="route")
        except Exception as exc:  # noqa: BLE001 - отчёт, а не падение
            print(f"  {question!r}: цепочка не ответила ({type(exc).__name__}: {exc})")
            continue
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(f"  route     {question!r}: {decision.provider} -> {decision.value!r} "
              f"(уверенность {decision.confidence:.2f}, {elapsed_ms} мс)")
    # ``addressed`` — та точка, где правила честно не уверены (0.6 без
    # подтверждённого пробуждения), и второй провайдер цепочки обязан ответить.
    started = time.perf_counter()
    try:
        decision = await chain.yes_no(
            "Was this utterance addressed to the assistant?",
            {"text": text, "wake_words": ["rowan ai"], "home_id": home,
             "heuristic": False},
            decision_type="addressed")
    except Exception as exc:  # noqa: BLE001 - отчёт, а не падение
        print(f"  addressed {text!r}: цепочка не ответила ({type(exc).__name__}: {exc})")
        return
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(f"  addressed {text!r}: {decision.provider} -> {decision.value!r} "
          f"(уверенность {decision.confidence:.2f}, {elapsed_ms} мс)")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
