"""One live Vertex AI image request, with the whole credential story printed.

Владелец 2026-09-23: «для генерации картинок теперь используй vertexai api (у
меня бесплатные 300$ credits)». ``scripts/check_image_generation.py`` уже умеет
проверять Nano Banana, но когда картинки не появляются, первым делом нужно
знать, чем именно подписан запрос. Этот скрипт отвечает на три вопроса одним
запуском: откуда взят доступ, куда ушёл запрос и что вернулось.

Without ``--send`` it only reports, so a broken setup costs nothing:

    python scripts/vertex_image_probe.py --dry-run
    python scripts/vertex_image_probe.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config
from hub.api_budget import CloudUnavailable
from hub.image_generation import ImageGenerator

PROMPT = ('A friendly small copper robot holding a bright yellow banana, cinematic studio '
          'lighting, dark charcoal background, polished 3D illustration, no text.')


def describe_access(generator: ImageGenerator) -> dict:
    """Where the credential comes from - paths and names, never secret values."""
    vertex = generator.vertex
    if vertex is None:
        import os

        return {'road': 'gemini', 'api_key_env': generator.cfg.api_key_env,
                'present': bool(os.environ.get(generator.cfg.api_key_env, '').strip())}
    if vertex.api_key:
        return {'road': 'vertex', 'credential': 'express api key',
                'api_key_env': vertex.api_key_env}
    if vertex.static_token:
        return {'road': 'vertex', 'credential': 'access token from the environment',
                'access_token_env': vertex.access_token_env}
    path = vertex.credentials_file()
    return {'road': 'vertex',
            'credential': str(path) if path is not None else 'none found',
            'searched': [str(candidate) for candidate in vertex.candidate_paths()]}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.openai.yaml',
                        help='active hub config (default: config.openai.yaml)')
    parser.add_argument('--dry-run', action='store_true',
                        help='report the credential and the URL without sending anything')
    args = parser.parse_args()
    cfg = load_config(args.config)
    image_cfg = cfg.server.image_generation
    generator = ImageGenerator(image_cfg, ledger_path=ROOT / 'data' / 'api_usage.sqlite3',
                               monthly_usd=cfg.server.llm.monthly_budget_usd)
    report: dict = {'provider': image_cfg.provider, 'model': image_cfg.model,
                    'enabled': image_cfg.enabled, 'access': describe_access(generator),
                    'ready': generator.ready}
    vertex = generator.vertex
    report['project'] = getattr(vertex, 'project', '') if vertex is not None else None
    report['location'] = getattr(vertex, 'location', '') if vertex is not None else None
    try:
        if vertex is not None:
            try:
                report['url'] = vertex.model_url(image_cfg.model)
            except Exception as exc:  # noqa: BLE001 - the reason IS the report
                report['url'] = f'{type(exc).__name__}: {exc}'
        if not generator.ready:
            report['ok'] = False
            report['error'] = ((vertex.missing_reason() if vertex is not None else '')
                               or 'the provider is not set up')
            if args.dry_run:
                print(json.dumps(report, indent=2, ensure_ascii=False))
                return 1
        if args.dry_run:
            report['ok'] = True
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        started = time.perf_counter()
        result = await generator.generate(PROMPT)
        output = ROOT / 'data' / ('vertex-probe-' + uuid.uuid4().hex + '.png')
        with output.open('xb') as file:
            file.write(result.png)
        report.update(ok=True, width=result.width, height=result.height, path=str(output),
                      elapsed_s=round(time.perf_counter() - started, 2),
                      spent_usd=generator.budget.status()['accounted_usd'])
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except CloudUnavailable as exc:
        report.update(ok=False, error=str(exc))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    finally:
        await generator.close()


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
