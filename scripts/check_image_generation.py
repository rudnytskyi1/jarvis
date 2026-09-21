"""One billed Nano Banana smoke check; synthetic prompt, shared budget, no camera.

Supply GEMINI_API_KEY in the process environment using the saved-key helper.
Never pass the key as an argument. This script sends one request with no retries.
"""
import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config
from hub.api_budget import CloudUnavailable
from hub.image_generation import MAX_IMAGE_BYTES, MIMES, ImageGenerator


class CheckedGenerator(ImageGenerator):
    """Keep only non-sensitive usage fields to diagnose accounting mismatches."""

    def _settle(self, reservation, data):
        usage = data.get('usageMetadata') or {}
        report = {'reservation': reservation, 'usageMetadata': {key: usage[key] for key in (
            'promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount',
            'totalTokenCount', 'candidatesTokensDetails', 'promptTokensDetails',
        ) if key in usage}}
        path = ROOT / 'data' / ('nano-banana-usage-' + reservation + '.json')
        path.write_text(json.dumps(report), encoding='utf-8')
        print(json.dumps({'usage_report': str(path), **report}))
        super()._settle(reservation, data)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        help='Upload this synthetic robot image to test adding a red party hat.')
    args = parser.parse_args()
    reference, mime = None, 'image/jpeg'
    if args.reference:
        if not args.reference.is_file() or args.reference.stat().st_size > MAX_IMAGE_BYTES:
            parser.error('Reference image is missing or exceeds the image size limit.')
        with Image.open(args.reference) as picture:
            mime = MIMES.get(picture.format)
        if mime is None:
            parser.error('Reference must be PNG, JPEG or WebP.')
        reference = args.reference.read_bytes()
    cfg = load_config(ROOT / 'config.openai.yaml')
    client = CheckedGenerator(cfg.server.image_generation, ledger_path=ROOT / 'data' / 'api_usage.sqlite3',
                            monthly_usd=cfg.server.llm.monthly_budget_usd)
    started = time.perf_counter()
    try:
        prompt = (
            'Edit the provided image: add a tall bright red party hat with a white pompom '
            'on the small copper robot\'s head. Preserve the same robot, banana, pose, '
            'camera angle, lighting, and dark background. Make only this change. No text.'
            if reference is not None else
            'A friendly small copper robot holding a bright yellow banana, cinematic studio lighting, '
            'dark charcoal background, playful polished 3D illustration, square composition, no text.'
        )
        result = await client.generate(prompt, reference, mime)
        output = ROOT / 'data' / ('nano-banana-check-' + uuid.uuid4().hex + '.png')
        with output.open('xb') as file:
            file.write(result.png)
        print(json.dumps({'ok': True, 'model': cfg.server.image_generation.model,
                          'reference_supplied': reference is not None,
                          'width': result.width, 'height': result.height, 'path': str(output),
                          'elapsed_s': round(time.perf_counter() - started, 2)}))
    except CloudUnavailable as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        return 1
    finally:
        await client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
