"""Read-only bot/group connectivity check; never post or consume updates."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config
from hub.telegram import TelegramError, TelegramProvider


async def main():
    cfg = load_config(ROOT / 'config.openai.yaml')
    provider = TelegramProvider(cfg.server.telegram)
    try:
        result = await provider.check_connection()
        webhook = await provider.get_webhook_info()
        result['webhook_configured'] = bool(webhook.get('url'))
        result['messages_sent'] = 0
        print(json.dumps(result, ensure_ascii=True))
    except TelegramError as exc:
        print(json.dumps({'ok': False, 'error': str(exc), 'messages_sent': 0}))
        return 1
    finally:
        await provider.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
