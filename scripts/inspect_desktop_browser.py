"""Read the ordinary room browser from an interactive Windows scheduled task."""
import asyncio
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def main():
    from client.actions.browser_desktop import DesktopBrowserController
    driver = DesktopBrowserController()
    try:
        result = json.loads(await driver.execute({'command': 'read', 'browser': 'Google Chrome'}))
        (ROOT / 'data' / 'desktop-browser-probe.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    finally:
        await driver.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception:
        (ROOT / 'data' / 'desktop-browser-probe.json').write_text(json.dumps({'error': traceback.format_exc()}), encoding='utf-8')
        raise
