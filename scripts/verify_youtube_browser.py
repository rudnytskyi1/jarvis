"""Inspect the requested public YouTube search in a temporary ordinary Chrome tab."""
import asyncio
import json
import sys
import traceback
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QUERY = 'MrBeast Squid Game'
URL = 'https://www.youtube.com/results?search_query=MrBeast+Squid+Game'


async def main():
    from client.actions import app_control, pc
    from client.actions.browser_desktop import DesktopBrowserController
    driver = DesktopBrowserController()
    report = {'snapshots': []}
    opened = False
    try:
        windows = [w for app in app_control.visible_apps() if app['image'] == 'chrome.exe' for w in app['windows']]
        if len(windows) != 1:
            raise RuntimeError('Need one unambiguous existing Chrome window.')
        pc._sync_focus_window(windows[0]['hwnd'])
        mods, keys, _ = pc.parse_hotkey('ctrl+t')
        pc._sync_hotkey(mods, keys)
        opened = True
        await asyncio.sleep(.3)
        args = {'command': 'navigate', 'url': URL, 'browser': 'Google Chrome'}
        for _ in range(4):
            snapshot = json.loads(await driver.execute(args))
            report['snapshots'].append(snapshot)
            if any('squid game' in e.get('text', '').casefold() for e in snapshot.get('elements', [])):
                report['search_results_accessible'] = True
                break
            args = {'command': 'read', 'browser': 'Google Chrome'}
            await asyncio.sleep(.8)
        report['existing_chrome_pid'] = windows[0]['pid']
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        if opened:
            try:
                current = json.loads(await driver.execute({'command': 'read', 'browser': 'Google Chrome'}))
                address = current.get('url', '')
                parsed = urlparse(address if '://' in address else 'https://' + address)
                if (parsed.hostname in {'www.youtube.com', 'youtube.com'} and parsed.path == '/results'
                    and parse_qs(parsed.query).get('search_query') == [QUERY]):
                    mods, keys, _ = pc.parse_hotkey('ctrl+w')
                    pc._sync_hotkey(mods, keys)
                    report['temporary_tab_closed'] = True
            except Exception:
                report['cleanup_error'] = traceback.format_exc()
        await driver.close()
        (ROOT / 'data' / 'youtube-browser-verification.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(main())
