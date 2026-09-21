"""Exercise the ordinary Chrome in one temporary local test tab, then close that tab."""
import asyncio
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HTML = b'''<!doctype html><html><head><title>Rowan local browser verification</title></head>
<body><h1>Local verification</h1><form onsubmit="event.preventDefault();document.getElementById('result').textContent='Found: '+document.getElementById('query').value">
<label>Search<input id="query" aria-label="Search"></label><button type="submit">Find video</button></form>
<p id="result"></p><a href="/next">Open verified result</a></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = (b'<html><title>Rowan verified destination</title><h1>Verified destination</h1></html>'
                   if self.path == '/next' else HTML)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


async def main():
    from client.actions import app_control, pc
    from client.actions.browser_desktop import DesktopBrowserController
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/'
    driver = DesktopBrowserController()
    started = time.monotonic()
    opened = False
    report = {}
    try:
        windows = [w for app in app_control.visible_apps() if app['image'] == 'chrome.exe' for w in app['windows']]
        if len(windows) != 1:
            raise RuntimeError('Verification requires one unambiguous existing Chrome window.')
        hwnd = windows[0]['hwnd']
        pc._sync_focus_window(hwnd)
        modifiers, keys, _ = pc.parse_hotkey('ctrl+t')
        pc._sync_hotkey(modifiers, keys)
        opened = True
        await asyncio.sleep(.3)
        result = json.loads(await driver.execute({'command': 'navigate', 'url': url, 'browser': 'Google Chrome'}))
        for _ in range(4):
            fields = [e for e in result.get('elements', []) if e.get('text') == 'Search']
            if fields:
                break
            await asyncio.sleep(.3)
            result = json.loads(await driver.execute({'command': 'read'}))
        field = next(e for e in fields if e.get('role') in ('edit', 'textbox', 'combobox', 'input'))
        report['initial_page'] = result
        result = json.loads(await driver.execute({'command': 'fill', 'ref': field['ref'], 'text': 'MrBeast Squid Game', 'submit': True}))
        assert 'Found: MrBeast Squid Game' in result.get('text', ''), result
        report['atomic_fill_submit'] = True
        link = next(e for e in result['elements'] if e.get('text') == 'Open verified result')
        result = json.loads(await driver.execute({'command': 'click', 'ref': link['ref']}))
        assert 'Verified destination' in result.get('text', '') or result.get('url', '').rstrip('/').endswith('/next'), result
        report['click_destination_verified'] = True
        result = json.loads(await driver.execute({'command': 'back'}))
        field = next(e for e in result['elements'] if e.get('text') == 'Search' and e.get('role') in ('edit', 'textbox', 'combobox', 'input'))
        await driver.execute({'command': 'fill', 'ref': field['ref'], 'text': 'Enter without ref'})
        result = json.loads(await driver.execute({'command': 'press', 'key': 'Enter'}))
        assert 'Found: Enter without ref' in result.get('text', ''), result
        report.update(ok=True, enter_without_ref=True, seconds=round(time.monotonic() - started, 2),
                      existing_chrome_pid=windows[0]['pid'], new_browser_profile=False)
    except Exception:
        report.update(ok=False, error=traceback.format_exc())
    finally:
        if opened:
            try:
                current = json.loads(await driver.execute({'command': 'read', 'browser': 'Google Chrome'}))
                report['final_page'] = current
                if (current.get('url', '').removeprefix('https://').removeprefix('http://').startswith(url.removeprefix('http://'))
                    and current.get('title') in ('Rowan local browser verification', 'Rowan verified destination')):
                    modifiers, keys, _ = pc.parse_hotkey('ctrl+w')
                    pc._sync_hotkey(modifiers, keys)
                    report['temporary_tab_closed'] = True
            except Exception:
                report['temporary_tab_closed'] = False
        await driver.close()
        server.shutdown()
        (ROOT / 'data' / 'desktop-browser-verification.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(main())
