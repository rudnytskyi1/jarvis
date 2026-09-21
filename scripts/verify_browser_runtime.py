"""Exercise the real Chrome DOM driver against a loopback-only test page."""
import asyncio
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client.actions.browser import BrowserController

HTML = b'''<!doctype html><html><title>Rowan browser test</title><body>
<h1>Local browser test</h1><label>Search<input aria-label="Search"></label>
<button onclick="document.getElementById('result').textContent='Found: '+document.querySelector('input').value">Find video</button>
<div id="result"></div><div style="height:900px"></div>
<p>Second viewport content</p><div style="height:800px"></div></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML)
    def log_message(self, *args):
        pass

async def verify(url, profile):
    browser = BrowserController(profile, headless=True)
    try:
        started = time.perf_counter()
        result = json.loads(await browser.execute({'command': 'navigate', 'url': url}))
        field = next(x for x in result['elements'] if x['text'] == 'Search')
        result = json.loads(await browser.execute({'command': 'fill', 'ref': field['ref'], 'text': 'Mars latest video'}))
        button = next(x for x in result['elements'] if x['text'] == 'Find video')
        raw = await browser.execute({'command': 'click', 'ref': button['ref']})
        assert 'Found: Mars latest video' in json.loads(raw)['text']
        assert len(raw) <= 4000
        try:
            await browser.execute({'command': 'click', 'ref': button['ref']})
        except ValueError:
            pass
        else:
            raise AssertionError('Old browser reference was accepted')
        scrolled = json.loads(await browser.execute({'command': 'scroll', 'direction': 'down'}))
        assert 'Second viewport content' in scrolled['text']
        assert 'Local browser test' not in scrolled['text']
        elapsed = time.perf_counter() - started
        print(json.dumps({'ok': True, 'navigate_fill_click_s': round(elapsed, 3),
                          'stale_ref_rejected': True, 'screenshots': 0}))
    finally:
        await browser.close()

if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix='rowan-browser-check-') as profile:
            asyncio.run(verify(f'http://127.0.0.1:{server.server_port}/', profile))
    finally:
        server.shutdown()
