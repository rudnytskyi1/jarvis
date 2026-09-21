"""Read-only installed/open-browser inventory on the actual Windows desktop."""
import asyncio
import ctypes
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client.actions.app_control import AppController
from client.actions.apps import AppIndex
from common.config import load_config


async def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'config.openai.yaml')
    controller = AppController(AppIndex(cfg.client.apps))
    report = {}
    for action in ('open', 'close'):
        result = await controller.inventory(action, 'browser')
        report[action] = [{k: v for k, v in c.items() if k != 'id'} for c in result['candidates']]
    # An invisible test window belongs only to this diagnostic. Never touch any
    # user window; verify native pid validation and graceful WM_CLOSE for real.
    from client.actions import pc
    from client.actions.app_control import close_windows
    user32 = pc._user32()
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = ctypes.c_ssize_t
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    hwnd = user32.CreateWindowExW(0, 'STATIC', 'Rowan diagnostic only', 0, 0, 0, 1, 1, None, None, None, None)
    if not hwnd:
        raise RuntimeError('Could not create the private test window')
    try:
        report['wrong_pid_refused'] = not close_windows([{'hwnd': hwnd, 'pid': os.getpid() + 999999}])
        future = asyncio.create_task(asyncio.to_thread(close_windows, [{'hwnd': hwnd, 'pid': os.getpid()}]))
        while not future.done():
            msg = wintypes.MSG()
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                user32.DispatchMessageW(ctypes.byref(msg))
            await asyncio.sleep(.01)
        report['graceful_close_verified'] = await future
    finally:
        if user32.IsWindow(hwnd):
            user32.DestroyWindow(hwnd)
    text = json.dumps(report)
    print(text)
    if '--output' in sys.argv:
        (root / 'data/app-request-check.json').write_text(text, encoding='utf-8')
    assert report['wrong_pid_refused'] and report['graceful_close_verified']


if __name__ == '__main__':
    asyncio.run(main())
