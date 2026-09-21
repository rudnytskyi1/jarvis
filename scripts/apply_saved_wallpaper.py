"""Apply an explicitly selected existing image in the room desktop session."""
import argparse
import base64
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from client.actions.wallpaper import set_wallpaper
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    previous = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Control Panel\Desktop') as key:
            previous = winreg.QueryValueEx(key, 'WallPaper')[0]
    except OSError:
        pass
    try:
        result = set_wallpaper({'image_base64': base64.b64encode(args.image.read_bytes()).decode('ascii')})
        result['ok'] = True
    except Exception:
        result = {'ok': False, 'applied': False, 'verified': False, 'error': traceback.format_exc()}
    result['previous_path'] = previous
    args.report.write_text(json.dumps(result), encoding='utf-8')
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
