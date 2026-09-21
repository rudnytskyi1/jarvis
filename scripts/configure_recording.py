"""Enable indefinite local recording. No auto-deletion or copied device settings."""
import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import Config


def configure(path, target, *, dry_run=False):
    original = path.read_bytes()
    data = yaml.safe_load(original.decode('utf-8-sig'))
    Config.model_validate(data)
    if target == 'server':
        settings = data.setdefault('server', {}).setdefault('audio_recording', {})
        camera_requests = data['server'].setdefault('camera_request_recording', {})
        camera_requests.update(enabled=True, retention_days=0, max_gb=0)
    elif target == 'room':
        camera = data.setdefault('client', {}).setdefault('camera', {})
        camera.update(fps=0, half=True)
        settings = camera.setdefault('frame_recording', {})
    else:
        raise ValueError('Expected server or room target')
    settings.update(enabled=True, retention_days=0, max_gb=0)
    Config.model_validate(data)
    result = {'target': target, 'recording': dict(settings), 'dry_run': dry_run}
    if target == 'server':
        result['camera_request_recording'] = dict(camera_requests)
    if target == 'room':
        result['yolo_fps_limit'] = 0
    if dry_run or yaml.safe_load(original.decode('utf-8-sig')) == data:
        return result
    backup = path.with_name(f'{path.name}.recording-backup-{datetime.now():%Y%m%d-%H%M%S-%f}')
    with backup.open('xb') as handle:
        handle.write(original)
    pending = path.with_name(f'{path.name}.{uuid.uuid4().hex}.pending')
    try:
        with pending.open('x', encoding='utf-8') as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
        if path.read_bytes() != original:
            raise RuntimeError('Configuration changed; refusing to overwrite it')
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    result['backup'] = str(backup)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['server', 'room'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.openai.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(configure(args.config, args.target, dry_run=args.dry_run)))
