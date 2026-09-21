"""Apply room speech settings without changing audio devices or server settings.

Run on the room PC after copying this file, then restart JarvisRoomClient.
Use --dry-run to validate and show the proposed VAD settings without writing.
"""
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


def update(path: Path, *, dry_run=False) -> dict:
    original = path.read_bytes()
    data = yaml.safe_load(original.decode('utf-8-sig'))
    cfg = Config.model_validate(data)
    previous = cfg.client.vad.model_dump()
    vad = data.setdefault('client', {}).setdefault('vad', {})
    # Allow a natural pause; avoid the most aggressive far-field voice gate.
    # Preserve even more patient settings if the user already chose them.
    vad.update(aggressiveness=min(cfg.client.vad.aggressiveness, 2),
               silence_ms=max(cfg.client.vad.silence_ms, 1100),
               max_utterance_s=max(cfg.client.vad.max_utterance_s, 25),
               pre_roll_ms=max(cfg.client.vad.pre_roll_ms, 1500))
    changed = Config.model_validate(data).client.vad.model_dump()
    result = {'dry_run': dry_run, 'before': previous, 'after': changed}
    if dry_run or previous == changed:
        return result
    if path.read_bytes() != original:
        raise RuntimeError('Configuration changed; refusing to overwrite it')
    backup = path.with_name(f'{path.name}.speech-backup-{datetime.now():%Y%m%d-%H%M%S-%f}')
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
    parser.add_argument('--config', type=Path, default=ROOT / 'config.openai.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(update(args.config, dry_run=args.dry_run)))
