"""Import verified legacy camera crops locally; never call an image API.

Only <24-hex-person-hash>/<timestamp>.jpg + .json pairs are considered. All
originals and rejected photographs remain untouched. Run once after deploying
the persistent gallery, or rerun safely after improving manual enrollment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hub.appearance import AppearanceGallery

MIN_IDENTITY = .65
MIN_MARGIN = .12
MAX_LEGACY_BYTES = 8 * 1024 * 1024


def normalized(raw):
    try:
        vector = np.asarray(raw, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        if vector.size and np.isfinite(vector).all() and norm > 0:
            return vector / norm
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def anchor_fingerprint(profiles):
    anchors = {}
    for name, vectors in profiles.items():
        valid = [value.tolist() for raw in vectors if (value := normalized(raw)) is not None]
        if valid:
            anchors[str(name)] = valid
    payload = json.dumps(anchors, sort_keys=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def identity_match(face, profiles):
    """Independently rank manual anchors, without adaptive recognition vectors."""
    vector = normalized(face.get('embedding'))
    if vector is None:
        return None, 0.0, 0.0
    scores = []
    for name, samples in profiles.items():
        matches = []
        for sample in samples:
            anchor = normalized(sample)
            if anchor is not None and anchor.shape == vector.shape:
                matches.append(float(vector @ anchor))
        if matches:
            scores.append((max(matches), name))
    scores.sort(reverse=True)
    if not scores:
        return None, 0.0, 0.0
    score, name = scores[0]
    margin = score - (scores[1][0] if len(scores) > 1 else 0.0)
    return (name if score >= MIN_IDENTITY and margin >= MIN_MARGIN else None), score, margin


def legacy_pairs(root):
    """Exclude generated images, new UUID sample directories and symlinks out."""
    root = Path(root).resolve()
    if not root.is_dir():
        return
    for directory in sorted(root.iterdir()):
        if not re.fullmatch(r'[0-9a-f]{24}', directory.name) or not directory.is_dir():
            continue
        if not directory.resolve().is_relative_to(root):
            continue
        for path in sorted(directory.glob('*.jpg')):
            metadata = path.with_suffix('.json')
            if (re.fullmatch(r'[0-9]{10,20}\.jpg', path.name)
                    and path.resolve().is_relative_to(root)
                    and metadata.is_file() and metadata.resolve().is_relative_to(root)):
                yield path, metadata


def bootstrap(root, engine, registry, *, gallery=None, manifest=None):
    root = Path(root).resolve()
    gallery = gallery or AppearanceGallery(root)
    profiles = registry.face_profiles()
    if not profiles:
        raise ValueError('No manually enrolled face profiles are available; no legacy photos were imported.')
    anchors = anchor_fingerprint(profiles)
    canonical = {name.casefold(): name for name in profiles}
    root.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest) if manifest is not None else root / 'bootstrap-imports.sqlite3'
    report = {'imported': 0, 'rejected': 0, 'already_imported': 0, 'unchanged_rejection': 0,
              'pending_review': 0, 'invalid_files': 0, 'items': [], 'paid_api_requests': 0}
    with sqlite3.connect(manifest, timeout=15) as database:
        database.row_factory = sqlite3.Row
        database.execute('''CREATE TABLE IF NOT EXISTS imports (
            source_key TEXT PRIMARY KEY, source_path TEXT NOT NULL,
            name TEXT NOT NULL, anchor_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL, sample_id TEXT, reason TEXT,
            captured_at REAL NOT NULL, updated_at REAL NOT NULL)''')
        database.commit()
        for path, metadata_path in legacy_pairs(root):
            relative = path.relative_to(root).as_posix()
            try:
                if path.stat().st_size > MAX_LEGACY_BYTES or metadata_path.stat().st_size > 64 * 1024:
                    raise ValueError('Legacy image or metadata exceeds the size limit')
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                if not isinstance(metadata, dict) or not isinstance(metadata.get('name'), str):
                    raise ValueError('Missing legacy person name')
                legacy_name = ' '.join(metadata['name'].split())
                captured = float(metadata['ts'])
                if not math.isfinite(captured) or captured <= 0 or captured > time.time() + 300:
                    raise ValueError('Invalid legacy capture timestamp')
                expected_directory = hashlib.sha256(legacy_name.casefold().encode()).hexdigest()[:24]
                if path.parent.name != expected_directory:
                    raise ValueError('Legacy person name does not match its source directory')
                name = canonical.get(legacy_name.casefold())
                if name is None:
                    raise ValueError('Legacy name has no current manual face enrollment')
                jpeg = path.read_bytes()
                source_key = hashlib.sha256(name.casefold().encode() + b'\0' + hashlib.sha256(jpeg).digest()).hexdigest()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                report['invalid_files'] += 1
                report['items'].append({'source': relative, 'status': 'invalid', 'reason': str(exc)})
                continue

            # Claim the source before inference/enrollment. A crash leaves a
            # visible pending row rather than importing an uncertain duplicate.
            database.execute('BEGIN IMMEDIATE')
            previous = database.execute('SELECT * FROM imports WHERE source_key=?', (source_key,)).fetchone()
            if previous is not None and (previous['status'] in {'imported', 'pending'}
                    or previous['anchor_fingerprint'] == anchors):
                status = {'imported': 'already_imported', 'pending': 'pending_review'}.get(previous['status'], 'unchanged_rejection')
                report[status] += 1
                report['items'].append({'source': relative, 'name': name, 'status': status,
                                        'sample_id': previous['sample_id']})
                database.commit()
                continue
            database.execute('''INSERT OR REPLACE INTO imports VALUES(?,?,?,?,?,?,?,?,?)''',
                (source_key, relative, name, anchors, 'pending', None, None, captured, time.time()))
            database.commit()

            reason, sample = None, None
            try:
                faces = engine.located_faces(jpeg)
                matching = []
                for face in faces:
                    found, score, margin = identity_match(face, profiles)
                    if found == name:
                        matching.append((face, score, margin))
                if len(matching) != 1:
                    reason = ('No unique face independently matches the current manual name '
                              'with score >= 0.65 and margin >= 0.12')
                else:
                    selected, score, margin = matching[0]
                    sample = gallery.enroll(jpeg, name, selected, faces=faces, now=captured)
                    if sample is None:
                        reason = 'The matched face did not pass gallery quality/crop checks'
            except Exception as exc:
                # Keep pending on unexpected failures: enrollment might have
                # committed before the failure. A rerun must not duplicate it.
                report['pending_review'] += 1
                report['items'].append({'source': relative, 'name': name, 'status': 'pending_review',
                                        'reason': type(exc).__name__})
                continue
            status = 'imported' if sample else 'rejected'
            sample_id = sample['sample_id'] if sample else None
            database.execute('UPDATE imports SET status=?, sample_id=?, reason=?, updated_at=? WHERE source_key=?',
                             (status, sample_id, reason, time.time(), source_key))
            database.commit()
            report[status] += 1
            report['items'].append({'source': relative, 'name': name, 'status': status,
                                    'sample_id': sample_id, 'reason': reason})
    return report


def main(argv=None):
    from common.config import load_config
    from hub.face import FaceEngine
    from hub.speaker import PEOPLE_FILENAME, VoiceRegistry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'config.openai.yaml')
    parser.add_argument('--root', type=Path, default=REPO_ROOT / 'data' / 'appearance')
    parser.add_argument('--report', type=Path, default=REPO_ROOT / 'data' / 'appearance-bootstrap.json')
    args = parser.parse_args(argv)
    if not (REPO_ROOT / 'data' / PEOPLE_FILENAME).is_file():
        parser.error('A current people.json registry is required; this script does not migrate profiles.')

    class ReadOnlyVoiceRegistry(VoiceRegistry):
        def _save_locked(self):
            # Registry construction may otherwise rewrite stale VOICE vectors.
            # This migration needs face anchors only and must never alter them.
            return None

    cfg = load_config(args.config)
    engine = FaceEngine(cfg.server.face)
    if not engine.available:
        parser.error('The local face detector is unavailable; no photos were imported.')
    registry = ReadOnlyVoiceRegistry(data_dir=REPO_ROOT / 'data', enabled=False, save_audio=False)
    result = bootstrap(args.root, engine, registry)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'items'}))
    print('Report: ' + str(args.report.resolve()))
    return 2 if result['pending_review'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
