"""Copy existing Rowan data into the permanent training archive, without inference.

Only the active registry, voice samples, appearance gallery, dated dialog logs,
and indexed request audio are considered. Originals are never changed. Repeated
runs deduplicate unchanged sources; changed sources remain separate snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hub.training_archive import TrainingArchive

_IMAGES = {'.jpg', '.jpeg', '.png', '.webp'}
_UNKNOWN = {'', 'unknown', 'anonymous', 'unidentified', 'guest', 'none', 'null'}


def _digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _object(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError('Expected a metadata object')
    return value


def _moment(value, fallback):
    if value is None or value == '':
        return fallback
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Invalid source timestamp')
    return result


class _Importer:
    def __init__(self, root, archive, dry_run):
        self.root = Path(root).resolve(strict=True)
        self.data = self.root / 'data'
        self.archive = archive
        self.dry_run = dry_run
        self.profiles = {}
        self.stats = dict(imported=0, already_imported=0, planned=0, errors=0,
                          missing_media=0, skipped=0, sources={}, paid_api_requests=0)
        self.known = set()
        database = archive.root / 'index.sqlite3'
        if database.is_file():
            if not database.resolve().is_relative_to(archive.root):
                raise ValueError('Archive index escaped its output folder')
            db = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
            try:
                self.known = {r[0] for r in db.execute('SELECT id FROM events')}
            finally:
                db.close()

    def _path(self, path, boundary, *, suffixes=None):
        boundary = Path(boundary).resolve()
        # A symlinked active source folder cannot redirect an import into a
        # sibling backup/test tree, even if that tree remains in the workspace.
        nominal = Path(path)
        candidate = nominal.resolve(strict=True)
        if not candidate.is_relative_to(boundary) or not candidate.is_relative_to(self.root):
            raise ValueError('Source escaped its allowed folder')
        if candidate != nominal.absolute():
            # Also covers Windows directory junctions on Python versions that
            # do not expose Path.is_junction yet, and noncanonical ../ paths.
            raise ValueError('Source links and noncanonical paths are not imported')
        relative = nominal.absolute().relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
                raise ValueError('Source links are not imported')
        if not candidate.is_file() or (suffixes and candidate.suffix.lower() not in suffixes):
            raise ValueError('Unsupported source file')
        return candidate

    def _database(self, path, boundary):
        path = self._path(path, boundary)
        connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _read(self, path, boundary, suffixes=None):
        path = self._path(path, boundary, suffixes=suffixes)
        return path, path.read_bytes()

    def _relative(self, path):
        return Path(path).relative_to(self.root).as_posix()

    def _person(self, name):
        value = str(name or '').strip()
        if value.casefold() in _UNKNOWN:
            return 'unknown'
        return next((key for key in self.profiles if key.casefold() == value.casefold()), value)

    def _emit(self, source, method, kind, person, *, source_key, source_hash,
              captured_at, metadata=None, **kwargs):
        identifier = 'bootstrap:v1:' + _digest([source_key, source_hash])
        event_id = hashlib.sha256((kind + '\0' + identifier).encode()).hexdigest()[:32]
        self.stats['sources'][source] = self.stats['sources'].get(source, 0) + 1
        if event_id in self.known:
            self.stats['already_imported'] += 1
            return
        name = self._person(person)
        profile = self.profiles.get(name, {}) if name != 'unknown' else {}
        metadata = dict(metadata or {})
        metadata['bootstrap'] = dict(source=source_key, sha256=source_hash,
                                     profile_snapshot='current registry at import, not historical proof')
        if self.dry_run:
            self.stats['planned'] += 1
        else:
            params = dict(metadata=metadata, profile=profile, captured_at=captured_at,
                          event_id=identifier, **kwargs)
            stable_id = profile.get('profile_id') or profile.get('id')
            if isinstance(stable_id, str) and stable_id.strip():
                params['profile_id'] = stable_id
            previous = self.archive.saved
            if method == 'record':
                self.archive.record(kind, name, **params)
            elif method == 'enrollment':
                self.archive.enrollment(name, 'voice', **params)
            else:
                self.archive.conversation(name, **params)
            self.stats['imported' if self.archive.saved > previous else 'already_imported'] += 1
        self.known.add(event_id)

    def profiles_import(self):
        path = self.data / 'people.json'
        if not path.exists():
            return
        try:
            path, raw = self._read(path, self.data, {'.json'})
            registry = _object(json.loads(raw))
            self.profiles = {str(name): profile for name, profile in _object(
                registry.get('people', {})).items() if isinstance(profile, dict)}
            for name, profile in self.profiles.items():
                self._emit('profiles', 'record', 'profile_snapshot', name,
                    source_key=self._relative(path) + '#' + name,
                    source_hash=_digest({'profile': profile, 'voice_model': registry.get('voice_model')}),
                    captured_at=path.stat().st_mtime,
                    metadata={'voice_model': registry.get('voice_model'),
                              'original_media_available': False,
                              'note': 'Embedding vectors are retained as profile data; no source photo or audio was reconstructed.'})
        except (OSError, ValueError, TypeError, sqlite3.Error):
            self.stats['errors'] += 1

    def voices_import(self):
        root = self.data / 'voices'
        if not root.is_dir():
            return
        names = {}
        for name in self.profiles:
            clean = ''.join(c if c.isalnum() or c in '-_ ' else '_' for c in name).strip()
            names.setdefault(clean.casefold(), []).append(name)
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            candidates = names.get(directory.name.casefold(), [])
            person = candidates[0] if len(candidates) == 1 else (
                'unknown' if len(candidates) > 1 else directory.name)
            for path in sorted(directory.glob('*.wav')):
                try:
                    path, wav = self._read(path, root, {'.wav'})
                    try:
                        captured = datetime.strptime(path.stem, '%Y%m%d-%H%M%S-%f').timestamp()
                    except ValueError:
                        captured = path.stat().st_mtime
                    self._emit('voices', 'enrollment', 'enrollment_voice', person,
                        source_key=self._relative(path), source_hash=_digest(wav), captured_at=captured,
                        wav=wav, metadata={'source_folder_label': directory.name,
                                           'identity_ambiguous': len(candidates) > 1})
                except (OSError, ValueError, TypeError, sqlite3.Error):
                    self.stats['errors'] += 1

    def _image_assets(self, root, row, quality):
        assets, source_files = {}, {}
        for label, field in (('face', 'face_path'), ('body', 'body_path'),
                             ('original', 'original_path'), ('frame', 'frame_path')):
            relative = row.get(field) or quality.get(field)
            if not relative:
                continue
            try:
                path, raw = self._read(root / relative, root, _IMAGES)
            except FileNotFoundError:
                self.stats['missing_media'] += 1
                continue
            filename = label + path.suffix.lower()
            assets[filename] = raw
            source_files[label] = {'path': self._relative(path), 'sha256': _digest(raw)}
        return assets, source_files

    def appearance_import(self):
        root = self.data / 'appearance'
        database = root / 'gallery.sqlite3'
        if database.exists():
            try:
                with closing(self._database(database, root)) as db:
                    rows = [dict(row) for row in db.execute(
                        'SELECT s.*, p.name AS current_name FROM samples s LEFT JOIN people p ON p.id=s.person_id ORDER BY s.captured_at,s.id')]
                for row in rows:
                    try:
                        quality = _object(row.get('quality') or '{}')
                        assets, source_files = self._image_assets(root, row, quality)
                        self._emit('appearance', 'record', 'appearance',
                            row.get('current_name') or row.get('captured_name'),
                            source_key=self._relative(database) + '#samples:' + str(row['id']),
                            source_hash=_digest({'row': row, 'files': source_files}),
                            captured_at=row['captured_at'], assets=assets,
                            metadata={'gallery_sample': row, 'quality': quality, 'source_files': source_files,
                                      'original_scene_available': any(k.startswith(('original.', 'frame.')) for k in assets)})
                    except (OSError, ValueError, TypeError, sqlite3.Error):
                        self.stats['errors'] += 1
            except (OSError, ValueError, sqlite3.Error):
                self.stats['errors'] += 1
        if not root.is_dir():
            return
        # This exact old layout stores already cropped pictures, not full scenes.
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not re.fullmatch(r'[0-9a-f]{24}', directory.name):
                continue
            for path in sorted(directory.glob('*.jpg')):
                if not re.fullmatch(r'[0-9]{10,20}\.jpg', path.name):
                    continue
                try:
                    path, raw = self._read(path, root, _IMAGES)
                    sidecar, payload = self._read(path.with_suffix('.json'), root, {'.json'})
                    metadata = _object(json.loads(payload))
                    self._emit('legacy_appearance', 'record', 'appearance_legacy', metadata.get('name'),
                        source_key=self._relative(path), source_hash=_digest([_digest(raw), _digest(payload)]),
                        captured_at=_moment(metadata.get('ts'), path.stat().st_mtime),
                        assets={'legacy_crop.jpg': raw}, metadata={'legacy_metadata': metadata,
                            'source_metadata': self._relative(sidecar), 'identity_status': 'historical unverified label',
                            'original_scene_available': False})
                except (OSError, ValueError, TypeError, sqlite3.Error):
                    self.stats['errors'] += 1

    def conversations_import(self):
        audio_root = self.data / 'request_audio'
        database = audio_root / 'index.sqlite3'
        recordings = []
        if database.exists():
            try:
                with closing(self._database(database, audio_root)) as db:
                    recordings = [dict(row) for row in db.execute('SELECT * FROM recordings ORDER BY captured_at,id')]
            except (OSError, ValueError, sqlite3.Error):
                self.stats['errors'] += 1
        by_id = {str(row['id']): row for row in recordings}
        by_path = {str(row['path']).replace('\\', '/'): row for row in recordings}
        matched = set()
        dialogs = self.data / 'dialogs'
        for file in sorted(dialogs.glob('*.jsonl')):
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}\.jsonl', file.name):
                continue
            try:
                path = self._path(file, dialogs, suffixes={'.jsonl'})
                with path.open('r', encoding='utf-8-sig') as handle:
                    for index, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            dialog = _object(json.loads(line))
                            link = dialog.get('audio_recording') or {}
                            link = link if isinstance(link, dict) else {}
                            audio = by_id.get(str(link.get('id', ''))) or by_path.get(str(link.get('path', '')).replace('\\', '/'))
                            wav, sources, audio_metadata = None, {}, {}
                            relative = audio['path'] if audio else link.get('path')
                            if relative:
                                try:
                                    audio_path, wav = self._read(audio_root / relative, audio_root, {'.wav'})
                                    sources = {'path': self._relative(audio_path), 'sha256': _digest(wav)}
                                except FileNotFoundError:
                                    self.stats['missing_media'] += 1
                            if audio:
                                audio_metadata = _object(audio.get('metadata') or '{}')
                                matched.add(str(audio['id']))
                            captured = _moment(dialog.get('ts'), audio['captured_at'] if audio else path.stat().st_mtime)
                            self._emit('dialogs', 'conversation', 'conversation', dialog.get('speaker'),
                                source_key=self._relative(path) + '#line:' + str(index),
                                source_hash=_digest({'dialog': dialog, 'audio': sources, 'audio_metadata': audio_metadata}),
                                captured_at=captured, wav=wav, transcript=dialog.get('transcript', ''),
                                reply=dialog.get('reply', ''), actions=dialog.get('actions', []),
                                metadata={'dialog': dialog, 'audio_metadata': audio_metadata, 'source_audio': sources})
                        except (OSError, ValueError, TypeError, sqlite3.Error):
                            self.stats['errors'] += 1
            except (OSError, ValueError, UnicodeError):
                self.stats['errors'] += 1
        for row in recordings:
            if str(row['id']) in matched:
                continue
            try:
                path, wav = self._read(audio_root / row['path'], audio_root, {'.wav'})
                metadata = _object(row.get('metadata') or '{}')
                self._emit('unmatched_audio', 'conversation', 'conversation', metadata.get('speaker'),
                    source_key=self._relative(database) + '#recordings:' + str(row['id']),
                    source_hash=_digest({'row': row, 'wav': _digest(wav)}),
                    captured_at=row['captured_at'], wav=wav,
                    transcript=metadata.get('transcript', ''), reply=metadata.get('reply', ''),
                    actions=metadata.get('actions', []),
                    metadata={'audio_metadata': metadata, 'audio_recording': row,
                              'source_audio': self._relative(path), 'dialog_available': False})
            except FileNotFoundError:
                self.stats['missing_media'] += 1
            except (OSError, ValueError, TypeError, sqlite3.Error):
                self.stats['errors'] += 1


def bootstrap(root, archive=None, *, dry_run=False, min_free_gb=5, timezone=None):
    """Return counts only. ``root`` is the project, not its data directory."""
    root = Path(root).resolve(strict=True)
    archive = archive if isinstance(archive, TrainingArchive) else TrainingArchive(
        Path(archive) if archive is not None else root / 'data' / 'training_archive',
        min_free_gb=min_free_gb, timezone=timezone)
    # An explicit output must never overwrite any existing active source tree.
    protected = [root / 'data' / key for key in ('voices', 'appearance', 'dialogs', 'request_audio')]
    if any(archive.root == path.resolve() or archive.root.is_relative_to(path.resolve())
           or path.resolve().is_relative_to(archive.root) for path in protected):
        raise ValueError('Archive output overlaps an active source directory')
    importer = _Importer(root, archive, dry_run)
    importer.profiles_import()
    importer.voices_import()
    importer.appearance_import()
    importer.conversations_import()
    return importer.stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--min-free-gb', type=float, default=5)
    parser.add_argument('--timezone', help='Optional IANA timezone; otherwise use the local machine timezone')
    args = parser.parse_args(argv)
    try:
        result = bootstrap(args.root, args.archive, dry_run=args.dry_run,
                           min_free_gb=args.min_free_gb, timezone=args.timezone)
    except (OSError, ValueError, sqlite3.Error):
        print(json.dumps({'errors': 1, 'imported': 0, 'paid_api_requests': 0}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
