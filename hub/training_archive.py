"""Permanent, dated person folders for inspecting/exporting local training data.

Media/event files are immutable; profile.json is a latest snapshot and every
event keeps its own profile snapshot. No retention or automatic deletion exists.
Face IDs group observations locally; they never grant permissions or enroll voices.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import time
import unicodedata
import uuid
import wave
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

_UNKNOWN = {'', 'unknown', 'anonymous', 'unidentified', 'guest', 'none', 'null'}
_SECRET_KEY = re.compile(r'^(?:.*(?:password|passwd|credential|secret).*|'
                         r'.*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|bot[_-]?token|private[_-]?key)|'
                         r'authorization|token|.*[_-]token|image_base64|audio_base64)$', re.I)
_TOKEN_TEXT = re.compile(r'\bsk-[A-Za-z0-9_-]{16,}|\b[0-9]{5,}:[A-Za-z0-9_-]{20,}|\bAIza[A-Za-z0-9_-]{30,}')
_SUFFIXES = {'.wav', '.jpg', '.jpeg', '.png', '.webp', '.txt', '.json'}
_FACE_ID = re.compile(r'face-[0-9a-f]{32}')


def _safe(value, depth=0):
    """Keep useful structured metadata, never credential fields or binary blobs."""
    if depth > 18:
        return '[nesting limit]'
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _TOKEN_TEXT.sub('[redacted credential]', value)
    if isinstance(value, bytes):
        return {'binary_bytes': len(value), 'note': 'Binary belongs in media assets.'}
    if isinstance(value, dict):
        return {str(key): '[redacted]' if _SECRET_KEY.fullmatch(str(key)) else _safe(item, depth + 1)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth + 1) for item in value]
    if hasattr(value, 'tolist'):
        return _safe(value.tolist(), depth + 1)
    return _safe(str(value), depth + 1)


def _name(value):
    name = ' '.join(unicodedata.normalize('NFKC', str(value or '')).split())
    return 'unknown' if name.casefold() in _UNKNOWN else name[:150]


def _slug(name):
    result = re.sub(r'[^\w .-]', '_', name, flags=re.UNICODE).strip(' .')
    result = re.sub(r'[ .]+', '_', result)[:72] or 'person'
    if result.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL',
                                       *{f'COM{i}' for i in range(1, 10)},
                                       *{f'LPT{i}' for i in range(1, 10)}}:
        result = '_' + result
    return result


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _wav(pcm, sample_rate):
    if not isinstance(pcm, bytes) or not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or not 8000 <= sample_rate <= 192000:
        raise ValueError('Audio needs signed-16-bit PCM bytes and a valid sample rate')
    if len(pcm) % 2:
        raise ValueError('Signed-16-bit PCM must contain complete samples')
    output = BytesIO()
    with wave.open(output, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


class TrainingArchive:
    def __init__(self, root, *, min_free_gb=5, timezone=None):
        self.root = Path(root).resolve()
        self.database = self.root / 'index.sqlite3'
        self.min_free_bytes = int(float(min_free_gb) * 1024 ** 3)
        if self.min_free_bytes < 0:
            raise ValueError('Free-space reserve cannot be negative')
        self.timezone = ZoneInfo(timezone) if isinstance(timezone, str) else timezone
        self._lock = threading.RLock()
        self.saved = self.failures = 0
        self._face_store = None

    def assign_face_ids(self, faces, *, frame_id, captured_at, source_id=''):
        """Assign the whole visible frame together so its people stay distinct."""
        from hub.face_identity import FaceIdentityStore
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.root).free < self.min_free_bytes + 16384:
                raise OSError('Face indexing stopped at the free-space reserve; no files were deleted')
            if self._face_store is None:
                self._face_store = FaceIdentityStore(self.root / 'face_identities')
            return self._face_store.assign_batch(faces, frame_id=frame_id,
                captured_at=captured_at, source_id=source_id)

    def _db(self):
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self._path('index.sqlite3'), timeout=30)
        db.row_factory = sqlite3.Row
        db.executescript('''
            CREATE TABLE IF NOT EXISTS identities (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS aliases (
                name_key TEXT PRIMARY KEY, person_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, person_id TEXT NOT NULL, kind TEXT NOT NULL,
                captured_at REAL NOT NULL, folder TEXT NOT NULL, record TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_person_time ON events(person_id,captured_at);
            CREATE TABLE IF NOT EXISTS passive_appearance_admissions (
                event_id TEXT PRIMARY KEY, face_id TEXT NOT NULL,
                continuity_key TEXT NOT NULL, saved_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS passive_face_time
                ON passive_appearance_admissions(face_id,saved_at);
            CREATE INDEX IF NOT EXISTS passive_track_time
                ON passive_appearance_admissions(continuity_key,saved_at);
            CREATE TABLE IF NOT EXISTS face_events (
                event_id TEXT PRIMARY KEY, face_id TEXT NOT NULL, record TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS face_events_identity ON face_events(face_id);
            CREATE TABLE IF NOT EXISTS face_manifest_offsets (
                face_id TEXT PRIMARY KEY, committed_bytes INTEGER NOT NULL
            );
        ''')
        return db

    def _path(self, relative):
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError('Archive path escapes its root')
        return candidate

    def _write(self, relative, data, *, replace=False):
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not replace:
            if path.read_bytes() != data:
                raise ValueError('An immutable archive event already contains different bytes')
            return
        pending = path.with_name(path.name + '.pending-' + uuid.uuid4().hex)
        with pending.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)

    def _moment(self, captured_at):
        if captured_at is None:
            captured_at = time.time()
        if isinstance(captured_at, str):
            captured_at = datetime.fromisoformat(captured_at.replace('Z', '+00:00'))
        if isinstance(captured_at, datetime):
            captured_at = captured_at.timestamp()
        captured_at = float(captured_at)
        if not math.isfinite(captured_at):
            raise ValueError('Invalid capture time')
        utc = datetime.fromtimestamp(captured_at, UTC)
        local = utc.astimezone(self.timezone)
        return captured_at, utc, local

    def _identity(self, db, name, profile_id, now):
        if name == 'unknown':
            return 'unknown'
        existing = db.execute('SELECT person_id FROM aliases WHERE name_key=?', (name.casefold(),)).fetchone()
        identity = str(profile_id).strip() if profile_id is not None else ''
        if not identity:
            identity = existing['person_id'] if existing else 'person-' + uuid.uuid5(uuid.NAMESPACE_URL, name.casefold()).hex
        if len(identity) > 200 or any(ord(c) < 32 for c in identity):
            raise ValueError('Invalid stable profile ID')
        identity = _safe(identity)
        db.execute('INSERT OR IGNORE INTO identities VALUES(?,?,?)', (identity, name, now))
        db.execute('UPDATE identities SET name=? WHERE id=?', (name, identity))
        db.execute('INSERT INTO aliases VALUES(?,?) ON CONFLICT(name_key) DO UPDATE SET person_id=excluded.person_id',
                   (name.casefold(), identity))
        return identity

    def record(self, kind, person=None, *, metadata=None, assets=None, profile=None,
               profile_id=None, captured_at=None, event_id=None, face_id=None):
        """Write one immutable event; supplied event IDs are idempotent per kind.

        ``assets`` maps simple filenames to bytes; this function never reads or
        copies caller-specified filesystem paths. ``profile`` is a snapshot, not
        a mutable registry. Exceptions leave original source media untouched.
        """
        try:
            if not isinstance(kind, str) or re.fullmatch(r'[a-z][a-z0-9_]{0,47}', kind) is None:
                raise ValueError('Invalid training event kind')
            name = _name(person)
            if face_id is not None and (not isinstance(face_id, str) or not _FACE_ID.fullmatch(face_id)):
                raise ValueError('Invalid face identity ID')
            captured, utc, local = self._moment(captured_at)
            if event_id is None:
                identifier = uuid.uuid4().hex
            else:
                if not isinstance(event_id, str) or not event_id or len(event_id) > 500:
                    raise ValueError('An event ID must be a nonempty short string')
                identifier = hashlib.sha256((kind + '\0' + event_id).encode()).hexdigest()[:32]
            assets = dict(assets or {})
            for filename, raw in assets.items():
                if (not isinstance(filename, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', filename) is None
                        or '..' in filename or Path(filename).suffix.lower() not in _SUFFIXES
                        or filename.lower() in {'event.json', 'profile.json', 'events.jsonl'}
                        or not isinstance(raw, bytes)):
                    raise ValueError('Media assets need safe filenames and byte contents')
            clean_metadata, clean_profile = _safe(metadata or {}), _safe(profile or {})
            self.root.mkdir(parents=True, exist_ok=True)
            incoming = sum(len(raw) for raw in assets.values()) + len(_json(clean_metadata)) + len(_json(clean_profile)) + 16384
            with self._lock:
                db = self._db()
                try:
                    db.execute('BEGIN IMMEDIATE')
                    existing = db.execute('SELECT record FROM events WHERE id=?', (identifier,)).fetchone()
                    if existing:
                        return json.loads(existing['record'])
                    admission_time = time.time()
                    continuity_key = str(clean_metadata.get('capture_continuity_key') or '')
                    passive = (kind == 'appearance' and (face_id or continuity_key)
                               and clean_metadata.get('capture_mode') == 'passive')
                    if passive:
                        # Persisted rolling window, shared across cameras and
                        # restarts. Active-request observations never consume it.
                        count = db.execute('''SELECT COUNT(*) FROM passive_appearance_admissions
                            WHERE saved_at>? AND ((face_id<>'' AND face_id=?) OR (continuity_key<>'' AND continuity_key=?))''',
                            (admission_time - 60., face_id or '', continuity_key)).fetchone()[0]
                        if count >= 50:
                            return {'skipped': True, 'reason': 'passive_face_rate_limit', 'face_id': face_id}
                    if passive:
                        db.execute('INSERT INTO passive_appearance_admissions VALUES(?,?,?,?)',
                                   (identifier, face_id or '', continuity_key, admission_time))
                    if shutil.disk_usage(self.root).free < self.min_free_bytes + incoming:
                        raise OSError('Training archive stopped at its free-space reserve; no files were deleted')
                    identity = self._identity(db, name, profile_id, captured)
                    if name == 'unknown' and face_id:
                        identity = face_id
                        db.execute('INSERT OR IGNORE INTO identities VALUES(?,?,?)', (identity, name, captured))
                    folder_name = ('unknown/' + face_id if face_id else 'unknown') if name == 'unknown' else _slug(name) + '--' + hashlib.sha256(identity.encode()).hexdigest()[:10]
                    folder = f'{local:%Y-%m-%d}/{folder_name}'
                    event_folder = f'{folder}/events/{local:%H%M%S-%f}-{identifier[:16]}'
                    event_file = f'{event_folder}/event.json'
                    # A completed file surviving an interrupted SQLite commit
                    # remains canonical. Retrying must not rewrite its history.
                    event_path = self._path(event_file)
                    recovered = event_path.exists()
                    if recovered:
                        record = json.loads(event_path.read_text(encoding='utf-8'))
                        if record.get('id') != identifier or record.get('kind') != kind:
                            raise ValueError('Archive event identity mismatch')
                    else:
                        files = {}
                        for filename, raw in assets.items():
                            relative = f'{event_folder}/{filename}'
                            self._write(relative, raw)
                            files[filename] = {'path': relative, 'bytes': len(raw),
                                               'sha256': hashlib.sha256(raw).hexdigest()}
                        record = {'schema': 1, 'id': identifier, 'kind': kind,
                                  'person': name, 'profile_id': identity,
                                  'captured_at': captured, 'timestamp': utc.isoformat(),
                                  'local_timestamp': local.isoformat(),
                                  'recorded_at': datetime.now(UTC).isoformat(),
                                  'folder': folder, 'event_path': event_file,
                                  'metadata': clean_metadata, 'profile_snapshot': clean_profile,
                                  'files': files}
                        if face_id:
                            record['face_id'] = face_id
                        self._write(event_file, _json(record))
                    # Serial SQLite transactions also serialize the human-readable
                    # JSONL stream across multiple archive objects/processes.
                    log_path = self._path(f'{folder}/events.jsonl')
                    already_logged = False
                    if recovered and log_path.exists():
                        with log_path.open('r', encoding='utf-8') as handle:
                            for line in handle:
                                try:
                                    if json.loads(line).get('id') == identifier:
                                        already_logged = True
                                        break
                                except (ValueError, AttributeError):
                                    continue
                    if not already_logged:
                        with log_path.open('a+b') as handle:
                            # Keep an interrupted JSONL tail rather than deleting
                            # it, and keep the next complete record parseable.
                            if handle.tell():
                                handle.seek(-1, os.SEEK_END)
                                if handle.read(1) != b'\n':
                                    handle.write(b'\n')
                            handle.write(_json(record) + b'\n')
                            handle.flush()
                            os.fsync(handle.fileno())
                    profile_snapshot = {'schema': 1, 'profile_id': identity, 'name': name,
                                        'updated_at': datetime.now(UTC).isoformat(),
                                        'last_event_id': identifier, 'profile': clean_profile,
                                        'identity_status': 'unidentified' if name == 'unknown' else 'caller_assigned'}
                    if face_id:
                        profile_snapshot['face_id'] = face_id
                    self._write(f'{folder}/profile.json', _json(profile_snapshot), replace=True)
                    db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',
                               (identifier, identity, kind, captured, folder, _json(record).decode()))
                    db.commit()
                    self.saved += 1
                    return record
                finally:
                    db.close()
        except Exception:
            with self._lock:
                self.failures += 1
            raise

    def conversation(self, person=None, *, pcm=None, sample_rate=16000, wav=None,
                     transcript='', reply='', actions=None, metadata=None, profile=None,
                     profile_id=None, captured_at=None, event_id=None):
        assets = {}
        if wav is not None and pcm is not None:
            raise ValueError('Provide WAV or PCM, not both')
        if wav is not None:
            assets['request.wav'] = wav
        elif pcm is not None:
            assets['request.wav'] = _wav(pcm, sample_rate)
        values = dict(metadata or {})
        values.update(transcript=transcript, reply=reply, actions=actions or [])
        return self.record('conversation', person, metadata=values, assets=assets,
                           profile=profile, profile_id=profile_id, captured_at=captured_at, event_id=event_id)

    @staticmethod
    def _images(jpeg, face, row):
        """Preserve source bytes and native-pixel crops, without quality gating."""
        if not isinstance(jpeg, bytes) or not jpeg:
            raise ValueError('A frame needs nonempty JPEG bytes')
        from PIL import Image
        with Image.open(BytesIO(jpeg)) as image:
            if image.format not in {'JPEG', 'PNG', 'WEBP'} or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Unsupported static frame')
            extension = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}[image.format]
            image.load()
            width, height = image.size
            assets = {f'original.{extension}': jpeg}
            crops = {}
            for label, selected in (('face', face), ('body', row)):
                if not isinstance(selected, dict) or (label == 'body' and selected.get('body_unambiguous') is False):
                    continue
                box = selected.get('box')
                try:
                    values = [float(v) for v in box]
                except (TypeError, ValueError):
                    continue
                if (len(values) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                        or values[2] <= values[0] or values[3] <= values[1]):
                    continue
                pixels = (max(0, math.floor(values[0] * width)), max(0, math.floor(values[1] * height)),
                          min(width, math.ceil(values[2] * width)), min(height, math.ceil(values[3] * height)))
                if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
                    continue
                output = BytesIO()
                image.crop(pixels).save(output, format='PNG')
                assets[label + '.png'] = output.getvalue()
                crops[label] = {'box_normalized_xyxy': values, 'box_pixels_xyxy': list(pixels)}
        return assets, {'width': width, 'height': height, 'crops': crops,
                        'face_observation': face or {}, 'body_observation': row or {}}

    def appearance(self, jpeg, person=None, *, face=None, row=None, metadata=None,
                   profile=None, profile_id=None, captured_at=None, event_id=None,
                   face_identity=None):
        existing = self._existing_media_event('appearance', event_id)
        if existing is not None:
            return existing
        assets, image_metadata = self._images(jpeg, face, row)
        values = dict(metadata or {})
        values.update(image_metadata)
        if face is not None and face_identity is None:
            captured_at = self._moment(captured_at)[0]
            frame_key = str(values.get('face_frame_key') or
                            str(values.get('frame_id') or uuid.uuid4().hex) + ':' + hashlib.sha256(jpeg).hexdigest())
            selected = dict(face)
            if _name(person) != 'unknown':
                selected['confirmed_name'] = _name(person)
            face_identity = self.assign_face_ids([selected], frame_id=frame_key,
                captured_at=captured_at, source_id=str(values.get('client_id') or ''))[0]
        if face_identity:
            values['face_identity'] = face_identity
        record = self.record('appearance', person, metadata=values, assets=assets,
            profile=profile, profile_id=profile_id, captured_at=captured_at, event_id=event_id,
            face_id=face_identity.get('face_id') if face_identity else None)
        if not record.get('skipped') and face_identity and face_identity.get('face_id'):
            self.index_face_event(record, face_identity)
        return record

    def index_face_event(self, record, assignment):
        """Link old or new immutable media to a persistent face's readable index."""
        face_id = assignment.get('face_id')
        if not isinstance(face_id, str) or not _FACE_ID.fullmatch(face_id):
            raise ValueError('Invalid face identity ID')
        identifier = record['id']
        relative = record['event_path']
        # This index references existing events, never arbitrary paths or copies.
        canonical = json.loads(self._path(relative).read_text(encoding='utf-8'))
        if canonical.get('id') != identifier:
            raise ValueError('Face index event does not match its original')
        link = {'event_id': identifier, 'face_id': face_id, 'event_path': relative,
                'timestamp': canonical.get('timestamp'), 'person': canonical.get('person'),
                'files': canonical.get('files', {}), 'assignment': _safe(assignment)}
        with self._lock:
            if shutil.disk_usage(self.root).free < self.min_free_bytes + 16384:
                raise OSError('Face indexing stopped at the free-space reserve; no files were deleted')
            db = self._db()
            try:
                db.execute('BEGIN IMMEDIATE')
                existing = db.execute('SELECT face_id FROM face_events WHERE event_id=?', (identifier,)).fetchone()
                if existing:
                    if existing['face_id'] != face_id:
                        raise ValueError('The event already belongs to another face identity')
                    return False
                relative_manifest = f'face_identities/profiles/{face_id}/events.jsonl'
                path = self._path(relative_manifest)
                path.parent.mkdir(parents=True, exist_ok=True)
                # Recover a completed JSONL append if a crash preceded DB commit.
                seen = False
                if path.exists():
                    previous = db.execute('SELECT committed_bytes FROM face_manifest_offsets WHERE face_id=?', (face_id,)).fetchone()
                    offset = previous['committed_bytes'] if previous else 0
                    with path.open('rb') as source:
                        source.seek(offset if 0 <= offset <= path.stat().st_size else 0)
                        for line in source:
                            try:
                                recovered = json.loads(line)
                            except (ValueError, UnicodeDecodeError):
                                continue
                            if not isinstance(recovered, dict) or recovered.get('face_id') != face_id:
                                continue
                            recovered_id = recovered.get('event_id')
                            source_event = db.execute('SELECT record FROM events WHERE id=?', (recovered_id,)).fetchone()
                            if source_event is None:
                                continue
                            source_record = json.loads(source_event['record'])
                            if source_record.get('event_path') != recovered.get('event_path'):
                                continue
                            indexed = db.execute('SELECT face_id FROM face_events WHERE event_id=?', (recovered_id,)).fetchone()
                            if indexed and indexed['face_id'] != face_id:
                                raise ValueError('An interrupted face index conflicts with another identity')
                            # Recover every complete orphan before moving the
                            # committed offset, including earlier events whose
                            # callers have not retried yet.
                            db.execute('INSERT OR IGNORE INTO face_events VALUES(?,?,?)',
                                (recovered_id, face_id, _json(recovered).decode()))
                            seen = seen or recovered_id == identifier
                if not seen:
                    with path.open('a+b') as handle:
                        if handle.tell():
                            handle.seek(-1, os.SEEK_END)
                            if handle.read(1) != b'\n':
                                handle.write(b'\n')
                        handle.write(_json(link) + b'\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                db.execute('INSERT OR IGNORE INTO face_events VALUES(?,?,?)', (identifier, face_id, _json(link).decode()))
                db.execute('INSERT OR REPLACE INTO face_manifest_offsets VALUES(?,?)', (face_id, path.stat().st_size))
                db.commit()
                return True
            finally:
                db.close()

    def _existing_media_event(self, kind, event_id):
        """Idempotent retries reuse the original face assignment and its files."""
        if event_id is None:
            return None
        if not isinstance(event_id, str) or not event_id or len(event_id) > 500:
            raise ValueError('An event ID must be a nonempty short string')
        identifier = hashlib.sha256((kind + '\0' + event_id).encode()).hexdigest()[:32]
        with self._lock:
            db = self._db()
            try:
                row = db.execute('SELECT record FROM events WHERE id=?', (identifier,)).fetchone()
            finally:
                db.close()
            if row is None:
                return None
            record = json.loads(row['record'])
            assignment = record.get('metadata', {}).get('face_identity')
            if assignment and assignment.get('face_id'):
                self.index_face_event(record, assignment)
            return record

    def enrollment(self, person, kind, *, jpeg=None, face=None, row=None, pcm=None,
                   sample_rate=16000, wav=None, metadata=None, profile=None,
                   profile_id=None, captured_at=None, event_id=None):
        if kind not in {'voice', 'face'}:
            raise ValueError('Enrollment kind must be voice or face')
        existing = self._existing_media_event('enrollment_' + kind, event_id)
        if existing is not None:
            return existing
        if wav is not None and pcm is not None:
            raise ValueError('Provide WAV or PCM, not both')
        assets, values = {}, dict(metadata or {})
        if jpeg is not None:
            image_assets, image_metadata = self._images(jpeg, face, row)
            assets.update(image_assets)
            values.update(image_metadata)
        if wav is not None:
            assets['enrollment.wav'] = wav
        elif pcm is not None:
            assets['enrollment.wav'] = _wav(pcm, sample_rate)
        if not assets:
            raise ValueError('Enrollment requires original audio or image data')
        assignment = None
        if kind == 'face' and face is not None:
            captured_at = self._moment(captured_at)[0]
            selected = {**face, 'confirmed_name': _name(person) if _name(person) != 'unknown' else None}
            frame_key = 'enrollment:' + str(event_id or uuid.uuid4().hex) + ':' + hashlib.sha256(jpeg).hexdigest()
            assignment = self.assign_face_ids([selected], frame_id=frame_key,
                captured_at=captured_at, source_id=str(values.get('client_id') or ''))[0]
            values['face_identity'] = assignment
        record = self.record('enrollment_' + kind, person, metadata=values, assets=assets,
            profile=profile, profile_id=profile_id, captured_at=captured_at, event_id=event_id,
            face_id=assignment.get('face_id') if assignment else None)
        if assignment and assignment.get('face_id'):
            self.index_face_event(record, assignment)
        return record

    def rename(self, old, new):
        """Preserve old folders; future records share the stable ID under new name."""
        old, new = _name(old), _name(new)
        if 'unknown' in {old, new}:
            raise ValueError('Unknown observations cannot be collectively assigned to a person')
        with self._lock:
            db = self._db()
            try:
                db.execute('BEGIN IMMEDIATE')
                previous = db.execute('SELECT person_id FROM aliases WHERE name_key=?', (old.casefold(),)).fetchone()
                if previous is None:
                    return False
                existing = db.execute('SELECT person_id FROM aliases WHERE name_key=?', (new.casefold(),)).fetchone()
                if existing is not None and existing['person_id'] != previous['person_id']:
                    raise ValueError('The new name belongs to a different archive identity')
                db.execute('INSERT OR REPLACE INTO aliases VALUES(?,?)', (new.casefold(), previous['person_id']))
                db.execute('UPDATE identities SET name=? WHERE id=?', (new, previous['person_id']))
                db.commit()
                return True
            finally:
                db.close()

    def close(self):
        """Connections are scoped to individual writes; nothing remains open."""
