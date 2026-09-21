"""Conservative, persistent face clusters for the raw training archive.

These IDs describe dataset observations. They grant no permissions and never
enroll a person in the assistant's face or voice recognition profiles. SQLite is
authoritative; the adjacent JSON profiles are atomically exported snapshots.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import unicodedata
import uuid
from pathlib import Path

_MAX_TEMPLATES = 5
_MAX_DIMENSION = 8192
_UNKNOWN_NAMES = {'', 'unknown', 'anonymous', 'unidentified', 'guest', 'none', 'null'}


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _number(value):
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _vector(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= _MAX_DIMENSION:
        return None
    numbers = [_number(item) for item in value]
    if any(item is None for item in numbers):
        return None
    scale = max(abs(item) for item in numbers)
    if scale == 0:
        return None
    scaled = [item / scale for item in numbers]
    norm = math.hypot(*scaled)
    return [item / norm for item in scaled]


def _box(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = [_number(item) for item in value]
    if any(item is None for item in numbers):
        return None
    x1, y1, x2, y2 = numbers
    return numbers if x2 > x1 and y2 > y1 else None


def _name(value):
    if not isinstance(value, str):
        return None
    name = ' '.join(unicodedata.normalize('NFKC', value).split())[:150]
    if any(ord(character) < 32 for character in name):
        return None
    return None if name.casefold() in _UNKNOWN_NAMES else name


def _cosine(left, right):
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right))))


class FaceIdentityStore:
    """Assign anonymous IDs with a fixed anchor and at most five templates.

    ``root`` should be a dedicated directory, for example
    ``training_archive/face_identities``. ``assign_batch`` must receive every
    detected face in a frame, in the same order on retries. A frame key must
    identify the source image, not merely a counter that resets on reconnect.
    """

    def __init__(self, root, *, model='buffalo_l', match_threshold=.62,
                 match_margin=.08, min_detection_score=.80):
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ValueError('A nonempty face embedding model is required')
        threshold, margin, quality = map(_number, (match_threshold, match_margin, min_detection_score))
        if threshold is None or not 0 < threshold <= 1:
            raise ValueError('Face identity threshold must be in (0, 1]')
        if margin is None or not 0 <= margin <= 1:
            raise ValueError('Face identity margin must be in [0, 1]')
        if quality is None or not 0 <= quality <= 1:
            raise ValueError('Face detector minimum score must be in [0, 1]')
        self.root = Path(root).resolve()
        self.database = self.root / 'identity.sqlite3'
        self.model = model.strip()
        self.match_threshold = threshold
        self.match_margin = margin
        self.min_detection_score = quality
        self._lock = threading.RLock()

    def _db(self):
        self.root.mkdir(parents=True, exist_ok=True)
        database = self.database.resolve()
        if not database.is_relative_to(self.root):
            raise ValueError('Face identity database escapes its root')
        db = sqlite3.connect(database, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.executescript('''
            CREATE TABLE IF NOT EXISTS face_identities (
                face_id TEXT PRIMARY KEY, model TEXT NOT NULL, dimension INTEGER,
                profile TEXT NOT NULL, needs_export INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS face_identity_model_dimension
                ON face_identities(model, dimension);
            CREATE TABLE IF NOT EXISTS face_observations (
                model TEXT NOT NULL, source_id TEXT NOT NULL, frame_id TEXT NOT NULL,
                face_index INTEGER NOT NULL, fingerprint TEXT NOT NULL, result TEXT NOT NULL,
                PRIMARY KEY(model, source_id, frame_id, face_index)
            );
        ''')
        return db

    def _export_pending(self, db):
        # A separate serialized transaction reads the latest committed profiles.
        # Failed exports leave the dirty flag set and are repaired on the next
        # call, including an idempotent replay after a process interruption.
        db.execute('BEGIN IMMEDIATE')
        try:
            for row in db.execute('SELECT face_id,profile FROM face_identities WHERE needs_export=1'):
                path = (self.root / 'profiles' / row['face_id'] / 'profile.json').resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError('Face profile path escapes its root')
                path.parent.mkdir(parents=True, exist_ok=True)
                pending = path.with_name(path.name + '.pending-' + uuid.uuid4().hex)
                try:
                    with pending.open('xb') as handle:
                        handle.write(row['profile'].encode('utf-8'))
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(pending, path)
                finally:
                    if pending.exists():
                        pending.unlink()
                db.execute('UPDATE face_identities SET needs_export=0 WHERE face_id=?', (row['face_id'],))
            db.commit()
        except BaseException:
            db.rollback()
            raise

    def _candidates(self, vector, name, profiles):
        candidates, conflicts = [], []
        for profile in profiles:
            templates = profile['templates']
            if not templates or profile['dimension'] != len(vector):
                continue
            # No chain of near-neighbor updates may move a cluster away from its
            # first reliable observation, even if a newer template scores well.
            anchor_similarity = _cosine(vector, templates[0])
            similarity = max(_cosine(vector, template) for template in templates)
            candidate = {'face_id': profile['face_id'], 'similarity': similarity,
                         'anchor_similarity': anchor_similarity}
            existing_name = profile['confirmed_name']
            if name and existing_name and name.casefold() != existing_name.casefold():
                conflicts.append(candidate)
            else:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (-item['similarity'], item['face_id']))
        conflicts.sort(key=lambda item: (-item['similarity'], item['face_id']))
        return candidates, conflicts

    def _new_profile(self, prepared, captured_at, source_id, candidates, *, learn):
        name, vector = prepared['name'] if learn else None, prepared['vector']
        return {
            'schema': 1, 'face_id': 'face-' + uuid.uuid4().hex,
            'identity_status': 'anonymous', 'model': self.model,
            'dimension': len(vector) if vector is not None else None,
            'first_seen': captured_at, 'last_seen': captured_at,
            'observation_count': 1,
            'matching_status': 'established' if learn else 'provisional',
            'confirmed_name': name, 'aliases': [name] if name else [],
            'source_ids': [source_id] if source_id else [],
            'templates': [vector] if learn else [],
            'template_limit': _MAX_TEMPLATES,
            'candidates': candidates[:3],
        }

    def _update_profile(self, profile, prepared, captured_at, source_id):
        profile['first_seen'] = min(profile['first_seen'], captured_at)
        profile['last_seen'] = max(profile['last_seen'], captured_at)
        profile['observation_count'] += 1
        if prepared['name']:
            profile['confirmed_name'] = profile['confirmed_name'] or prepared['name']
            if prepared['name'] not in profile['aliases'] and len(profile['aliases']) < 32:
                profile['aliases'].append(prepared['name'])
        if source_id and source_id not in profile['source_ids'] and len(profile['source_ids']) < 32:
            profile['source_ids'].append(source_id)
        vector, templates = prepared['vector'], profile['templates']
        # Retain a few distinct angles. The first anchor is immutable and each
        # accepted update independently clears it, avoiding transitive drift.
        update_threshold = min(1.0, self.match_threshold + .08)
        if (_cosine(vector, templates[0]) >= update_threshold
                and max(_cosine(vector, template) for template in templates) < .98):
            if len(templates) < _MAX_TEMPLATES:
                templates.append(vector)
            else:
                # Keep the anchor and the most recent four validated angles.
                templates[1:] = templates[2:] + [vector]
        profile['candidates'] = []

    @staticmethod
    def _result(face_id, assignment, *, similarity=None, reliable=False, name=None, candidates=None):
        return {'face_id': face_id, 'similarity': similarity, 'assignment': assignment,
                'confirmed_name': name, 'reliable': reliable, 'candidates': (candidates or [])[:3]}

    def assign_batch(self, faces, *, frame_id, captured_at, source_id=''):
        """Return one assignment per input face; retries never add observations.

        A valid positive-area ``box`` is required for a detected face. Missing or
        invalid embeddings and weak detector scores still receive individual
        provisional IDs, but never enter matching templates. Invalid boxes
        return ``face_id=None`` and ``assignment='invalid_box'``. Similarity alone
        never supplies a person's name; ``confirmed_name`` only comes from a
        name explicitly supplied on a reliable observation in this registry.

        The model, source, frame and face index form an idempotency key. A replay
        with different face content raises ValueError instead of silently
        attaching the wrong observation after an ordering/counter change.
        """
        if not isinstance(faces, (list, tuple)):
            raise ValueError('Faces must be an ordered list')
        if not isinstance(frame_id, str) or not frame_id or len(frame_id) > 1024:
            raise ValueError('A nonempty frame ID of at most 1024 characters is required')
        if not isinstance(source_id, str) or len(source_id) > 512:
            raise ValueError('Source ID must be a string of at most 512 characters')
        captured_at = _number(captured_at)
        if captured_at is None:
            raise ValueError('Capture time must be finite')
        if not faces:
            return []

        prepared = []
        for face in faces:
            face = face if isinstance(face, dict) else {}
            item = {'vector': _vector(face.get('embedding')), 'box': _box(face.get('box')),
                    'score': _number(face.get('score')), 'name': _name(face.get('confirmed_name'))}
            item['fingerprint'] = hashlib.sha256(_json(item).encode('utf-8')).hexdigest()
            prepared.append(item)

        with self._lock:
            db = self._db()
            try:
                db.execute('BEGIN IMMEDIATE')
                profiles = {row['face_id']: json.loads(row['profile']) for row in db.execute(
                    'SELECT face_id,profile FROM face_identities WHERE model=?', (self.model,))}
                # The snapshot excludes identities/templates learned within this
                # frame, so two simultaneous faces can never merge with each other.
                snapshot = json.loads(_json(list(profiles.values())))
                results, used_ids, pending = [None] * len(faces), set(), []
                for index, item in enumerate(prepared):
                    existing = db.execute('''SELECT fingerprint,result FROM face_observations
                        WHERE model=? AND source_id=? AND frame_id=? AND face_index=?''',
                        (self.model, source_id, frame_id, index)).fetchone()
                    if existing:
                        if existing['fingerprint'] != item['fingerprint']:
                            raise ValueError('Frame face ordering/content changed for an existing observation')
                        results[index] = json.loads(existing['result'])
                        if results[index]['face_id']:
                            used_ids.add(results[index]['face_id'])
                        continue
                    candidates, conflicts = (self._candidates(item['vector'], item['name'], snapshot)
                                             if item['vector'] is not None else ([], []))
                    pending.append((index, item, candidates, conflicts))

                # Allocate contested existing IDs to the strongest observation.
                # A second face keeps its own provisional ID, never a fallback
                # match to a weaker runner-up from the same frame.
                pending.sort(key=lambda entry: (
                    -(entry[2][0]['similarity'] if entry[2] else -1.0), entry[0]))
                for index, item, candidates, conflicts in pending:
                    top = candidates[0] if candidates else None
                    similarity = top['similarity'] if top else None
                    score = item['score']
                    good_quality = score is not None and self.min_detection_score <= score <= 1
                    assignment, matched, learn = 'new', None, True
                    if item['box'] is None:
                        results[index] = self._result(None, 'invalid_box')
                    else:
                        if item['vector'] is None:
                            assignment, learn = 'invalid_embedding_new', False
                        elif not good_quality:
                            assignment, learn = 'low_quality_new', False
                        elif top and top['similarity'] >= self.match_threshold:
                            runner_up = candidates[1]['similarity'] if len(candidates) > 1 else -1.0
                            if top['anchor_similarity'] < self.match_threshold:
                                assignment, learn = 'anchor_conflict_new', False
                            elif (top['similarity'] <= runner_up
                                  or top['similarity'] - runner_up < self.match_margin):
                                assignment, learn = 'ambiguous_new', False
                            elif top['face_id'] in used_ids:
                                assignment, learn = 'frame_conflict_new', False
                            else:
                                assignment, matched = 'matched', profiles[top['face_id']]
                        elif conflicts and conflicts[0]['similarity'] >= self.match_threshold:
                            assignment = 'name_conflict_new'

                        if matched is not None:
                            self._update_profile(matched, item, captured_at, source_id)
                            profile = matched
                        else:
                            profile = self._new_profile(item, captured_at, source_id, candidates, learn=learn)
                        face_id = profile['face_id']
                        used_ids.add(face_id)
                        profiles[face_id] = profile
                        db.execute('''INSERT INTO face_identities(face_id,model,dimension,profile,needs_export)
                            VALUES(?,?,?,?,1) ON CONFLICT(face_id) DO UPDATE
                            SET profile=excluded.profile,needs_export=1''',
                            (face_id, self.model, profile['dimension'], _json(profile)))
                        results[index] = self._result(face_id, assignment, similarity=similarity,
                            reliable=learn, name=profile['confirmed_name'], candidates=candidates)
                    db.execute('INSERT INTO face_observations VALUES(?,?,?,?,?,?)',
                        (self.model, source_id, frame_id, index, item['fingerprint'], _json(results[index])))
                db.commit()
                self._export_pending(db)
                return results
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()
