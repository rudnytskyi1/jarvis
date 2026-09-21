"""Short-lived visual tracks and conservative face identity continuity.

Track names are scene hints, never voice identities or permission credentials.
"""
from __future__ import annotations

import math
import time

TRACK_TTL_S = 3.0
IDENTITY_HOLD_S = 6.0
UNKNOWN_STABILITY_S = 0.6


def _track_jumped(previous, current):
    """A reused ID at a disjoint location needs a fresh face confirmation."""
    a, b = previous['box'], current['box']
    overlap = min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
    dx = abs((a[0] + a[2] - b[0] - b[2]) / 2)
    dy = abs((a[1] + a[3] - b[1] - b[3]) / 2)
    return not overlap and (dx > max(.18, .75 * max(a[2] - a[0], b[2] - b[0]))
                            or dy > max(.2, .75 * max(a[3] - a[1], b[3] - b[1])))


def _clear_identity(row, *, replacement=False):
    row.update(name=None, confirmed=0, face_score=0, face_source='unknown',
               unknown_since=None, unknown_observations=0)
    row.pop('face_anchor', None)
    if replacement:
        # Positive contradictory face evidence describes a different occupant.
        # Mere expiry or an ambiguous view must not start another greeting.
        row['greeted'] = False


def _accepts_face(row, observed_at):
    return (row is not None and observed_at - row['seen'] < TRACK_TTL_S
            and row['born'] <= observed_at and row.get('face_seen', 0) <= observed_at)


def valid_tracks(rows):
    clean = []
    for row in (rows or [])[:24]:
        try:
            box = [float(v) for v in row['box']]
            if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            clean.append(dict(id=str(row['id'])[:100], box=box))
        except (KeyError, TypeError, ValueError):
            continue
    return clean


def enclosing_track(face, tracks):
    x1, y1, x2, y2 = face['box']
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    candidates = [t for t in tracks if t['box'][0] <= cx <= t['box'][2]
                  and t['box'][1] <= cy <= t['box'][3]
                  and cy < t['box'][1] + .65 * (t['box'][3] - t['box'][1])]
    # Overlapping bodies create ambiguous face association. Do not guess.
    return candidates[0]['id'] if len(candidates) == 1 else None


class RoomState:
    def __init__(self):
        self.tracks = {}
        self.events = []
        self.updated = 0.0
        self._unknown_greeted_at = None
        self._unknown_greeted_count = 0

    def update(self, tracks, now=None):
        """Apply geometry at its SERVER receipt time, never rewind newer state.

        Image inference can finish after a newer camera-state update. Callers
        may pass the image's receipt clock to keep that old image from moving
        tracks backwards or resurrecting an ID already removed in newer data.
        Returns whether the geometry was applied.
        """
        now = time.monotonic() if now is None else now
        if now < self.updated:
            return False
        self.updated = now
        for track in valid_tracks(tracks):
            key = track['id']
            old = self.tracks.get(key)
            if old is None or now - old['seen'] >= TRACK_TTL_S or _track_jumped(old, track):
                old = self.tracks[key] = dict(born=now, name=None, face_seen=0, greeted=False)
                self.events.append(dict(event='entered', track=key, at=now))
            old.update(track, seen=now)
            if old.get('name') and now - old.get('confirmed', 0) > IDENTITY_HOLD_S:
                _clear_identity(old)
        for key, row in list(self.tracks.items()):
            if now - row['seen'] >= TRACK_TTL_S:
                self.events.append(dict(event='left', track=key, name=row['name'], at=now))
                del self.tracks[key]
        self.events = self.events[-100:]
        return True

    def bind(self, faces, tracks, matcher, profiles, now=None):
        """Return fresh, unambiguous face confirmations suitable for learning."""
        resolved = self.resolve_faces(faces, tracks, matcher, profiles, now=now)
        return [(self.tracks[result['track_id']], face) for face, result in zip(faces, resolved)
                if result['source'] == 'direct' and result['track_id'] in self.tracks]

    def resolve_faces(self, faces, tracks, matcher, profiles, now=None):
        """Resolve one frame without assigning two faces to a body or identity.

        Returned dictionaries align with ``faces``. Direct recognition can name
        a face without a body association, but only an unambiguous body binding
        may be used to learn appearance or carry identity through head turns.
        """
        now = time.monotonic() if now is None else now
        tracks = valid_tracks(tracks)
        matches = [matcher(face['embedding'], profiles) for face in faces]
        keys = [enclosing_track(face, tracks) for face in faces]
        names = [name.casefold() if name else None for name, _score in matches]
        ambiguous_keys = {key for key in keys if key is not None and keys.count(key) > 1}
        ambiguous_names = {name for name in names if name is not None and names.count(name) > 1}
        for key, name in zip(keys, names):
            row = self.tracks.get(key)
            if _accepts_face(row, now) and (key in ambiguous_keys or name in ambiguous_names):
                _clear_identity(row)
        results = []
        for face, key, name, match in zip(faces, keys, names, matches):
            stale = key is not None and not _accepts_face(self.tracks.get(key), now)
            duplicate = name in ambiguous_names
            if duplicate:
                match = (None, match[1])
            chosen_tracks = [] if key in ambiguous_keys or duplicate else tracks
            result = self.resolve_face(face, chosen_tracks, lambda *_args, value=match: value,
                                       profiles, now=now)
            if duplicate or key in ambiguous_keys:
                result['ambiguous'] = True
            if stale:
                # Keep this after ambiguity strips the track candidates: the
                # historical face must not become a fresh unbound presence.
                result['stale'] = True
            results.append(result)
        return results

    def resolve_face(self, face, tracks, matcher, profiles, now=None):
        """Recognize a face, with a short hold on a recently confirmed SAME track.

        ``source='tracked'`` is only a scene hint. It never updates the face
        anchor, confirmation time or appearance gallery. No room-wide presence
        TTL or nearby person's identity is used to label an unmatched face.
        Call ``resolve_faces`` when a frame may contain more than one face.
        """
        now = time.monotonic() if now is None else now
        key = enclosing_track(face, valid_tracks(tracks))
        row = self.tracks.get(key)
        if not _accepts_face(row, now):
            # The same ID may already refer to a new occupant, or a newer face
            # may have contradicted this result while inference was running.
            row = None
        name, score = matcher(face['embedding'], profiles)
        result = dict(name=name, score=score, source='direct' if name else 'unknown',
                      track_id=key if row is not None else None)
        if row is None:
            if key is not None:
                result['stale'] = True
            return result
        previous_face_seen = row.get('face_seen', 0)
        row['face_seen'] = now
        if name:
            if row.get('name') != name:
                row['greeted'] = False
            row.update(name=name, face_score=score, confirmed=now, face_source='direct',
                       face_anchor=face['embedding'], unknown_since=None, unknown_observations=0)
            return result
        if row.get('name'):
            import numpy as np

            from hub.face import cosine
            incompatible = False
            try:
                anchor = np.asarray(row.get('face_anchor'), dtype=np.float32).ravel()
                embedding = np.asarray(face['embedding'], dtype=np.float32).ravel()
                # A clear, large, frontal-enough detection of an incompatible
                # face is positive contradictory evidence, not a head-turn gap.
                incompatible = (float(face.get('score', 0)) >= .85
                                and float(face.get('area', 0)) >= 1600
                                and anchor.size == embedding.size
                                and np.isfinite(embedding).all()
                                and np.linalg.norm(embedding) > 0
                                and cosine(anchor, embedding) < .2)
            except (TypeError, ValueError):
                pass
            if not incompatible and now - row.get('confirmed', 0) <= IDENTITY_HOLD_S:
                row['face_source'] = 'tracked'
                return dict(name=row['name'], score=score, source='tracked', track_id=key)
            _clear_identity(row, replacement=incompatible)
        if row.get('unknown_since') is None or now - previous_face_seen >= 2.0:
            row['unknown_since'] = now
            row['unknown_observations'] = 0
        row['unknown_observations'] = row.get('unknown_observations', 0) + 1
        return result

    def active(self, now=None):
        now = time.monotonic() if now is None else now
        rows = [r for r in self.tracks.values() if now - r['seen'] < TRACK_TTL_S]
        for row in rows:
            if row.get('name') and now - row.get('confirmed', 0) > IDENTITY_HOLD_S:
                _clear_identity(row)
        return rows

    def description(self):
        rows = []
        for row in self.active():
            x = (row['box'][0] + row['box'][2]) / 2
            place = 'left' if x < .35 else 'right' if x > .65 else 'middle'
            name = row['name'] or 'unidentified person'
            rows.append(f"{name} at {place} (visual track)")
        return '; '.join(rows)

    def unknown_due(self, delay, now=None, *, cooldown=300):
        now = time.monotonic() if now is None else now
        unknown = [r for r in self.active(now) if not r['name']]
        if (self._unknown_greeted_at is not None and now - self._unknown_greeted_at < cooldown
                and len(unknown) <= self._unknown_greeted_count):
            return []
        return [r for r in unknown if not r['greeted']
                and r['face_seen'] and now - r['face_seen'] < 2 and now - r['born'] >= delay
                and (('unknown_since' not in r)
                     or (r['unknown_since'] is not None
                         and r.get('unknown_observations', 0) >= 2
                         and now - r['unknown_since'] >= UNKNOWN_STABILITY_S))]

    def mark_greeted(self, target, now=None):
        now = time.monotonic() if now is None else now
        rows = self.active(now)
        for row in rows:
            if row['name'] == target or (target == 'unknown' and row['name'] is None):
                row['greeted'] = True
        if target == 'unknown':
            self._unknown_greeted_at = now
            self._unknown_greeted_count = sum(not row['name'] for row in rows)
