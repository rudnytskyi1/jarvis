"""Conservative, persistent visual examples of manually enrolled people.

The archive is append-only: selection limits apply to recognition/reference
inputs, never to retained photographs. Automated examples cannot authorize
their own successors; every admission is checked against manual enrollment.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis.server.appearance")


class LearnedProfiles(dict):
    """Ordinary matching map with preserved manual-anchor provenance."""

    def __init__(self, manual):
        self.manual_profiles = {name: [list(v) for v in vectors] for name, vectors in manual.items()}
        super().__init__({name: [list(v) for v in vectors] for name, vectors in manual.items()})


def _vector(raw):
    try:
        value = np.asarray(raw, dtype=np.float32).ravel()
        length = float(np.linalg.norm(value))
        return value / length if value.size and np.isfinite(value).all() and length > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _box(raw):
    try:
        value = [float(v) for v in raw]
        if (len(value) == 4 and all(math.isfinite(v) and 0 <= v <= 1 for v in value)
                and value[2] > value[0] and value[3] > value[1]):
            return value
    except (TypeError, ValueError):
        pass
    return None


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _manual(profiles):
    return profiles.manual_profiles if isinstance(profiles, LearnedProfiles) else (profiles or {})


def _fingerprint(profiles):
    return hashlib.sha256(json.dumps(
        {n: [np.asarray(v).tolist() for v in vs] for n, vs in _manual(profiles).items()},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def _match(vector, profiles):
    scores = []
    for name, raw_vectors in _manual(profiles).items():
        vectors = [_vector(v) for v in raw_vectors]
        values = [float(vector @ v) for v in vectors if v is not None and v.shape == vector.shape]
        if values:
            scores.append((max(values), str(name)))
    scores.sort(reverse=True)
    if not scores:
        return None, 0.0, 0.0
    score, name = scores[0]
    return name, score, score - (scores[1][0] if len(scores) > 1 else 0.0)


class AppearanceGallery:
    MIN_SCORE = .60
    MIN_MARGIN = .12
    MIN_DETECTION = .80
    MIN_FACE_SIDE = 64
    MIN_SHARPNESS = 30.0
    MIN_INTERVAL = 60.0
    ACTIVE_LIMIT = 12

    def __init__(self, root):
        self.root = Path(root)
        self.database = self.root / "gallery.sqlite3"
        self._lock = threading.RLock()
        self._pending = {}
        self._learned_cache = None

    def _connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                name_key TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY, person_id TEXT NOT NULL,
                captured_name TEXT NOT NULL, captured_at REAL NOT NULL,
                face_path TEXT NOT NULL, body_path TEXT,
                embedding TEXT NOT NULL, quality TEXT NOT NULL,
                face_hash TEXT NOT NULL, appearance_hash TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES people(id)
            );
            CREATE INDEX IF NOT EXISTS samples_person_time
                ON samples(person_id, captured_at DESC);
        """)
        return connection

    @staticmethod
    def _crop(image, box):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        return image[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    @staticmethod
    def _hash(image):
        import cv2
        tiny = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
        return "".join("1" if v else "0" for v in (tiny[:, 1:] > tiny[:, :-1]).ravel())

    def observe(self, jpeg, row, face, profiles, *, faces=None, now=None):
        """Admit a high-quality face after repeated manual-anchor confirmation.

        Boxes are normalized xyxy. ``row['body_unambiguous']=False`` excludes
        a body overlapping another YOLO person. Missing ``faces`` excludes
        body capture. ``now`` is Unix seconds (mainly injectable for tests).
        """
        return self._observe(jpeg, row, face, profiles, faces=faces, now=now)

    def enroll(self, jpeg, name, face, *, faces=None, now=None):
        """Retain an explicitly confirmed enrollment, never an inferred identity.

        The caller must have completed the face-selection/enrollment flow.
        A manual example needs no temporal confirmation or capture cooldown.
        """
        name = str(name).strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            return None
        row = dict(id="enrollment:" + uuid.uuid4().hex, name=name)
        return self._observe(jpeg, row, face, {name: [face.get("embedding")]},
                             faces=faces, now=now, enrollment=True)

    def _observe(self, jpeg, row, face, profiles, *, faces=None, now=None, enrollment=False):
        now = time.time() if now is None else float(now)
        name = str(row.get("name") or "").strip()
        key = (str(row.get("id", "")), name.casefold())
        try:
            with self._lock:
                # A track changing its tentative identity must start a new run.
                for previous in [k for k in self._pending if k[0] == key[0] and k != key]:
                    del self._pending[previous]
            vector = _vector(face.get("embedding"))
            box = _box(face.get("box"))
            detector = float(face.get("score", 0))
            if not name or not key[0] or not math.isfinite(now) or vector is None or box is None:
                return None
            matched, score, margin = _match(vector, profiles)
            if (matched is None or matched.casefold() != name.casefold()
                    or score < self.MIN_SCORE or margin < self.MIN_MARGIN
                    or not math.isfinite(detector) or detector < self.MIN_DETECTION):
                with self._lock:
                    self._pending.pop(key, None)
                return None
            # Resolve spelling from the manually enrolled profile, never a track label.
            name = matched
            import cv2

            from hub.face import decode_jpeg
            image = decode_jpeg(jpeg)
            if image is None:
                return None
            crop = self._crop(image, box)
            if not crop.size or min(crop.shape[:2]) < self.MIN_FACE_SIDE:
                return None
            sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if not math.isfinite(sharpness) or sharpness < self.MIN_SHARPNESS:
                return None
            # Keep some hair/head context, but never include a second detected face.
            padding_x, padding_y = (box[2] - box[0]) * .20, (box[3] - box[1]) * .25
            padded = [max(0, box[0] - padding_x), max(0, box[1] - padding_y),
                      min(1, box[2] + padding_x), min(1, box[3] + padding_y)]
            other_boxes = []
            target_present = False
            invalid_other_box = False
            for other in faces or []:
                other_box = _box(other.get("box"))
                if other_box is None:
                    invalid_other_box = True
                elif np.allclose(other_box, box, atol=1e-5):
                    target_present = True
                else:
                    other_boxes.append(other_box)
            if any(_overlap(box, b) for b in other_boxes):
                return None
            face_crop = self._crop(image, box if any(_overlap(padded, b) for b in other_boxes) else padded)
            body = None
            body_box = _box(row.get("box"))
            if (target_present and not invalid_other_box and row.get("body_unambiguous", True) and body_box is not None
                    and body_box[0] <= box[0] and body_box[1] <= box[1]
                    and body_box[2] >= box[2] and body_box[3] >= box[3]
                    and not any(_overlap(body_box, b) for b in other_boxes)):
                body = self._crop(image, body_box)
                if not body.size or min(body.shape[:2]) < self.MIN_FACE_SIDE:
                    body = None
            quality = dict(identity_score=score, identity_margin=margin, detector_score=detector,
                           sharpness=sharpness, face_pixels=[crop.shape[1], crop.shape[0]],
                           face_box=box, body_box=body_box if body is not None else None,
                           track_id=key[0], admission="manual_enrollment" if enrollment else "manual_anchor_confirmation")
            # Different enrollments or a swapped occupant break the confirmation chain.
            anchor_fingerprint = _fingerprint(profiles)
            with self._lock:
                if enrollment:
                    return self._store(name, vector, face_crop, body, quality, now, force=True)
                for stale in [k for k, p in self._pending.items() if now - p["last"] > 10]:
                    del self._pending[stale]
                pending = self._pending.get(key)
                if (pending is None or pending["anchors"] != anchor_fingerprint
                        or not 0 < now - pending["last"] <= 5
                        or pending["vector"].shape != vector.shape
                        or float(pending["vector"] @ vector) < .65):
                    pending = dict(first=now, count=0, anchors=anchor_fingerprint)
                pending.update(last=now, count=pending["count"] + 1, vector=vector)
                self._pending[key] = pending
                if pending["count"] < 3 or now - pending["first"] < 1.0:
                    return None
                return self._store(name, vector, face_crop, body, quality, now)
        except Exception:
            log.exception("Appearance observation was not retained")
            return None

    def _write_jpeg(self, relative, image):
        import cv2
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise OSError("Could not encode appearance sample")
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with temporary.open("xb") as handle:
            handle.write(encoded.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    def _store(self, name, vector, face_crop, body_crop, quality, now, *, force=False):
        face_hash = self._hash(face_crop)
        appearance_hash = self._hash(body_crop if body_crop is not None else face_crop)
        connection = self._connect()
        try:
            # SQLite serializes all connections/processes, not only this object's lock.
            connection.execute("BEGIN IMMEDIATE")
            person = connection.execute("SELECT * FROM people WHERE name_key=?", (name.casefold(),)).fetchone()
            person_id = person["id"] if person else uuid.uuid4().hex
            last = connection.execute("SELECT * FROM samples WHERE person_id=? ORDER BY captured_at DESC LIMIT 1",
                                      (person_id,)).fetchone()
            if last and not force:
                if now - last["captured_at"] < self.MIN_INTERVAL:
                    return None
                try:
                    old_vector = _vector(json.loads(last["embedding"]))
                except (ValueError, TypeError):
                    # Preserve damaged archive rows, but do not let one bad
                    # comparison sample disable all future automatic captures.
                    old_vector = None
                old_hash = last["appearance_hash"]
                changed = (sum(a != b for a, b in zip(appearance_hash, old_hash)) / 64
                           if isinstance(old_hash, str) and len(old_hash) == 64 else 1.0)
                if (old_vector is not None and old_vector.shape == vector.shape
                        and float(vector @ old_vector) > .995 and changed < .12
                        and now - last["captured_at"] < 21600):
                    return None
            sample_id = uuid.uuid4().hex
            face_path = f"{person_id}/{sample_id}-face.jpg"
            body_path = f"{person_id}/{sample_id}-body.jpg" if body_crop is not None else None
            self._write_jpeg(face_path, face_crop)
            if body_path:
                self._write_jpeg(body_path, body_crop)
            if person is None:
                connection.execute("INSERT INTO people VALUES(?,?,?)", (person_id, name, name.casefold()))
            connection.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?)", (
                sample_id, person_id, name, now, face_path, body_path,
                json.dumps(vector.tolist()), json.dumps(quality), face_hash, appearance_hash))
            connection.commit()
            return dict(sample_id=sample_id, name=name, captured_at=now, quality=quality,
                        face_path=face_path, body_path=body_path)
        finally:
            connection.close()

    def _rows(self, name, limit=128):
        if not self.database.exists():
            return []
        with self._lock:
            connection = self._connect()
            try:
                return connection.execute("""SELECT s.*, p.name FROM samples s
                    JOIN people p ON p.id=s.person_id WHERE p.name_key=?
                    ORDER BY captured_at DESC LIMIT ?""", (str(name).casefold(), limit)).fetchall()
            finally:
                connection.close()

    def _read(self, relative):
        if not relative:
            return None
        try:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root.resolve()) or path.stat().st_size > 20_000_000:
                return None
            raw = path.read_bytes()
            from hub.face import decode_jpeg
            return raw if decode_jpeg(raw) is not None else None
        except (OSError, ValueError):
            return None

    def references(self, name, limit=2, profiles=None):
        """Return dated references, optionally revalidated against current anchors.

        Pass current manual profiles when supplying identity images to tools.
        An empty mapping deliberately rejects all archived identities; omitting
        it preserves raw archive access for maintenance and legacy callers.
        """
        limit = max(0, min(int(limit), 6))
        if not limit:
            return []
        try:
            candidates = []
            rows = self._rows(name)
            timestamps = [r["captured_at"] for r in rows
                          if isinstance(r["captured_at"], (int, float)) and math.isfinite(r["captured_at"])]
            newest = max(timestamps, default=0)
            for row in rows:
                try:
                    if profiles is not None:
                        vector = _vector(json.loads(row["embedding"]))
                        if vector is None:
                            continue
                        matched, score, margin = _match(vector, profiles)
                        if (matched is None or matched.casefold() != str(name).casefold()
                                or score < self.MIN_SCORE or margin < self.MIN_MARGIN):
                            continue
                    quality = json.loads(row["quality"])
                    captured_at = float(row["captured_at"])
                    captured_at_iso = datetime.fromtimestamp(captured_at, UTC).isoformat()
                    identity_score, sharpness = float(quality["identity_score"]), float(quality["sharpness"])
                    if not all(math.isfinite(v) for v in (captured_at, identity_score, sharpness)):
                        continue
                    # Prefer recent photographs, allowing quality to choose within a day.
                    rank = identity_score + min(sharpness, 1000) / 4000
                    rank -= min((newest - captured_at) / 86400, 10) * .1
                    candidates.append((rank, row, quality, captured_at_iso))
                except (ValueError, KeyError, TypeError, OverflowError, OSError):
                    continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            result = []
            # First image always supplies a face, then an optional body reference.
            for kind in ["face", "body"] + ["face"] * max(0, limit - 2):
                for _, row, quality, captured_at_iso in candidates:
                    if any(r["sample_id"] == row["id"] and r["kind"] == kind for r in result):
                        continue
                    raw = self._read(row[f"{kind}_path"])
                    if raw:
                        result.append(dict(name=row["name"], kind=kind, jpeg=raw,
                            captured_at=row["captured_at"], captured_name=row["captured_name"],
                            captured_at_iso=captured_at_iso,
                            sample_id=row["id"], quality=quality, source="appearance_gallery",
                            label=f"{row['name']}: {kind} reference captured at {captured_at_iso}; "
                                  "use for identity, not as evidence of current presence or clothing."))
                        break
                if kind == "face" and not result:
                    # A body crop alone cannot stand in for a usable identity
                    # photograph if its associated face archive is damaged.
                    return []
                if len(result) >= limit:
                    break
            return result
        except (OSError, sqlite3.Error):
            log.warning("Appearance references unavailable", exc_info=True)
            return []

    def list_people(self, profiles=None):
        if not self.database.exists():
            return []
        try:
            with self._lock:
                connection = self._connect()
                try:
                    names = [r[0] for r in connection.execute("SELECT name FROM people ORDER BY name_key")]
                finally:
                    connection.close()
            return [name for name in names if self.references(name, limit=1, profiles=profiles)]
        except (OSError, sqlite3.Error):
            log.warning("Appearance archive unavailable", exc_info=True)
            return []

    def learned_profiles(self, profiles):
        """Bound matching cost, revalidate samples against current manual anchors."""
        manual = _manual(profiles)
        try:
            stat = self.database.stat() if self.database.exists() else None
            cache_key = (stat.st_mtime_ns if stat else None, stat.st_size if stat else 0, _fingerprint(manual))
        except (OSError, ValueError, TypeError):
            cache_key = None
        with self._lock:
            if cache_key is not None and self._learned_cache is not None and self._learned_cache[0] == cache_key:
                cached = self._learned_cache[1]
                result = LearnedProfiles(manual)
                for name, values in cached.items():
                    result[name] = [list(v) for v in values]
                return result
        result = LearnedProfiles(manual)
        for name in manual:
            selected = []
            try:
                for row in self._rows(name):
                    try:
                        vector = _vector(json.loads(row["embedding"]))
                        if vector is None:
                            continue
                        matched, score, margin = _match(vector, manual)
                        if (matched is None or matched.casefold() != name.casefold()
                                or score < self.MIN_SCORE or margin < self.MIN_MARGIN):
                            continue
                        if any(float(vector @ v) > .985 for v in selected):
                            continue
                        if self._read(row["face_path"]) is None:
                            continue
                        selected.append(vector)
                        if len(selected) >= self.ACTIVE_LIMIT:
                            break
                    except (ValueError, TypeError, KeyError):
                        continue
            except (OSError, sqlite3.Error):
                log.warning("Adaptive appearance samples unavailable for %s", name, exc_info=True)
            result[name].extend(vector.tolist() for vector in selected)
        with self._lock:
            self._learned_cache = (cache_key, {name: [list(v) for v in values] for name, values in result.items()})
        return result

    def rename(self, old, new, *, allow_merge=False):
        """Change the current label; preserve stable IDs and historical labels."""
        new = str(new).strip()
        if not new or len(new) > 100 or any(ord(c) < 32 for c in new):
            raise ValueError("Invalid appearance profile name")
        if not self.database.exists():
            return False
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                old_row = connection.execute("SELECT id FROM people WHERE name_key=?", (str(old).casefold(),)).fetchone()
                if old_row is None:
                    return False
                existing = connection.execute("SELECT id FROM people WHERE name_key=?", (new.casefold(),)).fetchone()
                if existing and existing["id"] != old_row["id"]:
                    if not allow_merge:
                        raise ValueError("An appearance profile with this name already exists")
                    connection.execute("UPDATE samples SET person_id=? WHERE person_id=?", (existing["id"], old_row["id"]))
                    connection.execute("DELETE FROM people WHERE id=?", (old_row["id"],))
                else:
                    connection.execute("UPDATE people SET name=?, name_key=? WHERE id=?", (new, new.casefold(), old_row["id"]))
                connection.commit()
                self._pending.clear()
                return True
            finally:
                connection.close()
