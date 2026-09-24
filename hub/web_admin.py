"""The owner's web panel: FastAPI + Jinja, overlay network only (ТЗ F-705).

The panel shows what the owner needs to see and nothing that can be stolen: the
client list carries the *hash* of each token, never the token, and the panel
itself refuses every request that did not come from the overlay network —
including the login form, which is also a piece of information.

Login is a password from the environment, exchanged for a short-lived signed
session cookie. Signing uses the standard library (HMAC), so no extra
dependency enters the hub for this.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib.util
import ipaddress
import json
import logging
import math
import os
import secrets
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from hub import labelling, three_d, turn_trace
from hub.audit import AuditLog

log = logging.getLogger(__name__)

TEMPLATES = Path(__file__).resolve().parent / "templates"
COOKIE = "rowan_admin"
#: The panel knows one password, not which member typed it, so the audit says
#: honestly where the click came from (see ``DECISIONS.md``, P2-26).
PANEL_ACTOR = "web-panel"


class WebAdminAuth:
    """Password from the environment, exchanged for a signed cookie."""

    def __init__(self, *, password_env: str, session_minutes: int = 60,
                 secret: bytes | None = None, clock: Any = time.time) -> None:
        self.password_env, self.session_s = password_env, session_minutes * 60
        self.secret = secret or secrets.token_bytes(32)
        self.clock = clock

    def password(self) -> str:
        value = os.environ.get(self.password_env, "")
        if not value:
            raise RuntimeError(f"{self.password_env} is not set")
        return value

    def verify(self, given: str) -> bool:
        try:
            expected = self.password()
        except RuntimeError:
            return False
        return bool(given) and hmac.compare_digest(str(given), expected)

    def issue(self) -> str:
        expires = int(self.clock()) + self.session_s
        nonce = secrets.token_urlsafe(12)
        body = f"{expires}:{nonce}"
        return body + ":" + self._sign(body)

    def valid(self, cookie: str | None) -> bool:
        if not cookie:
            return False
        parts = str(cookie).split(":")
        if len(parts) != 3:
            return False
        body = f"{parts[0]}:{parts[1]}"
        if not hmac.compare_digest(parts[2], self._sign(body)):
            return False
        try:
            return int(parts[0]) >= int(self.clock())
        except ValueError:
            return False

    def _sign(self, body: str) -> str:
        return hmac.new(self.secret, body.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def overlay_only(allowed: list[str]):
    """The network gate: everything outside the overlay never sees the panel."""
    networks = [ipaddress.ip_network(str(item), strict=False) for item in allowed]

    def allowed_from(host: str | None) -> bool:
        if not host:
            return False
        try:
            address = ipaddress.ip_address(str(host))
        except ValueError:
            return False
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return any(address in network for network in networks)

    return allowed_from


def panel_names(allowed: list[str]):
    """The second gate: the panel answers under its own names, not a public one.

    The hub also answers on a public tunnel (ngrok and the like). There the
    request arrives *from loopback* with the tunnel's public name in ``Host``,
    so the peer address alone would let the whole internet reach the login form.
    The name therefore has to belong to the machine as well: an IP literal inside
    the allowed networks, or ``localhost``.
    """
    networks = [ipaddress.ip_network(str(item), strict=False) for item in allowed]

    def allowed_name(host_header: str | None) -> bool:
        value = str(host_header or "").strip().lower()
        if not value:
            return False
        if value.startswith("["):  # an IPv6 literal: [::1]:8770
            value = value[1:].split("]", 1)[0]
        elif value.count(":") == 1:  # a name or IPv4 with its port
            value = value.split(":", 1)[0]
        if value in {"localhost", "localhost.localdomain"}:
            return True
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return any(address in network for network in networks)

    return allowed_name


#: How many frames of one person the ring spins. Every archived frame is in the
#: strip and the table below; the ring is the overview, sampled evenly so the
#: whole archive is one turn of the wrist.
RING_FRAMES = 24
#: How many frames of one look (one outfit) the card shows.
LOOK_FRAMES = 8
#: How different two torso colour signatures must be to call it other clothes.
#: Measured on this hub's own archive (Lab histogram of the torso band of the
#: body crop): frames two minutes apart differ by 0.11 median, frames of another
#: day by 0.47. The bar sits between the two, and a look only splits after two
#: frames in a row agree, so one bad crop cannot invent a change of clothes.
OUTFIT_DISTANCE = 0.30
#: The clothes signatures of the frames that were already read (small JPEGs).
_clothing_cache: dict[str, tuple[float, ...]] = {}


class WebAdminData:
    """Read-only views of the hub database for the panel (ТЗ F-705).

    Every read opens its own short-lived connection: the panel is served from
    whatever thread the web server happens to use, and the hub's own connection
    belongs to the server loop (see ``hub.decision_log``). WAL makes the reads
    free of the writer.
    """

    #: How many archived photographs one profile page shows. The archive is
    #: append-only and grows for years; the page is a viewer, not a backup.
    PROFILE_SAMPLE_LIMIT = 120

    def __init__(self, path: str | Path, *, root: str | Path | None = None,
                 gallery: str | Path | None = None) -> None:
        self.path = str(path)
        #: Crops may only be served from here: the database stores a path, and a
        #: path that escaped the data directory must not reach the web server.
        self.root = Path(root) if root is not None else Path(self.path).resolve().parent
        #: The appearance archive (``hub.appearance.AppearanceGallery``) keeps its
        #: own SQLite file and its own crop files; both live under this directory.
        self.gallery_root = (Path(gallery) if gallery is not None
                             else self.root / "appearance")
        #: The training archive (F-303) holds the thousands of frames; a 3D build
        #: reads it, the panel only counts it.
        self.training_root = self.root / "training_archive"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)

    def _read(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def homes(self) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT home_id, name, tz, owner_person_id, config_rev,"
            " (SELECT COUNT(*) FROM clients WHERE clients.home_id = homes.home_id) AS clients,"
            " (SELECT COUNT(*) FROM memberships WHERE memberships.home_id = homes.home_id) AS people"
            " FROM homes ORDER BY name")
        return [{"home_id": row[0], "name": row[1], "tz": row[2], "owner": row[3] or "",
                 "config_rev": row[4], "clients": row[5], "people": row[6]} for row in rows]

    def clients(self) -> list[dict[str, Any]]:
        """Token *hashes* only: the panel must never show a usable token."""
        rows = self._read(
            "SELECT client_id, home_id, kind, token_hash, version, hw, last_seen"
            " FROM clients ORDER BY home_id, client_id")
        return [{"client_id": row[0], "home_id": row[1], "kind": row[2],
                 "token": _hash_hint(row[3]), "version": row[4] or "unknown",
                 "hw": row[5] or "", "last_seen": row[6] or "never"} for row in rows]

    def people(self) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT p.person_id, p.display_name, m.home_id, m.role, m.share_identity,"
            " m.share_presence FROM persons p LEFT JOIN memberships m ON m.person_id = p.person_id"
            " ORDER BY p.display_name, m.home_id")
        return [{"person_id": row[0], "name": row[1], "home_id": row[2] or "",
                 "role": row[3] or "not a member",
                 "shares": _shares(row[4], row[5])} for row in rows]

    def counts(self) -> dict[str, int]:
        profiles = self.profiles()
        return {"homes": len(self.homes()), "clients": len(self.clients()),
                "people": len({row["person_id"] for row in self.people()}),
                "profiles": len(profiles),
                "archive": sum(1 for entry in profiles if entry["gallery_id"]),
                "pending": len(self.unknown_tracks())}

    def audit(self, limit: int = 50) -> list[dict[str, Any]]:
        """The newest privileged actions of the hub (ТЗ F-706)."""
        rows = self._read(
            "SELECT ts, actor_person_id, home_id, action, target, result, detail_json"
            f" FROM audit ORDER BY ts DESC, rowid DESC LIMIT {int(limit)}")
        return [{"when": time.strftime("%Y-%m-%d %H:%M", time.localtime(row[0])),
                 "actor": row[1] or "—", "home_id": row[2] or "—", "action": row[3],
                 "target": row[4] or "—", "result": row[5] or "ok",
                 "detail": row[6] or "{}"} for row in rows]

    # --- the chain of one request -----------------------------------------

    def turns(self, limit: int = turn_trace.RECENT_TURNS) -> list[dict[str, Any]]:
        """Every recent request with a one-line summary of its chain.

        A turn is one request: a spoken turn in a room (the client's utterance
        id) or a Telegram message. This is the list the owner opens when they
        ask "what did the request actually do".
        """
        conn = self._connect()
        try:
            rows = turn_trace.TurnTraceStore(conn).recent(limit)
            lines = self._request_lines(conn, [str(row["turn_id"]) for row in rows])
        except sqlite3.Error as exc:
            log.warning("The panel could not read the request chain (%s)", exc)
            return []
        finally:
            conn.close()
        return [{**row,
                 "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["finished"])),
                 "request": lines.get(str(row["turn_id"]), {}).get("text", ""),
                 "reply": lines.get(str(row["turn_id"]), {}).get("reply", ""),
                 "href": quote(str(row["turn_id"]), safe="")} for row in rows]

    def turn_events(self, turn_id: str) -> list[dict[str, Any]]:
        """The steps of one request, in the order they happened."""
        conn = self._connect()
        try:
            events = turn_trace.TurnTraceStore(conn).events(turn_id)
        except sqlite3.Error as exc:
            log.warning("The panel could not read the steps of %s (%s)", turn_id, exc)
            return []
        finally:
            conn.close()
        for event in events:
            payload = event.get("payload") or {}
            event["summary"] = _step_summary(event.get("kind", ""), event.get("name", ""), payload)
            event["detail"] = json.dumps(payload, ensure_ascii=False, indent=2, default=str)[:6000]
        return events

    @staticmethod
    def _request_lines(conn: sqlite3.Connection,
                       turn_ids: list[str]) -> dict[str, dict[str, str]]:
        """What each request asked and what came back, from its own turn rows."""
        if not turn_ids:
            return {}
        placeholders = ",".join("?" * len(turn_ids))
        rows = conn.execute(
            "SELECT turn_id, payload_json FROM turn_events WHERE kind='turn'"
            f" AND turn_id IN ({placeholders}) ORDER BY event_id", tuple(turn_ids)).fetchall()
        lines: dict[str, dict[str, str]] = {}
        for turn_id, payload_json in rows:
            try:
                payload = json.loads(payload_json or "{}")
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            found = lines.setdefault(str(turn_id), {})
            text = payload.get("transcript") or payload.get("text") or ""
            if isinstance(text, str) and text and not found.get("text"):
                found["text"] = text
            reply = payload.get("reply") or ""
            if isinstance(reply, str) and reply:
                found["reply"] = reply
        return lines

    # --- manual labelling (ТЗ F-216) ---------------------------------------

    def unknown_tracks(self, *, day: str | None = None, home_id: str | None = None,
                       limit: int = labelling.QUEUE_LIMIT) -> list[dict[str, Any]]:
        """The tracks of one day that nobody could name, with their evidence."""
        conn = self._connect()
        try:
            tracks = labelling.queue(conn, home_id=home_id, day=day, limit=limit)
        finally:
            conn.close()
        return [self._track_view(track) for track in tracks]

    def labelling_people(self) -> list[dict[str, Any]]:
        """Who the track may be bound to: the click names a person, not a name."""
        rows = self._read("SELECT person_id, display_name FROM persons"
                          " ORDER BY display_name, person_id")
        return [{"person_id": row[0], "name": row[1] or row[0]} for row in rows]

    def save_label(self, track_id: str, person_id: str, *, day: str | None = None,
                   crop_id: str = "") -> dict[str, Any]:
        """One click: name the track, link that day's samples, keep the label."""
        conn = self._connect()
        try:
            result = labelling.label(conn, track_id, person_id, day=day, crop_id=crop_id,
                                     actor=PANEL_ACTOR, source="admin", audit=AuditLog(conn))
        except sqlite3.Error as exc:
            log.warning("The panel could not label track %s (%s)", track_id, exc)
            return {"ok": False, "track_id": str(track_id), "person_id": str(person_id),
                    "note": f"database error: {exc}"}
        finally:
            conn.close()
        return result.summary()

    def crop_path(self, crop_id: str) -> Path | None:
        """The JPEG of a crop, and only when it lives under the data directory."""
        conn = self._connect()
        try:
            return labelling.crop_file(conn, crop_id, self.root)
        finally:
            conn.close()

    # --- the profile classifier (ТЗ F-203, F-207, F-209, F-211, F-216) -----

    def _archive(self) -> sqlite3.Connection | None:
        """Read-only access to the appearance archive, or ``None``.

        The hub writes that archive through :class:`hub.appearance.AppearanceGallery`
        in the server loop; the panel must never open it for writing, so the
        connection is explicit about mode=ro and simply absent when the archive
        does not exist yet.
        """
        database = self.gallery_root / "gallery.sqlite3"
        if not database.exists():
            return None
        try:
            conn = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro",
                                   uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as exc:
            log.warning("The panel could not open the appearance archive (%s)", exc)
            return None

    def _archive_file(self, relative: Any) -> Path | None:
        """One archived photograph, and only from under the archive directory."""
        if not relative or not isinstance(relative, str):
            return None
        try:
            path = (self.gallery_root / relative).resolve()
            if not path.is_relative_to(self.gallery_root.resolve()) or not path.is_file():
                return None
            if path.stat().st_size > 20_000_000:
                return None
            return path
        except (OSError, ValueError):
            return None

    def appearance_people(self) -> list[dict[str, Any]]:
        """One line per archived person: how much of them exists and when.

        This is the archive's own view (F-209 look-alike samples). The names come
        from the archive, which resolves spelling from the manual enrollment, so
        a track label can never invent a person here.
        """
        conn = self._archive()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT p.id, p.name, COUNT(s.id) AS samples,"
                " SUM(CASE WHEN s.body_path IS NOT NULL THEN 1 ELSE 0 END) AS bodies,"
                " COUNT(DISTINCT s.appearance_hash) AS outfits,"
                " MIN(s.captured_at) AS first_seen, MAX(s.captured_at) AS last_seen"
                " FROM people p LEFT JOIN samples s ON s.person_id = p.id"
                " GROUP BY p.id, p.name ORDER BY p.name").fetchall()
            covers = {str(row["person_id"]): row for row in conn.execute(
                "SELECT person_id, id, captured_at FROM ("
                " SELECT person_id, id, captured_at,"
                " ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY captured_at DESC) AS n"
                " FROM samples WHERE face_path IS NOT NULL) WHERE n = 1")}
        except sqlite3.Error as exc:
            log.warning("The panel could not read the appearance archive (%s)", exc)
            return []
        finally:
            conn.close()
        people = []
        for row in rows:
            person_id, name = str(row["id"]), str(row["name"])
            cover = covers.get(person_id)
            people.append({
                "person_id": person_id, "name": name,
                "samples": int(row["samples"] or 0), "bodies": int(row["bodies"] or 0),
                "outfits": int(row["outfits"] or 0),
                "first_seen": _stamp(row["first_seen"]), "last_seen": _stamp(row["last_seen"]),
                "cover": (_archive_url(person_id, str(cover["id"]), "face")
                          if cover is not None else ""),
                "cover_at": _stamp(cover["captured_at"]) if cover is not None else "",
            })
        return people

    def appearance_samples(self, gallery_id: str,
                           limit: int | None = None) -> list[dict[str, Any]]:
        """Every archived frame of one person, newest first, with its numbers."""
        conn = self._archive()
        if conn is None:
            return []
        limit = self.PROFILE_SAMPLE_LIMIT if limit is None else max(1, int(limit))
        try:
            rows = conn.execute("SELECT * FROM samples WHERE person_id=?"
                                " ORDER BY captured_at DESC LIMIT ?",
                                (str(gallery_id), limit)).fetchall()
        except sqlite3.Error as exc:
            log.warning("The panel could not read archived samples (%s)", exc)
            return []
        finally:
            conn.close()
        samples = []
        for row in rows:
            quality = _quality(row["quality"])
            face = self._archive_file(row["face_path"])
            body = self._archive_file(row["body_path"])
            if face is None and body is None:
                continue
            captured = float(row["captured_at"])
            samples.append({
                "sample_id": str(row["id"]),
                "gallery_id": str(row["person_id"]),
                "captured_at": _stamp(captured), "day": _day(captured),
                "face": _archive_url(str(row["person_id"]), str(row["id"]), "face") if face else "",
                "body": _archive_url(str(row["person_id"]), str(row["id"]), "body") if body else "",
                "identity": _number(quality.get("identity_score")),
                "margin": _number(quality.get("identity_margin")),
                "detector": _number(quality.get("detector_score")),
                "sharpness": _number(quality.get("sharpness")),
                "pixels": _pixels(quality.get("face_pixels")),
                "admission": str(quality.get("admission") or "unknown"),
                "track": str(quality.get("track_id") or ""),
                "outfit": str(row["appearance_hash"])[:12],
                "appearance_hash": str(row["appearance_hash"]),
                "body_file": str(row["body_path"] or ""),
                "captured_name": str(row["captured_name"]),
            })
        return samples

    def clothing(self, samples: list[dict[str, Any]]) -> dict[str, tuple[float, ...]]:
        """A colour signature of the clothes in each frame, keyed by sample id.

        The archive's own difference hash is too jumpy to see a change of clothes
        (see :meth:`outfits`), so the panel reads the body crop itself: the torso
        band, converted to Lab and histogrammed. Reading 100 small JPEGs is a few
        tens of milliseconds, and the cache keeps it off the hot path.
        """
        signatures: dict[str, tuple[float, ...]] = {}
        for sample in samples:
            path = self._archive_file(sample.get("body_file"))
            if path is None:
                continue
            signature = _clothing_cache.get(str(path))
            if signature is None:
                signature = _clothing_signature(path)
                _clothing_cache[str(path)] = signature
            if signature:
                signatures[sample["sample_id"]] = signature
        return signatures

    @staticmethod
    def outfits(samples: list[dict[str, Any]],
                clothing: dict[str, tuple[float, ...]] | None = None,
                threshold: float = OUTFIT_DISTANCE) -> list[dict[str, Any]]:
        """Group the archive into looks: runs of frames in the same clothes.

        A day is not a look — people change. Frames are walked in time order; a
        look is a run of frames whose torso colour is close to the look's own
        running signature, and a new look starts only after TWO frames in a row
        are far away, so one bad crop cannot invent a change of clothes. Frames
        without a signature (no body crop) stay in the look they arrived in.
        """
        clothing = clothing or {}
        by_day: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            by_day.setdefault(sample["day"], []).append(sample)
        looks: list[dict[str, Any]] = []
        for day in sorted(by_day, reverse=True):
            frames = sorted(by_day[day], key=lambda item: item["captured_at"])
            runs: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            pending: list[dict[str, Any]] = []
            signature: tuple[float, ...] | None = None
            for frame in frames:
                own = clothing.get(frame["sample_id"])
                far = bool(own and signature and _clothing_distance(own, signature) > threshold)
                if far:
                    pending.append(frame)
                    if len(pending) >= 2 and current:
                        runs.append(current)
                        current, pending, signature = pending[:], [], None
                    continue
                current.extend(pending)
                pending = []
                current.append(frame)
                own = clothing.get(frame["sample_id"])
                if own:
                    signature = own if signature is None else _blend(signature, own)
            current.extend(pending)
            if current:
                runs.append(current)
            for index, run in enumerate(reversed(runs), start=1):
                looks.append({
                    "day": day, "count": len(run), "index": index,
                    "changed": len(runs) > 1,
                    "first": run[0]["captured_at"][11:16],
                    "last": run[-1]["captured_at"][11:16],
                    "face": next((item["face"] for item in reversed(run) if item["face"]), ""),
                    "body": next((item["body"] for item in reversed(run) if item["body"]), ""),
                    "identity": max((item["identity"] for item in run
                                     if item["identity"] != "—"), default="—"),
                    "frames": _ring(run, LOOK_FRAMES),
                })
        return looks

    def gallery_image(self, person_id: str, sample_id: str, kind: str) -> Path | None:
        """One archived photograph of one person, never a path of somebody else."""
        if kind not in {"face", "body"}:
            return None
        conn = self._archive()
        if conn is None:
            return None
        try:
            row = conn.execute(f"SELECT {kind}_path FROM samples WHERE id=? AND person_id=?",
                               (str(sample_id), str(person_id))).fetchone()
        except sqlite3.Error:
            return None
        finally:
            conn.close()
        return self._archive_file(row[0] if row is not None else None)

    def three_d(self, gallery_id: str) -> dict[str, Any] | None:
        """The carved 3D build of one person, when ``build-3d-profile.py`` made one.

        The build is a command, never a service: the panel only reports what is
        on disk, so a person without a model simply shows the command that makes
        one (``scripts/build-3d-profile.py``).
        """
        if not gallery_id:
            return None
        directory = self.gallery_root / str(gallery_id) / "model"
        try:
            if not (directory / "model.bin").is_file() or not (directory / "model.json").is_file():
                return None
            meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict) or not meta.get("points"):
                return None
            bytes_on_disk = (directory / "model.bin").stat().st_size
            built = time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime((directory / "model.bin").stat().st_mtime))
        except (OSError, ValueError) as exc:
            log.warning("The panel could not read the 3D build of %s (%s)", gallery_id, exc)
            return None
        details = meta.get("meta") if isinstance(meta.get("meta"), dict) else {}
        return {
            "points": int(meta.get("points") or 0),
            "triangles": int(meta.get("triangles") or 0),
            "bytes": int(bytes_on_disk), "built": built,
            "views": int(details.get("views") or 0),
            "of_frames": int(details.get("of_frames") or 0),
            "yaw": details.get("yaw_range") or [],
            "method": str(details.get("method") or ""),
            "source": str(details.get("source") or "appearance archive"),
            "skipped": int(details.get("skipped") or 0),
            "bin": _model_url(gallery_id, "model.bin"),
            "json": _model_url(gallery_id, "model.json"),
        }

    def model_file(self, gallery_id: str, name: str) -> Path | None:
        """``model.bin``/``model.json``/``model.ply`` of one person, and nothing else."""
        if name not in {"model.bin", "model.json", "model.ply"}:
            return None
        try:
            path = (self.gallery_root / str(gallery_id) / "model" / name).resolve()
            root = (self.gallery_root / str(gallery_id) / "model").resolve()
            if not path.is_relative_to(root) or not path.is_file():
                return None
            if path.stat().st_size > 60_000_000:
                return None
            return path
        except (OSError, ValueError):
            return None

    def body_crops(self, person_id: str, limit: int = 600) -> list[tuple[str, float]]:
        """Per-track body crops of one person from ``data/homes`` (newest first).

        These are the crops the hub itself made while the person was tracked, so
        they are the raw material of the photographic build beside the archive.
        """
        if not person_id:
            return []
        try:
            rows = self._read(
                "SELECT b.path, b.ts FROM body_crops b JOIN tracks t ON t.track_id = b.track_id"
                " WHERE t.person_id = ? ORDER BY b.ts DESC LIMIT ?",
                (str(person_id), int(max(1, limit))))
        except sqlite3.Error as exc:
            log.debug("The panel could not read body crops (%s)", exc)
            return []
        found: list[tuple[str, float]] = []
        for path, stamp in rows:
            try:
                if Path(str(path)).is_file():
                    found.append((str(path), float(stamp or 0.0)))
            except (OSError, ValueError):
                continue
        return found

    def build_sources(self, person: dict[str, Any]) -> list[dict[str, Any]]:
        """What a 3D build could be made of, with the real numbers of this hub."""
        gallery = self.appearance_samples(person["gallery_id"]) if person["gallery_id"] else []
        from_archive = (len(three_d.training_frames(self.training_root, person["name"],
                                                    limit=20000))
                        if person["name"] else 0)
        crops = len(self.body_crops(person["person_id"], limit=20000)) if person["person_id"] else 0
        return [{"key": "all", "title": "Everything (default)",
                 "note": "archive, gallery and body crops together, newest first",
                 "frames": from_archive + len(gallery) + crops},
                {"key": "archive", "title": "Training archive",
                 "note": "every crop the hub stored for this person (F-303)",
                 "frames": from_archive},
                {"key": "gallery", "title": "Appearance archive",
                 "note": "the curated, quality-gated frames the panel shows (F-209)",
                 "frames": len(gallery)},
                {"key": "bodies", "title": "Track body crops",
                 "note": "what the hub cut out while it followed the person",
                 "frames": crops}]

    def build_status(self, gallery_id: str) -> dict[str, Any]:
        """What the last 3D build of this person is doing, or did."""
        if not gallery_id:
            return {"state": "never"}
        try:
            path = self.gallery_root / str(gallery_id) / "model" / "status.json"
            if not path.is_file():
                return {"state": "never"}
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {"state": "never"}
            payload["age_s"] = int(max(0.0, time.time() - path.stat().st_mtime))
            return payload
        except (OSError, ValueError):
            return {"state": "never"}

    def start_build(self, person: dict[str, Any], *, source: str = "all",
                    method: str = "photos", limit: int = 600, step: int = 3,
                    standing: str = "on", look: str = "newest",
                    pose: str = "same", device: str = "auto") -> dict[str, Any]:
        """Run ``scripts/build-3d-profile.py`` for one person, in the background.

        The panel never builds a model inside the request: it writes the status
        file the page polls and starts one detached process. A build that is
        already running is not started twice.
        """
        gallery_id = person["gallery_id"] or person["person_id"]
        if not gallery_id:
            return {"state": "error", "error": "This person has nothing archived to build from."}
        current = self.build_status(gallery_id)
        # A build writes its status at least every few frames; a file that has not
        # moved for ten minutes is a dead build, not a busy one.
        if current.get("state") == "running" and current.get("age_s", 999) < 600:
            return {"state": "running", "note": "A build is already running."}
        source = source if source in {"all", "archive", "gallery", "bodies"} else "all"
        method = method if method in {"photos", "hull"} else "photos"
        standing = standing if standing in {"on", "off"} else "on"
        look = look if look in {"newest", "biggest", "all"} else "newest"
        pose = pose if pose in {"same", "any"} else "same"
        device = device if device in {"auto", "cpu", "cuda"} else "auto"
        limit = max(8, min(int(limit), 20000))
        step = max(1, min(int(step), 12))
        script = Path(__file__).resolve().parent.parent / "scripts" / "build-3d-profile.py"
        status = (self.gallery_root / str(gallery_id) / "model" / "status.json")
        interpreter = sys.executable or "python"
        command = [interpreter, str(script), "--person", str(gallery_id),
                   "--root", self.path, "--archive", str(self.training_root),
                   "--source", source, "--method", method, "--limit", str(limit),
                   "--step", str(step), "--standing", standing, "--look", look,
                   "--pose", pose, "--device", device, "--status", str(status)]
        try:
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps(
                {"state": "running", "done": 0, "total": limit, "seconds": 0.0,
                 "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "options": {"person": person["name"], "source": source, "method": method,
                             "limit": limit, "step": step, "standing": standing,
                             "look": look, "pose": pose, "device": device}},
                ensure_ascii=False, indent=2),
                encoding="utf-8")
            log_dir = self.gallery_root / str(gallery_id) / "model"
            log_file = (log_dir / "build.log").open("ab")
            subprocess.Popen(command, cwd=str(script.parent.parent), stdout=log_file,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=_detached_flags())
        except OSError as exc:
            log.warning("The panel could not start a 3D build (%s)", exc)
            return {"state": "error", "error": f"could not start the build: {exc}"}
        log.info("A 3D build of %s started (%s, %s frames, %s)", person["name"], method,
                 limit, device)
        return {"state": "running", "options": {"source": source, "method": method,
                                               "limit": limit, "step": step,
                                               "standing": standing, "look": look,
                                               "pose": pose, "device": device}}

    def profiles(self) -> list[dict[str, Any]]:
        """Everybody Rowan knows: the manual people plus the photo archive.

        One row per name, because that is what the owner thinks in. The manual
        person row carries the vectors (F-211) and the membership; the archive
        carries the photographs (F-209). A name may exist in either place alone,
        and the panel says which.
        """
        people = self._read("SELECT person_id, display_name FROM persons"
                            " ORDER BY display_name, person_id")
        homes: dict[str, list[str]] = {}
        roles: dict[str, str] = {}
        for person_id, home_id, role in self._read(
                "SELECT person_id, home_id, role FROM memberships ORDER BY home_id"):
            key = str(person_id)
            homes.setdefault(key, []).append(str(home_id))
            word = str(role or "").strip()
            if word and roles.get(key, "") != "admin":
                roles[key] = word
        faces = self._count_by("face_embeddings", "person_id")
        voices = self._count_by("voice_embeddings", "person_id")
        bodies = self._count_by("body_embeddings", "person_id")
        labels = self._count_by("identity_labels", "person_id")
        archive = {item["name"].strip().casefold(): item for item in self.appearance_people()}
        merged: dict[str, dict[str, Any]] = {}
        for row in people:
            person_id, name = str(row[0]), str(row[1] or "").strip()
            merged[name.casefold()] = {
                "profile_id": person_id, "person_id": person_id, "gallery_id": "",
                "name": name or person_id, "homes": homes.get(person_id, []),
                "role": roles.get(person_id, ""),
                "vectors": {"face": faces.get(person_id, 0), "voice": voices.get(person_id, 0),
                            "body": bodies.get(person_id, 0)},
                "labels": labels.get(person_id, 0),
                "gallery": _empty_gallery(),
            }
        for key, item in archive.items():
            entry = merged.get(key)
            if entry is None:
                entry = merged[key] = {
                    "profile_id": "archive:" + item["person_id"], "person_id": "",
                    "gallery_id": item["person_id"], "name": item["name"], "homes": [],
                    "role": "", "vectors": {"face": 0, "voice": 0, "body": 0}, "labels": 0,
                    "gallery": _empty_gallery(),
                }
            entry["gallery_id"] = item["person_id"]
            entry["gallery"] = item
        return sorted(merged.values(), key=lambda entry: entry["name"].strip().casefold())

    def profile(self, profile_id: str) -> dict[str, Any] | None:
        """One person with the frames the carousel spins and their label history."""
        for entry in self.profiles():
            if entry["profile_id"] != profile_id:
                continue
            entry["samples"] = (self.appearance_samples(entry["gallery_id"])
                                if entry["gallery_id"] else [])
            entry["outfits"] = self.outfits(entry["samples"], self.clothing(entry["samples"]))
            entry["ring"] = _ring(entry["samples"], RING_FRAMES)
            entry["stats"] = _profile_stats(entry["samples"])
            entry["model"] = self.three_d(entry["gallery_id"])
            entry["history"] = self.label_history(entry["person_id"]) if entry["person_id"] else []
            return entry
        return None

    def label_history(self, person_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """The manual labels of one person: who bound which track, and when."""
        try:
            rows = self._read("SELECT day, track_id, actor, source, at, crop_id"
                              " FROM identity_labels WHERE person_id=?"
                              " ORDER BY at DESC, rowid DESC LIMIT ?",
                              (str(person_id), int(limit)))
        except sqlite3.Error as exc:
            log.debug("The panel could not read label history (%s)", exc)
            return []
        return [{"day": str(row[0] or ""), "track": str(row[1] or ""),
                 "actor": str(row[2] or "—"), "source": str(row[3] or "—"),
                 "when": _stamp(row[4]), "crop": str(row[5] or "")} for row in rows]

    def _count_by(self, table: str, column: str) -> dict[str, int]:
        """Rows per person in one table; a table this deployment lacks is no data."""
        try:
            rows = self._read(f"SELECT {column}, COUNT(*) FROM {table}"
                              f" WHERE {column} IS NOT NULL GROUP BY {column}")
        except sqlite3.Error as exc:
            log.debug("The panel could not count %s (%s)", table, exc)
            return {}
        return {str(row[0]): int(row[1]) for row in rows}

    @staticmethod
    def _track_view(track: labelling.UnknownTrack) -> dict[str, Any]:
        belief = dict(track.belief)
        sources = belief.get("sources") if isinstance(belief.get("sources"), dict) else {}
        return {"track_id": track.track_id, "home_id": track.home_id or "—",
                "first_seen": track.first_seen, "last_seen": track.last_seen,
                "faces": track.faces, "bodies": track.bodies, "voices": track.voices,
                "quality": f"{track.quality:.2f}",
                "belief_p": f"{float(belief.get('p') or 0.0):.2f}",
                "belief_person": belief.get("person_id") or "—",
                "sources": ", ".join(f"{name} {float(value):.2f}"
                                     for name, value in sorted(sources.items())) or "—",
                "crops": [{"crop_id": crop.crop_id} for crop in track.crops]}


def _hash_hint(token_hash: str) -> str:
    """A fingerprint of the stored hash: enough to compare, useless to present."""
    digest = hashlib.sha256(str(token_hash).encode("utf-8")).hexdigest()
    return f"{digest[:6]}…{digest[-4:]}"


def _body_reid_available() -> bool:
    """Is the body (ReID) engine importable in THIS deployment?

    The panel says ``degraded`` instead of a threshold nobody reaches: the owner
    already saw the honest line in the log ("torchreid is not installed - body
    ReID stays off"), and the page has to agree with it.
    """
    try:
        return importlib.util.find_spec("torchreid") is not None
    except (ImportError, ValueError):
        return False


def _stamp(value: Any) -> str:
    """Unix seconds as the owner's own clock; anything unusable is a dash."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number) or number <= 0:
        return "—"
    try:
        return datetime.fromtimestamp(number).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "—"


def _day(value: Any) -> str:
    stamp = _stamp(value)
    return stamp[:10] if stamp != "—" else "unknown"


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.3f}" if math.isfinite(number) else "—"


def _brief(value: Any) -> str:
    """A threshold as the owner reads it: ``0.45``, ``30`` — no padding zeros."""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.3f}".rstrip("0").rstrip(".") or "0"


def _pixels(value: Any) -> str:
    """A face box in pixels, as ``78×94``."""
    try:
        width, height = (int(part) for part in value)
    except (TypeError, ValueError):
        return "—"
    return f"{width}×{height}"


def _quality(raw: Any) -> dict[str, Any]:
    """The quality JSON of one archived sample; damaged rows keep their picture."""
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _archive_url(person_id: str, sample_id: str, kind: str) -> str:
    return (f"/admin/appearance/{quote(str(person_id), safe='')}"
            f"/{quote(str(sample_id), safe='')}/{kind}.jpg")


def _model_url(gallery_id: str, name: str) -> str:
    return f"/admin/model/{quote(str(gallery_id), safe='')}/{name}"


def _detached_flags() -> int:
    """Start the builder without a console window of its own (Windows)."""
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _int_or(value: Any, default: int) -> int:
    """A number from a form field, or the default when somebody typed rubbish."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)


def _empty_gallery() -> dict[str, Any]:
    return {"person_id": "", "name": "", "samples": 0, "bodies": 0, "outfits": 0,
            "first_seen": "—", "last_seen": "—", "cover": "", "cover_at": ""}


def _clothing_signature(path: Path) -> tuple[float, ...]:
    """A Lab histogram of the torso band of one body crop.

    Colour, not shape: two photographs of one shirt under different light land
    close together, another shirt lands far away. The band avoids the head and
    the legs, which move.
    """
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is None or not image.size:
            return ()
        small = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
        torso = small[17:74, 9:55]  # rows ~13-58%, columns ~14-86%
        if not torso.size:
            return ()
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        histogram = cv2.calcHist([lab], [0, 1, 2], None, [6, 5, 5],
                                 [0, 256, 0, 256, 0, 256]).ravel()
        total = float(histogram.sum())
        if not total:
            return ()
        return tuple(float(value) / total for value in histogram)
    except Exception:  # a broken JPEG must not take the panel down
        log.debug("The panel could not read a body crop for its clothes", exc_info=True)
        return ()


def _clothing_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Histogram intersection distance: 0 is the same clothes, 1 is nothing alike."""
    if not left or not right or len(left) != len(right):
        return 1.0
    return 1.0 - sum(min(one, other) for one, other in zip(left, right))


def _blend(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    """The running signature of one look: mean of what it has seen, renormalised."""
    merged = [0.5 * one + 0.5 * other for one, other in zip(left, right)]
    total = sum(merged)
    return tuple(value / total for value in merged) if total else left


def _ring(samples: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Evenly spaced frames of the archive, for the turntable."""
    pictures = [sample for sample in samples if sample["face"] or sample["body"]]
    if count <= 0 or len(pictures) <= count:
        return pictures
    step = len(pictures) / float(count)
    return [pictures[int(index * step)] for index in range(count)]


def _profile_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers of one archive: how sharp, how confident, how long a span."""
    identity = [float(sample["identity"]) for sample in samples if sample["identity"] != "—"]
    sharpness = [float(sample["sharpness"]) for sample in samples if sample["sharpness"] != "—"]
    days = {sample["day"] for sample in samples}
    return {
        "frames": len(samples),
        "bodies": sum(1 for sample in samples if sample["body"]),
        "days": len(days),
        "best_identity": f"{max(identity):.3f}" if identity else "—",
        "best_sharpness": f"{max(sharpness):.1f}" if sharpness else "—",
        "first": samples[-1]["captured_at"] if samples else "—",
        "last": samples[0]["captured_at"] if samples else "—",
        "manual": sum(1 for sample in samples if sample["admission"] == "manual_enrollment"),
    }


def classifier_stages(cfg: Any, *, body_reid: bool | None = None) -> list[dict[str, Any]]:
    """The recognition pipeline as it is configured right now (ТЗ 7, F-203-F-211).

    The owner asked for the classifier itself on a page, not only its results:
    which signal runs, at which threshold, and which one is missing in THIS
    deployment. Every number below is read from the live config, so the page
    cannot drift away from what the hub actually does.
    """
    face = getattr(cfg.server, "face", None)
    speaker = getattr(cfg.server, "speaker", None)
    identity = getattr(cfg.server, "identity", None)
    reid = getattr(identity, "reid", None)
    learning = getattr(identity, "learning", None)
    anti = getattr(identity, "anti_spoofing", None)
    guest = getattr(identity, "guest", None)

    def flag(value: Any) -> str:
        return "on" if value else "off"

    def stage(key: str, title: str, state: str, note: str,
              values: list[tuple[str, Any]]) -> dict[str, Any]:
        # The key is ``numbers`` and not ``values``: in Jinja, ``stage.values``
        # is the dict's own method and the page would render a bound method.
        return {"key": key, "title": title, "state": state, "note": note,
                "numbers": [(label, _brief(value)) for label, value in values]}

    reid_on = bool(getattr(reid, "enabled", False))
    reid_state = "on" if (reid_on and body_reid is not False) else ("degraded" if reid_on else "off")
    engine = ("torchreid is importable" if body_reid else
              "torchreid is not installed — no body vector is produced")
    return [
        stage("detect", "1 · Detection and tracking",
              flag(getattr(face, "enabled", False)),
              "The room PC runs YOLO on every camera frame; the hub keeps one track "
              "per person and one body crop per track. Nothing here is a name yet.",
              [("face engine", "insightface"), ("presence ttl, s",
                                                getattr(face, "presence_ttl_s", 0))]),
        stage("face", "2 · Face signature",
              flag(getattr(face, "enabled", False)),
              "512-d insightface vector per detected face; the newest manual anchor "
              "wins and the archive is compared when adaptive_recognition is on.",
              [("match threshold", getattr(face, "threshold", 0)),
               ("adaptive samples", flag(getattr(face, "adaptive_recognition", False)))]),
        stage("voice", "3 · Voice signature",
              flag(getattr(speaker, "enabled", False)),
              "ECAPA vector of the utterance; it names the speaker of the turn, "
              "not the person in front of the camera.",
              [("match threshold", getattr(speaker, "threshold", 0)),
               ("winner margin", getattr(speaker, "margin", 0)),
               ("admin threshold", getattr(speaker, "admin_threshold", 0)),
               ("min speech, s", getattr(speaker, "min_speech_s", 0))]),
        stage("body", "4 · Body (ReID)",
              reid_state,
              "512-d OSNet vector of a body crop, kept per day: the same clothes "
              "link a track to a person when the face is turned away. " + engine,
              [("model", str(getattr(reid, "model", "—"))),
               ("same-day threshold", getattr(reid, "threshold", 0)),
               ("margin", getattr(reid, "match_margin", 0))]),
        stage("fusion", "5 · Fusion into one person",
              flag(getattr(identity, "enabled", False)),
              "Face, voice and body become one belief per track; a privileged "
              "action additionally needs the second factor below.",
              [("admin voice", getattr(identity, "admin_voice_threshold", 0)),
               ("admin face", getattr(identity, "admin_face_threshold", 0)),
               ("phone voice", getattr(identity, "phone_admin_threshold", 0)),
               ("second factor", flag(getattr(identity, "admin_second_factor", False))),
               ("spoken PIN", flag(getattr(identity, "phone_pin_required", False)))]),
        stage("spoof", "6 · Anti-spoofing",
              flag(getattr(anti, "face", False)),
              "A photograph or a screen in front of the camera is rejected before "
              "any of the numbers above are trusted.",
              [("feature checks", flag(getattr(anti, "face", False))),
               ("liveness model", str(getattr(anti, "model", "") or "not installed")),
               ("model required", flag(getattr(anti, "require_model", False))),
               ("burst frames", getattr(anti, "window_frames", 0))]),
        stage("archive", "7 · Appearance archive",
              flag(getattr(face, "appearance_enabled", False)),
              "A confirmed sighting is kept only after several distinct frames agree, "
              "and only the newest usable frame becomes an identity reference.",
              [("frames to admit", 3), ("min identity", 0.60), ("min margin", 0.12),
               ("min sharpness", 30.0), ("capture cooldown, s", 60.0),
               ("reference identity", 0.65), ("reference sharpness", 40.0),
               ("reference age, days", getattr(identity, "appearance_retention_days", 7))]),
        stage("learning", "8 · Adaptive learning",
              flag(getattr(learning, "enabled", False)),
              "New vectors join a profile permanently only above a higher belief "
              "than recognition needs, and never when they look like somebody else.",
              [("min belief", getattr(learning, "min_p", 0)),
               ("face vectors", getattr(learning, "face_max_vectors", 0)),
               ("voice vectors", getattr(learning, "voice_max_vectors", 0)),
               ("body vectors per day", getattr(learning, "body_per_day", 0)),
               ("conflict at", getattr(learning, "conflict_similarity", 0))]),
        stage("guest", "9 · Guest registration",
              flag(getattr(guest, "enabled", False)),
              "A visitor is remembered for a window instead of joining the household: "
              "5-10 frames of the face and a spoken phrase.",
              [("frames", f"{getattr(guest, 'min_frames', 0)}–{getattr(guest, 'max_frames', 0)}"),
               ("clean speech, s", getattr(guest, "min_voice_seconds", 0)),
               ("flow window, s", getattr(guest, "flow_ttl_s", 0))]),
    ]


def _shares(identity: Any, presence: Any) -> str:
    words = []
    if identity:
        words.append("identity")
    if presence:
        words.append("presence")
    return ", ".join(words) or "nothing shared"


def _short(value: Any, limit: int = 200) -> str:
    """One readable piece of a payload: never the whole base64 blob."""
    if isinstance(value, str):
        text = " ".join(value.split())
        return text[:limit] + ("…" if len(text) > limit else "")
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)[:limit]
    except (TypeError, ValueError):
        return str(value)[:limit]


def _message_line(message: Any, limit: int = 140) -> str:
    """One message of a prompt as ``role: text`` (tool calls named, not dumped)."""
    if not isinstance(message, dict):
        return _short(message, limit)
    role = str(message.get("role") or "?")
    parts = []
    content = message.get("content")
    if isinstance(content, list):  # Responses-style content blocks
        content = " ".join(_short(block.get("text", block), limit) for block in content
                           if isinstance(block, dict))
    if content:
        parts.append(_short(content, limit))
    calls = message.get("tool_calls")
    if isinstance(calls, list) and calls:
        names = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            names.append(str((function or {}).get("name") or call.get("name") or "tool")
                         if isinstance(call, dict) else "tool")
        parts.append("→ " + ", ".join(names))
    if message.get("name"):
        role = f"{role} {message['name']}"
    return f"{role}: " + " ".join(part for part in parts if part)


def _step_summary(kind: str, name: str, payload: Any) -> str:
    """One human line per step: what happened, without reading the JSON below."""
    if not isinstance(payload, dict):
        return _short(payload)
    if kind == "turn":
        if payload.get("text"):
            where = payload.get("room") or payload.get("chat") or ""
            return f"Telegram {payload.get('from', '')} {where}: {_short(payload['text'], 300)}"
        if payload.get("transcript"):
            return (f"{payload.get('speaker', 'unknown')}: {_short(payload['transcript'], 300)}"
                    f" → {_short(payload.get('reply', ''), 300)}")
        if payload.get("reply"):
            return f"answer: {_short(payload['reply'], 300)}"
        if payload.get("durations_ms"):
            stages = payload["durations_ms"]
            degraded = ", ".join(payload.get("degraded") or []) or "none"
            return (f"stt {stages.get('stt', 0)} ms, llm {stages.get('llm', 0)} ms,"
                    f" tts {stages.get('tts', 0)} ms, total {stages.get('total', 0)} ms;"
                    f" degraded: {degraded}")
        return "home " + _short(payload.get("home") or "—", 40) + \
               f", client {_short(payload.get('client') or '—', 40)}"
    if kind == "decision":
        return (f"{payload.get('type', '?')} = {_short(payload.get('value'), 120)}"
                f" (p {payload.get('confidence', '?')}) → {payload.get('outcome', '')}")
    if kind == "tool":
        return (f"{_short(payload.get('args'), 240)} → {_short(payload.get('result'), 300)}")
    if kind == "llm":
        text = _short(payload.get("text"), 300)
        tools = payload.get("tools") or []
        calls = (" called " + ", ".join(str(tool) for tool in tools)) if tools else " no tool call"
        return f"round {payload.get('round', '?')}/{payload.get('of', '?')}{calls}: {text}"
    if kind == "prompt":
        messages = payload.get("messages")
        if isinstance(messages, list):
            shown = " | ".join(_message_line(message) for message in messages[:6])
            more = f" (+{len(messages) - 6} more)" if len(messages) > 6 else ""
            return f"{payload.get('provider', 'llm')} ← {shown}{more}"
        return (f"{payload.get('provider', 'llm')} {payload.get('model', '')} ←"
                f" {_short(payload.get('prompt'), 300)}")
    if kind == "say":
        return f"said out loud ({name}): {_short(payload.get('text'), 300)}"
    if kind == "image":
        return (f"the image model refused ({payload.get('reason', '?')},"
                f" {payload.get('model', '')})")
    return _short(payload, 300)


def build_router(*, cfg: Any, data: WebAdminData | None, auth: WebAdminAuth,
                 preview: Any = None) -> APIRouter:
    """The panel's routes; every one of them sits behind the overlay gate."""
    settings = cfg.server.web_admin
    allowed_from = overlay_only(list(settings.allowed_networks))
    allowed_name = panel_names(list(settings.allowed_networks))
    templates = Jinja2Templates(directory=str(TEMPLATES))
    router = APIRouter()

    def denied(request: Request):
        peer = request.client.host if request.client else None
        if not allowed_from(peer) or not allowed_name(request.headers.get("host")):
            log.warning("The web panel refused %s (host %r)",
                        peer or "?", request.headers.get("host"))
            return HTMLResponse("<h1>404</h1><p>This panel is only reachable from the "
                                "overlay network.</p>", status_code=404)
        return None

    def signed_in(request: Request) -> bool:
        return auth.valid(request.cookies.get(COOKIE))

    @router.get("/admin", response_class=HTMLResponse)
    async def home(request: Request):
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return templates.TemplateResponse(request, "admin/login.html",
                                              {"error": "Sign in to open the panel."})
        if data is None:
            return templates.TemplateResponse(request, "admin/login.html",
                                              {"error": "The hub database is unavailable."})
        return templates.TemplateResponse(request, "admin/dashboard.html",
                                          {"counts": data.counts()})

    @router.post("/admin/login", response_class=HTMLResponse)
    async def login(request: Request, password: str = Form("")):  # noqa: S107 - a form field
        if (blocked := denied(request)) is not None:
            return blocked
        if not auth.verify(password):
            log.warning("A web panel login failed from %s",
                        request.client.host if request.client else "?")
            return templates.TemplateResponse(request, "admin/login.html",
                                              {"error": "Wrong password."}, status_code=403)
        answer = RedirectResponse("/admin/homes", status_code=303)
        answer.set_cookie(COOKIE, auth.issue(), httponly=True, samesite="lax",
                          max_age=auth.session_s)
        return answer

    @router.get("/admin/logout")
    async def logout(request: Request):
        if (blocked := denied(request)) is not None:
            return blocked
        answer = RedirectResponse("/admin", status_code=303)
        answer.delete_cookie(COOKIE)
        return answer

    @router.get("/admin/preview", response_class=HTMLResponse)
    async def preview_view(request: Request, room: str = "", on: int = 1):
        """Владелец 2026-09-24: открыть/закрыть живое окно камеры на ПК комнаты.

        Тот же переключатель, что у голосового инструмента ``camera_preview``:
        комната показывает свою камеру у себя на экране, кадры никуда не
        уезжают. Здесь только кнопка - сама работа у ``admin_preview``.
        """
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        if preview is None:
            return HTMLResponse("<h1>503</h1><p>The hub cannot reach the rooms "
                                "right now.</p>", status_code=503)
        try:
            result = await preview(room, bool(on))
        except Exception as exc:  # noqa: BLE001 - the panel answers, it never throws
            log.warning("The panel could not switch the live preview (%s)", exc)
            result = {"ok": False, "error": f"Could not switch it: {type(exc).__name__}"}
        done = bool(result.get("ok"))
        detail = (f"Live camera view is {'on' if on else 'off'} for "
                  f"{result.get('label') or room or 'this room'}." if done else
                  str(result.get("error") or "It did not work."))
        return HTMLResponse(f"<h1>{'Done' if done else 'No'}</h1><p>{detail}</p>"
                            "<p><a href=\"/admin\">Back to the panel</a></p>")

    def page(template: str, rows_key: str, rows_fn):
        async def view(request: Request):
            if (blocked := denied(request)) is not None:
                return blocked
            if not signed_in(request):
                return RedirectResponse("/admin", status_code=303)
            rows = rows_fn() if data is not None else []
            return templates.TemplateResponse(request, template,
                                              {rows_key: rows, "counts": data.counts() if data else {}})

        return view

    router.add_api_route("/admin/homes", page("admin/homes.html", "homes",
                                              lambda: data.homes() if data else []),
                         methods=["GET"], response_class=HTMLResponse)
    router.add_api_route("/admin/clients", page("admin/clients.html", "clients",
                                                lambda: data.clients() if data else []),
                         methods=["GET"], response_class=HTMLResponse)
    router.add_api_route("/admin/people", page("admin/people.html", "people",
                                               lambda: data.people() if data else []),
                         methods=["GET"], response_class=HTMLResponse)
    router.add_api_route("/admin/audit", page("admin/audit.html", "events",
                                              lambda: data.audit(50) if data else []),
                         methods=["GET"], response_class=HTMLResponse)

    def turns_page(request: Request):
        """Every recent request, newest first, with the shape of its chain."""
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        return templates.TemplateResponse(request, "admin/turns.html", {
            "turns": data.turns() if data is not None else [],
            "counts": data.counts() if data is not None else {}})

    router.add_api_route("/admin/turns", turns_page, methods=["GET"],
                         response_class=HTMLResponse)

    def turn_page(request: Request, turn_id: str):
        """One request: every step of its chain in the order it happened."""
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        events = data.turn_events(turn_id) if data is not None else []
        return templates.TemplateResponse(request, "admin/turn.html", {
            "turn_id": turn_id, "events": events,
            "counts": data.counts() if data is not None else {}})

    # ``:path`` keeps a Telegram turn id (chat and message ids, colons) intact.
    router.add_api_route("/admin/turns/{turn_id:path}", turn_page, methods=["GET"],
                         response_class=HTMLResponse)

    def tracks_page(request: Request, day: str = ""):
        """ТЗ F-216: the unrecognized tracks of a day, with crops and numbers."""
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        wanted = str(day)[:10] or labelling.day_of()
        return templates.TemplateResponse(request, "admin/tracks.html", {
            "tracks": data.unknown_tracks(day=wanted) if data is not None else [],
            "people": data.labelling_people() if data is not None else [],
            "day": wanted, "message": request.query_params.get("message", ""),
            "counts": data.counts() if data is not None else {}})

    router.add_api_route("/admin/tracks", tracks_page, methods=["GET"],
                         response_class=HTMLResponse)

    def profiles_page(request: Request):
        """The classifier itself: every signal, its thresholds, and every person."""
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        return templates.TemplateResponse(request, "admin/profiles.html", {
            "profiles": data.profiles() if data is not None else [],
            "stages": classifier_stages(cfg, body_reid=_body_reid_available()),
            "message": request.query_params.get("message", ""),
            "counts": data.counts() if data is not None else {}})

    router.add_api_route("/admin/profiles", profiles_page, methods=["GET"],
                         response_class=HTMLResponse)

    # These two live above the profile page: its ``:path`` converter would
    # otherwise swallow ``/admin/profiles/<id>/build`` as a profile id.
    @router.post("/admin/profiles/{profile_id}/build")
    async def start_build(request: Request, profile_id: str, source: str = Form("all"),
                          method: str = Form("photos"), limit: str = Form("600"),
                          step: str = Form("3"), standing: str = Form("on"),
                          look: str = Form("newest"), pose: str = Form("same"),
                          device: str = Form("auto")):
        """The menu's Build button: one detached process, progress in a file."""
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        person = data.profile(profile_id) if data is not None else None
        if person is None or data is None:
            return HTMLResponse("<h1>404</h1><p>No such profile.</p>", status_code=404)
        result = data.start_build(person, source=source, method=method,
                                  limit=_int_or(limit, 600), step=_int_or(step, 3),
                                  standing=standing, look=look, pose=pose,
                                  device=device)
        message = (f"A build is already running for {person['name']}."
                   if result.get("state") == "running" and result.get("note")
                   else (f"Build failed to start: {result.get('error')}"
                         if result.get("state") == "error"
                        else f"Building {person['name']} from {source} ({method}, "
                             f"{device}). "
                              "The page shows the progress."))
        query = urlencode({"message": message})
        return RedirectResponse(f"/admin/profiles/{quote(profile_id, safe='')}?{query}",
                                status_code=303)

    @router.get("/admin/profiles/{profile_id}/build-status")
    async def build_status(request: Request, profile_id: str):
        """What the build is doing, for the page's own polling."""
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return Response('{"state": "unknown"}', media_type="application/json",
                            status_code=403)
        person = data.profile(profile_id) if data is not None else None
        if person is None:
            return Response('{"state": "unknown"}', media_type="application/json",
                            status_code=404)
        return Response(json.dumps(data.build_status(person["gallery_id"]), default=str),
                        media_type="application/json")

    def profile_page(request: Request, profile_id: str):
        """One person: the frames of the archive, their looks, and their labels."""
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        person = data.profile(profile_id) if data is not None else None
        if person is None:
            return HTMLResponse("<h1>404</h1><p>No such profile.</p>", status_code=404)
        return templates.TemplateResponse(request, "admin/profile.html", {
            "person": person, "stages": classifier_stages(cfg, body_reid=_body_reid_available()),
            "sources": data.build_sources(person) if data is not None else [],
            "status": data.build_status(person["gallery_id"]) if data is not None else {},
            "message": request.query_params.get("message", ""),
            "counts": data.counts() if data is not None else {}})

    # ``:path`` keeps a profile id (an archive id may carry a prefix) intact.
    router.add_api_route("/admin/profiles/{profile_id:path}", profile_page, methods=["GET"],
                         response_class=HTMLResponse)

    @router.get("/admin/appearance/{person_id}/{sample_id}/{kind}.jpg")
    async def appearance(request: Request, person_id: str, sample_id: str, kind: str):
        """One archived frame: the sample has to belong to that person."""
        if (blocked := denied(request)) is not None:
            return blocked
        path = (data.gallery_image(person_id, sample_id, kind)
                if (data is not None and signed_in(request)) else None)
        if path is None:
            return HTMLResponse("<h1>404</h1><p>No such archived frame.</p>", status_code=404)
        return Response(content=path.read_bytes(), media_type="image/jpeg")

    @router.get("/admin/model/{person_id}/{name}")
    async def model_file(request: Request, person_id: str, name: str):
        """The carved 3D build of one person: points, metadata, or the PLY."""
        if (blocked := denied(request)) is not None:
            return blocked
        path = (data.model_file(person_id, name)
                if (data is not None and signed_in(request)) else None)
        if path is None:
            return HTMLResponse("<h1>404</h1><p>No such 3D build.</p>", status_code=404)
        media = {"model.bin": "application/octet-stream", "model.json": "application/json",
                 "model.ply": "application/octet-stream"}[name]
        return Response(content=path.read_bytes(), media_type=media)

    @router.post("/admin/label")
    async def apply_label(request: Request, track_id: str = Form(""),
                          person_id: str = Form(""), day: str = Form(""),
                          crop_id: str = Form("")):
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        wanted = str(day)[:10]
        query: dict[str, str] = {"day": wanted} if wanted else {}
        if data is None or not track_id or not person_id:
            query["message"] = "Nothing changed: pick a track and a person."
        else:
            result = data.save_label(track_id, person_id, day=wanted or None, crop_id=crop_id)
            query["message"] = (f"Track {track_id} is now {result['name']}."
                                if result.get("ok")
                                else f"Nothing changed: {result.get('note', 'failed')}")
        return RedirectResponse("/admin/tracks?" + urlencode(query), status_code=303)

    @router.get("/admin/crop/{crop_id}.jpg")
    async def crop(request: Request, crop_id: str):
        """The crop of a track, from the data directory and nowhere else."""
        if (blocked := denied(request)) is not None:
            return blocked
        path = data.crop_path(crop_id) if (data is not None and signed_in(request)) else None
        if path is None:
            return HTMLResponse("<h1>404</h1><p>No such crop.</p>", status_code=404)
        return Response(content=path.read_bytes(), media_type="image/jpeg")

    return router


def mount(app: Any, *, cfg: Any, data: WebAdminData | None, preview: Any = None) -> Any:
    """Attach the panel to the hub's FastAPI app when it is switched on."""
    settings = getattr(cfg.server, "web_admin", None)
    if settings is None or not settings.enabled:
        return None
    auth = WebAdminAuth(password_env=settings.password_env,
                        session_minutes=settings.session_minutes)
    if not os.environ.get(settings.password_env):
        log.warning("The web panel is enabled but %s is not set: nobody can sign in",
                    settings.password_env)
    app.include_router(build_router(cfg=cfg, data=data, auth=auth, preview=preview))
    # The router is part of the hub's own app, so the panel answers on the hub's
    # own port (``server.host``/``server.port``); the panel settings only name
    # the overlay address the deployment expects.
    log.info("The owner web panel serves /admin on the hub's own port (overlay names only;"
             " expected at http://%s:%s/admin)", settings.host, settings.port)
    return auth


__all__ = ["COOKIE", "PANEL_ACTOR", "TEMPLATES", "WebAdminAuth", "WebAdminData", "build_router",
           "classifier_stages", "mount", "overlay_only", "panel_names", "quote"]
