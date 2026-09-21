"""Room media storage and retention (ТЗ section 4.6, F-304).

Media files are kept under ``data/homes/<home_id>/media/YYYY-MM-DD/`` and are
tracked by the ``media`` table. Frames/crops and clips have separate TTLs; the
cleanup job removes both the file and the database row, while embeddings and
presence events remain untouched.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hub.homes import ensure_home

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

_HOME_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

FRAME_KINDS = {"frame", "crop"}
CLIP_KINDS = {"clip"}
MEDIA_KINDS = FRAME_KINDS | CLIP_KINDS | {"audio"}

_KIND_SUFFIX = {
    "frame": ".jpg",
    "crop": ".jpg",
    "clip": ".mp4",
    "audio": ".wav",
}

log = logging.getLogger("jarvis.server.media")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


class MediaStore:
    """Create, register and expire room media files."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        *,
        media_ttl_days: int = 3,
        clip_ttl_days: int = 7,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.conn = conn
        self.base = Path(data_dir)
        self.media_ttl_days = media_ttl_days
        self.clip_ttl_days = clip_ttl_days
        self.clock = clock
        self.homes_root = self.base / "homes"

    @staticmethod
    def validate_home_id(home_id: str) -> str:
        if not _HOME_ID_RE.fullmatch(home_id):
            raise ValueError(f"invalid home_id: {home_id!r}")
        return home_id

    def path_for(self, home_id: str, when: datetime | None = None) -> Path:
        """Date directory for media captured at ``when`` (today by default)."""
        home_id = self.validate_home_id(home_id)
        moment = when or self.clock()
        return self.homes_root / home_id / "media" / moment.strftime("%Y-%m-%d")

    def media_root(self, home_id: str) -> Path:
        return self.homes_root / self.validate_home_id(home_id) / "media"

    def ttl_days(self, kind: str) -> int:
        if kind in CLIP_KINDS:
            return self.clip_ttl_days
        if kind in FRAME_KINDS or kind == "audio":
            return self.media_ttl_days
        raise ValueError(f"unknown media kind: {kind!r}")

    def expires_at(self, kind: str, ts: float) -> str:
        if ts < 0:
            raise ValueError("media timestamp cannot be negative")
        moment = datetime.fromtimestamp(ts, tz=UTC)
        return (moment + timedelta(days=self.ttl_days(kind))).isoformat(timespec="seconds")

    def register(self, home_id: str, kind: str, path: Path | str, *, ts: float | None = None) -> str:
        """Record an existing file in ``media``; return its stable ``media_ref``."""
        home_id = self.validate_home_id(home_id)
        if kind not in MEDIA_KINDS:
            raise ValueError(f"unknown media kind: {kind!r}")
        ensure_home(self.conn, home_id, name=home_id)
        media_path = Path(path).resolve()
        allowed = self.media_root(home_id).resolve()
        if not media_path.is_relative_to(allowed):
            raise ValueError(f"media path {media_path} is outside {allowed}")
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        timestamp = float(media_path.stat().st_mtime if ts is None else ts)
        media_ref = f"media-{_sha(str(media_path) + '|' + kind)}"
        self.conn.execute(
            "INSERT OR IGNORE INTO media(media_ref, home_id, path, kind, ts, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (media_ref, home_id, str(media_path), kind, timestamp, self.expires_at(kind, timestamp)),
        )
        self.conn.commit()
        return media_ref

    def save_bytes(
        self,
        home_id: str,
        kind: str,
        data: bytes,
        *,
        filename: str | None = None,
        ts: float | None = None,
    ) -> tuple[str, Path]:
        """Write media bytes into the dated directory and register the row."""
        home_id = self.validate_home_id(home_id)
        if kind not in MEDIA_KINDS:
            raise ValueError(f"unknown media kind: {kind!r}")
        timestamp = self.clock().timestamp() if ts is None else float(ts)
        date_dir = self.path_for(home_id, datetime.fromtimestamp(timestamp, tz=UTC))
        date_dir.mkdir(parents=True, exist_ok=True)
        safe_name = filename or (uuid.uuid4().hex + _KIND_SUFFIX[kind])
        target = date_dir / Path(safe_name).name
        target.write_bytes(data)
        return self.register(home_id, kind, target, ts=timestamp), target

    def cleanup_expired(self, *, now: datetime | None = None) -> dict[str, int]:
        """Delete expired files and their ``media`` rows; never touch other paths."""
        moment = now or self.clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        now_iso = moment.astimezone(UTC).isoformat(timespec="seconds")
        rows = self.conn.execute(
            "SELECT media_ref, home_id, path FROM media "
            "WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now_iso,),
        ).fetchall()
        result = {"expired_rows": len(rows), "deleted_files": 0, "missing_files": 0, "unsafe_skipped": 0}
        for media_ref, home_id, path_text in rows:
            media_path = Path(path_text).resolve()
            allowed = self.media_root(str(home_id)).resolve()
            if media_path.is_relative_to(allowed) and media_path.is_file():
                media_path.unlink()
                result["deleted_files"] += 1
            elif media_path.is_relative_to(allowed):
                result["missing_files"] += 1
            else:
                result["unsafe_skipped"] += 1
            self.conn.execute("DELETE FROM media WHERE media_ref=?", (media_ref,))
        self.conn.commit()
        self._remove_empty_date_dirs()
        return result

    def _remove_empty_date_dirs(self) -> None:
        """Best-effort cleanup of empty dated media directories."""
        if not self.homes_root.exists():
            return
        for media_root in self.homes_root.glob("*/media"):
            if not media_root.is_dir():
                continue
            for date_dir in sorted(media_root.iterdir(), reverse=True):
                if date_dir.is_dir():
                    try:
                        date_dir.rmdir()
                    except OSError:
                        pass


__all__ = [
    "DEFAULT_DATA_DIR",
    "MEDIA_KINDS",
    "FRAME_KINDS",
    "CLIP_KINDS",
    "MediaStore",
]
