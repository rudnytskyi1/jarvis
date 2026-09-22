"""Make an alert video playable on a phone, not only on a desktop player.

The room client records through OpenCV's ``mp4v`` writer, which produces
MPEG-4 Part 2 (``mp4v``). Desktop Telegram plays that; the phone apps do not -
the message arrives and cannot be opened. The hub is the only place that has
an encoder, so a clip is converted to H.264 (``avc1``) with the ``moov`` atom
at the front (``+faststart``) right before it is sent.

Every step is best-effort: a room must still get its recording when ffmpeg is
missing, busy or fed a clip it does not like. In that case the original bytes
come back and the caller says so, instead of losing the alert.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

#: Point this at a specific binary when ffmpeg is not on PATH.
FFMPEG_ENV = "ROWAN_FFMPEG"
#: The encoder settings that all phone players agree on: H.264 baseline-ish
#: profile, ``yuv420p`` chroma (4:4:4 trips iOS), no audio track, and the
#: metadata first so Telegram can stream it. ``veryfast`` keeps a 60-second
#: chunk to a few seconds of CPU on the hub.
_ARGS = ("-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.0",
         "-movflags", "+faststart", "-an")


def ffmpeg_path() -> str | None:
    """The encoder to use, or ``None`` when this hub has none."""
    override = str(os.environ.get(FFMPEG_ENV) or "").strip()
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("ffmpeg")


def is_phone_ready(data: bytes) -> bool:
    """True when this MP4 already carries an H.264 track.

    MP4 stores the sample description inside ``stsd``: the four-character code
    (``avc1``) is enough to tell H.264 from the MPEG-4 Part 2 the room client
    writes, without decoding anything.
    """
    return isinstance(data, bytes) and b"avc1" in data[:4096]


def phone_ready_mp4(data: bytes, *, timeout_s: float = 90.0) -> bytes | None:
    """``data`` re-encoded for phone players, or ``None`` if that is impossible."""
    if not isinstance(data, bytes) or len(data) < 32:
        return None
    if is_phone_ready(data):
        return data
    encoder = ffmpeg_path()
    if not encoder:
        log.info("No ffmpeg on this hub: alert videos stay in %s and phone apps "
                 "may refuse them", "mp4v")
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="rowan-transcode-") as folder:
            source = Path(folder) / "in.mp4"
            target = Path(folder) / "out.mp4"
            source.write_bytes(data)
            completed = subprocess.run(
                [encoder, "-y", "-loglevel", "error", "-i", str(source), *_ARGS, str(target)],
                capture_output=True, timeout=max(5.0, float(timeout_s)), check=False)
            if completed.returncode != 0 or not target.is_file():
                log.warning("ffmpeg could not convert an alert video (exit %s): %s",
                            completed.returncode,
                            completed.stderr.decode("utf-8", "replace").strip()[:200])
                return None
            converted = target.read_bytes()
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("ffmpeg could not convert an alert video (%s)", type(exc).__name__)
        return None
    if len(converted) < 32 or not is_phone_ready(converted):
        log.warning("ffmpeg returned a file without an H.264 track")
        return None
    return converted
