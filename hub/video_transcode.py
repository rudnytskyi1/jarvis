"""Make an alert video playable on a phone, not only on a desktop player.

The room client records through OpenCV's ``mp4v`` writer, which produces
MPEG-4 Part 2 (``mp4v``). Desktop Telegram plays that; the phone apps do not -
the message arrives and cannot be opened. The hub is the only place that has
an encoder, so a clip is converted to H.264 (``avc1``) with the ``moov`` atom
at the front (``+faststart``) right before it is sent.

Two encoders, in this order (2026-09-23):

* **ffmpeg on PATH** - it is the only one that can write ``+faststart`` (the
  ``moov`` atom in front, which is what lets a phone stream the file).
* **PyAV in-process** (``av``, libx264) - no child process and no temp file, so
  it still works when spawning ``ffmpeg.exe`` is refused. That refusal was the
  live failure: ``subprocess.run`` came back as ``PermissionError`` and every
  alert went out in ``mp4v`` («опять видео в телеге на мобилке не грузятся»).
  PyAV cannot write faststart into a pipe, so its MP4 keeps ``moov`` at the
  end: still H.264 and still playable on a phone, just downloaded first.

Which one produced the bytes matters for the Telegram call, so
:func:`is_stream_ready` says whether the metadata is in front.

Every step is best-effort: a room must still get its recording when ffmpeg is
missing, busy or fed a clip it does not like. In that case the original bytes
come back and the caller says so, instead of losing the alert.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

#: Where the ffmpeg fallback writes its scratch files. ``%TEMP%`` is where the
#: old code failed with ``WinError 5`` on cleanup, so the hub keeps its own
#: directory inside the working tree, which it definitely owns.
SCRATCH_DIR_NAME = "rowan-transcode"

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


def _av_available() -> bool:
    """Есть ли второй энкодер (PyAV с libx264) — без импорта в горячем пути."""
    return importlib.util.find_spec("av") is not None


def is_phone_ready(data: bytes) -> bool:
    """True when this MP4 already carries an H.264 track.

    MP4 stores the sample description inside ``stsd``: the four-character code
    (``avc1``) is enough to tell H.264 from the MPEG-4 Part 2 the room client
    writes, without decoding anything.
    """
    return isinstance(data, bytes) and b"avc1" in data[:4096]


def is_stream_ready(data: bytes) -> bool:
    """True when ``moov`` comes before ``mdat``: the phone may stream it."""
    if not isinstance(data, bytes) or len(data) < 16:
        return False
    moov, mdat = data.find(b"moov"), data.find(b"mdat")
    return 0 <= moov < mdat


def _encode_with_av(data: bytes) -> bytes | None:
    """H.264 MP4 through PyAV, in this process: no subprocess, no temp file.

    Returns ``None`` when PyAV is not installed or cannot make sense of the
    clip; the caller then tries ffmpeg. Video only: alert clips carry no audio,
    and ``yuv420p`` + ``main`` profile is what both phone platforms accept.

    ``movflags`` is deliberately NOT passed: PyAV writes ``+faststart`` into a
    plain pipe as ``ArgumentError: Invalid argument: '<none>' returned 22``
    (measured 2026-09-23), and an H.264 clip with the metadata at the end is
    still a clip every phone opens. That is why ffmpeg is tried first.
    """
    if not _av_available():
        return None
    import av
    try:
        with av.open(io.BytesIO(data), mode="r") as source:
            if not source.streams.video:
                return None
            input_stream = source.streams.video[0]
            rate = input_stream.average_rate or input_stream.base_rate
            stream = io.BytesIO()
            with av.open(stream, mode="w", format="mp4") as output:
                encoded = output.add_stream("libx264", rate=rate)
                encoded.width = input_stream.codec_context.width
                encoded.height = input_stream.codec_context.height
                encoded.pix_fmt = "yuv420p"
                encoded.options = {"preset": "veryfast", "crf": "26", "profile": "main"}
                for frame in source.decode(input_stream):
                    for packet in encoded.encode(frame):
                        output.mux(packet)
                for packet in encoded.encode(None):
                    output.mux(packet)
            converted = stream.getvalue()
    except Exception as exc:  # noqa: BLE001 - PyAV is an optional shortcut
        log.info("PyAV could not convert an alert video (%s: %s)",
                 type(exc).__name__, exc)
        return None
    if len(converted) < 32 or not is_phone_ready(converted):
        log.info("PyAV returned a file without an H.264 track")
        return None
    return converted


def _encode_with_ffmpeg(data: bytes, timeout_s: float) -> bytes | None:
    """The external-encoder path: same settings, scratch files in the hub tree."""
    encoder = ffmpeg_path()
    if not encoder:
        log.info("No ffmpeg on this hub: alert videos stay in %s and phone apps "
                 "may refuse them", "mp4v")
        return None
    folder = None
    try:
        scratch = Path(tempfile.gettempdir()) / SCRATCH_DIR_NAME
        scratch.mkdir(parents=True, exist_ok=True)
        folder = tempfile.mkdtemp(prefix="clip-", dir=str(scratch))
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
        log.warning("ffmpeg could not convert an alert video (%s: %s)",
                    type(exc).__name__, exc)
        return None
    finally:
        # Чистка не решает дело: клип уже в памяти. Прежний код терял клип
        # именно здесь - ``rmtree``/``chmod`` на каталоге в ``%TEMP%`` отдавал
        # WinError 5, исключение уходило наверх и перекодированный клип
        # выбрасывался, а в Телеграм уходил неиграемый ``mp4v``.
        if folder:
            try:
                shutil.rmtree(folder, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 - never lose the clip
                log.debug("Could not remove the transcode scratch dir (%s)", exc)
    if len(converted) < 32 or not is_phone_ready(converted):
        log.warning("ffmpeg returned a file without an H.264 track")
        return None
    return converted


def phone_ready_mp4(data: bytes, *, timeout_s: float = 90.0) -> bytes | None:
    """``data`` re-encoded for phone players, or ``None`` if that is impossible."""
    if not isinstance(data, bytes) or len(data) < 32:
        return None
    if is_phone_ready(data):
        return data
    # ffmpeg first: only it writes +faststart. PyAV second: it keeps working
    # when the hub is not allowed to spawn a child process at all, which is
    # exactly what was happening live (every alert left as unplayable mp4v).
    converted = _encode_with_ffmpeg(data, timeout_s)
    if converted is None:
        converted = _encode_with_av(data)
    return converted
