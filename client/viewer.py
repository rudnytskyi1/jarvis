"""Detections-photo viewer for the room TV (SPEC v1.6).

Whenever the server's ``find_object`` tool succeeds it draws the boxes on the
pulled frame and pushes it as ``image_show`` (a JSON header, then one binary
JPEG — SPEC §4). This module owns showing that photo on the room screen:

* a borderless, always-on-top OpenCV window, driven from ONE dedicated
  thread — ``cv2``'s HighGUI event loop (``imshow``/``waitKey``) is not safe
  to call from more than one thread, and must never be driven from asyncio;
* it auto-closes after ``ttl_s`` seconds (default :data:`DEFAULT_TTL_S`);
* a newer image simply replaces whatever is currently showing;
* when ``cv2`` is not installed, the photo is instead written to
  ``data/last_detections.jpg`` and opened with the default viewer
  (``os.startfile``) — the camera stack already ships ``cv2`` per SPEC, so
  this is a defensive fallback, not the expected path.

Everything here is best-effort: a failure logs a warning and is otherwise
swallowed — the voice pipeline must never notice that a photo could not be
shown.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: How long a shown photo stays up when the caller passes no/invalid ttl_s.
DEFAULT_TTL_S = 60.0
#: Title of the OpenCV window (also used to find its HWND for the
#: borderless/topmost styling below).
WINDOW_NAME = "Jarvis - detections"
#: Fallback when cv2 is unavailable (SPEC v1.6).
FALLBACK_IMAGE_PATH = REPO_ROOT / "data" / "last_detections.jpg"
#: How often the viewer thread wakes to pump the window and check the ttl.
POLL_S = 0.2
#: How long :meth:`ImageViewer.close` waits for the thread to stop.
JOIN_TIMEOUT_S = 2.0

# -- best-effort Windows borderless/always-on-top styling (ctypes) ---------
_GWL_STYLE = -16
_WS_CAPTION = 0x00C00000
_WS_THICKFRAME = 0x00040000
_WS_SYSMENU = 0x00080000
_WS_MINIMIZEBOX = 0x00020000
_WS_MAXIMIZEBOX = 0x00010000
_HWND_TOPMOST = -1
_SWP_NOMOVE = 0x0002
_SWP_NOSIZE = 0x0001
_SWP_FRAMECHANGED = 0x0020
_SWP_SHOWWINDOW = 0x0040

__all__ = [
    "DEFAULT_TTL_S",
    "WINDOW_NAME",
    "FALLBACK_IMAGE_PATH",
    "ImageViewer",
]


class _ShowRequest:
    """One photo handed to the viewer thread."""

    __slots__ = ("jpeg", "title", "ttl_s")

    def __init__(self, jpeg: bytes, title: str, ttl_s: float) -> None:
        self.jpeg = jpeg
        self.title = title
        self.ttl_s = ttl_s


#: Sentinel telling the viewer thread to stop.
_STOP = object()


def _make_borderless_topmost(window_name: str) -> None:
    """Best-effort: strip the title bar/border and pin the window on top.

    ``cv2``'s HighGUI has no "borderless" flag of its own, so this reaches for
    the window by its exact title through ``user32`` (Windows-only, matching
    the rest of this project's ctypes usage in ``client/actions/pc.py``).
    Failures here are purely cosmetic and must never affect whether the photo
    is actually shown.
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return
        style = user32.GetWindowLongW(hwnd, _GWL_STYLE)
        style &= ~(_WS_CAPTION | _WS_THICKFRAME | _WS_SYSMENU | _WS_MINIMIZEBOX | _WS_MAXIMIZEBOX)
        user32.SetWindowLongW(hwnd, _GWL_STYLE, style)
        user32.SetWindowPos(
            hwnd,
            _HWND_TOPMOST,
            0,
            0,
            0,
            0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_FRAMECHANGED | _SWP_SHOWWINDOW,
        )
    except Exception:  # noqa: BLE001 - cosmetic only
        log.debug("Could not make the detections window borderless/topmost", exc_info=True)


class ImageViewer:
    """Shows the latest ``find_object`` detections photo (SPEC v1.6).

    ``show()`` may be called from any thread (the client hands it the raw
    JPEG straight off the reader task); it only enqueues the request and
    starts the dedicated window thread if needed, so it is cheap and never
    touches ``cv2`` itself off that one thread.
    """

    def __init__(self) -> None:
        self._cv2: Any = None
        self._cv2_checked = False
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # cv2 (lazy, like client/camera.py)
    # ------------------------------------------------------------------
    def _import_cv2(self) -> Any:
        if self._cv2_checked:
            return self._cv2
        self._cv2_checked = True
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except Exception as exc:  # noqa: BLE001 - any half-installed OpenCV too
            log.info(
                "OpenCV is not available for the detections viewer (%s) - "
                "falling back to the default image viewer",
                exc,
            )
            self._cv2 = None
        return self._cv2

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def show(self, jpeg: bytes, title: str = "", ttl_s: float = DEFAULT_TTL_S) -> None:
        """Show ``jpeg`` on the room screen, replacing whatever is showing.

        Never raises: any failure is logged and swallowed, so a broken viewer
        can never take down the caller (the client's socket reader).
        """
        if not jpeg:
            return
        try:
            ttl = float(ttl_s) if ttl_s and float(ttl_s) > 0 else DEFAULT_TTL_S
        except (TypeError, ValueError):
            ttl = DEFAULT_TTL_S
        cv2 = self._import_cv2()
        if cv2 is None:
            self._show_fallback(jpeg)
            return
        try:
            self._ensure_thread()
            self._queue.put(_ShowRequest(jpeg, str(title or "Jarvis"), ttl))
        except Exception:  # noqa: BLE001 - never break the caller
            log.exception("Could not queue the detections photo for display")

    def close(self) -> None:
        """Stop the viewer thread and close its window (safe to call twice)."""
        try:
            self._queue.put_nowait(_STOP)
        except Exception:  # pragma: no cover - unbounded queue cannot fail
            pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=JOIN_TIMEOUT_S)
        self._thread = None

    # ------------------------------------------------------------------
    # fallback: no cv2 (SPEC v1.6)
    # ------------------------------------------------------------------
    def _show_fallback(self, jpeg: bytes) -> None:
        try:
            FALLBACK_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FALLBACK_IMAGE_PATH.write_bytes(jpeg)
            os.startfile(str(FALLBACK_IMAGE_PATH))  # noqa: S606 - Windows client, by design
            log.info("Showing the detections photo via %s (no OpenCV)", FALLBACK_IMAGE_PATH)
        except Exception as exc:  # noqa: BLE001 - never break the caller
            log.warning("Could not show the detections photo: %s", exc)

    # ------------------------------------------------------------------
    # the dedicated cv2 thread
    # ------------------------------------------------------------------
    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="jarvis-image-viewer", daemon=True
            )
            self._thread.start()

    def _decode(self, cv2: Any, jpeg: bytes) -> Any:
        try:
            import numpy as np

            buffer = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception:
            log.exception("Could not decode a detections photo")
            return None
        if frame is None or getattr(frame, "size", 0) == 0:
            log.warning("Could not decode a detections photo (%d bytes)", len(jpeg))
            return None
        return frame

    def _run(self) -> None:
        """The window/message-pump loop — the ONLY thread that touches cv2's UI."""
        cv2 = self._cv2
        shown_deadline: Optional[float] = None
        try:
            while True:
                timeout = POLL_S if shown_deadline is not None else None
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    item = None

                if item is _STOP:
                    break
                if isinstance(item, _ShowRequest):
                    frame = self._decode(cv2, item.jpeg)
                    if frame is not None:
                        try:
                            height, width = frame.shape[0], frame.shape[1]
                            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                            cv2.imshow(WINDOW_NAME, frame)
                            cv2.waitKey(1)
                            cv2.resizeWindow(WINDOW_NAME, max(1, int(width)), max(1, int(height)))
                            _make_borderless_topmost(WINDOW_NAME)
                            cv2.waitKey(1)
                        except Exception:
                            log.exception("Could not display the detections photo")
                        else:
                            shown_deadline = time.monotonic() + item.ttl_s
                            log.info(
                                "Showing the detections photo (%r) for %.0f s",
                                item.title, item.ttl_s,
                            )

                if shown_deadline is not None:
                    try:
                        cv2.waitKey(1)  # pump the window's message loop
                    except Exception:
                        pass
                    if time.monotonic() >= shown_deadline:
                        try:
                            cv2.destroyWindow(WINDOW_NAME)
                        except Exception:
                            pass
                        shown_deadline = None
        except Exception:  # noqa: BLE001 - the thread must never crash silently
            log.exception("The detections viewer thread crashed")
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
