"""Modern HUD overlay for the room TV (self-contained, OPTIONAL).

A transparent, borderless, always-on-top, click-through full-screen window
that gives the room TV a subtle "presence" for the assistant: a single soft
luminous bloom, centered on screen, that stays completely hidden while the
assistant is idle and only fades in while something is actually happening -
listening, thinking, speaking, a click ping, a screen scan, the typing dots,
a status caption, or a quick wake/error flash. When the activity ends it
fades back out and disappears again.

Everything is drawn on a plain :class:`tkinter.Canvas` (no external image
assets) from ONE dedicated UI thread — like :mod:`client.viewer`'s ``cv2``
window and :mod:`client.camera`'s capture/inference threads, Tkinter's own
event loop is not safe to drive from more than one thread and must never be
driven from asyncio. The public methods below only enqueue a small event and
return immediately; the UI thread drains that queue on its own ~30 FPS timer
and is the only thing that ever touches ``tkinter`` or ``ctypes``.

Click-through, on Windows, is done with a best-effort ``ctypes`` call that
sets ``WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE`` on the window,
so mouse clicks fall straight through to whatever is running underneath and
the overlay can never steal focus or block the desktop. The fade in/out is
the Tk ``-alpha`` window attribute (combined with ``-transparentcolor`` for
the click-through chroma key), eased over ~180 ms in and ~260 ms out; at
rest the window is fully ``withdraw()``-n, not just blank, so nothing is on
screen at all until something happens.

Everything here is OPTIONAL and best-effort: if the window cannot be created
for ANY reason (no display, headless CI, Tk not installed, the ``ctypes``
styling call failing) exactly ONE warning is logged, :attr:`OverlayHUD.enabled`
flips to ``False``, and every public method silently becomes a no-op — the
voice pipeline must never notice or care that the HUD could not be shown.

Wiring (a human/another worker wires this into :mod:`client.main`)::

    overlay = OverlayHUD(_attr(cfg.client, "overlay"))
    overlay.start()
    ...
    overlay.set_state("listening")
    overlay.click_at(0.42, 0.61)
    ...
    overlay.stop()

Demo, run directly on the room PC to eyeball it on the TV without the rest
of the system::

    python -m client.overlay
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Mapping
from typing import Any, List, Optional, Tuple

log = logging.getLogger(__name__)

# --- palette (soft luminous bloom: near-white core -> indigo/violet -> bg) ---
BLOOM_CORE = "#F4F1FF"
BLOOM_MID = "#8B7DFF"
BLOOM_ERROR_MID = "#FF6B6B"
TEXT_COLOR = "#D8D4F0"

# --- state machine ------------------------------------------------------------
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"
VALID_STATES = (STATE_IDLE, STATE_LISTENING, STATE_THINKING, STATE_SPEAKING)

# --- flash kinds ---------------------------------------------------------------
FLASH_WAKE = "wake"
FLASH_ERROR = "error"
_FLASH_KINDS = (FLASH_WAKE, FLASH_ERROR)

# --- config defaults -----------------------------------------------------------
DEFAULT_POSITION = "bottom_right"
_POSITIONS = ("bottom_right", "bottom_left", "top_right", "top_left", "center")
DEFAULT_CHROMA = "#010101"

# --- layout / timing tuning -----------------------------------------------------
BASE_BLOOM_R = 130.0  # ~260px diameter at scale 1.0
BLOOM_STEPS = 48  # concentric ovals in the precomputed colour ramp
TICK_MS = 33  # ~30 FPS
START_TIMEOUT_S = 5.0
JOIN_TIMEOUT_S = 2.0
QUEUE_MAXSIZE = 256
STATUS_MAX_CHARS = 60
TRAIL_STEPS = 8
FADE_IN_S = 0.18
FADE_OUT_S = 0.26
_ALPHA_EPS = 0.004
CLICK_TOTAL_S = 0.45
CLICK_CONVERGE_S = CLICK_TOTAL_S  # kept as a separate name for API stability
FLASH_DURATION_S = 0.45
LISTEN_PERIOD_S = 2.5
SPEAK_PERIOD_S = 1.1
THINK_CRESCENT_PERIOD_S = 4.5
SCAN_PERIOD_S = 2.5
SCAN_BAND_H = 130.0
TYPING_PERIOD_S = 1.2

# --- best-effort Windows click-through styling (ctypes) -------------------------
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_NOACTIVATE = 0x08000000

__all__ = [
    "STATE_IDLE",
    "STATE_LISTENING",
    "STATE_THINKING",
    "STATE_SPEAKING",
    "VALID_STATES",
    "FLASH_WAKE",
    "FLASH_ERROR",
    "norm_to_px",
    "OverlayHUD",
]


class _StopRequested(Exception):
    """Internal signal: the UI thread saw :data:`_STOP` and should tear down."""


#: Sentinel telling the UI thread to stop (mirrors client.viewer's own _STOP).
_STOP = object()


# ------------------------------------------------------------------------------
# defensive config reading (same pattern as client.camera._attr et al.)
# ------------------------------------------------------------------------------
def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a pydantic model / dataclass / mapping / ``None``."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        return default
    try:
        return bool(value)
    except Exception:  # pragma: no cover - defensive
        return default


# ------------------------------------------------------------------------------
# pure helpers (importable and testable without ever opening a window)
# ------------------------------------------------------------------------------
def _validate_state(state: Any) -> str:
    """Normalise ``state``, raising :class:`ValueError` for anything unknown.

    Case-insensitive, strips whitespace. Never called on a hot path with
    anything the caller cannot afford to have rejected: :meth:`OverlayHUD.set_state`
    catches this and simply ignores an invalid state (never raises to the
    voice client), but the validation itself is a plain, pure function so it
    can be unit-tested on its own.
    """
    if not isinstance(state, str):
        raise ValueError(f"overlay state must be a string, got {state!r}")
    normalized = state.strip().lower()
    if normalized not in VALID_STATES:
        raise ValueError(
            f"unknown overlay state {state!r} (expected one of {VALID_STATES})"
        )
    return normalized


def _truncate_status(text: Any, limit: int = STATUS_MAX_CHARS) -> str:
    """Trim a status caption to ``limit`` characters, adding an ellipsis.

    ``None`` becomes ``""`` (clears the caption, per the public API). Never
    raises: anything is coerced through ``str()`` first.
    """
    value = "" if text is None else str(text).strip()
    limit = max(0, int(limit))
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1].rstrip() + "…"


def _clamp01(value: Any) -> float:
    number = float(value)
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def norm_to_px(x_norm: Any, y_norm: Any, width: int, height: int) -> Tuple[int, int]:
    """Map normalized ``(0..1, 0..1)`` screen coordinates to clamped pixel ints.

    Out-of-range values are clamped rather than rejected (a slightly
    over/under-shooting vision-model coordinate is common and should still
    animate the click ring at the nearest edge, not raise).
    """
    x = _clamp01(x_norm)
    y = _clamp01(y_norm)
    px = int(round(x * max(0, int(width) - 1)))
    py = int(round(y * max(0, int(height) - 1)))
    return px, py


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _reticle_trail(
    start_px: Tuple[float, float], end_px: Tuple[float, float], steps: int = TRAIL_STEPS
):
    """``steps + 1`` points interpolated from ``start_px`` to ``end_px``.

    Pure geometry helper for a chain of consecutive :meth:`OverlayHUD.click_at`
    targets. Kept as a small, independently-testable utility even though the
    current click visual (a single contracting ring) no longer renders a
    dotted trail between points. Deterministic: the first point is always
    ``start_px`` and the last is always ``end_px``.
    """
    steps = max(1, int(steps))
    sx, sy = start_px
    ex, ey = end_px
    return [(_lerp(sx, ex, i / steps), _lerp(sy, ey, i / steps)) for i in range(steps + 1)]


def _anchor_point(position: str, width: int, height: int, margin: float) -> Tuple[float, float]:
    """Where a ``position`` config value points on screen.

    The bloom itself is always centered now (see :meth:`OverlayHUD._redraw`);
    this pure helper is kept for API/test stability and is available for a
    future "shift the status caption toward a corner" tweak. Unknown
    positions fall back to :data:`DEFAULT_POSITION` rather than raising,
    matching this module's "never break the caller" philosophy.
    """
    pos = position if position in _POSITIONS else DEFAULT_POSITION
    if pos == "bottom_right":
        return width - margin, height - margin
    if pos == "bottom_left":
        return margin, height - margin
    if pos == "top_right":
        return width - margin, margin
    if pos == "top_left":
        return margin, margin
    return width / 2.0, height / 2.0


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    text = str(color).lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {color!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, int(round(v)))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Interpolate two ``#rrggbb`` colours; ``t=0`` -> ``c1``, ``t=1`` -> ``c2``."""
    t = _clamp01(t)
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((_lerp(r1, r2, t), _lerp(g1, g2, t), _lerp(b1, b2, t)))


def _build_bloom_ramp(core: str, mid: str, edge: str, steps: int = BLOOM_STEPS) -> List[str]:
    """Precompute a ``steps``-long colour ramp: ``core`` -> ``mid`` -> ``edge``.

    Drives the "soft bloom" glow: ~48 concentric filled ovals, each one
    index further out, painted with the matching ramp colour so the whole
    thing reads as a genuine soft blur with no visible edge. Index ``0`` is
    the brightest core colour; the last index is ``edge`` (normally the
    chroma background colour, so the outermost ovals fade to fully
    transparent via ``-transparentcolor``). Pure and deterministic, so the
    interpolation can be unit-tested without ever opening a window.
    """
    steps = max(2, int(steps))
    ramp: List[str] = []
    for i in range(steps):
        t = i / (steps - 1)
        if t <= 0.5:
            ramp.append(_lerp_color(core, mid, t / 0.5))
        else:
            ramp.append(_lerp_color(mid, edge, (t - 0.5) / 0.5))
    return ramp


def _ease_in_out(t: float) -> float:
    """Smoothstep ease curve, clamped to ``[0, 1]`` on both ends."""
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def _fade_alpha(elapsed_s: float, duration_s: float, start_alpha: float, end_alpha: float) -> float:
    """The window ``-alpha`` value ``elapsed_s`` into an eased fade of ``duration_s``."""
    if duration_s <= 0:
        return end_alpha
    return _lerp(start_alpha, end_alpha, _ease_in_out(elapsed_s / duration_s))


def _is_active(
    state: str,
    status: str,
    scanning: bool,
    typing_on: bool,
    flash_active: bool,
    click_active: bool,
) -> bool:
    """Hidden-at-idle decision: is there anything worth showing right now?

    ``True`` whenever the assistant is doing something - a non-idle state
    (listening/thinking/speaking), a screen scan, the typing dots, a flash,
    a click ping, or a status caption. ``False`` only when all of those are
    quiet, which is when the overlay is fully withdrawn. This is
    unconditional: :attr:`OverlayHUD.idle_hidden` is accepted for backwards
    compatibility but no longer changes this decision.
    """
    if state != STATE_IDLE:
        return True
    return bool(scanning or typing_on or flash_active or click_active or status)


class OverlayHUD:
    """Transparent click-through HUD - a single centered soft bloom, hidden at rest.

    Construction never fails and never touches ``tkinter``/``ctypes`` — only
    :meth:`start` does, off on its own thread. Every public method is safe to
    call from any thread (the voice client's asyncio loop, a worker thread,
    ...): each one just validates its argument defensively and posts a small
    event onto a queue the UI thread drains on its own timer, so none of them
    ever block.
    """

    def __init__(self, cfg: Any = None) -> None:
        self.enabled = _as_bool(_attr(cfg, "enabled", True), True)
        position = str(_attr(cfg, "position", DEFAULT_POSITION) or DEFAULT_POSITION).strip().lower()
        self.position = position if position in _POSITIONS else DEFAULT_POSITION
        # Vestigial: hiding at idle is unconditional now. Still accepted (and
        # stored) so existing configs/callers do not break.
        self.idle_hidden = _as_bool(_attr(cfg, "idle_hidden", False), False)
        scale = _as_float(_attr(cfg, "scale", 1.0), 1.0)
        self.scale = scale if scale > 0 else 1.0
        self.chroma = str(_attr(cfg, "chroma", DEFAULT_CHROMA) or DEFAULT_CHROMA)

        # Colour ramps for the bloom are pure functions of (core, mid, edge)
        # and never change after construction, so precompute them once.
        # ``chroma`` is normally a "#rrggbb" hex string, but construction
        # must never fail even if a config hands us something odd (a named
        # Tk colour, garbage, ...) - fall back to the default hex for the
        # ramp's edge colour in that case; ``self.chroma`` itself (used for
        # the actual window/-transparentcolor) is left untouched.
        try:
            _ramp_edge = self.chroma if len(str(self.chroma).lstrip("#")) == 6 else DEFAULT_CHROMA
            self._ramp_normal = _build_bloom_ramp(BLOOM_CORE, BLOOM_MID, _ramp_edge, BLOOM_STEPS)
            self._ramp_error = _build_bloom_ramp(BLOOM_CORE, BLOOM_ERROR_MID, _ramp_edge, BLOOM_STEPS)
        except (ValueError, TypeError):  # pragma: no cover - defensive, chroma is normally valid hex
            self._ramp_normal = _build_bloom_ramp(BLOOM_CORE, BLOOM_MID, DEFAULT_CHROMA, BLOOM_STEPS)
            self._ramp_error = _build_bloom_ramp(BLOOM_CORE, BLOOM_ERROR_MID, DEFAULT_CHROMA, BLOOM_STEPS)

        self._lock = threading.Lock()
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._warned = False

        # -- UI-thread-only state: written by _apply_event, read by _redraw --
        # (both run inside the same Tk `after` callback chain on the UI
        # thread, so nothing here needs a lock; only __init__, running before
        # that thread exists, touches it from the outside.)
        self._root: Any = None
        self._canvas: Any = None
        self._canvas_w = 0
        self._canvas_h = 0
        self._state = STATE_IDLE
        self._status = ""
        self._scanning = False
        self._typing_on = False
        self._flash: Optional[Tuple[str, float]] = None
        self._click_anim: Optional[dict] = None
        self._last_click_px: Optional[Tuple[float, float]] = None
        self._bloom_item_ids: Optional[List[Any]] = None
        self._mapped = False
        self._alpha = 0.0
        self._fade_target = 0.0
        self._fade_from = 0.0
        self._fade_t0 = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the UI thread and open the overlay window (no-op if disabled).

        Blocks briefly (at most :data:`START_TIMEOUT_S`) for the window to
        either come up or fail, so callers can check :attr:`enabled` right
        after this returns. Any failure - no display, Tk missing, the
        ``ctypes`` styling call failing - logs exactly ONE warning and flips
        :attr:`enabled` to ``False``; every other method then becomes a no-op.
        The window starts fully hidden (``withdraw()``-n): nothing appears
        on screen until an actual event (state/status/scan/click/typing/flash)
        makes it fade in.
        """
        if not self.enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready_event.clear()
            self._thread = threading.Thread(
                target=self._run_ui, name="jarvis-overlay-ui", daemon=True
            )
            self._thread.start()
        if not self._ready_event.wait(timeout=START_TIMEOUT_S):
            self._disable("the overlay window did not start in time")

    def stop(self) -> None:
        """Stop the UI thread and close the window (safe to call twice, or never-started)."""
        if not self.enabled:
            return
        with self._lock:
            thread = self._thread
            self._thread = None
            if thread is None or not thread.is_alive():
                return
            try:
                self._queue.put_nowait(_STOP)
            except Exception:  # pragma: no cover - the queue cannot be full of nothing
                pass
        thread.join(timeout=JOIN_TIMEOUT_S)

    def _disable(self, reason: str) -> None:
        """Flip :attr:`enabled` off, logging exactly one warning (like client.camera._fail)."""
        self.enabled = False
        if not self._warned:
            self._warned = True
            log.warning("Overlay HUD disabled: %s. The voice client keeps working normally.", reason)
        else:  # pragma: no cover - a second failure after the first warning
            log.debug("Overlay HUD failure after it was already disabled: %s", reason)

    # ------------------------------------------------------------------
    # public API - thread-safe, never blocks, never raises
    # ------------------------------------------------------------------
    def set_state(self, state: Any) -> None:
        """Switch the bloom's animation: one of :data:`VALID_STATES`.

        An unknown state is logged at DEBUG and ignored - never raised to the
        caller (see :func:`_validate_state` for the pure validation logic).
        """
        if not self.enabled:
            return
        try:
            normalized = _validate_state(state)
        except ValueError:
            log.debug("Ignoring unknown overlay state %r", state)
            return
        self._post(("state", normalized))

    def set_status(self, text: Any) -> None:
        """Short caption shown just below the bloom; ``""`` (or ``None``) clears it."""
        if not self.enabled:
            return
        self._post(("status", _truncate_status(text)))

    def scan_screen(self, on: bool = True) -> None:
        """Toggle the soft screen-scan sweep (while a screenshot/vision call is in flight)."""
        if not self.enabled:
            return
        self._post(("scan", bool(on)))

    def click_at(self, x_norm: Any, y_norm: Any) -> None:
        """Animate a soft ring contracting onto ``(x_norm, y_norm)`` and fading out.

        Coordinates are normalized 0..1 (clamped, never raise on an
        out-of-range or slightly malformed value); non-numeric input is
        logged at DEBUG and dropped.
        """
        if not self.enabled:
            return
        try:
            x = _clamp01(x_norm)
            y = _clamp01(y_norm)
        except (TypeError, ValueError):
            log.debug("Ignoring click_at with non-numeric coordinates: %r, %r", x_norm, y_norm)
            return
        self._post(("click", x, y))

    def typing(self, on: bool = True) -> None:
        """Toggle the three small pulsing "typing" dots below the bloom."""
        if not self.enabled:
            return
        self._post(("typing", bool(on)))

    def flash(self, kind: str) -> None:
        """A quick accent flash: ``"wake"`` (bright bloom expansion) or ``"error"`` (soft red)."""
        if not self.enabled:
            return
        normalized = str(kind or "").strip().lower()
        if normalized not in _FLASH_KINDS:
            log.debug("Ignoring unknown overlay flash kind %r", kind)
            return
        self._post(("flash", normalized))

    def _post(self, event: Any) -> None:
        """Enqueue one event for the UI thread; never blocks, never raises."""
        try:
            self._queue.put_nowait(event)
        except Exception:  # noqa: BLE001 - a full/closed queue must not break the caller
            log.debug("Dropping an overlay event - the queue is full")

    # ------------------------------------------------------------------
    # the dedicated UI thread
    # ------------------------------------------------------------------
    def _run_ui(self) -> None:
        """Thread entry point: build the window, then run Tk's own event loop."""
        try:
            self._make_tk_root()
        except Exception as exc:  # noqa: BLE001 - no display / Tk missing / ctypes failure
            self._disable(f"could not create the overlay window ({exc})")
            self._ready_event.set()
            return
        self._ready_event.set()
        try:
            self._root.after(TICK_MS, self._tick)
            self._root.mainloop()
        except Exception as exc:  # noqa: BLE001 - the UI thread must never crash the process
            self._disable(f"the overlay UI loop failed ({exc})")
        finally:
            self._teardown_tk()

    def _make_tk_root(self) -> None:
        """Build the fullscreen, borderless, transparent, click-through window.

        Everything here is guarded as ONE unit by the caller (:meth:`_run_ui`):
        a Tk root that cannot go borderless/transparent, or a ctypes call that
        cannot make it click-through, is just as useless to us as no display
        at all, so any failure anywhere in this method disables the whole HUD
        rather than showing a half-working (possibly click-blocking) window.

        The window is built at full size and then immediately withdrawn:
        hidden-at-idle is the resting state, and it only ever deiconifies
        from :meth:`_redraw` once something is actually happening.
        """
        import tkinter as tk

        root = tk.Tk()
        try:
            width = int(root.winfo_screenwidth())
            height = int(root.winfo_screenheight())
            root.geometry(f"{width}x{height}+0+0")
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-transparentcolor", self.chroma)
            root.attributes("-alpha", 0.0)
            root.config(bg=self.chroma)
            root.resizable(False, False)
            canvas = tk.Canvas(
                root, width=width, height=height, bg=self.chroma,
                highlightthickness=0, bd=0,
            )
            canvas.pack(fill="both", expand=True)
            root.update_idletasks()
            self._apply_click_through(root)
            root.withdraw()
        except Exception:
            try:
                root.destroy()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
            raise

        self._root = root
        self._canvas = canvas
        self._canvas_w = width
        self._canvas_h = height
        self._last_click_px = None
        self._bloom_item_ids = None
        self._mapped = False
        self._alpha = 0.0
        self._fade_target = 0.0
        self._fade_from = 0.0
        self._fade_t0 = 0.0

    @staticmethod
    def _apply_click_through(root: Any) -> None:
        """Best-effort ``ctypes``: WS_EX_TRANSPARENT|LAYERED|NOACTIVATE (Windows)."""
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = root.winfo_id()
        parent = user32.GetParent(hwnd)
        if parent:
            hwnd = parent
        ex_style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd, _GWL_EXSTYLE, ex_style | _WS_EX_LAYERED | _WS_EX_TRANSPARENT | _WS_EX_NOACTIVATE
        )

    # -- the ~30 FPS timer: drain events, then redraw -----------------------
    def _tick(self) -> None:
        try:
            if self._drain_queue():
                raise _StopRequested()
            self._redraw(time.monotonic())
            self._root.after(TICK_MS, self._tick)
        except _StopRequested:
            self._teardown_tk()
        except Exception as exc:  # noqa: BLE001 - the UI thread must never crash the process
            self._disable(f"the overlay UI loop failed ({exc})")
            self._teardown_tk()

    def _drain_queue(self) -> bool:
        """Apply every pending event; returns ``True`` if :data:`_STOP` was seen."""
        stopping = False
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            if event is _STOP:
                stopping = True
                continue
            self._apply_event(event)
        return stopping

    def _apply_event(self, event: Any) -> None:
        tag = event[0]
        if tag == "state":
            self._state = event[1]
        elif tag == "status":
            self._status = event[1]
        elif tag == "scan":
            self._scanning = event[1]
        elif tag == "typing":
            self._typing_on = event[1]
        elif tag == "flash":
            self._flash = (event[1], time.monotonic())
        elif tag == "click":
            _, x, y = event
            end_px = norm_to_px(x, y, self._canvas_w, self._canvas_h)
            start_px = self._last_click_px or end_px
            self._click_anim = {"start": start_px, "end": end_px, "t0": time.monotonic()}
            self._last_click_px = end_px

    def _teardown_tk(self) -> None:
        root, self._root = self._root, None
        self._canvas = None
        self._bloom_item_ids = None
        self._mapped = False
        if root is None:
            return
        try:
            root.quit()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        try:
            root.destroy()
        except Exception:  # pragma: no cover - best-effort teardown
            pass

    # ------------------------------------------------------------------
    # drawing (Canvas 2D, no external assets) - UI thread only
    # ------------------------------------------------------------------
    def _redraw(self, now: float) -> None:
        canvas = self._canvas
        root = self._root
        if canvas is None or root is None:
            return

        flash_active = self._flash_active(now)
        click_active = self._click_active(now)
        active = _is_active(
            self._state, self._status, self._scanning, self._typing_on,
            flash_active, click_active,
        )
        target = 1.0 if active else 0.0
        if target != self._fade_target:
            self._fade_from = self._alpha
            self._fade_target = target
            self._fade_t0 = now
        duration = FADE_IN_S if target >= 1.0 else FADE_OUT_S
        alpha = _fade_alpha(now - self._fade_t0, duration, self._fade_from, target)
        self._alpha = alpha

        if alpha <= _ALPHA_EPS and target <= 0.0:
            if self._mapped:
                self._hide_window()
            return

        if not self._mapped:
            self._show_window()
        try:
            root.attributes("-alpha", max(0.0, min(1.0, alpha)))
        except Exception:  # pragma: no cover - best-effort, window may be closing
            pass

        width, height = self._canvas_w, self._canvas_h
        cx, cy = width / 2.0, height / 2.0
        radius = BASE_BLOOM_R * self.scale

        if self._scanning:
            self._draw_scan(canvas, width, height, now)
        else:
            canvas.delete("scan")

        self._draw_bloom(canvas, cx, cy, radius, now, flash_active)

        if self._status:
            self._draw_status(canvas, cx, cy, radius)
        else:
            canvas.delete("status")

        if self._typing_on:
            self._draw_typing(canvas, cx, cy, radius, now)
        else:
            canvas.delete("typing")

        if click_active:
            self._draw_click(canvas, now)
        else:
            canvas.delete("click")

    def _show_window(self) -> None:
        try:
            self._root.deiconify()
        except Exception:  # pragma: no cover - best-effort
            pass
        self._mapped = True

    def _hide_window(self) -> None:
        try:
            self._root.withdraw()
        except Exception:  # pragma: no cover - best-effort
            pass
        self._mapped = False

    # -- the bloom: ~48 reused concentric ovals, recoloured every frame -----
    def _ensure_bloom_items(self, c: Any) -> None:
        if self._bloom_item_ids is not None:
            return
        self._bloom_item_ids = [
            c.create_oval(0, 0, 0, 0, fill=self.chroma, outline="")
            for _ in range(BLOOM_STEPS)
        ]

    def _draw_bloom(self, c: Any, cx: float, cy: float, base_radius: float, now: float, flash_active: bool) -> None:
        state = self._state
        ramp = self._ramp_normal
        brightness = 1.0
        size_mult = 1.0
        crescent_phase: Optional[float] = None

        if state == STATE_LISTENING:
            wave = math.sin(2.0 * math.pi * (now % LISTEN_PERIOD_S) / LISTEN_PERIOD_S)
            size_mult = 1.0 + 0.05 * wave
            brightness = 1.05 + 0.05 * wave
        elif state == STATE_THINKING:
            brightness = 0.8
            crescent_phase = (now / THINK_CRESCENT_PERIOD_S) % 1.0
        elif state == STATE_SPEAKING:
            wave = math.sin(2.0 * math.pi * (now % SPEAK_PERIOD_S) / SPEAK_PERIOD_S)
            pulse = max(0.0, wave)
            size_mult = 1.0 + 0.04 * pulse
            brightness = 0.95 + 0.15 * pulse
        else:
            brightness = 0.72

        flash_boost = 0.0
        if flash_active and self._flash is not None:
            kind, t0 = self._flash
            progress = _clamp01((now - t0) / FLASH_DURATION_S)
            eased = _ease_in_out(progress)
            flash_boost = 1.0 - eased
            size_mult = max(size_mult, 1.0 + 0.9 * flash_boost)
            brightness = max(brightness, 1.0 + 0.5 * flash_boost)
            ramp = self._ramp_error if kind == FLASH_ERROR else self._ramp_normal

        radius = max(1.0, base_radius * size_mult)
        self._ensure_bloom_items(c)
        ids = self._bloom_item_ids or []
        n = len(ids)
        if n == 0:
            return
        core_color = ramp[0]
        for idx in range(n):
            step = n - 1 - idx
            frac = step / (n - 1)
            r = radius * frac
            color = ramp[step]
            if brightness > 1.0:
                color = _lerp_color(color, core_color, min(1.0, brightness - 1.0))
            elif brightness < 1.0:
                color = _lerp_color(color, self.chroma, (1.0 - brightness) * 0.6)
            item = ids[idx]
            if r < 0.5:
                c.coords(item, cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
            else:
                c.coords(item, cx - r, cy - r, cx + r, cy + r)
            c.itemconfig(item, fill=color)

        if crescent_phase is not None:
            self._draw_crescent(c, cx, cy, radius, crescent_phase)
        else:
            c.delete("crescent")

    def _draw_crescent(self, c: Any, cx: float, cy: float, radius: float, phase: float) -> None:
        """A softly brighter crescent drifting under the "thinking" bloom.

        A couple of large, low-contrast arcs blended into the ramp colour -
        deliberately NOT a thin spinner - so it reads as something slowly
        moving under frosted glass.
        """
        c.delete("crescent")
        mid_color = self._ramp_normal[len(self._ramp_normal) // 2]
        glow = _lerp_color(mid_color, BLOOM_CORE, 0.45)
        angle = phase * 360.0
        r1 = radius * 0.72
        w1 = max(6.0, radius * 0.55)
        c.create_arc(cx - r1, cy - r1, cx + r1, cy + r1, start=angle, extent=120,
                     style="arc", outline=glow, width=w1, tags="crescent")
        r2 = radius * 0.5
        w2 = max(4.0, radius * 0.35)
        c.create_arc(cx - r2, cy - r2, cx + r2, cy + r2, start=angle + 190, extent=80,
                     style="arc", outline=glow, width=w2, tags="crescent")

    def _draw_scan(self, c: Any, width: int, height: int, now: float) -> None:
        """A barely-there, feathered horizontal light band sweeping down once per cycle."""
        c.delete("scan")
        band_h = SCAN_BAND_H * self.scale
        phase = (now % SCAN_PERIOD_S) / SCAN_PERIOD_S
        y = -band_h + phase * (height + 2.0 * band_h)
        strips = 14
        for i in range(strips):
            t = i / (strips - 1)
            feather = max(0.0, 1.0 - abs(t - 0.5) * 2.0)
            color = _lerp_color(self.chroma, BLOOM_MID, feather * 0.35)
            strip_top = y - band_h / 2.0 + t * band_h
            strip_h = band_h / strips + 1.0
            c.create_rectangle(0, strip_top, width, strip_top + strip_h, fill=color, outline="", tags="scan")

    def _draw_status(self, c: Any, cx: float, cy: float, radius: float) -> None:
        c.delete("status")
        font = ("Segoe UI Light", max(9, int(round(13 * self.scale))))
        y = cy + radius + 26.0 * self.scale
        c.create_text(cx, y, text=self._status, anchor="n", fill=TEXT_COLOR, font=font, tags="status")

    def _draw_typing(self, c: Any, cx: float, cy: float, radius: float, now: float) -> None:
        c.delete("typing")
        y = cy + radius + 26.0 * self.scale + (24.0 * self.scale if self._status else 0.0)
        spacing = 14.0 * self.scale
        start_x = cx - spacing
        for i in range(3):
            phase = ((now / TYPING_PERIOD_S) + i * 0.28) % 1.0
            glow = 0.5 - 0.5 * math.cos(phase * 2.0 * math.pi)
            color = _lerp_color(self.chroma, TEXT_COLOR, 0.25 + 0.75 * glow)
            r = 3.0 * self.scale
            x = start_x + i * spacing
            c.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="", tags="typing")

    # -- click: one soft ring, contracts onto the point and fades out -------
    def _click_active(self, now: float) -> bool:
        anim = self._click_anim
        if anim is None:
            return False
        if now - anim["t0"] >= CLICK_TOTAL_S:
            self._click_anim = None
            return False
        return True

    def _draw_click(self, c: Any, now: float) -> None:
        c.delete("click")
        anim = self._click_anim
        if anim is None:  # pragma: no cover - guarded by _click_active already
            return
        t = now - anim["t0"]
        progress = _ease_in_out(t / CLICK_TOTAL_S)
        ex, ey = anim["end"]
        radius = _lerp(46.0 * self.scale, 3.0 * self.scale, progress)
        color = _lerp_color(BLOOM_CORE, self.chroma, progress)
        width = max(1.0, _lerp(2.5, 0.4, progress) * self.scale)
        c.create_oval(ex - radius, ey - radius, ex + radius, ey + radius,
                       outline=color, width=width, tags="click")

    # -- accent flash: state is read by _draw_bloom, this just tracks timing -
    def _flash_active(self, now: float) -> bool:
        flash = self._flash
        if flash is None:
            return False
        _, t0 = flash
        if now - t0 >= FLASH_DURATION_S:
            self._flash = None
            return False
        return True


# ------------------------------------------------------------------------------
# __main__ demo: cycle through every state/animation so it can be eyeballed
# on the TV without the rest of the system running.
# ------------------------------------------------------------------------------
def _run_demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    hud = OverlayHUD({"enabled": True})
    hud.start()
    if not hud.enabled:
        print("Overlay could not start (no display / Tk unavailable?) - demo aborted")
        return

    print(
        "Overlay demo running - watch the centre of the TV. It stays completely "
        "hidden until something happens, then fades in. Ctrl+C to stop early."
    )
    try:
        print("... hidden (idle, nothing on screen) ...")
        time.sleep(2.0)

        print("... wake flash ...")
        hud.flash(FLASH_WAKE)
        time.sleep(1.0)

        hud.set_state(STATE_LISTENING)
        hud.set_status("listening")
        time.sleep(2.5)

        hud.set_state(STATE_THINKING)
        hud.set_status("thinking")
        time.sleep(2.5)

        hud.set_status("looking at the screen")
        hud.scan_screen(True)
        time.sleep(2.5)
        hud.scan_screen(False)

        hud.set_status("clicking around")
        hud.click_at(0.3, 0.4)
        time.sleep(0.6)
        hud.click_at(0.7, 0.6)
        time.sleep(0.6)

        hud.typing(True)
        hud.set_status("typing")
        time.sleep(2.0)
        hud.typing(False)

        hud.set_state(STATE_SPEAKING)
        hud.set_status("speaking")
        time.sleep(3.0)

        print("... fading out and hidden again ...")
        hud.set_state(STATE_IDLE)
        hud.set_status("")
        time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        hud.stop()
        print("Overlay demo stopped")


if __name__ == "__main__":
    _run_demo()
