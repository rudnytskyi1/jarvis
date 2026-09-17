"""Sci-fi HUD overlay for the room TV (self-contained, OPTIONAL).

A transparent, borderless, always-on-top, click-through full-screen window
that gives the room TV a visible "presence" for the assistant: a glowing orb
that breathes at rest, reacts while listening/thinking/speaking, sweeps a
scan-line while the screen is being looked at, and flies a targeting reticle
to wherever ``mouse_click`` is about to land.

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
the overlay can never steal focus or block the desktop.

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
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)

# --- palette (sci-fi: cyan/teal, deep blue, soft white) ----------------------
COLOR_CYAN = "#16f2e0"
COLOR_DEEP_BLUE = "#0a1e3f"
COLOR_SOFT_WHITE = "#eaf6ff"
COLOR_RED = "#ff3b3b"

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
BASE_ORB_R = 42.0
BASE_MARGIN = 130.0
TICK_MS = 33  # ~30 FPS
START_TIMEOUT_S = 5.0
JOIN_TIMEOUT_S = 2.0
QUEUE_MAXSIZE = 256
STATUS_MAX_CHARS = 60
TRAIL_STEPS = 8
CLICK_CONVERGE_S = 0.35
CLICK_PING_S = 0.45
CLICK_TOTAL_S = CLICK_CONVERGE_S + CLICK_PING_S
FLASH_DURATION_S = 0.6

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
    animate a reticle at the nearest edge, not raise).
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

    Used to draw the brief cursor trail between two consecutive
    :meth:`OverlayHUD.click_at` targets, so a chain of clicks reads as the
    assistant "moving the mouse". Pure and deterministic: the first point is
    always ``start_px`` and the last is always ``end_px``.
    """
    steps = max(1, int(steps))
    sx, sy = start_px
    ex, ey = end_px
    return [(_lerp(sx, ex, i / steps), _lerp(sy, ey, i / steps)) for i in range(steps + 1)]


def _anchor_point(position: str, width: int, height: int, margin: float) -> Tuple[float, float]:
    """Where the orb's center sits for a given ``position`` config value.

    Unknown positions fall back to :data:`DEFAULT_POSITION` rather than
    raising, matching this module's "never break the caller" philosophy.
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


class OverlayHUD:
    """Transparent click-through sci-fi HUD, driven from one dedicated UI thread.

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
        self.idle_hidden = _as_bool(_attr(cfg, "idle_hidden", False), False)
        scale = _as_float(_attr(cfg, "scale", 1.0), 1.0)
        self.scale = scale if scale > 0 else 1.0
        self.chroma = str(_attr(cfg, "chroma", DEFAULT_CHROMA) or DEFAULT_CHROMA)

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
        """Switch the orb's animation: one of :data:`VALID_STATES`.

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
        """Short caption shown near the orb; ``""`` (or ``None``) clears it."""
        if not self.enabled:
            return
        self._post(("status", _truncate_status(text)))

    def scan_screen(self, on: bool = True) -> None:
        """Toggle the screen-scan sweep (while a screenshot/vision call is in flight)."""
        if not self.enabled:
            return
        self._post(("scan", bool(on)))

    def click_at(self, x_norm: Any, y_norm: Any) -> None:
        """Animate a targeting reticle flying to and pinging at ``(x_norm, y_norm)``.

        Coordinates are normalized 0..1 (clamped, never raise on an
        out-of-range or slightly malformed value); non-numeric input is
        logged at DEBUG and dropped. A trail is drawn from the previous
        reticle position to this one, so a chain of clicks reads as the
        assistant moving the mouse.
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
        """Toggle the small pulsing "typing" glyph near the orb."""
        if not self.enabled:
            return
        self._post(("typing", bool(on)))

    def flash(self, kind: str) -> None:
        """A quick accent flash: ``"wake"`` (bright rings) or ``"error"`` (red)."""
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
            root.config(bg=self.chroma)
            root.resizable(False, False)
            canvas = tk.Canvas(
                root, width=width, height=height, bg=self.chroma,
                highlightthickness=0, bd=0,
            )
            canvas.pack(fill="both", expand=True)
            root.update_idletasks()
            self._apply_click_through(root)
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
        if canvas is None:
            return
        canvas.delete("all")
        width, height = self._canvas_w, self._canvas_h
        margin = BASE_MARGIN * self.scale
        cx, cy = _anchor_point(self.position, width, height, margin)
        radius = BASE_ORB_R * self.scale

        flash_active = self._flash_active(now)
        click_active = self._click_active(now)
        quiet = (
            self._state == STATE_IDLE
            and not self._scanning
            and not self._typing_on
            and not flash_active
            and not click_active
            and not self._status
        )
        if self.idle_hidden and quiet:
            return  # nothing at all while idle and quiet, per idle_hidden=True

        if self._scanning:
            self._draw_scan(canvas, width, height, now)

        self._draw_orb(canvas, cx, cy, radius, now)

        if self._status:
            self._draw_status(canvas, cx, cy, radius)

        if self._typing_on:
            self._draw_typing(canvas, cx, cy, radius, now)

        if click_active:
            self._draw_click(canvas, now)

        if flash_active:
            self._draw_flash(canvas, cx, cy, now)

    def _draw_orb(self, c: Any, cx: float, cy: float, r: float, now: float) -> None:
        breathing = 1.0 + 0.08 * math.sin(now * 1.6)
        br = r * breathing
        if self._state == STATE_LISTENING:
            self._draw_listening(c, cx, cy, br, now)
        elif self._state == STATE_THINKING:
            self._draw_thinking(c, cx, cy, br, now)
        elif self._state == STATE_SPEAKING:
            self._draw_speaking(c, cx, cy, br, now)
        else:
            self._draw_idle_orb(c, cx, cy, br)

    def _draw_idle_orb(self, c: Any, cx: float, cy: float, r: float) -> None:
        # soft halo (stipple degrades gracefully to a solid blob on platforms
        # without canvas stipple support - never a hard failure either way)
        c.create_oval(cx - r * 1.8, cy - r * 1.8, cx + r * 1.8, cy + r * 1.8,
                       fill=COLOR_CYAN, outline="", stipple="gray12")
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=COLOR_CYAN, outline="")
        core = r * 0.45
        c.create_oval(cx - core, cy - core, cx + core, cy + core, fill=COLOR_SOFT_WHITE, outline="")

    def _draw_listening(self, c: Any, cx: float, cy: float, r: float, now: float) -> None:
        core = r * 0.55
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=COLOR_CYAN, outline="")
        c.create_oval(cx - core, cy - core, cx + core, cy + core, fill=COLOR_SOFT_WHITE, outline="")
        period = 1.6
        rings = 3
        for i in range(rings):
            phase = ((now / period) + i / rings) % 1.0
            ring_r = r * (1.3 + phase * 2.2)
            width = max(0, int(round((1.0 - phase) * 3 * self.scale)))
            if width <= 0:
                continue
            c.create_oval(cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                           outline=COLOR_CYAN, width=width)

    def _draw_thinking(self, c: Any, cx: float, cy: float, r: float, now: float) -> None:
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=COLOR_DEEP_BLUE,
                       outline=COLOR_CYAN, width=max(1, int(round(1.5 * self.scale))))
        core = r * 0.4
        c.create_oval(cx - core, cy - core, cx + core, cy + core, fill=COLOR_CYAN, outline="")
        spin_r = r * 1.6
        angle = (now * 220.0) % 360.0
        c.create_arc(cx - spin_r, cy - spin_r, cx + spin_r, cy + spin_r,
                     start=angle, extent=70, style="arc", outline=COLOR_CYAN,
                     width=max(1, int(round(2.5 * self.scale))))
        angle2 = (-now * 140.0) % 360.0
        spin_r2 = spin_r * 0.75
        c.create_arc(cx - spin_r2, cy - spin_r2, cx + spin_r2, cy + spin_r2,
                     start=angle2, extent=40, style="arc", outline=COLOR_SOFT_WHITE,
                     width=max(1, int(round(1.5 * self.scale))))
        for i in range(6):
            pangle = math.radians((now * 60.0 + i * 60.0) % 360.0)
            prad = r * (1.9 + 0.25 * math.sin(now * 2.0 + i))
            px = cx + prad * math.cos(pangle)
            py = cy + prad * math.sin(pangle)
            ps = max(1.0, 1.5 * self.scale)
            c.create_oval(px - ps, py - ps, px + ps, py + ps, fill=COLOR_SOFT_WHITE, outline="")

    def _draw_speaking(self, c: Any, cx: float, cy: float, r: float, now: float) -> None:
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=COLOR_CYAN, outline="")
        core = r * 0.5
        c.create_oval(cx - core, cy - core, cx + core, cy + core, fill=COLOR_SOFT_WHITE, outline="")
        bars = 24
        base = r * 1.3
        for i in range(bars):
            angle = 2.0 * math.pi * i / bars
            wobble = (
                (0.5 + 0.5 * math.sin(now * 6.0 + i * 0.9))
                * (0.5 + 0.5 * math.sin(now * 2.3 - i * 0.4))
            )
            outer = base + r * 0.5 * wobble
            x1, y1 = cx + base * math.cos(angle), cy + base * math.sin(angle)
            x2, y2 = cx + outer * math.cos(angle), cy + outer * math.sin(angle)
            c.create_line(x1, y1, x2, y2, fill=COLOR_CYAN, width=max(1, int(round(2 * self.scale))))
        pulse = (math.sin(now * 3.0) + 1.0) / 2.0
        pr = r * (1.15 + 0.15 * pulse)
        c.create_oval(cx - pr, cy - pr, cx + pr, cy + pr, outline=COLOR_SOFT_WHITE,
                       width=max(1, int(round(1.5 * self.scale))))

    def _draw_scan(self, c: Any, width: int, height: int, now: float) -> None:
        period = 2.2
        phase = (now / period) % 1.0
        y = phase * height
        step = max(40, int(round(80 * self.scale)))
        for gx in range(0, width, step):
            c.create_line(gx, 0, gx, height, fill=COLOR_DEEP_BLUE, width=1)
        for gy in range(0, height, step):
            c.create_line(0, gy, width, gy, fill=COLOR_DEEP_BLUE, width=1)
        for i, dy in enumerate((0, -14, -28, -44)):
            yy = y + dy
            if yy < 0:
                yy += height
            line_width = max(1, 3 - i)
            c.create_line(0, yy, width, yy, fill=COLOR_CYAN, width=line_width)

    def _draw_status(self, c: Any, cx: float, cy: float, r: float) -> None:
        font = ("Consolas", max(9, int(round(11 * self.scale))))
        if "left" in self.position:
            x, y, anchor = cx + r * 1.6, cy, "w"
        elif self.position == "center":
            x, y, anchor = cx, cy - r * 1.8, "s"
        else:
            x, y, anchor = cx - r * 1.6, cy, "e"
        c.create_text(x, y, text=self._status, anchor=anchor, fill=COLOR_SOFT_WHITE, font=font)

    def _draw_typing(self, c: Any, cx: float, cy: float, r: float, now: float) -> None:
        base_x = cx - r * 1.2
        base_y = cy + r * 1.4
        for i in range(3):
            t = (now * 3.0 + i * 0.9) % (2 * math.pi)
            pulse = (math.sin(t) + 1.0) / 2.0
            radius = (1.5 + 2.0 * pulse) * self.scale
            x = base_x - i * 10.0 * self.scale
            c.create_oval(x - radius, base_y - radius, x + radius, base_y + radius,
                           fill=COLOR_SOFT_WHITE, outline="")

    # -- click reticle: rotating brackets converge, then ping ---------------
    def _click_active(self, now: float) -> bool:
        anim = self._click_anim
        if anim is None:
            return False
        if now - anim["t0"] >= CLICK_TOTAL_S:
            self._click_anim = None
            return False
        return True

    def _draw_click(self, c: Any, now: float) -> None:
        anim = self._click_anim
        if anim is None:  # pragma: no cover - guarded by _click_active already
            return
        t = now - anim["t0"]
        sx, sy = anim["start"]
        ex, ey = anim["end"]

        if (sx, sy) != (ex, ey):
            trail = _reticle_trail((sx, sy), (ex, ey))
            fade = max(0.0, 1.0 - t / CLICK_TOTAL_S)
            for i, (px, py) in enumerate(trail):
                frac = i / max(1, len(trail) - 1)
                size = max(1.0, 2.0 * self.scale * (1.0 - frac * 0.5) * fade)
                c.create_oval(px - size, py - size, px + size, py + size,
                               fill=COLOR_CYAN, outline="")

        if t < CLICK_CONVERGE_S:
            progress = t / CLICK_CONVERGE_S
            radius = (30.0 * (1.0 - progress) + 8.0) * self.scale
            angle = 360.0 * (1.0 - progress) * 2.0
            self._draw_crosshair(c, ex, ey, radius, angle)
        else:
            pt = t - CLICK_CONVERGE_S
            progress = min(1.0, pt / CLICK_PING_S)
            ping_r = (8.0 + progress * 40.0) * self.scale
            width = max(0, int(round((1.0 - progress) * 3 * self.scale)))
            if width > 0:
                c.create_oval(ex - ping_r, ey - ping_r, ex + ping_r, ey + ping_r,
                               outline=COLOR_CYAN, width=width)
            self._draw_crosshair(c, ex, ey, 8.0 * self.scale, 0.0)

    def _draw_crosshair(self, c: Any, x: float, y: float, r: float, angle_deg: float) -> None:
        bracket = max(4.0, r * 0.5)
        for base_angle in (45, 135, 225, 315):
            a = math.radians(base_angle + angle_deg)
            bx = x + r * math.cos(a)
            by = y + r * math.sin(a)
            c.create_line(
                bx, by,
                bx + bracket * math.cos(a + math.pi), by + bracket * math.sin(a + math.pi),
                fill=COLOR_CYAN, width=max(1, int(round(2 * self.scale))),
            )
        dot = 2.0 * self.scale
        c.create_oval(x - dot, y - dot, x + dot, y + dot, fill=COLOR_SOFT_WHITE, outline="")

    # -- accent flash: expanding rings ---------------------------------------
    def _flash_active(self, now: float) -> bool:
        flash = self._flash
        if flash is None:
            return False
        _, t0 = flash
        if now - t0 >= FLASH_DURATION_S:
            self._flash = None
            return False
        return True

    def _draw_flash(self, c: Any, cx: float, cy: float, now: float) -> None:
        flash = self._flash
        if flash is None:  # pragma: no cover - guarded by _flash_active already
            return
        kind, t0 = flash
        color = COLOR_RED if kind == FLASH_ERROR else COLOR_SOFT_WHITE
        progress = min(1.0, (now - t0) / FLASH_DURATION_S)
        for i in range(3):
            ring_progress = min(1.0, progress + i * 0.15)
            if i > 0 and ring_progress >= 1.0:
                continue
            radius = (20.0 + ring_progress * 260.0) * self.scale
            width = max(0, int(round((1.0 - ring_progress) * 4 * self.scale)))
            if width <= 0:
                continue
            c.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                           outline=color, width=width)


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

    print("Overlay demo running - watch the bottom-right corner of the TV. Ctrl+C to stop early.")
    try:
        hud.set_state(STATE_IDLE)
        time.sleep(2.0)

        hud.set_state(STATE_LISTENING)
        hud.set_status("listening")
        time.sleep(2.5)

        hud.set_state(STATE_THINKING)
        hud.set_status("thinking")
        time.sleep(2.5)

        hud.scan_screen(True)
        hud.set_status("looking at the screen")
        time.sleep(2.5)
        hud.scan_screen(False)

        hud.set_status("clicking around")
        for point in ((0.2, 0.3), (0.8, 0.2), (0.5, 0.8)):
            hud.click_at(*point)
            time.sleep(0.9)

        hud.typing(True)
        hud.set_status("typing")
        time.sleep(2.0)
        hud.typing(False)

        hud.set_status("")
        hud.flash(FLASH_WAKE)
        time.sleep(1.0)

        hud.set_state(STATE_SPEAKING)
        hud.set_status("speaking")
        time.sleep(3.0)

        hud.flash(FLASH_ERROR)
        time.sleep(1.0)

        hud.set_state(STATE_IDLE)
        hud.set_status("")
        time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        hud.stop()
        print("Overlay demo stopped")


if __name__ == "__main__":
    _run_demo()
