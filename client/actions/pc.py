"""Windows actions on the room PC — ``pc_control`` and ``run_command`` (SPEC §5/§8).

Implemented with pycaw (master volume) and plain ctypes ``SendInput`` (media keys,
unicode typing, hotkey combos, mouse jiggle), plus Win32 calls for monitor power
and suspend. Applications are resolved through :mod:`client.actions.apps`, which
indexes everything installed (Start Menu + Store apps) on top of the
``cfg.client.apps`` overrides.

All blocking work runs in worker threads (:func:`asyncio.to_thread`); the COM
apartment needed by pycaw is initialised inside the worker thread.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from ctypes import wintypes
from typing import Any, Iterator, Mapping, NamedTuple, Sequence

from .apps import (
    AppEntry,
    AppError,
    AppIndex,
    decode_console_output,
    powershell_executable,
)

log = logging.getLogger(__name__)

# --- pc_control command names (SPEC §5) --------------------------------------
CMD_VOLUME_SET = "volume_set"
CMD_VOLUME_UP = "volume_up"
CMD_VOLUME_DOWN = "volume_down"
CMD_MUTE = "mute"
CMD_UNMUTE = "unmute"
CMD_MEDIA_PLAY_PAUSE = "media_play_pause"
CMD_MEDIA_NEXT = "media_next"
CMD_MEDIA_PREV = "media_prev"
CMD_DISPLAY_OFF = "display_off"
CMD_DISPLAY_ON = "display_on"
CMD_SLEEP = "sleep"
CMD_OPEN_APP = "open_app"
CMD_CLOSE_APP = "close_app"
CMD_TYPE_TEXT = "type_text"
CMD_HOTKEY = "hotkey"

PC_COMMANDS = frozenset(
    {
        CMD_VOLUME_SET,
        CMD_VOLUME_UP,
        CMD_VOLUME_DOWN,
        CMD_MUTE,
        CMD_UNMUTE,
        CMD_MEDIA_PLAY_PAUSE,
        CMD_MEDIA_NEXT,
        CMD_MEDIA_PREV,
        CMD_DISPLAY_OFF,
        CMD_DISPLAY_ON,
        CMD_SLEEP,
        CMD_OPEN_APP,
        CMD_CLOSE_APP,
        CMD_TYPE_TEXT,
        CMD_HOTKEY,
    }
)

#: volume step for ``volume_up`` / ``volume_down`` (5 %)
VOLUME_STEP = 0.05

#: how long to wait for the suspend thread to report a failure before answering
SLEEP_GRACE_S = 0.7

#: ``run_command`` limits (SPEC §5): wall clock and returned output size
RUN_COMMAND_TIMEOUT_S = 30.0
RUN_COMMAND_OUTPUT_LIMIT = 4000

#: upper bound for a single ``type_text`` action — a voice command never needs more
MAX_TYPE_CHARS = 4000

#: how many SendInput events are pushed in one call while typing
TYPE_CHUNK_EVENTS = 100

#: pause between typing chunks so slower windows keep up
TYPE_CHUNK_PAUSE_S = 0.005

# --- Win32 constants ---------------------------------------------------------
_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_MOUSEEVENTF_MOVE = 0x0001

VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_F1 = 0x70

_HWND_BROADCAST = 0xFFFF
_WM_SYSCOMMAND = 0x0112
_SC_MONITORPOWER = 0xF170
_MONITOR_POWER_OFF = 2

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

_IS_WINDOWS = sys.platform == "win32"


class PCActionError(RuntimeError):
    """Recoverable failure of a ``pc_control`` / ``run_command`` action.

    ``output`` carries extra data for the LLM (e.g. the closest app names when
    ``open_app`` could not resolve what the user said) — the dispatcher copies it
    into ``action_result.output`` (SPEC §4).
    """

    def __init__(self, message: str, output: str | None = None) -> None:
        super().__init__(message)
        self.output = output


class PCResult(NamedTuple):
    """Outcome of a successful command: log/summary text plus optional output."""

    detail: str
    output: str | None = None


# --- hotkey vocabulary (SPEC §8) ---------------------------------------------

#: modifier name -> virtual-key code
MODIFIER_KEYS: dict[str, int] = {
    "ctrl": VK_CONTROL,
    "control": VK_CONTROL,
    "alt": VK_MENU,
    "menu": VK_MENU,
    "shift": VK_SHIFT,
    "win": VK_LWIN,
    "windows": VK_LWIN,
    "super": VK_LWIN,
    "meta": VK_LWIN,
    "cmd": VK_LWIN,
}

#: named key -> (virtual-key code, extended-key flag)
NAMED_KEYS: dict[str, tuple[int, bool]] = {
    "enter": (VK_RETURN, False),
    "return": (VK_RETURN, False),
    "esc": (VK_ESCAPE, False),
    "escape": (VK_ESCAPE, False),
    "tab": (VK_TAB, False),
    "space": (VK_SPACE, False),
    "spacebar": (VK_SPACE, False),
    "backspace": (VK_BACK, False),
    "bksp": (VK_BACK, False),
    "del": (VK_DELETE, True),
    "delete": (VK_DELETE, True),
    "up": (VK_UP, True),
    "down": (VK_DOWN, True),
    "left": (VK_LEFT, True),
    "right": (VK_RIGHT, True),
}

_HOTKEY_HELP = (
    "supported: ctrl, alt, shift, win + a letter, a digit, f1-f24, enter, esc, "
    "tab, space, backspace, del, up, down, left, right"
)

# --- ctypes plumbing ---------------------------------------------------------

ULONG_PTR = wintypes.WPARAM


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT))


class _INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))


_dll_lock = threading.Lock()
_user32_dll: Any = None


def _require_windows() -> None:
    if not _IS_WINDOWS:
        raise PCActionError("pc_control commands only work on Windows")


def _user32() -> Any:
    """Return a configured ``user32`` handle (loaded once, thread-safe)."""

    global _user32_dll
    _require_windows()
    with _dll_lock:
        if _user32_dll is None:
            dll = ctypes.WinDLL("user32", use_last_error=True)
            dll.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
            dll.SendInput.restype = wintypes.UINT
            dll.SendMessageW.argtypes = (
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )
            dll.SendMessageW.restype = wintypes.LPARAM
            _user32_dll = dll
        return _user32_dll


def _send_input(*events: _INPUT) -> None:
    _send_input_batch(events)


def _send_input_batch(events: Sequence[_INPUT]) -> None:
    """Push a batch of input events in one ``SendInput`` call (order preserved)."""

    count = len(events)
    if not count:
        return
    user32 = _user32()
    array = (_INPUT * count)(*events)
    sent = user32.SendInput(count, array, ctypes.sizeof(_INPUT))
    if sent != count:
        raise PCActionError(
            f"SendInput delivered {sent} of {count} events "
            f"(error code {ctypes.get_last_error()})"
        )


def _key_event(vk: int, key_up: bool, extended: bool = False) -> _INPUT:
    flags = (_KEYEVENTF_EXTENDEDKEY if extended else 0) | (
        _KEYEVENTF_KEYUP if key_up else 0
    )
    return _INPUT(
        type=_INPUT_KEYBOARD,
        union=_INPUTUNION(
            ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
        ),
    )


def _unicode_event(code_unit: int, key_up: bool) -> _INPUT:
    """A KEYEVENTF_UNICODE event carrying one UTF-16 code unit."""

    flags = _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if key_up else 0)
    return _INPUT(
        type=_INPUT_KEYBOARD,
        union=_INPUTUNION(
            ki=_KEYBDINPUT(wVk=0, wScan=code_unit, dwFlags=flags, time=0, dwExtraInfo=0)
        ),
    )


def _mouse_move_event(dx: int, dy: int) -> _INPUT:
    return _INPUT(
        type=_INPUT_MOUSE,
        union=_INPUTUNION(
            mi=_MOUSEINPUT(
                dx=dx, dy=dy, mouseData=0, dwFlags=_MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0
            )
        ),
    )


@contextmanager
def _com_apartment() -> Iterator[None]:
    """Initialise COM for the current (worker) thread for the pycaw calls."""

    try:
        import comtypes  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise PCActionError(
            "the comtypes library is missing (pip install comtypes pycaw)"
        ) from exc

    initialized = False
    try:
        comtypes.CoInitialize()
        initialized = True
    except OSError as exc:
        # RPC_E_CHANGED_MODE etc. — the thread already lives in another apartment
        log.debug("CoInitialize failed, continuing without it: %s", exc)
    try:
        yield
    finally:
        if initialized:
            try:
                comtypes.CoUninitialize()
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask errors
                log.debug("CoUninitialize failed: %s", exc)


def _endpoint_volume() -> Any:
    """Return ``IAudioEndpointVolume`` for the default output device.

    Recent pycaw releases hand back an ``AudioDevice`` wrapper with a ready
    ``EndpointVolume`` property, older ones a raw ``IMMDevice`` that still has to
    be activated — both are supported here.
    """

    try:
        from comtypes import CLSCTX_ALL  # type: ignore[import-not-found]
        from pycaw.pycaw import (  # type: ignore[import-not-found]
            AudioUtilities,
            IAudioEndpointVolume,
        )
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise PCActionError(
            "the pycaw library is missing (pip install pycaw comtypes)"
        ) from exc

    speakers = AudioUtilities.GetSpeakers()
    if speakers is None:
        raise PCActionError("no audio output device found")

    if hasattr(type(speakers), "EndpointVolume"):
        return speakers.EndpointVolume

    activate = getattr(speakers, "Activate", None)
    if not callable(activate):
        raise PCActionError(
            "pycaw returned an unexpected speaker object "
            f"({type(speakers).__name__}) — cannot reach the volume interface"
        )
    interface = activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))


# --- blocking workers (run via asyncio.to_thread) ----------------------------


def _clamp_scalar(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sync_set_volume(scalar: float) -> float:
    with _com_apartment():
        volume = _endpoint_volume()
        level = _clamp_scalar(scalar)
        volume.SetMasterVolumeLevelScalar(level, None)
        if level > 0.0:
            volume.SetMute(0, None)
        return level


def _sync_step_volume(delta: float) -> float:
    with _com_apartment():
        volume = _endpoint_volume()
        current = float(volume.GetMasterVolumeLevelScalar())
        level = _clamp_scalar(current + delta)
        volume.SetMasterVolumeLevelScalar(level, None)
        if delta > 0.0 and level > 0.0:
            volume.SetMute(0, None)
        return level


def _sync_set_mute(muted: bool) -> None:
    with _com_apartment():
        volume = _endpoint_volume()
        volume.SetMute(1 if muted else 0, None)


def _sync_media_key(vk: int) -> None:
    # media keys are extended keys on a PC/AT keyboard
    _send_input(_key_event(vk, key_up=False, extended=True), _key_event(vk, key_up=True, extended=True))


def _sync_display_off() -> None:
    user32 = _user32()
    user32.SendMessageW(
        _HWND_BROADCAST, _WM_SYSCOMMAND, _SC_MONITORPOWER, _MONITOR_POWER_OFF
    )


def _sync_display_on() -> None:
    # a 1 px mouse move is the reliable way to wake the monitor back up
    _send_input(_mouse_move_event(1, 0))
    time.sleep(0.05)
    _send_input(_mouse_move_event(-1, 0))


def _sync_suspend() -> None:
    _require_windows()
    powrprof = ctypes.WinDLL("powrprof", use_last_error=True)
    powrprof.SetSuspendState.argtypes = (ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte)
    powrprof.SetSuspendState.restype = ctypes.c_ubyte
    # hibernate=False, force=True, wakeup events enabled
    result = powrprof.SetSuspendState(0, 1, 0)
    if not result:
        raise PCActionError(
            f"SetSuspendState failed (error code {ctypes.get_last_error()})"
        )


def _text_events(text: str) -> list[_INPUT]:
    """Build the SendInput events that type ``text`` (SPEC §8).

    Every character is sent as UTF-16 code units with ``KEYEVENTF_UNICODE``, so
    surrogate pairs (emoji, rare CJK) become two consecutive code-unit events —
    Windows joins them into one character. Line breaks and tabs have no unicode
    equivalent that applications act on, so they are sent as real key presses.
    """

    events: list[_INPUT] = []
    for char in text:
        if char == "\r":
            continue
        if char == "\n":
            events.append(_key_event(VK_RETURN, key_up=False))
            events.append(_key_event(VK_RETURN, key_up=True))
            continue
        if char == "\t":
            events.append(_key_event(VK_TAB, key_up=False))
            events.append(_key_event(VK_TAB, key_up=True))
            continue
        units = _utf16_units(char)
        for unit in units:
            events.append(_unicode_event(unit, key_up=False))
        for unit in units:
            events.append(_unicode_event(unit, key_up=True))
    return events


def _utf16_units(char: str) -> tuple[int, ...]:
    """Return the UTF-16 code units of one character (1, or 2 for a surrogate pair)."""

    encoded = char.encode("utf-16-le")
    return tuple(
        encoded[index] | (encoded[index + 1] << 8) for index in range(0, len(encoded), 2)
    )


def _sync_type_text(text: str) -> int:
    """Type ``text`` into the focused window. Returns the character count."""

    _require_windows()
    events = _text_events(text)
    for start in range(0, len(events), TYPE_CHUNK_EVENTS):
        _send_input_batch(events[start : start + TYPE_CHUNK_EVENTS])
        if start + TYPE_CHUNK_EVENTS < len(events):
            time.sleep(TYPE_CHUNK_PAUSE_S)
    return len(text)


def parse_hotkey(combo: Any) -> tuple[list[int], list[tuple[int, bool]], str]:
    """Parse ``"ctrl+shift+t"`` into ``(modifier vks, [(vk, extended)], label)``.

    Raises :class:`PCActionError` for an empty combo or an unsupported key name.
    """

    text = "" if combo is None else str(combo)
    parts = [part.strip().casefold() for part in text.replace(" ", "+").split("+")]
    parts = [part for part in parts if part]
    if not parts:
        raise PCActionError("no hotkey given (value), e.g. 'ctrl+shift+t'")

    modifiers: list[int] = []
    keys: list[tuple[int, bool]] = []
    labels: list[str] = []
    for part in parts:
        labels.append(part)
        modifier = MODIFIER_KEYS.get(part)
        if modifier is not None:
            if modifier not in modifiers:
                modifiers.append(modifier)
            continue
        named = NAMED_KEYS.get(part)
        if named is not None:
            keys.append(named)
            continue
        if len(part) == 1 and (part.isalpha() or part.isdigit()) and part.isascii():
            keys.append((ord(part.upper()), False))
            continue
        if part.startswith("f") and part[1:].isdigit():
            number = int(part[1:])
            if 1 <= number <= 24:
                keys.append((VK_F1 + number - 1, False))
                continue
        raise PCActionError(f"unknown hotkey key '{part}' ({_HOTKEY_HELP})")

    if not modifiers and not keys:
        raise PCActionError(f"hotkey '{text}' has no keys to press ({_HOTKEY_HELP})")
    return modifiers, keys, "+".join(labels)


def _sync_hotkey(modifiers: Sequence[int], keys: Sequence[tuple[int, bool]]) -> None:
    """Press modifiers, then the keys, and release everything in reverse order."""

    _require_windows()
    # modifiers down -> keys down -> keys up -> modifiers up; a modifier-only
    # combo (e.g. "win") simply has no keys in the middle.
    events: list[_INPUT] = [_key_event(vk, key_up=False) for vk in modifiers]
    events.extend(_key_event(vk, key_up=False, extended=extended) for vk, extended in keys)
    events.extend(
        _key_event(vk, key_up=True, extended=extended) for vk, extended in reversed(keys)
    )
    events.extend(_key_event(vk, key_up=True) for vk in reversed(modifiers))
    _send_input_batch(events)


def truncate_output(text: str, limit: int = RUN_COMMAND_OUTPUT_LIMIT) -> str:
    """Cut command output down to ``limit`` characters, marking what was dropped."""

    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    marker = f"\n... [truncated, {len(clean)} chars total]"
    keep = max(0, limit - len(marker))
    return clean[:keep] + marker


def _kill_process_tree(pid: int) -> None:
    """Terminate a spawned process and everything it started (SPEC §8)."""

    try:
        subprocess.run(  # noqa: S603 - fixed command, pid from our own Popen
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=_CREATE_NO_WINDOW,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - best effort, the caller kills anyway
        log.warning("taskkill could not stop the process tree of pid %s: %s", pid, exc)


def _sync_run_command(command: str, timeout_s: float) -> tuple[int | None, str, bool]:
    """Run PowerShell. Returns ``(exit code, merged output, timed out)``."""

    _require_windows()
    argv = [
        powershell_executable(),
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        command,
    ]
    try:
        process = subprocess.Popen(  # noqa: S603 - the LLM's command, by design (SPEC §5)
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=_CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP,
        )
    except OSError as exc:
        raise PCActionError(f"could not start PowerShell: {exc}") from exc

    try:
        stdout, _ = process.communicate(timeout=timeout_s)
        return process.returncode, decode_console_output(stdout), False
    except subprocess.TimeoutExpired:
        _kill_process_tree(process.pid)
        try:
            stdout, _ = process.communicate(timeout=5)
        except Exception:  # noqa: BLE001 - the tree is already being killed
            stdout = b""
            try:
                process.kill()
            except Exception as exc:  # noqa: BLE001 - nothing left to do
                log.debug("could not kill pid %s: %s", process.pid, exc)
        return None, decode_console_output(stdout), True


def _sync_close_process(image_name: str) -> str:
    """``taskkill /IM <image> /F`` — returns the tool's output on success."""

    _require_windows()
    completed = subprocess.run(  # noqa: S603 - fixed command, name from the app index
        ["taskkill", "/IM", image_name, "/F"],
        capture_output=True,
        creationflags=_CREATE_NO_WINDOW,
        check=False,
    )
    output = (
        decode_console_output(completed.stdout).strip()
        or decode_console_output(completed.stderr).strip()
    )
    if completed.returncode != 0:
        detail = output or f"taskkill exited with code {completed.returncode}"
        raise PCActionError(f"could not close '{image_name}': {detail}")
    return output or f"process {image_name} terminated"


# --- controller --------------------------------------------------------------


def _parse_volume_value(value: Any) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PCActionError("volume_set needs a level between 0 and 100")
    if isinstance(value, bool):
        raise PCActionError(f"unclear volume level {value!r}: expected 0..100")
    if isinstance(value, str):
        text = value.strip().rstrip("%").replace(",", ".").strip()
        try:
            number = float(text)
        except ValueError as exc:
            raise PCActionError(f"unclear volume level {value!r}: expected 0..100") from exc
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise PCActionError(f"unclear volume level {value!r}: expected 0..100")
    if 0.0 < number <= 1.0 and isinstance(value, float):
        # tolerate a 0..1 scalar coming from the model
        number *= 100.0
    return max(0, min(100, int(round(number))))


class PCController:
    """Executes ``pc_control`` commands and ``run_command`` on the client machine."""

    def __init__(
        self,
        apps: Mapping[str, Any] | None = None,
        app_index: AppIndex | None = None,
    ) -> None:
        self.apps = app_index if app_index is not None else AppIndex(apps)

    # -- lifecycle ----------------------------------------------------------

    async def prepare(self) -> None:
        """Build the installed-app index ahead of the first ``open_app``."""

        await self.apps.ensure_ready()

    # -- pc_control ---------------------------------------------------------

    async def execute(self, command: Any, value: Any = None) -> PCResult:
        """Run one ``pc_control`` command. Raises :class:`PCActionError` on failure."""

        name = command.strip().casefold() if isinstance(command, str) else ""
        if not name:
            raise PCActionError("no pc_control command given")
        if name not in PC_COMMANDS:
            known = ", ".join(sorted(PC_COMMANDS))
            raise PCActionError(f"unknown pc_control command '{command}' (known: {known})")
        _require_windows()

        if name == CMD_VOLUME_SET:
            level = _parse_volume_value(value)
            await asyncio.to_thread(_sync_set_volume, level / 100.0)
            return PCResult(f"volume {level}%")

        if name in (CMD_VOLUME_UP, CMD_VOLUME_DOWN):
            delta = VOLUME_STEP if name == CMD_VOLUME_UP else -VOLUME_STEP
            level = await asyncio.to_thread(_sync_step_volume, delta)
            return PCResult(f"volume {int(round(level * 100))}%")

        if name in (CMD_MUTE, CMD_UNMUTE):
            muted = name == CMD_MUTE
            await asyncio.to_thread(_sync_set_mute, muted)
            return PCResult("sound muted" if muted else "sound unmuted")

        if name in (CMD_MEDIA_PLAY_PAUSE, CMD_MEDIA_NEXT, CMD_MEDIA_PREV):
            vk = {
                CMD_MEDIA_PLAY_PAUSE: VK_MEDIA_PLAY_PAUSE,
                CMD_MEDIA_NEXT: VK_MEDIA_NEXT_TRACK,
                CMD_MEDIA_PREV: VK_MEDIA_PREV_TRACK,
            }[name]
            await asyncio.to_thread(_sync_media_key, vk)
            return PCResult(f"media key {name}")

        if name == CMD_DISPLAY_OFF:
            await asyncio.to_thread(_sync_display_off)
            return PCResult("display off")

        if name == CMD_DISPLAY_ON:
            await asyncio.to_thread(_sync_display_on)
            return PCResult("display on")

        if name == CMD_SLEEP:
            return PCResult(await self._suspend())

        if name == CMD_TYPE_TEXT:
            return await self._type_text(value)

        if name == CMD_HOTKEY:
            return await self._hotkey(value)

        if name == CMD_OPEN_APP:
            return await self._open_app(value)

        # CMD_CLOSE_APP
        return await self._close_app(value)

    # -- run_command --------------------------------------------------------

    async def run_command(self, command: Any) -> tuple[bool, str | None, str | None]:
        """Run a PowerShell command (SPEC §5). Returns ``(ok, error, output)``."""

        text = "" if command is None else str(command).strip()
        if not text:
            raise PCActionError("run_command needs a 'command' string")
        _require_windows()

        log.info("run_command: %s", text)
        code, output, timed_out = await asyncio.to_thread(
            _sync_run_command, text, RUN_COMMAND_TIMEOUT_S
        )
        clipped = truncate_output(output) or None

        if timed_out:
            message = f"command timed out after {int(RUN_COMMAND_TIMEOUT_S)} s and was terminated"
            log.warning("run_command: %s", message)
            return False, message, clipped
        if code != 0:
            message = f"command exited with code {code}"
            log.warning("run_command: %s", message)
            return False, message, clipped
        log.info("run_command: finished, %d chars of output", len(clipped or ""))
        return True, None, clipped

    # -- helpers ------------------------------------------------------------

    async def _suspend(self) -> str:
        """Start the suspend in a background thread.

        ``SetSuspendState`` only returns once the machine resumes, so waiting for
        it would hang the action; instead we give the thread a short grace period
        to report an immediate failure.
        """

        failure: dict[str, str] = {}

        def worker() -> None:
            try:
                _sync_suspend()
            except Exception as exc:  # noqa: BLE001 - reported through the box/log
                failure["error"] = str(exc)
                log.error("pc_control: could not suspend the PC: %s", exc)

        thread = threading.Thread(target=worker, name="jarvis-suspend", daemon=True)
        thread.start()
        await asyncio.sleep(SLEEP_GRACE_S)
        if "error" in failure:
            raise PCActionError(failure["error"])
        return "the PC is going to sleep"

    async def _type_text(self, value: Any) -> PCResult:
        text = "" if value is None else str(value)
        if not text:
            raise PCActionError("type_text needs the text to type in 'value'")
        if len(text) > MAX_TYPE_CHARS:
            raise PCActionError(
                f"type_text got {len(text)} characters, the limit is {MAX_TYPE_CHARS}"
            )
        typed = await asyncio.to_thread(_sync_type_text, text)
        log.info("pc_control: typed %d characters", typed)
        return PCResult(f"typed {typed} characters")

    async def _hotkey(self, value: Any) -> PCResult:
        modifiers, keys, label = parse_hotkey(value)
        await asyncio.to_thread(_sync_hotkey, modifiers, keys)
        log.info("pc_control: pressed %s", label)
        return PCResult(f"pressed {label}")

    async def _resolve_app(self, value: Any) -> AppEntry:
        """Resolve an app name through the index, or raise with close matches."""

        if value is None or not str(value).strip():
            raise PCActionError("no application name given (value)")
        name = str(value).strip()
        entry = await self.apps.resolve(name)
        if entry is None:
            hint = self.apps.miss_hint(name)
            raise PCActionError(
                f"no installed application matches '{name}' ({hint})", output=hint
            )
        return entry

    async def _open_app(self, value: Any) -> PCResult:
        entry = await self._resolve_app(value)
        try:
            launched = await asyncio.to_thread(entry.launch)
        except AppError as exc:
            raise PCActionError(str(exc)) from exc
        log.info("pc_control: launched '%s' (%s)", entry.name, launched)
        return PCResult(f"opened: {entry.name}")

    async def _close_app(self, value: Any) -> PCResult:
        entry = await self._resolve_app(value)
        image_name = entry.process_name()
        if image_name is None:
            kind = "a Store app" if entry.is_store_app else "registered by AppID only"
            message = (
                f"'{entry.name}' is {kind} ('{entry.target}'), so close_app cannot map "
                f"it to a process image; use run_command with something like "
                f"\"Stop-Process -Name <process> -Force\" instead"
            )
            raise PCActionError(message, output=message)
        detail = await asyncio.to_thread(_sync_close_process, image_name)
        log.info("pc_control: closed '%s' (%s)", entry.name, image_name)
        return PCResult(f"closed: {entry.name} ({detail})")


__all__ = [
    "MODIFIER_KEYS",
    "NAMED_KEYS",
    "PC_COMMANDS",
    "PCActionError",
    "PCController",
    "PCResult",
    "RUN_COMMAND_OUTPUT_LIMIT",
    "RUN_COMMAND_TIMEOUT_S",
    "parse_hotkey",
    "truncate_output",
]
