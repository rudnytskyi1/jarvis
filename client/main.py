"""Jarvis room client — entry point (SPEC §7).

Single asyncio loop:

1. connect to the brain server over WebSocket (auto-reconnect, 3 s backoff,
   ``hello`` re-sent every time),
2. listen for the wake word with Vosk, play a generated ack beep,
3. record the utterance with webrtcvad, including the ``pre_roll_ms`` of audio
   captured before the trigger,
4. stream it to the server in 30 ms chunks and handle every server message,
5. run the actions through :mod:`client.actions.dispatcher` and report results
   (``ok``/``error``/``output``), and answer ``screenshot_request`` with a JPEG
   grabbed by :mod:`client.screen`,
6. play the TTS stream as it arrives,
7. optionally keep listening for ``followup_window_s`` seconds without the
   wake word.

Per utterance the server may send several rounds of ``actions`` and/or
``screenshot_request`` (one per LLM tool round) before the spoken reply, so the
response loop keeps handling messages until ``tts_end`` or ``error``.

v1.4 — the socket is read ALL the time
--------------------------------------
The server talks between utterances too: it greets a face it does not know and
it pulls camera frames for ``look_at_camera``/``enroll_face``. So exactly one
background task (:meth:`JarvisClient._reader_loop`) owns ``ws.recv()`` for the
lifetime of a connection and routes what it reads:

* ``camera_request`` — answered at any moment from :mod:`client.camera`;
* during a conversation — everything (binary frames included) goes into an
  ``asyncio.Queue`` that :meth:`JarvisClient._receive_response` consumes;
* while idle — a proactive ``say`` + ``tts_start``…``tts_end`` block is played
  through the speakers, and the wake word cuts it short and starts listening.

The mode flag is owned by the conversation loop and flips exactly at
``utterance_start``/end of reply, so no message is ever consumed twice or lost
in between. The reader dies with the connection and is restarted by
:meth:`JarvisClient._ensure_link` after every reconnect; the camera keeps
running across reconnects and resumes its pushes by itself.

Run from the repository root: ``python -m client.main --config config.yaml``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import protocol as _protocol
from common.config import load_config
from common.protocol import (
    MSG_ACTION_RESULT,
    MSG_ACTIONS,
    MSG_CAMERA_ERROR,
    MSG_CAMERA_REQUEST,
    MSG_ERROR,
    MSG_HELLO,
    MSG_READY,
    MSG_SAY,
    MSG_SCREENSHOT,
    MSG_SCREENSHOT_ERROR,
    MSG_SCREENSHOT_REQUEST,
    MSG_TRANSCRIPT,
    MSG_TTS_END,
    MSG_TTS_START,
    MSG_UTTERANCE_END,
    MSG_UTTERANCE_START,
)

from client.actions.dispatcher import Dispatcher
from client.audio import (
    BEEP_FREQ_HZ,
    BEEP_MS,
    ERROR_BEEP_FREQ_HZ,
    ERROR_BEEP_MS,
    FRAME_MS,
    SAMPLE_WIDTH,
    AudioInput,
    AudioOutput,
    RingBuffer,
    frame_bytes,
)
from client.devices.registry import build_registry
from client.screen import SCREENSHOT_FORMAT, Capture, capture_jpeg
from client.vad import VadRecorder
from client.wakeword import WakeWordDetector
from client.ws_client import WAIT_FOREVER, WSClient, WSDisconnected

log = logging.getLogger("client")

#: The camera stack (OpenCV + Ultralytics) is optional and lives behind lazy
#: imports, but even importing this thin module must not be able to stop the
#: voice client — a broken checkout simply means "no camera on this machine".
_CAMERA_IMPORT_ERROR: Optional[str] = None
try:
    from client.camera import CameraService
except Exception as _camera_exc:  # noqa: BLE001 - pragma: no cover
    _CAMERA_IMPORT_ERROR = f"{type(_camera_exc).__name__}: {_camera_exc}"
    CameraService = None  # type: ignore[assignment]

#: Audio format announced in ``utterance_start`` (SPEC §4).
PCM_FORMAT = getattr(_protocol, "AUDIO_FORMAT", "pcm_s16le")
CHANNELS = int(getattr(_protocol, "AUDIO_CHANNELS", 1))
#: How long we wait for a batch of actions before moving on. Must cover the
#: slowest action: ``run_command`` runs PowerShell for up to 30 s (SPEC §8) and
#: BLE devices can take a while too.
ACTION_TIMEOUT_S = 60.0
#: Upper bound for ``action_result.output`` (SPEC §4, C->S #5).
MAX_OUTPUT_CHARS = 4000
#: Short "nothing heard" chirp after a false wake-word trigger.
NO_SPEECH_BEEP_FREQ_HZ = 440.0
NO_SPEECH_BEEP_MS = 90
#: Pause before the follow-up window opens: the room is still ringing with the
#: tail of our own reply (speaker-to-mic echo), which VAD would otherwise pick
#: up as speech and send to the server as a phantom empty utterance.
FOLLOWUP_ECHO_GUARD_S = 0.5
#: "Thinking" sounds: when the server takes longer than this to start replying
#: (vision, tool rounds), a soft two-tone blip repeats so the user knows Jarvis
#: is working rather than stuck. Silenced the moment the reply begins.
THINKING_DELAY_S = 1.5
THINKING_INTERVAL_S = 2.2
THINKING_BLIP_FREQS_HZ = (520.0, 660.0)
THINKING_BLIP_MS = 60
THINKING_VOLUME = 0.14
#: Mic audio recorded while the ack beep was playing: everything captured in the
#: last ``BEEP_MS + BEEP_ECHO_GUARD_MS`` is our own beep (tone + output latency)
#: and is dropped; anything older is the user already speaking and is kept, so a
#: command said in one breath ("rowan, turn on the light") is not cut off.
BEEP_ECHO_GUARD_MS = 250

#: After a proactive message (a greeting that asks a question) the client
#: listens this long WITHOUT the wake word, so the person can just answer.
PROACTIVE_LISTEN_S = 8.0

RESULT_OK = "ok"
RESULT_NO_SPEECH = "no_speech"
RESULT_ERROR = "error"
#: The user said the wake word while the reply was playing: playback was cut
#: and the client goes straight back to listening.
RESULT_BARGE_IN = "barge_in"

#: Routing modes of the reader task (v1.4). The conversation loop owns the flag.
#: ``idle`` — proactive audio is played and camera/screen requests answered;
#: ``conversation`` — every message belongs to the utterance in flight and is
#: buffered for :meth:`JarvisClient._receive_response`.
MODE_IDLE = "idle"
MODE_CONVERSATION = "conversation"
#: How often a waiting conversation re-checks that the reader is still alive.
INBOX_POLL_S = 0.5


class _LinkDown:
    """Sentinel put into the inbox when the reader task ends."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<link down>"


#: Singleton sentinel: waking a waiting conversation when the socket dies.
LINK_DOWN = _LinkDown()


class _Stopping(Exception):
    """Internal: Ctrl+C was pressed, unwind the audio loops."""


def _attr(obj: Any, name: str) -> Any:
    """Read ``name`` from a pydantic model / dataclass / mapping."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clip_output(value: Any) -> Optional[str]:
    """Normalise the dispatcher's ``output`` for ``action_result`` (SPEC §4).

    ``None``/empty stays ``None``; anything longer than :data:`MAX_OUTPUT_CHARS`
    is cut so one chatty command cannot flood the LLM context.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if not text:
        return None
    if len(text) > MAX_OUTPUT_CHARS:
        suffix = "... (truncated)"
        text = text[: max(0, MAX_OUTPUT_CHARS - len(suffix))] + suffix
    return text


def build_hello(cfg_client: Any) -> Dict[str, Any]:
    """Build the ``hello`` payload from the client config (SPEC §4.1)."""
    devices: List[Dict[str, Any]] = []
    for dev in (_attr(cfg_client, "devices") or []):
        name = _opt_str(_attr(dev, "name"))
        if not name:
            continue
        devices.append(
            {
                "name": name,
                "type": str(_attr(dev, "type") or ""),
                "area": _opt_str(_attr(dev, "area")),
                "description": _opt_str(_attr(dev, "description")),
            }
        )
    return {
        "type": MSG_HELLO,
        "client_id": str(_attr(cfg_client, "client_id") or "client"),
        "devices": devices,
    }


def resolve_path(raw: Any) -> Path:
    """Resolve a config path relative to the current dir, then to the repo root."""
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


class JarvisClient:
    """Wires audio, wake word, VAD, WebSocket, screen capture and actions together."""

    def __init__(self, cfg: Any) -> None:
        self._stopping = False
        self.cfg = cfg
        self.ccfg = cfg.client

        audio_cfg = self.ccfg.audio
        self.sample_rate = int(audio_cfg.sample_rate)
        self.frame_ms = FRAME_MS
        self.frame_bytes = frame_bytes(self.sample_rate, self.frame_ms)
        self.audio_in = AudioInput(
            device=audio_cfg.input_device,
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
        )
        self.audio_out = AudioOutput(
            device=audio_cfg.output_device,
            default_sample_rate=self.sample_rate,
        )

        vad_cfg = self.ccfg.vad
        pre_roll_ms = max(0, int(vad_cfg.pre_roll_ms))
        self.vad = VadRecorder(
            aggressiveness=int(vad_cfg.aggressiveness),
            silence_ms=int(vad_cfg.silence_ms),
            max_utterance_s=float(vad_cfg.max_utterance_s),
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
            pre_roll_ms=pre_roll_ms,
            min_speech_ms=int(getattr(vad_cfg, "min_speech_ms", 250)),
        )
        self.preroll = RingBuffer(int(math.ceil(pre_roll_ms / float(self.frame_ms))))

        self.followup_window_s = float(_attr(self.ccfg, "followup_window_s") or 0.0)
        raw_thinking = _attr(self.ccfg, "thinking_sounds")
        self.thinking_sounds = True if raw_thinking is None else bool(raw_thinking)
        self._thinking_task: Optional[asyncio.Task] = None
        self._barge_task: Optional[asyncio.Task] = None
        self._barged = False

        self.registry = build_registry(self.ccfg)
        self.dispatcher = Dispatcher(self.ccfg, self.registry)
        self.ws = WSClient(
            url=str(self.ccfg.server_url),
            hello=build_hello(self.ccfg),
            should_stop=lambda: self._stopping,
        )

        self.wake: Optional[WakeWordDetector] = None
        self._started = False
        self._action_task: Optional[asyncio.Task] = None
        self._tts_active = False
        self._tts_bytes = 0
        self._last_say = ""
        #: Server's follow-up window request from the last reply (say.listen_s).
        self._listen_hint_s = 0.0
        #: Set when a proactive message finished playing: answer without wake word.
        self._proactive_listen_s = 0.0

        # -- v1.4: one reader task owns the socket -----------------------
        #: Messages belonging to the utterance in flight (text and binary).
        self._inbox: "asyncio.Queue[Any]" = asyncio.Queue()
        self._mode = MODE_IDLE
        self._reader_task: Optional[asyncio.Task] = None
        #: Held around every header+binary pair we send, and for the whole
        #: duration of a streamed utterance: the server routes incoming binary
        #: frames by the header that announced them, so a camera JPEG must never
        #: interleave with microphone audio or with a screenshot.
        self._wire_lock = asyncio.Lock()
        # -- proactive (unprompted) playback state, owned by the reader ---
        self._idle_stream_active = False   # between a proactive tts_start/tts_end
        self._idle_tts_active = False      # ...and the speaker is accepting it
        self._idle_tts_bytes = 0
        self._idle_interrupted = False     # the wake word cut the greeting
        self._idle_playing = False         # audio still queued for the speaker
        self._idle_drain_task: Optional[asyncio.Task] = None

        camera_cfg = _attr(self.ccfg, "camera")
        self.camera: Optional[Any] = None
        if CameraService is None:
            log.debug("Camera support is not importable: %s", _CAMERA_IMPORT_ERROR)
        elif camera_cfg is None:
            log.debug("No client.camera section in the config - running voice only")
        else:
            self.camera = CameraService(camera_cfg)

    # ------------------------------------------------------------------
    # setup / teardown
    # ------------------------------------------------------------------
    def _install_signal_handler(self) -> None:
        def handler(signum, frame):  # noqa: ANN001
            if self._stopping:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                raise KeyboardInterrupt
            self._stopping = True
            log.info("Ctrl+C received - shutting down...")

        try:
            signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError) as exc:  # pragma: no cover - non-main thread
            log.debug("Could not install the SIGINT handler: %s", exc)

    async def _setup_wakeword(self) -> None:
        wake_cfg = self.ccfg.wakeword
        phrases = [p for p in (list(_attr(wake_cfg, "phrases") or [])) if str(p).strip()]
        if not phrases:
            phrases = [str(wake_cfg.word)]
        model_path = resolve_path(wake_cfg.vosk_model)
        self.wake = await asyncio.to_thread(
            WakeWordDetector, model_path, phrases, self.sample_rate
        )

    async def _shutdown(self) -> None:
        self._stopping = True
        await self._stop_reader()
        await self._cancel_task(self._idle_drain_task, "proactive playback")
        self._idle_drain_task = None
        camera = self.camera
        if camera is not None:
            try:
                await asyncio.to_thread(camera.stop)
            except Exception as exc:  # pragma: no cover - teardown
                log.debug("Error while stopping the camera: %s", exc)
        task, self._action_task = self._action_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - teardown
                log.debug("Error while finishing actions: %s", exc)
        try:
            self.audio_in.close()
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while closing the microphone: %s", exc)
        try:
            await self.audio_out.aclose()
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while closing the audio output: %s", exc)
        try:
            await self.ws.close()
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while closing the connection: %s", exc)
        if self._started:
            log.info("Client stopped")

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task], what: str) -> None:
        """Cancel a helper task and swallow whatever it ends with."""
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - helpers must not break teardown
            log.debug("The %s task ended with: %s", what, exc)

    def _start_camera(self) -> None:
        """Start the optional camera service (SPEC v1.4); never fatal."""
        camera = self.camera
        if camera is None:
            if _CAMERA_IMPORT_ERROR is not None:
                log.warning(
                    "Camera support is unavailable (%s) - running voice only",
                    _CAMERA_IMPORT_ERROR,
                )
            return
        try:
            camera.start(
                asyncio.get_running_loop(),
                self.ws.send_json,
                self.ws.send_bytes,
                send_lock=self._wire_lock,
            )
        except Exception as exc:  # noqa: BLE001 - the camera is never worth a crash
            log.warning("Could not start the camera service: %s - running voice only", exc)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._install_signal_handler()
        try:
            await self._setup_wakeword()
            self.audio_in.start()
            self._start_camera()
            self._started = True
            log.info(
                "Jarvis client started: client_id=%s, server=%s",
                _attr(self.ccfg, "client_id"),
                self.ccfg.server_url,
            )
            while not self._stopping:
                try:
                    await self._ensure_link()
                    await self._conversation()
                except _Stopping:
                    break
                except WSDisconnected as exc:
                    if self._stopping:
                        break
                    log.warning("Lost the connection to the server: %s", exc)
                    await self._beep(ERROR_BEEP_FREQ_HZ, ERROR_BEEP_MS)
        finally:
            await self._shutdown()

    # ------------------------------------------------------------------
    # v1.4: the connection and its single reader task
    # ------------------------------------------------------------------
    def _reader_alive(self) -> bool:
        task = self._reader_task
        return task is not None and not task.done()

    async def _ensure_link(self) -> None:
        """Guarantee a live connection with exactly ONE task reading it.

        The handshake in ``ws.ensure_connected`` consumes messages itself, so
        the reader is always stopped first: two consumers on one socket would
        each swallow half of the reply.
        """
        if self.ws.connected and self._reader_alive():
            return
        await self._stop_reader()
        await self.ws.ensure_connected()
        self._start_reader()

    def _start_reader(self) -> None:
        if self._reader_alive():
            return
        self._drain_inbox("stale message")
        self._reader_task = asyncio.get_running_loop().create_task(
            self._reader_loop(), name="jarvis-ws-reader"
        )
        log.debug("The socket reader task is running")

    async def _stop_reader(self) -> None:
        task, self._reader_task = self._reader_task, None
        await self._cancel_task(task, "socket reader")
        self._idle_stream_active = False
        self._idle_tts_active = False
        self._drain_inbox("message from the previous connection")

    def _drain_inbox(self, what: str) -> int:
        """Throw away buffered messages that can no longer belong to a reply."""
        dropped = 0
        while True:
            try:
                item = self._inbox.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not LINK_DOWN:
                dropped += 1
        if dropped:
            log.debug("Discarded %d %s(s)", dropped, what)
        return dropped

    async def _reader_loop(self) -> None:
        """Own ``ws.recv()`` for the lifetime of this connection (SPEC v1.4)."""
        try:
            while not self._stopping:
                msg = await self.ws.recv(timeout=WAIT_FOREVER)
                await self._route_message(msg)
        except asyncio.CancelledError:
            raise
        except WSDisconnected as exc:
            if not self._stopping:
                log.info("The socket reader stopped: %s", exc)
        except Exception as exc:  # noqa: BLE001 - one bad message must not deafen us
            log.error("The socket reader crashed: %s", exc)
            log.debug("Details:", exc_info=True)
            self.ws.drop()
        finally:
            self._idle_stream_active = False
            self._idle_tts_active = False
            try:
                self._inbox.put_nowait(LINK_DOWN)
            except Exception:  # pragma: no cover - an unbounded queue cannot fail
                pass

    async def _route_message(self, msg: Any) -> None:
        """Send one server message where it belongs (see the module docstring)."""
        if isinstance(msg, bytes):
            if self._idle_stream_active:
                await self._on_idle_tts_chunk(msg)
            elif self._mode == MODE_CONVERSATION:
                self._inbox.put_nowait(msg)
            else:
                log.debug("Binary frame outside of any stream (%d bytes) - ignored", len(msg))
            return

        mtype = msg.get("type")
        if mtype == MSG_CAMERA_REQUEST:
            # Answered in both modes: the server pulls frames for
            # look_at_camera / enroll_face whenever it likes.
            await self._handle_camera_request(msg)
            return
        if self._idle_stream_active:
            # The tail of a proactive block stays with the reader even if a
            # conversation started meanwhile (the wake word interrupted it):
            # its binary frames are NOT the reply the conversation waits for.
            if mtype == MSG_TTS_END:
                await self._on_idle_tts_end()
                return
            if mtype == MSG_TTS_START:  # pragma: no cover - defensive
                log.debug("A new TTS stream began before the proactive one ended")
                await self._on_idle_tts_end()
        if self._mode == MODE_CONVERSATION:
            self._inbox.put_nowait(msg)
            return
        await self._handle_idle_message(mtype, msg)

    async def _handle_idle_message(self, mtype: Any, msg: Dict[str, Any]) -> None:
        """Handle a message that arrived between utterances (SPEC v1.4)."""
        if mtype == MSG_SAY:
            self._last_say = str(msg.get("text") or "").strip()
            log.info("Unprompted message: %s", self._last_say or "(empty)")
        elif mtype == MSG_TTS_START:
            await self._on_idle_tts_start(msg)
        elif mtype == MSG_TTS_END:
            log.debug("tts_end without a proactive stream - ignored")
        elif mtype == MSG_SCREENSHOT_REQUEST:
            await self._handle_screenshot_request(msg)
        elif mtype == MSG_ACTIONS:
            log.info("Actions received between utterances")
            await self._start_actions(msg.get("items") or [])
        elif mtype == MSG_TRANSCRIPT:
            log.debug("Transcript outside of a conversation: %s", msg.get("text"))
        elif mtype == MSG_ERROR:
            log.error("Server error between utterances: %s", msg.get("message"))
        elif mtype == MSG_READY:
            log.debug("The server sent ready")
        else:
            log.warning("Unknown message type from the server: %r", mtype)

    # -- proactive playback (greetings): only ever while idle ----------------

    async def _on_idle_tts_start(self, msg: Dict[str, Any]) -> None:
        sample_rate = int(msg.get("sr") or self.sample_rate)
        fmt = str(msg.get("format") or PCM_FORMAT)
        channels = int(msg.get("channels") or CHANNELS)
        if fmt != PCM_FORMAT or channels != CHANNELS:
            log.warning(
                "The server announced an unexpected TTS format: %s, %d channel(s) - playing it as %s mono",
                fmt, channels, PCM_FORMAT,
            )
        self._idle_tts_bytes = 0
        self._idle_interrupted = False
        # Set before opening the device: even if playback fails, the frames of
        # this block belong to the reader and must not reach the inbox.
        self._idle_stream_active = True
        try:
            await self.audio_out.open(sample_rate)
        except Exception as exc:  # noqa: BLE001 - audio device issues
            log.error("Could not open the audio output at %d Hz: %s", sample_rate, exc)
            self._idle_tts_active = False
            return
        self._idle_tts_active = True
        self._idle_playing = True
        log.info("Playing an unprompted message (%d Hz)", sample_rate)

    async def _on_idle_tts_chunk(self, data: bytes) -> None:
        if not self._idle_tts_active or self._idle_interrupted:
            return  # interrupted or muted: swallow the rest of the block
        self._idle_tts_bytes += len(data)
        await self.audio_out.write(data)

    async def _on_idle_tts_end(self) -> None:
        self._idle_stream_active = False
        self._idle_tts_active = False
        if self._idle_interrupted:
            log.debug("The unprompted message was cut after %d bytes", self._idle_tts_bytes)
        elif self._idle_tts_bytes:
            log.debug("Played %d bytes of the unprompted message", self._idle_tts_bytes)
        else:
            log.info("The unprompted message carried no audio")
        # Waiting for the speaker to go quiet must not stop the reader from
        # reading, so the drain runs in its own task.
        if self._idle_drain_task is None or self._idle_drain_task.done():
            self._idle_drain_task = asyncio.get_running_loop().create_task(
                self._finish_idle_playback(), name="jarvis-proactive-drain"
            )

    async def _finish_idle_playback(self) -> None:
        try:
            await self.audio_out.drain()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Error while finishing the unprompted playback: %s", exc)
        finally:
            if not self._idle_stream_active:
                interrupted = self._idle_interrupted
                self._idle_playing = False
                self._idle_interrupted = False
                if not interrupted and self._idle_tts_bytes:
                    # The greeting asked a question: give the person a window to
                    # simply answer instead of demanding the wake word first.
                    self._proactive_listen_s = PROACTIVE_LISTEN_S
                    log.info(
                        "Proactive message finished - listening for an answer "
                        "for %.0f s without the wake word", PROACTIVE_LISTEN_S,
                    )

    def _interrupt_idle_playback(self) -> bool:
        """Cut an unprompted message short; ``True`` if there was one.

        Called when the wake word fires and when a new utterance starts. The
        ``_idle_stream_active`` flag stays on so the rest of the block (binary
        frames and its ``tts_end``) is still swallowed by the reader.
        """
        if not (self._idle_stream_active or self._idle_tts_active or self._idle_playing):
            return False
        already = self._idle_interrupted
        self._idle_interrupted = True
        self._idle_tts_active = False
        dropped = self.audio_out.cancel_pending()
        if not already:
            log.info("Interrupting the unprompted message")
        log.debug("Dropped %d queued playback chunk(s)", dropped)
        return True

    # -- conversation mode flag (owned by the conversation loop) -------------

    def _enter_conversation(self) -> None:
        """From now on every server message belongs to the utterance in flight."""
        self._interrupt_idle_playback()
        self._drain_inbox("stale message")
        self._mode = MODE_CONVERSATION

    def _leave_conversation(self) -> None:
        """Back to idle routing; anything left over cannot belong to a reply."""
        self._mode = MODE_IDLE
        self._drain_inbox("leftover reply message")

    async def _next_message(self) -> Any:
        """Next message of the conversation, buffered by the reader task.

        Raises :class:`WSDisconnected` when the link died or the server went
        quiet for longer than the transport's receive timeout — the same
        contract ``ws.recv()`` had before the reader existed.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(getattr(self.ws, "recv_timeout", 420.0))
        while True:
            try:
                item = await asyncio.wait_for(self._inbox.get(), timeout=INBOX_POLL_S)
            except (asyncio.TimeoutError, TimeoutError):
                if self._stopping:
                    raise _Stopping()
                if not self._reader_alive() and self._inbox.empty():
                    raise WSDisconnected("the connection was lost while waiting for the reply")
                if loop.time() >= deadline:
                    self.ws.drop()
                    raise WSDisconnected("the server is not responding")
                continue
            if item is LINK_DOWN:
                raise WSDisconnected("the connection was lost while waiting for the reply")
            return item

    async def _conversation(self) -> None:
        """One wake-word trigger plus any follow-up turns."""
        if not await self._wait_for_wakeword():
            return
        proactive = self._proactive_listen_s
        self._proactive_listen_s = 0.0
        pre_roll = self.preroll.snapshot()
        self.preroll.clear()
        await self._beep(BEEP_FREQ_HZ, BEEP_MS)
        pre_roll += self._drain_beep_window()
        lead_in: Optional[float] = proactive if proactive > 0 else None
        while not self._stopping:
            result = await self._handle_utterance(pre_roll, lead_in)
            if result == RESULT_BARGE_IN:
                # The wake word cut the reply short: acknowledge and listen for
                # the new command right away, no wake word needed.
                await self._beep(BEEP_FREQ_HZ, BEEP_MS)
                pre_roll = self._drain_beep_window()
                lead_in = None
                continue
            # SPEC §4: the server may ask for a longer follow-up window via
            # say.listen_s (voice enrollment needs room to keep talking).
            hint = self._listen_hint_s
            self._listen_hint_s = 0.0
            window = max(self.followup_window_s, hint)
            if result != RESULT_OK or window <= 0:
                return
            pre_roll = b""
            lead_in = window
            await asyncio.sleep(FOLLOWUP_ECHO_GUARD_S)
            self.audio_in.clear()
            log.info(
                "Listening for a follow-up for %.1f s (no wake word needed)...",
                window,
            )

    def _drain_beep_window(self) -> bytes:
        """Return the speech captured while the ack beep played (beep itself dropped).

        The queue holds everything recorded since the wake word fired: first the
        user's next words (the output stream is still opening), then the echo of
        our own tone at the very end. Keep the former, drop the latter — feeding
        the beep to the VAD would make every false trigger look like speech.
        """
        frames: List[bytes] = []
        while True:
            frame = self.audio_in.read_frame_nowait()
            if frame is None:
                break
            frames.append(frame)
        if not frames:
            return b""
        skip = int(math.ceil((BEEP_MS + BEEP_ECHO_GUARD_MS) / float(self.frame_ms)))
        kept = frames[:-skip] if skip < len(frames) else []
        if kept:
            log.debug("Kept %d frame(s) recorded while the beep played", len(kept))
        return b"".join(kept)

    async def _wait_for_wakeword(self) -> bool:
        if self.wake is None:  # pragma: no cover - run() always initialises it
            raise RuntimeError("The wake-word detector is not initialised")
        word = _attr(self.ccfg.wakeword, "word")
        log.info("Waiting for the wake word '%s'...", word)
        self.preroll.clear()
        self.audio_in.clear()
        self.wake.reset()
        while not self._stopping:
            if not self.ws.connected or not self._reader_alive():
                log.info("No connection to the server - reconnecting...")
                await self._ensure_link()
                self.audio_in.clear()
                self.preroll.clear()
                self.wake.reset()
                log.info("Waiting for the wake word '%s'...", word)
            if self._proactive_listen_s > 0:
                log.info("Answer window after a proactive message - no wake word needed")
                self.wake.reset()
                return True
            frame = await self.audio_in.read_frame(timeout=0.5)
            if not frame:
                continue
            self.preroll.push(frame)
            if self.wake.accept_frame(frame):
                log.info("Wake word detected")
                # While idle this loop is also the barge-in watcher: an
                # unprompted greeting is cut off here and the client goes
                # straight on to record what the user wants (SPEC v1.4).
                self._interrupt_idle_playback()
                self.wake.reset()
                return True
        return False

    # ------------------------------------------------------------------
    # one utterance
    # ------------------------------------------------------------------
    async def _read_frame(self) -> Optional[bytes]:
        if self._stopping:
            raise _Stopping()
        frame = await self.audio_in.read_frame(timeout=0.5)
        if self._stopping:
            raise _Stopping()
        return frame

    def _split_frames(self, chunk: bytes) -> Iterator[bytes]:
        """Split buffered audio into ~30 ms pieces for streaming (SPEC §7 step 4)."""
        size = self.frame_bytes
        for start in range(0, len(chunk), size):
            piece = chunk[start : start + size]
            if piece:
                yield piece

    async def _handle_utterance(
        self, pre_roll: bytes, lead_in_s: Optional[float]
    ) -> str:
        sent_start = False
        holding_wire = False

        async def on_audio(chunk: bytes) -> None:
            nonlocal sent_start, holding_wire
            if not sent_start:
                # Everything binary the server receives between utterance_start
                # and utterance_end is microphone audio (SPEC §4), so the wire is
                # held for the whole stream: a camera frame would corrupt it.
                await self._wire_lock.acquire()
                holding_wire = True
                # ...and from this exact point on every incoming message belongs
                # to this utterance, not to an unprompted greeting.
                self._enter_conversation()
                await self.ws.send_json(
                    {
                        "type": MSG_UTTERANCE_START,
                        "sr": self.sample_rate,
                        "format": PCM_FORMAT,
                        "channels": CHANNELS,
                    }
                )
                sent_start = True
                log.info("Streaming the utterance to the server...")
            for piece in self._split_frames(chunk):
                await self.ws.send_bytes(piece)

        try:
            try:
                audio = await self.vad.record(
                    self._read_frame,
                    pre_roll=pre_roll,
                    lead_in_s=lead_in_s,
                    on_audio=on_audio,
                )
                if audio is None:
                    # on_audio was never called: nothing was sent, so this was a
                    # false trigger and the mode flag was never flipped.
                    log.info("No speech detected - back to waiting for the wake word")
                    await self._beep(NO_SPEECH_BEEP_FREQ_HZ, NO_SPEECH_BEEP_MS, volume=0.22)
                    return RESULT_NO_SPEECH

                await self.ws.send_json({"type": MSG_UTTERANCE_END})
                log.debug(
                    "Sent %.2f s of audio",
                    len(audio) / float(self.sample_rate * SAMPLE_WIDTH),
                )
            finally:
                # Released before waiting for the reply: the reply's rounds may
                # need the wire themselves (screenshot), and the camera should
                # get its turn again while the server thinks.
                if holding_wire:
                    holding_wire = False
                    self._wire_lock.release()
            return await self._receive_response()
        finally:
            self._leave_conversation()

    async def _receive_response(self) -> str:
        """Handle every server message for one utterance (SPEC §4).

        ``actions`` and ``screenshot_request`` may arrive several times and in
        any order before the reply, so both are handled inside this loop; it
        ends with ``tts_end`` or ``error``. Since v1.4 the messages come from
        the reader task through :meth:`_next_message` instead of the socket —
        whatever arrived while the microphone was still streaming is already
        waiting in the inbox.
        """
        self._tts_active = False
        self._tts_bytes = 0
        self._barged = False
        result = RESULT_OK
        self._start_thinking()
        try:
            while True:
                msg = await self._next_message()
                if isinstance(msg, bytes):
                    await self._on_tts_chunk(msg)
                    continue

                mtype = msg.get("type")
                if mtype == MSG_TRANSCRIPT:
                    text = str(msg.get("text") or "").strip()
                    language = str(msg.get("language") or "?")
                    log.info("Recognised [%s]: %s", language, text or "(empty)")
                elif mtype == MSG_ACTIONS:
                    await self._start_actions(msg.get("items") or [])
                elif mtype == MSG_SCREENSHOT_REQUEST:
                    await self._handle_screenshot_request(msg)
                elif mtype == MSG_SAY:
                    await self._stop_thinking()
                    self._last_say = str(msg.get("text") or "").strip()
                    try:
                        self._listen_hint_s = float(msg.get("listen_s") or 0.0)
                    except (TypeError, ValueError):
                        self._listen_hint_s = 0.0
                    log.info("Reply: %s", self._last_say or "(empty)")
                elif mtype == MSG_TTS_START:
                    await self._stop_thinking()
                    await self._on_tts_start(msg)
                    self._start_barge_watch()
                elif mtype == MSG_TTS_END:
                    self._tts_active = False
                    await self._stop_barge_watch()
                    await self.audio_out.drain()
                    if self._barged:
                        result = RESULT_BARGE_IN
                    elif self._tts_bytes == 0:
                        log.info("The TTS stream was empty - nothing to play")
                    else:
                        log.debug("Played %d bytes of TTS", self._tts_bytes)
                    break
                elif mtype == MSG_ERROR:
                    await self._stop_thinking()
                    log.error("Server error: %s", msg.get("message"))
                    self._tts_active = False
                    await self._beep(ERROR_BEEP_FREQ_HZ, ERROR_BEEP_MS)
                    result = RESULT_ERROR
                    break
                elif mtype == MSG_READY:
                    log.debug("The server sent ready")
                else:
                    log.warning("Unknown message type from the server: %r", mtype)
        finally:
            await self._stop_thinking()
            await self._stop_barge_watch()
            await self._await_actions()
        return result

    # -- barge-in: the wake word interrupts playback -------------------------

    async def _barge_loop(self) -> None:
        """Listen for the wake word while the reply is playing."""
        if self.wake is None:
            return
        self.audio_in.clear()  # drop stale audio buffered while the server thought
        self.wake.reset()
        while True:
            frame = await self.audio_in.read_frame(timeout=0.5)
            if not frame:
                continue
            if self.wake.accept_frame(frame):
                log.info("Wake word during playback - interrupting the reply")
                self._barged = True
                dropped = self.audio_out.cancel_pending()
                log.debug("Dropped %d queued playback chunk(s)", dropped)
                self.wake.reset()
                return

    def _start_barge_watch(self) -> None:
        if self._barge_task is None:
            self._barge_task = asyncio.get_running_loop().create_task(
                self._barge_loop(), name="jarvis-barge-in"
            )

    async def _stop_barge_watch(self) -> None:
        task, self._barge_task = self._barge_task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - the watcher must never break the loop
            log.debug("Barge-in watcher ended with: %s", exc)

    # -- thinking sounds (the "I'm working on it" blips) ---------------------

    async def _thinking_loop(self) -> None:
        """Soft repeating blips while the server is still working on a reply."""
        await asyncio.sleep(THINKING_DELAY_S)
        while True:
            for freq in THINKING_BLIP_FREQS_HZ:
                await self._beep(freq, THINKING_BLIP_MS, volume=THINKING_VOLUME)
            await asyncio.sleep(THINKING_INTERVAL_S)

    def _start_thinking(self) -> None:
        if self.thinking_sounds and self._thinking_task is None:
            self._thinking_task = asyncio.get_running_loop().create_task(
                self._thinking_loop(), name="jarvis-thinking-sounds"
            )

    async def _stop_thinking(self) -> None:
        task, self._thinking_task = self._thinking_task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - a sound must never break the loop
            log.debug("Thinking-sound task ended with: %s", exc)

    async def _on_tts_start(self, msg: Dict[str, Any]) -> None:
        sample_rate = int(msg.get("sr") or self.sample_rate)
        fmt = str(msg.get("format") or PCM_FORMAT)
        channels = int(msg.get("channels") or CHANNELS)
        if fmt != PCM_FORMAT or channels != CHANNELS:
            log.warning(
                "The server announced an unexpected TTS format: %s, %d channel(s) - playing it as %s mono",
                fmt, channels, PCM_FORMAT,
            )
        self._tts_bytes = 0
        try:
            await self.audio_out.open(sample_rate)
        except Exception as exc:
            log.error("Could not open the audio output at %d Hz: %s", sample_rate, exc)
            self._tts_active = False
            return
        self._tts_active = True
        log.debug("Receiving TTS: %d Hz", sample_rate)

    async def _on_tts_chunk(self, data: bytes) -> None:
        if not self._tts_active:
            log.debug("Binary frame outside of a TTS stream (%d bytes) - skipping", len(data))
            return
        if self._barged:
            return  # interrupted: swallow the rest of the stream silently
        self._tts_bytes += len(data)
        await self.audio_out.write(data)

    async def _beep(self, freq: float, ms: int, volume: float = 0.35) -> None:
        try:
            await self.audio_out.play_beep(freq, ms, volume)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - audio device issues
            log.warning("Could not play the beep: %s", exc)

    # ------------------------------------------------------------------
    # screen vision (SPEC §4 S->C #4, §7)
    # ------------------------------------------------------------------
    async def _handle_screenshot_request(self, msg: Dict[str, Any]) -> None:
        """Answer ``screenshot_request``: a header plus exactly ONE binary frame.

        The header carries both sizes (SPEC §4, C->S #6): ``w``/``h`` of the
        downscaled image that is actually sent, and ``screen_w``/``screen_h`` of
        the real desktop. The server needs both to turn the vision model's pixel
        coordinates into the normalized ones a ``mouse_click`` action expects.

        A capture failure is reported as ``screenshot_error`` and no binary
        frame is sent, so the server can turn it into a tool result instead of
        waiting for the full 120 s.
        """
        request_id = str(msg.get("id") or "")
        log.info("Screenshot requested (id=%s) - capturing the screen", request_id or "?")
        try:
            capture: Capture = await asyncio.to_thread(capture_jpeg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = str(exc).strip() or exc.__class__.__name__
            log.error("Screen capture failed: %s", error)
            await self.ws.send_json(
                {"type": MSG_SCREENSHOT_ERROR, "id": request_id, "error": error}
            )
            return

        # The header and its single binary frame must stay adjacent on the wire.
        async with self._wire_lock:
            await self.ws.send_json(
                {
                    "type": MSG_SCREENSHOT,
                    "id": request_id,
                    "format": SCREENSHOT_FORMAT,
                    "w": capture.w,
                    "h": capture.h,
                    "screen_w": capture.screen_w,
                    "screen_h": capture.screen_h,
                }
            )
            await self.ws.send_bytes(capture.jpeg)
        log.info(
            "Screenshot sent (id=%s): %dx%d image of a %dx%d screen, %d bytes",
            request_id or "?",
            capture.w,
            capture.h,
            capture.screen_w,
            capture.screen_h,
            len(capture.jpeg),
        )

    # ------------------------------------------------------------------
    # room camera (SPEC v1.4, S->C camera_request)
    # ------------------------------------------------------------------
    async def _handle_camera_request(self, msg: Dict[str, Any]) -> None:
        """Answer ``camera_request`` with the newest camera frame, in any mode.

        :mod:`client.camera` owns the reply, including ``camera_error`` when it
        has no picture to give (the mirror of ``screenshot_error``). A client
        without a camera answers the error itself so the server's tool call
        fails immediately instead of waiting for its timeout.
        """
        request_id = str(msg.get("id") or "")
        log.info("Camera frame requested (id=%s)", request_id or "?")
        camera = self.camera
        if camera is None:
            await self.ws.send_json(
                {
                    "type": MSG_CAMERA_ERROR,
                    "id": request_id,
                    "error": "this client has no camera",
                }
            )
            return
        try:
            await camera.serve_request(request_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the camera never breaks the client
            log.warning("Could not answer the camera request: %s", exc)

    # ------------------------------------------------------------------
    # actions (executed by W3's dispatcher)
    # ------------------------------------------------------------------
    async def _start_actions(self, items: Any) -> None:
        if not isinstance(items, list) or not items:
            log.debug("No actions in this batch")
            return
        await self._await_actions()
        log.info("Received %d action(s)", len(items))
        self._action_task = asyncio.get_running_loop().create_task(
            self._execute_actions(list(items)), name="jarvis-actions"
        )

    async def _execute_actions(self, items: List[Any]) -> None:
        reporting = True
        for item in items:
            if not isinstance(item, dict):
                log.warning("Skipping a malformed action: %r", item)
                continue
            action_id = str(item.get("id") or "")
            tool = str(item.get("tool") or "")
            log.info("Executing %s (%s): %s", action_id or "?", tool, item.get("args") or {})
            try:
                ok, error, output = await self.dispatcher.execute(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                ok, error, output = False, f"{type(exc).__name__}: {exc}", None
                log.warning("Action %s crashed: %s", action_id or "?", exc)
            ok = bool(ok)
            output_text = clip_output(output)
            if ok:
                log.info("Action %s done", action_id or "?")
            else:
                log.warning("Action %s failed: %s", action_id or "?", error)
            if output_text:
                log.debug("Action %s output: %s", action_id or "?", output_text[:200])
            if not reporting:
                continue
            try:
                await self.ws.send_json(
                    {
                        "type": MSG_ACTION_RESULT,
                        "id": action_id,
                        "ok": ok,
                        "error": None if ok else (str(error) if error else "unknown error"),
                        "output": output_text,
                    }
                )
            except WSDisconnected as exc:
                # The server waits for these results (SPEC §4) and falls back to
                # a timeout result, but a dead socket must not cancel the actions
                # the user asked for: finish them, stop reporting, and let
                # run()/ensure_connected handle the reconnect.
                log.warning("Could not send the action result: %s", exc)
                reporting = False

    async def _await_actions(self) -> None:
        task = self._action_task
        if task is None:
            return
        if task.done():
            self._action_task = None
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=ACTION_TIMEOUT_S)
            self._action_task = None
        except (asyncio.TimeoutError, TimeoutError):
            log.warning(
                "Actions are taking longer than %.0f s - continuing, will wait later",
                ACTION_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - dispatcher must not raise
            self._action_task = None
            log.warning("Error while executing actions: %s", exc)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="client.main",
        description="Jarvis room client: wake word, VAD, streaming to the brain server",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config.yaml"),
        help="path to config.yaml (default: config.yaml in the repository root)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level: DEBUG, INFO, WARNING, ERROR",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    # Transcripts and app names may contain non-ASCII text that a cp125x/cp866
    # Windows console cannot encode - never let that kill a log call.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - redirected/exotic streams
            pass
    resolved = getattr(logging, str(level).upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(level=resolved, format=fmt, datefmt="%H:%M:%S")
    # The client normally runs with a hidden console, so the log also goes to a
    # file — that is the only way to debug it after the window went away.
    try:
        log_path = REPO_ROOT / "data" / "client.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
            log_path.unlink()  # crude rotation: start over past 5 MB
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)
    except Exception as exc:  # noqa: BLE001 - file logging is best-effort
        logging.getLogger(__name__).warning("File logging unavailable: %s", exc)
    for noisy in ("websockets", "websockets.client", "comtypes", "bleak", "asyncio", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    config_path = resolve_path(args.config)
    if not config_path.exists():
        log.error(
            "Config file not found: %s (copy config.example.yaml to config.yaml)",
            config_path,
        )
        return 2

    try:
        cfg = load_config(str(config_path))
        client = JarvisClient(cfg)
    except Exception as exc:
        log.error("Could not start the client: %s", exc)
        log.debug("Details:", exc_info=True)
        return 1

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        log.info("Interrupted by the user")
    except Exception as exc:
        log.error("The client stopped because of an error: %s", exc)
        log.debug("Details:", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
