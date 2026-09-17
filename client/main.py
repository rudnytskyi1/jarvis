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
from client.ws_client import WSClient, WSDisconnected

log = logging.getLogger("client")

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
#: Mic audio recorded while the ack beep was playing: everything captured in the
#: last ``BEEP_MS + BEEP_ECHO_GUARD_MS`` is our own beep (tone + output latency)
#: and is dropped; anything older is the user already speaking and is kept, so a
#: command said in one breath ("rowan, turn on the light") is not cut off.
BEEP_ECHO_GUARD_MS = 250

RESULT_OK = "ok"
RESULT_NO_SPEECH = "no_speech"
RESULT_ERROR = "error"


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

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._install_signal_handler()
        try:
            await self._setup_wakeword()
            self.audio_in.start()
            self._started = True
            log.info(
                "Jarvis client started: client_id=%s, server=%s",
                _attr(self.ccfg, "client_id"),
                self.ccfg.server_url,
            )
            while not self._stopping:
                try:
                    await self.ws.ensure_connected()
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

    async def _conversation(self) -> None:
        """One wake-word trigger plus any follow-up turns."""
        if not await self._wait_for_wakeword():
            return
        pre_roll = self.preroll.snapshot()
        self.preroll.clear()
        await self._beep(BEEP_FREQ_HZ, BEEP_MS)
        pre_roll += self._drain_beep_window()
        lead_in: Optional[float] = None
        while not self._stopping:
            result = await self._handle_utterance(pre_roll, lead_in)
            if result != RESULT_OK or self.followup_window_s <= 0:
                return
            pre_roll = b""
            lead_in = self.followup_window_s
            await asyncio.sleep(FOLLOWUP_ECHO_GUARD_S)
            self.audio_in.clear()
            log.info(
                "Listening for a follow-up for %.1f s (no wake word needed)...",
                self.followup_window_s,
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
            if not self.ws.connected:
                log.info("No connection to the server - reconnecting...")
                await self.ws.ensure_connected()
                self.audio_in.clear()
                self.preroll.clear()
                self.wake.reset()
                log.info("Waiting for the wake word '%s'...", word)
            frame = await self.audio_in.read_frame(timeout=0.5)
            if not frame:
                continue
            self.preroll.push(frame)
            if self.wake.accept_frame(frame):
                log.info("Wake word detected")
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

        async def on_audio(chunk: bytes) -> None:
            nonlocal sent_start
            if not sent_start:
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

        audio = await self.vad.record(
            self._read_frame,
            pre_roll=pre_roll,
            lead_in_s=lead_in_s,
            on_audio=on_audio,
        )
        if audio is None:
            log.info("No speech detected - back to waiting for the wake word")
            await self._beep(NO_SPEECH_BEEP_FREQ_HZ, NO_SPEECH_BEEP_MS, volume=0.22)
            return RESULT_NO_SPEECH

        await self.ws.send_json({"type": MSG_UTTERANCE_END})
        log.debug("Sent %.2f s of audio", len(audio) / float(self.sample_rate * SAMPLE_WIDTH))
        return await self._receive_response()

    async def _receive_response(self) -> str:
        """Handle every server message for one utterance (SPEC §4).

        ``actions`` and ``screenshot_request`` may arrive several times and in
        any order before the reply, so both are handled inside this loop; it
        ends with ``tts_end`` or ``error``.
        """
        self._tts_active = False
        self._tts_bytes = 0
        result = RESULT_OK
        try:
            while True:
                msg = await self.ws.recv()
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
                    self._last_say = str(msg.get("text") or "").strip()
                    log.info("Reply: %s", self._last_say or "(empty)")
                elif mtype == MSG_TTS_START:
                    await self._on_tts_start(msg)
                elif mtype == MSG_TTS_END:
                    self._tts_active = False
                    await self.audio_out.drain()
                    if self._tts_bytes == 0:
                        log.info("The TTS stream was empty - nothing to play")
                    else:
                        log.debug("Played %d bytes of TTS", self._tts_bytes)
                    break
                elif mtype == MSG_ERROR:
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
            await self._await_actions()
        return result

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
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
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
