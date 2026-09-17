"""FastAPI application with the ``/ws`` WebSocket endpoint (SPEC §3, §4).

One connection = one session. Per utterance the server sends, in this order:
``transcript`` -> zero or more tool rounds (``actions`` / ``screenshot_request``,
each awaited) -> ``say`` -> ``tts_start`` … binary … ``tts_end``.

The utterance pipeline runs in its own task so the receive loop keeps reading
``action_result`` and ``screenshot`` messages while the LLM waits for them. All
model inference (Whisper, LLM, Silero) runs in worker threads via
``asyncio.to_thread`` so the event loop never blocks.

Server-side tools (SPEC §5): ``look_at_screen`` describes a screenshot,
``click_screen`` locates the described element in that screenshot and sends the
client a single ``mouse_click`` action, and ``remember`` writes to the memory
file. Everything else is forwarded to the client verbatim.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from common import protocol as proto
from common.config import load_config
from server.llm import LlmClient
from server.session import Session
from server.storage import DialogLog, Memory
from server.stt import SttEngine
from server.tools import (
    CLIENT_TOOLS,
    MOUSE_CLICK_TOOL,
    action_item,
    mouse_click_args,
    normalize_click_button,
)
from server.tts import TtsEngine
from server.vision import VisionClient

log = logging.getLogger("jarvis.server.app")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
CONFIG_ENV_VAR = "JARVIS_CONFIG"

DEFAULT_INPUT_SAMPLE_RATE = proto.MIC_SAMPLE_RATE
#: ~120 s of 16 kHz mono s16le — hard cap so a broken client cannot eat all RAM.
MAX_UTTERANCE_BYTES = DEFAULT_INPUT_SAMPLE_RATE * 2 * 120
#: ~85 ms of audio at 48 kHz mono s16le per binary frame.
TTS_CHUNK_BYTES = 8192
#: 8 MB is far above a 1600 px JPEG — a bigger frame means a broken client.
MAX_SCREENSHOT_BYTES = 8 * 1024 * 1024

#: SPEC §4: how long the server waits for the client (per action / per screenshot).
ACTION_TIMEOUT_S = 35.0
SCREENSHOT_TIMEOUT_S = 120.0

#: SPEC §4: the error sent when STT produced nothing (shared with the client).
ERROR_EMPTY_TRANSCRIPT = proto.ERR_EMPTY_TRANSCRIPT
SAY_AFTER_ACTIONS = "Done."
SAY_NOT_UNDERSTOOD = "Sorry, I did not catch that."


@dataclass(frozen=True)
class Screenshot:
    """One screenshot from the client, with the sizes from its header (SPEC §4).

    ``w``/``h`` describe the JPEG itself — the vision model answers in those
    pixels — while ``screen_w``/``screen_h`` are the real desktop resolution the
    client multiplies the normalized click coordinates by.
    """

    jpeg: bytes
    w: int
    h: int
    screen_w: int
    screen_h: int


def _positive_int(value: Any) -> int | None:
    """Return ``value`` as a positive int, or ``None`` when it is missing/bogus."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Read ``(width, height)`` out of the JPEG's own SOF marker.

    Only used when a screenshot header arrives without ``w``/``h`` — per SPEC §4
    the client always sends them, but an older client must not break the
    ``look_at_screen`` tool. Returns ``None`` when the bytes are not a JPEG.
    """
    total = len(data)
    if total < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    index = 2
    while index + 3 < total:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker == 0xFF:  # fill byte: the real marker follows
            index -= 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:  # standalone markers
            continue
        if marker == 0xDA:  # start of scan: entropy-coded data, no sizes left
            return None
        if index + 2 > total:
            return None
        length = int.from_bytes(data[index : index + 2], "big")
        if length < 2:
            return None
        # SOF0..SOF15 carry the frame size; C4/C8/CC are Huffman/arithmetic tables.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if index + 7 > total:
                return None
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            if width > 0 and height > 0:
                return width, height
            return None
        index += length
    return None


_config: Any = None
_stt: SttEngine | None = None
_llm: LlmClient | None = None
_tts: TtsEngine | None = None
_vision: VisionClient | None = None
_memory: Memory | None = None
_dialogs: DialogLog | None = None


def configure(cfg: Any) -> None:
    """Inject the loaded config before starting uvicorn (used by ``server.main``)."""
    global _config
    _config = cfg


def get_config() -> Any:
    """Return the config, loading it from ``JARVIS_CONFIG``/``config.yaml`` if needed."""
    global _config
    if _config is None:
        path = os.environ.get(CONFIG_ENV_VAR) or str(DEFAULT_CONFIG_PATH)
        log.info("No config was injected — loading %s", path)
        _config = load_config(path)
    return _config


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load STT/LLM/TTS/vision and the storage once at startup (SPEC §3)."""
    global _stt, _llm, _tts, _vision, _memory, _dialogs
    cfg = get_config()
    log.info("Starting the Jarvis brain: %s:%s", cfg.server.host, cfg.server.port)

    _memory = Memory()
    _dialogs = DialogLog()
    _stt = await asyncio.to_thread(SttEngine, cfg.server.stt)
    _llm = LlmClient(cfg.server.llm)
    _vision = VisionClient(cfg.server.llm)
    _tts = TtsEngine(cfg.server.tts)
    await asyncio.to_thread(_tts.load)
    log.info("Server ready for connections on /ws")
    try:
        yield
    finally:
        log.info("Shutting the server down")
        if _llm is not None:
            _llm.close()
        if _vision is not None:
            _vision.close()
        _stt = None
        _llm = None
        _tts = None
        _vision = None
        _memory = None
        _dialogs = None


app = FastAPI(title="Jarvis brain server", version="1.2", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Tiny status endpoint — handy when checking the firewall/port from the client."""
    return {
        "status": "ok",
        "stt": _stt is not None,
        "llm": _llm is not None,
        "tts": bool(_tts is not None and _tts.available),
    }


class Connection:
    """Handles one client WebSocket connection."""

    def __init__(self, websocket: WebSocket, cfg: Any) -> None:
        self.ws = websocket
        self.cfg = cfg
        self.session: Session | None = None
        self.audio = bytearray()
        self.receiving = False
        self.overflow = False
        self.sample_rate = DEFAULT_INPUT_SAMPLE_RATE
        self.peer = "?"
        client = getattr(websocket, "client", None)
        if client is not None:
            self.peer = f"{getattr(client, 'host', '?')}:{getattr(client, 'port', '?')}"

        # --- awaited client answers (SPEC §4) ---
        self._pending_actions: dict[str, asyncio.Future] = {}
        self._screenshot_future: asyncio.Future | None = None
        self._screenshot_id: str | None = None
        #: Set by a ``screenshot`` header: the next binary frame is the JPEG.
        self._expect_screenshot_bytes = False
        #: Sizes from the last ``screenshot`` header, consumed by that JPEG frame.
        self._screenshot_header: dict[str, int | None] = {}

        # --- per-utterance state ---
        self._action_seq = 1
        self._screenshot_seq = 1
        self._memory_seq = 1
        self._utterance_actions: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ sending

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.ws.client_state is not WebSocketState.CONNECTED:
            raise WebSocketDisconnect(code=1006)
        await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def send_error(self, message: str) -> None:
        log.warning("Error sent to client %s: %s", self.peer, message)
        try:
            await self.send_json({"type": proto.MSG_ERROR, "message": message})
        except (WebSocketDisconnect, RuntimeError):
            log.info("Client %s is gone — the error could not be delivered", self.peer)

    # ------------------------------------------------------------------ receiving

    async def run(self) -> None:
        while True:
            message = await self.ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1000))
            text = message.get("text")
            if text is not None:
                await self._on_text(text)
                continue
            data = message.get("bytes")
            if data is not None:
                self._on_binary(data)

    async def _on_text(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            await self.send_error("bad json")
            return
        if not isinstance(payload, dict):
            await self.send_error("bad message")
            return

        msg_type = payload.get("type")
        if msg_type == proto.MSG_HELLO:
            await self._on_hello(payload)
        elif msg_type == proto.MSG_UTTERANCE_START:
            self._on_utterance_start(payload)
        elif msg_type == proto.MSG_UTTERANCE_END:
            await self._on_utterance_end()
        elif msg_type == proto.MSG_ACTION_RESULT:
            self._on_action_result(payload)
        elif msg_type == proto.MSG_SCREENSHOT:
            self._on_screenshot_header(payload)
        elif msg_type == proto.MSG_SCREENSHOT_ERROR:
            self._on_screenshot_error(payload)
        else:
            log.warning("Unknown message type from %s: %r", self.peer, msg_type)

    async def _on_hello(self, payload: dict[str, Any]) -> None:
        devices = payload.get("devices")
        if not isinstance(devices, list):
            devices = []
        client_id = payload.get("client_id")
        facts: list[str] = []
        if _memory is not None:
            facts = await asyncio.to_thread(_memory.facts)
        self.session = Session(
            client_id=str(client_id) if client_id else None,
            devices=devices,
            history_turns=self.cfg.server.llm.history_turns,
            memory_facts=facts,
        )
        log.info(
            "Client %s (%s) connected, devices: %s",
            self.session.client_id,
            self.peer,
            ", ".join(self.session.device_names) or "none",
        )
        await self.send_json({"type": proto.MSG_READY})

    def _on_utterance_start(self, payload: dict[str, Any]) -> None:
        try:
            self.sample_rate = int(payload.get("sr") or DEFAULT_INPUT_SAMPLE_RATE)
        except (TypeError, ValueError):
            self.sample_rate = DEFAULT_INPUT_SAMPLE_RATE
        audio_format = payload.get("format") or proto.AUDIO_FORMAT
        if audio_format != proto.AUDIO_FORMAT:
            log.warning(
                "Client %s announced format %r — expecting %s",
                self.peer, audio_format, proto.AUDIO_FORMAT,
            )
        self.audio = bytearray()
        self.receiving = True
        self.overflow = False
        log.info("Utterance started by %s (%d Hz)", self.peer, self.sample_rate)

    def _on_binary(self, data: bytes) -> None:
        # SPEC §4: a binary frame is either the JPEG announced by a screenshot
        # header or a chunk of the utterance currently being recorded.
        if self._expect_screenshot_bytes:
            self._deliver_screenshot(data)
            return
        if not self.receiving:
            log.warning("Audio without utterance_start from %s — dropping %d bytes", self.peer, len(data))
            return
        if len(self.audio) + len(data) > MAX_UTTERANCE_BYTES:
            if not self.overflow:
                log.warning("Utterance from %s is over the limit — truncating", self.peer)
                self.overflow = True
            room = MAX_UTTERANCE_BYTES - len(self.audio)
            if room > 0:
                self.audio.extend(data[:room])
            return
        self.audio.extend(data)

    # ------------------------------------------------------- awaited client answers

    def _on_action_result(self, payload: dict[str, Any]) -> None:
        """Hand the client's result to the tool call waiting for it (SPEC §4)."""
        action_id = str(payload.get("id") or "")
        ok = bool(payload.get("ok"))
        error = payload.get("error")
        output = payload.get("output")
        log.log(
            logging.INFO if ok else logging.WARNING,
            "Action result %s from %s: ok=%s error=%s output=%s",
            action_id or "?",
            self.peer,
            ok,
            error,
            f"{len(str(output))} chars" if output else "-",
        )
        future = self._pending_actions.get(action_id)
        if future is None:
            log.warning("No tool call is waiting for action result %r", action_id)
            return
        result: dict[str, Any] = {"ok": ok}
        if error:
            result["error"] = str(error)
        if output is not None:
            result["output"] = str(output)
        if not future.done():
            future.set_result(result)

    def _on_screenshot_header(self, payload: dict[str, Any]) -> None:
        """A ``screenshot`` header announces exactly one binary frame (SPEC §4)."""
        shot_id = str(payload.get("id") or "")
        if self._screenshot_future is None or self._screenshot_future.done():
            log.warning("Screenshot header %r arrived with nothing waiting for it", shot_id)
            return
        if shot_id and self._screenshot_id and shot_id != self._screenshot_id:
            log.warning(
                "Screenshot id mismatch: got %r, waiting for %r", shot_id, self._screenshot_id
            )
        # SPEC §4: the header carries the image size and the desktop resolution;
        # both are kept next to the bytes so click_screen can aim the cursor.
        self._screenshot_header = {
            "w": _positive_int(payload.get("w")),
            "h": _positive_int(payload.get("h")),
            "screen_w": _positive_int(payload.get("screen_w")),
            "screen_h": _positive_int(payload.get("screen_h")),
        }
        self._expect_screenshot_bytes = True

    def _on_screenshot_error(self, payload: dict[str, Any]) -> None:
        error = str(payload.get("error") or "screenshot failed")
        log.warning("Client %s could not capture the screen: %s", self.peer, error)
        self._expect_screenshot_bytes = False
        self._screenshot_header = {}
        future = self._screenshot_future
        if future is not None and not future.done():
            future.set_result({"error": error})

    def _deliver_screenshot(self, data: bytes) -> None:
        self._expect_screenshot_bytes = False
        header = self._screenshot_header
        self._screenshot_header = {}
        future = self._screenshot_future
        if future is None or future.done():
            log.warning("Screenshot bytes arrived with nothing waiting for them")
            return
        if len(data) > MAX_SCREENSHOT_BYTES:
            log.warning("Screenshot from %s is too large (%d bytes)", self.peer, len(data))
            future.set_result({"error": "screenshot too large"})
            return

        jpeg = bytes(data)
        width = header.get("w")
        height = header.get("h")
        if width is None or height is None:
            # The client always sends the size (SPEC §4); measure the JPEG itself
            # only so an older client keeps working for look_at_screen.
            measured = _jpeg_size(jpeg)
            if measured is not None:
                width = width or measured[0]
                height = height or measured[1]
                log.info("Screenshot header had no size — the JPEG says %dx%d", width, height)
            else:
                log.warning("Screenshot header had no size and the JPEG could not be measured")
        shot = Screenshot(
            jpeg=jpeg,
            w=int(width or 0),
            h=int(height or 0),
            screen_w=int(header.get("screen_w") or width or 0),
            screen_h=int(header.get("screen_h") or height or 0),
        )
        log.info(
            "Screenshot received from %s (%d KB, image %dx%d, screen %dx%d)",
            self.peer, len(jpeg) // 1024, shot.w, shot.h, shot.screen_w, shot.screen_h,
        )
        future.set_result(shot)

    # ------------------------------------------------------------------ tool executor

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """ToolExecutor for :meth:`server.llm.LlmClient.generate` (SPEC §5 matrix)."""
        if name in CLIENT_TOOLS:
            return await self._run_client_action(name, args)
        if name == "look_at_screen":
            return await self._run_look_at_screen(args)
        if name == "click_screen":
            return await self._run_click_screen(args)
        if name == "remember":
            return await self._run_remember(args)
        log.warning("Tool %r has no server-side handler", name)
        return {"ok": False, "error": f"unknown tool: {name}"}

    async def _run_client_action(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Send one action to the client and wait for its ``action_result``."""
        action_id = f"a{self._action_seq}"
        self._action_seq += 1
        item = action_item(action_id, name, args)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_actions[action_id] = future
        record: dict[str, Any] = {"id": action_id, "tool": name, "args": item["args"]}
        self._utterance_actions.append(record)

        try:
            await self.send_json({"type": proto.MSG_ACTIONS, "items": [item]})
            log.info("Sent action %s: %s%s", action_id, name, item["args"])
            result = await asyncio.wait_for(future, timeout=ACTION_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("Action %s (%s) timed out after %.0f s", action_id, name, ACTION_TIMEOUT_S)
            result = {"ok": False, "error": proto.ERR_CLIENT_TIMEOUT}
        except (WebSocketDisconnect, RuntimeError) as exc:
            log.warning("Could not send action %s: %s", action_id, exc)
            result = {"ok": False, "error": "client disconnected"}
        finally:
            self._pending_actions.pop(action_id, None)

        record["result"] = result
        return result

    async def _request_screenshot(self, shot_id: str) -> Screenshot | str:
        """Ask the client for a screenshot (SPEC §4).

        Returns the :class:`Screenshot` on success, or an error message ready to
        be handed to the LLM as a tool result.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._screenshot_future = future
        self._screenshot_id = shot_id
        try:
            await self.send_json({"type": proto.MSG_SCREENSHOT_REQUEST, "id": shot_id})
            log.info("Requested a screenshot (%s)", shot_id)
            captured = await asyncio.wait_for(future, timeout=SCREENSHOT_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("Screenshot %s timed out after %.0f s", shot_id, SCREENSHOT_TIMEOUT_S)
            return proto.ERR_CLIENT_TIMEOUT
        except (WebSocketDisconnect, RuntimeError) as exc:
            log.warning("Could not request screenshot %s: %s", shot_id, exc)
            return "client disconnected"
        finally:
            self._screenshot_future = None
            self._screenshot_id = None
            self._expect_screenshot_bytes = False
            self._screenshot_header = {}

        if isinstance(captured, Screenshot):
            return captured
        if isinstance(captured, dict):
            return str(captured.get("error") or "screenshot failed")
        return "screenshot failed"

    async def _run_look_at_screen(self, args: dict[str, Any]) -> dict[str, Any]:
        """Ask the client for a screenshot and describe it with the vision model."""
        query = " ".join(str(args.get("query") or "").split())
        shot_id = f"s{self._screenshot_seq}"
        self._screenshot_seq += 1
        record: dict[str, Any] = {"id": shot_id, "tool": "look_at_screen", "args": dict(args)}
        self._utterance_actions.append(record)

        if _vision is None:
            record["result"] = {"ok": False, "error": "vision model is not loaded"}
            return record["result"]

        log.info("look_at_screen (%s): %r", shot_id, query)
        captured = await self._request_screenshot(shot_id)
        if isinstance(captured, str):
            result: dict[str, Any] = {"ok": False, "error": captured}
        else:
            answer = await _vision.describe_screenshot(captured.jpeg, query)
            result = {"ok": True, "answer": answer}

        record["result"] = result
        return result

    async def _run_click_screen(self, args: dict[str, Any]) -> dict[str, Any]:
        """Locate a described element on the screen and click it (SPEC §5, tool 6).

        Screenshot -> ``vision.locate_on_screen`` (pixels of the received image)
        -> one ``mouse_click`` action with normalized coordinates -> the client's
        own ``action_result``, which is what the LLM gets back.
        """
        target = " ".join(str(args.get("target") or "").split())
        button = normalize_click_button(args.get("button"))
        shot_id = f"s{self._screenshot_seq}"
        self._screenshot_seq += 1
        record: dict[str, Any] = {"id": shot_id, "tool": "click_screen", "args": dict(args)}
        self._utterance_actions.append(record)

        if not target:
            record["result"] = {
                "ok": False,
                "error": "click_screen needs a target: describe what to click",
            }
            return record["result"]
        if _vision is None:
            record["result"] = {"ok": False, "error": "vision model is not loaded"}
            return record["result"]

        log.info("click_screen (%s): %r with the %s button", shot_id, target, button)
        captured = await self._request_screenshot(shot_id)
        if isinstance(captured, str):
            record["result"] = {"ok": False, "error": captured}
            return record["result"]
        if captured.w <= 0 or captured.h <= 0:
            log.warning("Screenshot %s has no usable size — cannot aim the click", shot_id)
            record["result"] = {
                "ok": False,
                "error": "the screenshot came without its size, so the click cannot be aimed",
            }
            return record["result"]

        point = await _vision.locate_on_screen(captured.jpeg, target, captured.w, captured.h)
        if point is None:
            result: dict[str, Any] = {
                "ok": False,
                "error": f"could not find {target} on the screen",
            }
            record["result"] = result
            return result

        click_args = mouse_click_args(point[0], point[1], button)
        record["point"] = dict(click_args)
        result = await self._run_client_action(MOUSE_CLICK_TOOL, click_args)
        record["result"] = result
        return result

    async def _run_remember(self, args: dict[str, Any]) -> dict[str, Any]:
        """Append a fact to the long-term memory on this machine."""
        fact = str(args.get("fact") or "")
        record_id = f"m{self._memory_seq}"
        self._memory_seq += 1
        record: dict[str, Any] = {"id": record_id, "tool": "remember", "args": dict(args)}
        self._utterance_actions.append(record)

        if _memory is None:
            record["result"] = {"ok": False, "error": "memory storage is not available"}
            return record["result"]
        try:
            stored = await asyncio.to_thread(_memory.add, fact)
        except ValueError as exc:
            result = {"ok": False, "error": str(exc)}
        except Exception as exc:
            log.exception("Could not store a fact")
            result = {"ok": False, "error": f"could not store the fact: {exc}"}
        else:
            if self.session is not None:
                self.session.add_fact(stored)
            result = {"ok": True}

        record["result"] = result
        return result

    # ------------------------------------------------------------------ pipeline

    async def _on_utterance_end(self) -> None:
        if not self.receiving:
            await self.send_error("utterance_end without utterance_start")
            return
        pcm = bytes(self.audio)
        self.audio = bytearray()
        self.receiving = False
        if self.session is None:
            log.warning("Utterance from %s arrived without hello — session without devices", self.peer)
            facts: list[str] = []
            if _memory is not None:
                facts = await asyncio.to_thread(_memory.facts)
            self.session = Session(
                client_id=None,
                devices=[],
                history_turns=self.cfg.server.llm.history_turns,
                memory_facts=facts,
            )
        if self._task is not None and not self._task.done():
            log.warning("Client %s sent a new utterance while the previous one is running", self.peer)
            await self.send_error("busy with the previous utterance")
            return
        # A separate task: the receive loop must keep delivering action_result
        # and screenshot messages while the tool loop waits for them.
        self._task = asyncio.create_task(self._process_utterance(pcm))

    async def _process_utterance(self, pcm: bytes) -> None:
        try:
            await self._handle_utterance(pcm)
        except asyncio.CancelledError:
            raise
        except (WebSocketDisconnect, RuntimeError):
            log.info("Client %s disconnected while the reply was in flight", self.peer)
        except Exception:
            log.exception("Failed to handle an utterance from %s", self.peer)
            try:
                await self.send_error("internal server error")
            except Exception:
                log.debug("Could not report the failure to the client", exc_info=True)

    async def _handle_utterance(self, pcm: bytes) -> None:
        session = self.session
        engine, brain, voice = _stt, _llm, _tts
        if session is None or engine is None or brain is None or voice is None:
            await self.send_error("server not ready")
            return

        self._action_seq = 1
        self._screenshot_seq = 1
        self._memory_seq = 1
        self._utterance_actions = []
        started_at = datetime.now()
        t_start = time.perf_counter()

        # 1. STT
        try:
            text, language = await asyncio.to_thread(
                engine.transcribe_pcm,
                pcm,
                self.sample_rate,
                self.cfg.server.stt.language,
            )
        except Exception:
            log.exception("Speech recognition failed")
            await self.send_error("stt failed")
            return
        stt_ms = int((time.perf_counter() - t_start) * 1000)

        await self.send_json(
            {"type": proto.MSG_TRANSCRIPT, "text": text, "language": language or ""}
        )
        if not text.strip():
            await self.send_error(ERROR_EMPTY_TRANSCRIPT)
            await self._log_dialog(
                started_at,
                session,
                text,
                language,
                "",
                {"stt": stt_ms, "llm": 0, "tts": 0, "total": stt_ms},
                note=ERROR_EMPTY_TRANSCRIPT,
            )
            return

        # 2. LLM with the tool loop — tools are executed for real (SPEC §3, §5)
        t_llm = time.perf_counter()
        try:
            result = await brain.generate(session.messages(text), self._execute_tool)
        except (WebSocketDisconnect, RuntimeError):
            raise
        except Exception:
            log.exception("LLM request failed")
            await self.send_error("llm failed")
            return
        llm_ms = int((time.perf_counter() - t_llm) * 1000)

        say_text = result.text.strip()
        if not say_text:
            say_text = SAY_AFTER_ACTIONS if self._utterance_actions else SAY_NOT_UNDERSTOOD
        session.remember(text, say_text)

        # 3. say -> tts stream (order fixed by SPEC §4)
        await self.send_json({"type": proto.MSG_SAY, "text": say_text})
        t_tts = time.perf_counter()
        await self._stream_tts(voice, say_text)
        tts_ms = int((time.perf_counter() - t_tts) * 1000)

        total_ms = int((time.perf_counter() - t_start) * 1000)
        log.info(
            "Utterance done in %d ms (stt %d, llm %d, tts %d), %d tool call(s)",
            total_ms, stt_ms, llm_ms, tts_ms, len(self._utterance_actions),
        )
        await self._log_dialog(
            started_at,
            session,
            text,
            language,
            say_text,
            {"stt": stt_ms, "llm": llm_ms, "tts": tts_ms, "total": total_ms},
        )

    async def _log_dialog(
        self,
        started_at: datetime,
        session: Session,
        transcript: str,
        language: str | None,
        reply: str,
        durations_ms: dict[str, int],
        note: str | None = None,
    ) -> None:
        """Append one line to ``data/dialogs/YYYY-MM-DD.jsonl`` (SPEC §3)."""
        if _dialogs is None:
            return
        entry: dict[str, Any] = {
            "ts": started_at.isoformat(timespec="seconds"),
            "client_id": session.client_id,
            "transcript": transcript,
            "language": language or "",
            "reply": reply,
            "actions": list(self._utterance_actions),
            "durations_ms": durations_ms,
        }
        if note:
            entry["note"] = note
        # DialogLog.append never raises; the write runs off the event loop.
        await asyncio.to_thread(_dialogs.append, entry)

    async def _stream_tts(self, voice: TtsEngine, text: str) -> None:
        pcm = b""
        try:
            pcm = await asyncio.to_thread(voice.synth, text)
        except Exception:
            log.exception("Speech synthesis failed — sending an empty stream")

        await self.send_json(
            {
                "type": proto.MSG_TTS_START,
                # == cfg.server.tts.sample_rate, normalized to int by TtsEngine
                "sr": voice.sample_rate,
                "format": proto.AUDIO_FORMAT,
                "channels": proto.AUDIO_CHANNELS,
            }
        )
        for offset in range(0, len(pcm), TTS_CHUNK_BYTES):
            if self.ws.client_state is not WebSocketState.CONNECTED:
                raise WebSocketDisconnect(code=1006)
            await self.ws.send_bytes(pcm[offset : offset + TTS_CHUNK_BYTES])
        await self.send_json({"type": proto.MSG_TTS_END})

    async def close(self) -> None:
        """Cancel a reply still in flight and fail every pending wait."""
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                log.debug("The utterance task was cancelled on disconnect")
            except Exception:
                log.debug("The utterance task failed on disconnect", exc_info=True)
        for future in list(self._pending_actions.values()):
            if not future.done():
                future.set_result({"ok": False, "error": "client disconnected"})
        self._pending_actions.clear()
        if self._screenshot_future is not None and not self._screenshot_future.done():
            self._screenshot_future.set_result({"error": "client disconnected"})
        self._screenshot_future = None
        self._screenshot_header = {}
        self._expect_screenshot_bytes = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    cfg = get_config()
    await websocket.accept()
    connection = Connection(websocket, cfg)
    log.info("New connection: %s", connection.peer)
    try:
        await connection.run()
    except WebSocketDisconnect:
        log.info("Client %s disconnected", connection.peer)
    except asyncio.CancelledError:
        log.info("Connection with %s was cancelled", connection.peer)
        raise
    except Exception:
        log.exception("Unhandled error on the connection with %s", connection.peer)
        try:
            await connection.send_error("internal server error")
        except Exception:
            log.debug("Could not send the error message", exc_info=True)
    finally:
        await connection.close()
        if websocket.client_state is WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                log.debug("Could not close the socket", exc_info=True)
