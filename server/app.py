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
client a single ``mouse_click`` action, ``remember`` writes to the memory file,
v1.4's ``look_at_camera`` / ``enroll_face`` pull one frame of the room camera,
and v1.5's ``find_object`` pulls a camera or screen frame and runs it through
SAM3 (``server/segment.py``) to count and locate objects. Everything else is
forwarded to the client verbatim.

v1.4 camera (SPEC "camera, faces, presence"): screenshots and camera frames
share one request/binary-frame machinery, tagged by SOURCE. Unsolicited
``camera_frame`` frames with ``reason: "presence"`` go to this connection's
:class:`PresenceTracker` (face matching runs off the event loop), which feeds
the ``{presence}`` block of the system prompt and the proactive greeting of an
unknown face. Everything camera-related is best-effort: a missing, broken or
disabled camera/face stack never disturbs the voice pipeline.

v1.4 burst extension: a ``camera_request`` may ask for several frames at once
(``burst``, up to :data:`common.protocol.CAMERA_BURST_MAX`) and presence
pushes are themselves small bursts, each frame header carrying ``seq``/``of``
(see :meth:`Connection._request_image`, :meth:`Connection._buffer_presence_frame`).
Staged face enrollment (:meth:`Connection._run_enroll_face`) uses this to take
its first sample immediately and a background task (per connection, cancelled
on disconnect) to keep sampling a few seconds longer while the person turns
their head, mirroring how voice enrollment collects extra samples over several
utterances instead of blocking the first reply.
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
from typing import Any, AsyncIterator, Sequence

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from common import protocol as proto
from common.config import load_config
from server import speaker as speaker_mod
from server.face import FaceEngine
from server.llm import LlmClient
from server.segment import Sam3Engine, draw_boxes
from server.session import NO_PRESENCE_TEXT, Session
from server.speaker import VoiceRegistry
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
#: v1.4: the same cap covers camera frames (largest side <= 1280 per SPEC).
MAX_IMAGE_BYTES = MAX_SCREENSHOT_BYTES

#: SPEC §4: how long the server waits for the client (per action / per screenshot).
ACTION_TIMEOUT_S = 35.0
SCREENSHOT_TIMEOUT_S = 120.0
#: A camera grab is one frame of an already-running capture: far quicker than a
#: screenshot, and the client answers with camera_error when it has no camera.
CAMERA_TIMEOUT_S = 30.0
#: v1.4 burst: per-pair timeout once a request asks for more than one frame,
#: so a multi-frame pull's overall wait is capped around ``burst * 5s`` instead
#: of ``burst * CAMERA_TIMEOUT_S`` -- a burst frame is just as quick as a single
#: one (same already-running capture), it is only the head-turn spacing in
#: enroll_face that takes real seconds, and that lives in the background task,
#: not in this per-frame wait.
CAMERA_BURST_FRAME_TIMEOUT_S = 5.0

#: Image sources sharing the request/binary-frame machinery (SPEC §4, v1.4).
SOURCE_SCREEN = "screen"
SOURCE_CAMERA = "camera"

#: Why a camera frame arrived (SPEC v1.4): pushed by the client while somebody
#: is visible, or as the answer to our ``camera_request``.
REASON_PRESENCE = proto.CAMERA_REASON_PRESENCE
REASON_REQUEST = proto.CAMERA_REASON_REQUEST

#: Presence label for a detected face that matches no enrolled profile.
LABEL_UNKNOWN = speaker_mod.ROLE_UNKNOWN

#: How often the greeting task re-checks whether it may speak.
GREETING_POLL_S = 1.0
#: A proactive greeting never starts right on top of audio the client may still
#: be playing, nor immediately after an exchange ended.
GREETING_QUIET_S = 5.0
#: v1.6: hold the greeting when a KNOWN voice spoke on THIS connection within
#: this window - a recent real conversation makes an unknown face on camera
#: almost certainly that same person seen from a bad angle, not a stranger.
KNOWN_VOICE_HOLDOFF_S = 180.0
#: v1.6: how long an annotated detections photo stays on the room screen.
IMAGE_SHOW_TTL_S = 60.0

#: The camera frame is a photo of a room — the vision prompt is written for
#: screenshots, so the query says what it is actually looking at.
CAMERA_QUERY_PREFIX = (
    "This image is a photo taken by the room's webcam, not a computer screen. "
    "Answer about the room and the people, objects and surroundings in it."
)

#: v1.4: the one-off instruction that makes the model greet an unknown face.
#: It is a user turn (the models take instructions there far more reliably than
#: in a second system message) and it is marked as coming from the system.
GREETING_REQUEST = (
    "[system event: nobody is speaking to you right now. The room camera has "
    "been seeing a person you do not recognize for several seconds.] Greet them "
    "out loud, following your persona: one or two short sentences, introduce "
    "yourself, ASK FOR THEIR NAME, and offer once to remember their voice. Do "
    "not call any tools, just speak."
)
#: Appended to the greeting when known people are in the room with the
#: stranger, so the hello is personal ("you're here with Anton") instead of
#: addressing an empty room.
GREETING_COMPANY_HINT = (
    " The people you DO recognize in the room right now are: {names}. Mention "
    "naturally that you can see the stranger is here with them (use their "
    "names), then ask the stranger's name."
)
#: Spoken when the LLM is unreachable or answers nothing at all.
SAY_FALLBACK_GREETING = "Good day. I am Rowan, the assistant of this room."
#: While voice enrollment is collecting samples the user needs room to speak:
#: the follow-up window the client holds open after our reply (SPEC §4 say.listen_s).
ENROLL_LISTEN_S = 12.0

#: Cap for the self-check pass so it can never hang a reply.
VERIFY_TIMEOUT_S = 60.0
#: Tools that CHANGE something (vs. only looking): a turn using one of these is
#: worth a self-check. Pure chat or a lone look_at_* never triggers the judge.
STATE_CHANGING_TOOLS = frozenset(
    {
        "pc_control",
        "run_command",
        "click_screen",
        "set_light",
        "set_switch",
        "enroll_voice",
        "enroll_face",
        "set_role",
        "rename_person",
        "remember",
    }
)
#: v1.4 burst: gap between the background bursts of a staged face enrollment
#: ("2-3 s apart" per SPEC); at the default 3 background bursts this totals
#: the ~9 s SPEC describes.
ENROLL_FACE_INTERVAL_S = 3.0

#: SPEC §4: the error sent when STT produced nothing (shared with the client).
ERROR_EMPTY_TRANSCRIPT = proto.ERR_EMPTY_TRANSCRIPT
SAY_AFTER_ACTIONS = "Done."
SAY_NOT_UNDERSTOOD = "Sorry, I did not catch that."


@dataclass(frozen=True)
class ImageFrame:
    """One JPEG from the client with the sizes from its header (SPEC §4, v1.4).

    ``w``/``h`` describe the JPEG itself — the vision model answers in those
    pixels — while ``screen_w``/``screen_h`` are the real desktop resolution the
    client multiplies the normalized click coordinates by (a camera frame has
    no desktop, so they simply repeat ``w``/``h``). ``source`` is
    :data:`SOURCE_SCREEN` or :data:`SOURCE_CAMERA`, ``reason`` tells a pulled
    frame from a pushed presence frame. ``seq``/``of`` (v1.4 burst) are the
    1-based position of this frame within its burst and the burst's total size
    — ``1``/``1`` for a plain single frame, which is every screenshot and every
    pre-burst camera frame.
    """

    jpeg: bytes
    w: int
    h: int
    screen_w: int
    screen_h: int
    source: str = SOURCE_SCREEN
    reason: str = REASON_REQUEST
    seq: int = 1
    of: int = 1
    #: The id from the announcing header (presence bursts group by it).
    id: str = ""


#: v1.1 name of the same record; screenshots are just the ``screen`` source.
Screenshot = ImageFrame


def _should_show_detections(count: int, show_requested: bool) -> bool:
    """True when find_object's annotated photo should be pushed (BUG 4).

    Always for a successful match (``count > 0``, unchanged from before), and
    ALSO when the user explicitly asked to see/show the result
    (``show_requested``, the tool's ``show`` argument) even though nothing was
    found — a flat "nothing found" is a lot less convincing than the actual
    photo of an empty desk. Pure and side-effect free on purpose: it is the
    one piece of :meth:`Connection._run_find_object` worth unit-testing
    without a whole fake connection.
    """
    return count > 0 or bool(show_requested)


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


@dataclass
class _Presence:
    """One label the camera has seen: when it appeared and when it was last seen."""

    first_seen: float
    last_seen: float


class PresenceTracker:
    """Who the room camera has seen recently (SPEC v1.4), per connection.

    Keyed by LABEL — an enrolled person's name, or :data:`LABEL_UNKNOWN` for
    every face that matched nobody. Entries expire ``presence_ttl_s`` after the
    last sighting, and a ``camera_state`` that reports zero people clears the
    whole map once that same ttl has passed (the camera may simply have lost
    the face while the person is still there, so it is not cleared at once).

    ``first_seen`` is kept next to the last sighting because the greeting has to
    know for how long an unknown face has been in the room, not just that it
    was seen recently.
    """

    def __init__(self, ttl_s: float = 30.0) -> None:
        try:
            self.ttl_s = max(1.0, float(ttl_s))
        except (TypeError, ValueError):
            self.ttl_s = 30.0
        self._seen: dict[str, _Presence] = {}
        #: How many faces of the last frame matched nobody (for the wording).
        self._unknown_count = 0
        #: Since when ``camera_state`` has been reporting nobody at all.
        self._empty_since: float | None = None

    # -- updates ---------------------------------------------------------

    def note_persons(self, persons: int) -> None:
        """Feed a ``camera_state`` person count into the tracker."""
        if persons > 0:
            self._empty_since = None
        elif self._empty_since is None:
            self._empty_since = time.monotonic()

    def note_faces(self, labels: Sequence[str]) -> None:
        """Record the labels matched in one presence frame."""
        now = time.monotonic()
        unknown = 0
        #: How many faces the last burst actually saw: a stranger leaning in
        #: next to a known person means more faces than known labels.
        self.last_face_count = len(labels)
        for label in labels:
            entry = self._seen.get(label)
            if entry is None or now - entry.last_seen >= self.ttl_s:
                # First sighting, or a fresh arrival after an absence.
                self._seen[label] = _Presence(first_seen=now, last_seen=now)
            else:
                entry.last_seen = now
            if label == LABEL_UNKNOWN:
                unknown += 1
        if labels:
            self._empty_since = None
        if unknown or LABEL_UNKNOWN not in self._seen:
            self._unknown_count = unknown

    def clear(self) -> None:
        self._seen.clear()
        self._unknown_count = 0
        self._empty_since = None
        self.last_face_count = 0

    def reconcile(self, persons: int) -> None:
        """Drop the unknown bucket when named labels already cover YOLO (v1.6).

        The same not-yet-enrolled (or badly-angled) person can be seen as BOTH
        a named label (a good-angle frame matched them) and the unknown bucket
        (a worse-angle frame, or an older presence sighting that has not
        expired yet) at once, which used to tell the owner "you and an
        unknown person" while they were alone. If YOLO's own person count is
        already met or exceeded by the named labels currently tracked, that
        unknown sighting cannot be a second real person, so it is dropped.
        Does nothing when YOLO reports zero (no fresh signal to reconcile
        against) or there is no unknown bucket to begin with.
        """
        if persons <= 0:
            return
        self._expire()
        if LABEL_UNKNOWN not in self._seen:
            return
        named = sum(1 for label in self._seen if label != LABEL_UNKNOWN)
        if named >= persons:
            log.debug(
                "Presence reconcile: %d named label(s) already cover the %d "
                "YOLO person(s) - dropping the unknown bucket",
                named, persons,
            )
            del self._seen[LABEL_UNKNOWN]
            self._unknown_count = 0

    # -- queries ---------------------------------------------------------

    def _expire(self) -> None:
        now = time.monotonic()
        if self._empty_since is not None and now - self._empty_since >= self.ttl_s:
            if self._seen:
                log.info("Presence cleared: the camera has seen nobody for %.0f s", self.ttl_s)
            self._seen.clear()
            self._unknown_count = 0
            return
        for label, entry in list(self._seen.items()):
            if now - entry.last_seen >= self.ttl_s:
                del self._seen[label]
                if label == LABEL_UNKNOWN:
                    self._unknown_count = 0

    def present(self) -> dict[str, float]:
        """``{label: last_seen}`` for everybody still counted as present."""
        self._expire()
        return {label: entry.last_seen for label, entry in self._seen.items()}

    @property
    def unknown_count(self) -> int:
        """How many unrecognized faces the last matched frame contained."""
        self._expire()
        if LABEL_UNKNOWN not in self._seen:
            return 0
        return max(1, self._unknown_count)

    def unknown_present_for(self) -> float:
        """Seconds an unknown face has been continuously present (0 when none)."""
        self._expire()
        entry = self._seen.get(LABEL_UNKNOWN)
        if entry is None:
            return 0.0
        return max(0.0, time.monotonic() - entry.first_seen)

    def has_fresh_unknown_face(self) -> bool:
        """True when a CURRENTLY-FRESH unknown FACE match is on record (BUG 2).

        "Fresh" means :meth:`note_faces` recorded it (a real face-engine
        detection that matched no enrolled profile) and it has not yet expired
        (:meth:`_expire` runs first, dropping anything last seen more than
        ``ttl_s`` ago). Deliberately narrower than ``unknown_count`` or
        ``unknown_present_for`` being non-zero would already imply: this is
        the single source of truth the proactive greeting gates on, so it
        must never be satisfied by a bare YOLO person count (``note_persons``
        never touches :attr:`_seen`) or by a stale label that lingered past
        its ttl.
        """
        self._expire()
        return LABEL_UNKNOWN in self._seen


_config: Any = None
_stt: SttEngine | None = None
_llm: LlmClient | None = None
_tts: TtsEngine | None = None
_vision: VisionClient | None = None
_memory: Memory | None = None
_dialogs: DialogLog | None = None
_voices: VoiceRegistry | None = None
_face: FaceEngine | None = None
_segment: Sam3Engine | None = None


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
    global _stt, _llm, _tts, _vision, _memory, _dialogs, _voices, _face, _segment
    cfg = get_config()
    log.info("Starting the Jarvis brain: %s:%s", cfg.server.host, cfg.server.port)

    _memory = Memory()
    _dialogs = DialogLog()
    speaker_cfg = getattr(cfg.server, "speaker", None)
    _voices = VoiceRegistry(
        threshold=getattr(speaker_cfg, "threshold", 0.72),
        min_speech_s=getattr(speaker_cfg, "min_speech_s", 0.8),
        enabled=getattr(speaker_cfg, "enabled", True),
    )
    # The face model itself is loaded lazily on the first camera frame, so a
    # server without insightface still starts and serves the voice pipeline.
    _face = FaceEngine(getattr(cfg.server, "face", None))
    # SAM3 (v1.5, find_object) is just as lazy: nothing is imported or put on
    # the GPU until the first find_object call actually needs it.
    _segment = Sam3Engine(getattr(cfg.server, "segment", None))
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
        _voices = None
        _face = None
        _segment = None


app = FastAPI(title="Jarvis brain server", version="1.6", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Tiny status endpoint — handy when checking the firewall/port from the client."""
    return {
        "status": "ok",
        "stt": _stt is not None,
        "llm": _llm is not None,
        "tts": bool(_tts is not None and _tts.available),
        # v1.4: face recognition is enabled and insightface can be imported.
        "face": bool(_face is not None and _face.available),
        # v1.5: SAM3 (find_object) is loaded, or enabled and not yet known broken.
        "sam": bool(_segment is not None and _segment.available),
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
        #: Image source tag -> the future waiting for that JPEG (SPEC §4, v1.4).
        self._image_futures: dict[str, asyncio.Future] = {}
        #: Image source tag -> lock serializing pulls of that source (v1.4
        #: burst): the enroll_face background sampler and a fresh utterance's
        #: own look_at_camera/enroll_face could otherwise both try to pull a
        #: camera frame at once and race on the single _image_futures slot.
        self._image_pull_locks: dict[str, asyncio.Lock] = {}
        #: Image source tag -> the request id that future is waiting for.
        self._image_ids: dict[str, str] = {}
        #: Set by an image header: whose JPEG the next binary frame carries.
        self._expect_image: str | None = None
        #: The last image header, consumed by that JPEG frame.
        self._image_header: dict[str, Any] = {}

        # --- per-utterance state ---
        self._action_seq = 1
        self._screenshot_seq = 1
        self._camera_seq = 1
        self._memory_seq = 1
        #: v1.6: id counter for image_show pushes (find_object's annotated photo).
        self._image_seq = 1
        #: Last frame pulled per source, so ``show_photo`` can display exactly
        #: the picture that was just described instead of taking a new one.
        self._last_frames: dict[str, Any] = {}
        #: Last annotated detections photo (jpeg, w, h, title) from find_object.
        self._last_annotated: tuple[bytes, int, int, str] | None = None
        #: When each cached picture was produced, so "show me the photo" picks
        #: the freshest one - and prefers the ANNOTATED frame over the plain
        #: capture it was drawn from (the boxes are the point of showing it).
        self._last_frame_ts: dict[str, float] = {}
        self._last_annotated_ts: float = 0.0
        self._utterance_actions: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None

        # --- speaker recognition (SPEC v1.3) ---
        self._speaker_name = speaker_mod.ROLE_UNKNOWN
        self._speaker_role = speaker_mod.ROLE_UNKNOWN
        self._speaker_score = 0.0
        self._current_pcm: bytes = b""
        #: Set by ``enroll_voice``: the next utterances add samples for ``name``.
        self._enroll_pending: dict[str, Any] | None = None

        # --- camera, faces and presence (SPEC v1.4) ---
        face_cfg = getattr(getattr(cfg, "server", None), "face", None)
        self.face_enabled = bool(getattr(face_cfg, "enabled", True))
        self.presence = PresenceTracker(getattr(face_cfg, "presence_ttl_s", 30.0))
        #: Last ``camera_state``: ``{"persons": int, "objects": dict, "ts": float}``.
        self.camera_state: dict[str, Any] | None = None
        #: True while a presence burst is being matched (later ones are dropped).
        self._presence_busy = False
        self._presence_tasks: set[asyncio.Task] = set()
        #: v1.4 burst: frames of the presence mini-burst currently being
        #: assembled, keyed by the burst's own id so an incomplete burst never
        #: gets mixed up with the next one.
        self._presence_burst_id = ""
        self._presence_burst_frames: list[ImageFrame] = []
        #: Serializes reply streaming: an utterance reply and a proactive
        #: greeting can never interleave on the wire.
        self._reply_lock = asyncio.Lock()
        self._greet_task: asyncio.Task | None = None
        #: v1.4 burst: background task collecting the extra staged samples of
        #: the enroll_face currently in progress on this connection, if any.
        self._enroll_face_task: asyncio.Task | None = None
        #: Monotonic clock of the last TTS stream pushed to the client.
        self._last_audio_at = 0.0
        #: Monotonic clock of the last proactive greeting (0 = never greeted).
        self._last_greeting_at = 0.0
        #: v1.6: monotonic clock of the last utterance a KNOWN voice spoke on
        #: this connection (0 = never) - holds the greeting off (see _may_greet).
        self._last_known_voice_at = 0.0

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
            self._on_image_header(SOURCE_SCREEN, payload)
        elif msg_type == proto.MSG_SCREENSHOT_ERROR:
            self._on_image_error(SOURCE_SCREEN, payload)
        elif msg_type == proto.MSG_CAMERA_STATE:
            self._on_camera_state(payload)
        elif msg_type == proto.MSG_CAMERA_FRAME:
            self._on_image_header(SOURCE_CAMERA, payload)
        elif msg_type == proto.MSG_CAMERA_ERROR:
            self._on_image_error(SOURCE_CAMERA, payload)
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
            # Resolved per completion: the camera view changes between turns.
            presence=self.presence_text,
        )
        log.info(
            "Client %s (%s) connected, devices: %s",
            self.session.client_id,
            self.peer,
            ", ".join(self.session.device_names) or "none",
        )
        self._start_greeting_task()
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
        # or camera_frame header, or a chunk of the utterance being recorded.
        if self._expect_image is not None:
            self._deliver_image(data)
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

    def _on_camera_state(self, payload: dict[str, Any]) -> None:
        """Store what the client's YOLO pass currently sees (SPEC v1.4)."""
        persons = _positive_int(payload.get("persons")) or 0
        raw_objects = payload.get("objects")
        objects: dict[str, int] = {}
        if isinstance(raw_objects, dict):
            for label, count in raw_objects.items():
                number = _positive_int(count)
                if number:
                    objects[str(label)] = number
        previous = self.camera_state or {}
        self.camera_state = {"persons": persons, "objects": objects, "ts": time.time()}
        self.presence.note_persons(persons)
        self.presence.reconcile(persons)
        if previous.get("persons") != persons or previous.get("objects") != objects:
            # Only person-count changes earn a console line; object-label churn
            # (a phone appearing/disappearing) goes to DEBUG to keep it readable.
            log.log(
                logging.INFO if previous.get("persons") != persons else logging.DEBUG,
                "Camera state from %s: %d person(s), objects: %s",
                self.peer,
                persons,
                ", ".join(f"{name} x{count}" for name, count in sorted(objects.items()))
                or "none",
            )

    def _on_image_header(self, source: str, payload: dict[str, Any]) -> None:
        """A ``screenshot``/``camera_frame`` header announces ONE binary frame.

        Screenshots always answer a pending ``screenshot_request``. A camera
        frame either answers a pending ``camera_request`` or is one of the
        unsolicited presence frames the client pushes while somebody is visible
        (SPEC v1.4) — both are consumed here, so a presence frame can never be
        mistaken for audio.
        """
        frame_id = str(payload.get("id") or "")
        reason = str(payload.get("reason") or "").strip().lower()
        future = self._image_futures.get(source)
        waiting = future is not None and not future.done()

        if source == SOURCE_CAMERA:
            # Anything that does not answer an open request is a presence frame.
            reason = REASON_REQUEST if (waiting and reason != REASON_PRESENCE) else REASON_PRESENCE
        else:
            if not waiting:
                log.warning(
                    "A %s header (%r) arrived with nothing waiting for it", source, frame_id
                )
                return
            reason = REASON_REQUEST

        if reason == REASON_REQUEST:
            expected = self._image_ids.get(source)
            if frame_id and expected and frame_id != expected:
                log.warning(
                    "%s id mismatch: got %r, waiting for %r", source, frame_id, expected
                )

        # SPEC §4: the header carries the image size and (for screenshots) the
        # desktop resolution; both are kept next to the bytes so click_screen
        # can aim the cursor.
        self._image_header = {
            "source": source,
            "reason": reason,
            "id": frame_id,
            "w": _positive_int(payload.get("w")),
            "h": _positive_int(payload.get("h")),
            "screen_w": _positive_int(payload.get("screen_w")),
            "screen_h": _positive_int(payload.get("screen_h")),
        }
        self._expect_image = source

    def _on_image_error(self, source: str, payload: dict[str, Any]) -> None:
        """The client could not capture the screen / a camera frame (SPEC §4, v1.4)."""
        error = str(payload.get("error") or f"{source} capture failed")
        log.warning("Client %s could not capture the %s: %s", self.peer, source, error)
        if self._expect_image == source:
            self._expect_image = None
            self._image_header = {}
        future = self._image_futures.get(source)
        if future is not None and not future.done():
            future.set_result({"error": error})

    def _build_frame(self, data: bytes, header: dict[str, Any]) -> ImageFrame:
        """Turn the announced bytes plus their header into an :class:`ImageFrame`."""
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
                log.info("An image header had no size — the JPEG says %dx%d", width, height)
            else:
                log.warning("An image header had no size and the JPEG could not be measured")
        # v1.4 burst: seq/of default to 1/1 so a header without them (an older
        # client, or a screenshot which never carries them) is just a plain
        # single frame, exactly like before the burst extension.
        seq = _positive_int(header.get("seq")) or 1
        of = _positive_int(header.get("of")) or 1
        return ImageFrame(
            jpeg=jpeg,
            w=int(width or 0),
            h=int(height or 0),
            # A camera frame has no desktop behind it: the image is its own size.
            screen_w=int(header.get("screen_w") or width or 0),
            screen_h=int(header.get("screen_h") or height or 0),
            source=str(header.get("source") or SOURCE_SCREEN),
            reason=str(header.get("reason") or REASON_REQUEST),
            seq=seq,
            of=max(of, seq),
            id=str(header.get("id") or ""),
        )

    def _deliver_image(self, data: bytes) -> None:
        """Route the announced JPEG: to its waiting tool call, or to presence."""
        source = self._expect_image or SOURCE_SCREEN
        self._expect_image = None
        header = self._image_header
        self._image_header = {}
        reason = str(header.get("reason") or REASON_REQUEST)
        future = self._image_futures.get(source)
        waiting = future is not None and not future.done()

        if len(data) > MAX_IMAGE_BYTES:
            log.warning("A %s frame from %s is too large (%d bytes)", source, self.peer, len(data))
            if waiting and future is not None:
                future.set_result({"error": f"{source} frame too large"})
            return

        frame = self._build_frame(data, header)
        if source == SOURCE_CAMERA and reason == REASON_PRESENCE:
            log.debug(
                "Presence frame from %s (%d KB, %dx%d) %d/%d",
                self.peer, len(frame.jpeg) // 1024, frame.w, frame.h, frame.seq, frame.of,
            )
            self._buffer_presence_frame(frame)
            return
        if not waiting or future is None:
            log.warning("%s bytes arrived with nothing waiting for them", source)
            return
        log.info(
            "%s frame received from %s (%d KB, image %dx%d, screen %dx%d)",
            source, self.peer, len(frame.jpeg) // 1024, frame.w, frame.h,
            frame.screen_w, frame.screen_h,
        )
        future.set_result(frame)

    def _turn_changed_state(self) -> bool:
        """True when this utterance ran a tool that changed something (§ judge)."""
        return any(
            rec.get("tool") in STATE_CHANGING_TOOLS for rec in self._utterance_actions
        )

    # ------------------------------------------------------------------ tool executor

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """ToolExecutor for :meth:`server.llm.LlmClient.generate` (SPEC §5 matrix)."""
        denial = speaker_mod.check_permission(
            self._speaker_role,
            name,
            args,
            self._speaker_name,
            speaker_score=self._speaker_score,
            admin_threshold=getattr(self.cfg.server.speaker, "admin_threshold", 0.70),
        )
        if denial is not None:
            log.info(
                "Denied %s for %s (%s)", name, self._speaker_name, self._speaker_role
            )
            return {"ok": False, "error": denial}
        if name in CLIENT_TOOLS:
            return await self._run_client_action(name, args)
        if name == "look_at_screen":
            return await self._run_look_at_screen(args)
        if name == "click_screen":
            return await self._run_click_screen(args)
        if name == "remember":
            return await self._run_remember(args)
        if name == "show_photo":
            return await self._run_show_photo(args)
        if name == "enroll_voice":
            return await self._run_enroll_voice(args)
        if name == "set_role":
            return await self._run_set_role(args)
        if name == "rename_person":
            return await self._run_rename_person(args)
        if name == "look_at_camera":
            return await self._run_look_at_camera(args)
        if name == "enroll_face":
            return await self._run_enroll_face(args)
        if name == "find_object":
            return await self._run_find_object(args)
        log.warning("Tool %r has no server-side handler", name)
        return {"ok": False, "error": f"unknown tool: {name}"}

    async def _run_enroll_voice(self, args: dict[str, Any]) -> dict[str, Any]:
        """SPEC v1.3/v1.6: store the current utterance as a voice sample.

        v1.6: rejects placeholder names (Guest/User/Friend, ...) outright —
        the model must ask for the person's real name instead — and starts
        the pending enrollment tracking both accepted SAMPLES and total voiced
        SECONDS (see :meth:`_enroll_progress_note`); it only completes once
        both :data:`speaker_mod.ENROLL_MIN_SAMPLES` and
        :data:`speaker_mod.MIN_ENROLL_SPEECH_S` are met.
        """
        if _voices is None or not _voices.enabled:
            return {"ok": False, "error": "speaker recognition is disabled"}
        if not self._current_pcm:
            return {"ok": False, "error": "no utterance audio to sample"}
        name = str(args.get("name") or "").strip()
        if speaker_mod.is_placeholder_name(name):
            return {
                "ok": False,
                "error": (
                    f"{name!r} is a placeholder name, not a real one - ask for "
                    "their actual name before enrolling them"
                ),
            }
        try:
            role, status = await asyncio.to_thread(
                _voices.enroll, name, self._current_pcm, self.sample_rate
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self._enroll_pending = {
            "name": name,
            "samples": 1,
            "total_speech_s": speaker_mod.estimate_speech_seconds(self._current_pcm),
        }
        record = {
            "id": f"v{self._memory_seq}",
            "tool": "enroll_voice",
            "args": {"name": name},
            "result": {"ok": True, "role": role, "status": status},
        }
        self._memory_seq += 1
        self._utterance_actions.append(record)
        return {
            "ok": True,
            "role": role,
            "status": status,
            "next": self._enroll_progress_note(self._enroll_pending),
        }

    @staticmethod
    def _enroll_progress_note(pending: dict[str, Any]) -> str:
        """Progress phrase for the model to relay (SPEC v1.6, both notes below)."""
        name = pending.get("name") or "the speaker"
        remaining_s = max(
            0.0, speaker_mod.MIN_ENROLL_SPEECH_S - float(pending.get("total_speech_s") or 0.0)
        )
        if remaining_s > 0:
            return (
                f"about {remaining_s:.0f} more second(s) of speech needed for "
                f"{name} - ask them to keep talking"
            )
        return f"one more full sentence needed for {name} - ask them to keep talking"

    async def _run_set_role(self, args: dict[str, Any]) -> dict[str, Any]:
        """SPEC v1.3: admin-only role change (permission already checked)."""
        if _voices is None or not _voices.enabled:
            return {"ok": False, "error": "speaker recognition is disabled"}
        try:
            applied = await asyncio.to_thread(
                _voices.set_role, args.get("name"), args.get("role")
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "role": applied}

    async def _run_rename_person(self, args: dict[str, Any]) -> dict[str, Any]:
        """SPEC v1.6: rename (or merge) an enrolled person.

        Permission (admin, or the speaker renaming themselves) is already
        checked by :func:`server.speaker.check_permission` before this ever
        runs. Also updates an in-progress enrollment's pending name, so a
        rename mid-enrollment keeps collecting samples under the corrected
        name instead of losing them.
        """
        if _voices is None or not _voices.enabled:
            return {"ok": False, "error": "speaker recognition is disabled"}
        old_name = " ".join(str(args.get("old_name") or "").split())
        new_name = " ".join(str(args.get("new_name") or "").split())
        try:
            role, status = await asyncio.to_thread(
                _voices.rename_person, old_name, new_name
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        pending = self._enroll_pending
        if pending and str(pending.get("name") or "").strip().lower() == old_name.lower():
            pending["name"] = new_name
            log.info("Enrollment in progress for %s renamed to %s", old_name, new_name)
        if self._speaker_name.strip().lower() == old_name.lower():
            self._speaker_name = new_name
            if status == "merged":
                self._speaker_role = role

        log.info("rename_person: %s -> %s (%s, %s)", old_name, new_name, role, status)
        return {"ok": True, "role": role, "status": status}

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

    async def _request_image(
        self,
        source: str,
        request_id: str,
        request_type: str,
        timeout_s: float,
        burst: int = 1,
        full: bool = False,
    ) -> list[ImageFrame] | str:
        """Pull one or more JPEGs from the client and wait for them (SPEC §4, v1.4).

        Shared by ``screenshot_request`` (always ``burst=1``) and
        ``camera_request`` (``burst`` 1-5, v1.4's burst extension): only the
        source tag, the message type, the timeout and the burst size differ.
        Sends ONE request, then waits for up to ``burst`` header+binary pairs
        sharing its id, each pair awaited with its own timeout — a per-pair
        timeout, same mechanism as the single-frame wait always used, so a slow
        client fails one pair at a time instead of the whole burst together —
        and stops as soon as a frame reports ``seq >= of``. A burst request
        also asks for at most :data:`proto.CAMERA_BURST_MAX` frames.

        :returns: the frames collected so far as a list — non-empty on any
            success, even a partial burst (a caller doing best-of-burst
            selection, like ``enroll_face``, can use what arrived) — or an
            error string ready to be handed to the LLM as a tool result when
            NOTHING could be captured at all. Single-frame callers
            (``burst=1``) get a length-1 list on success, unchanged otherwise.

        Serialized per source (:attr:`_image_pull_locks`): the enroll_face
        background task and a fresh utterance's own camera tool call could
        otherwise both be pulling a camera frame at the same time and stomp on
        each other's single ``_image_futures[source]`` slot.

        :param full: v1.6, camera only - ask the client to skip its usual
            downscale for this pull (``find_object`` wants native resolution).
        """
        async with self._pull_lock(source):
            return await self._request_image_locked(
                source, request_id, request_type, timeout_s, burst, full
            )

    def _pull_lock(self, source: str) -> asyncio.Lock:
        lock = self._image_pull_locks.get(source)
        if lock is None:
            lock = asyncio.Lock()
            self._image_pull_locks[source] = lock
        return lock

    async def _request_image_locked(
        self,
        source: str,
        request_id: str,
        request_type: str,
        timeout_s: float,
        burst: int = 1,
        full: bool = False,
    ) -> list[ImageFrame] | str:
        """The body of :meth:`_request_image`, run under its per-source lock."""
        requested = max(1, min(int(burst or 1), proto.CAMERA_BURST_MAX))
        payload: dict[str, Any] = {"type": request_type, "id": request_id}
        if source == SOURCE_CAMERA and requested != 1:
            payload["burst"] = requested
        if source == SOURCE_CAMERA and full:
            payload["full"] = True
        try:
            await self.send_json(payload)
        except (WebSocketDisconnect, RuntimeError) as exc:
            log.warning("Could not request a %s frame (%s): %s", source, request_id, exc)
            return "client disconnected"
        log.info(
            "Requested a %s frame (%s)%s",
            source, request_id, f", burst={requested}" if requested != 1 else "",
        )

        per_pair_timeout = timeout_s if requested <= 1 else min(timeout_s, CAMERA_BURST_FRAME_TIMEOUT_S)
        loop = asyncio.get_running_loop()
        collected: list[ImageFrame] = []
        error: str | None = None
        for _ in range(requested):
            future: asyncio.Future = loop.create_future()
            self._image_futures[source] = future
            self._image_ids[source] = request_id
            try:
                captured = await asyncio.wait_for(future, timeout=per_pair_timeout)
            except asyncio.TimeoutError:
                log.warning(
                    "%s request %s timed out after %.0f s (%d/%d frame(s) received)",
                    source, request_id, per_pair_timeout, len(collected), requested,
                )
                error = proto.ERR_CLIENT_TIMEOUT
                break
            except (WebSocketDisconnect, RuntimeError) as exc:
                log.warning("Could not wait for a %s frame (%s): %s", source, request_id, exc)
                error = "client disconnected"
                break
            finally:
                self._image_futures.pop(source, None)
                self._image_ids.pop(source, None)
                # Only drop an announcement that belongs to THIS request: a
                # presence frame may have been announced while we were waiting.
                if (
                    self._expect_image == source
                    and str(self._image_header.get("reason") or "") != REASON_PRESENCE
                ):
                    self._expect_image = None
                    self._image_header = {}

            if isinstance(captured, ImageFrame):
                collected.append(captured)
                if captured.seq >= max(captured.of, 1):
                    break
                continue
            if isinstance(captured, dict):
                error = str(captured.get("error") or f"{source} capture failed")
            else:
                error = f"{source} capture failed"
            break

        if collected:
            return collected
        return error or f"{source} capture failed"

    def _remember_frame(self, frame: Any, kind: str) -> Any:
        """Cache the last pulled frame so ``show_photo`` can re-show it.

        The owner asked to SEE the photo that was just described, and every
        tool used to take a brand new one; the last frame per source is kept
        here so showing it costs nothing and shows exactly what was described.
        """
        if isinstance(frame, ImageFrame):
            self._last_frames[kind] = frame
            self._last_frame_ts[kind] = time.monotonic()
        return frame

    async def _request_screenshot(self, shot_id: str) -> ImageFrame | str:
        """Ask the client for a screenshot of the room PC (SPEC §4)."""
        result = await self._request_image(
            SOURCE_SCREEN, shot_id, proto.MSG_SCREENSHOT_REQUEST, SCREENSHOT_TIMEOUT_S
        )
        first = result[0] if isinstance(result, list) else result
        return self._remember_frame(first, SOURCE_SCREEN)

    async def _request_camera_frame(self, frame_id: str) -> ImageFrame | str:
        """Ask the client for one frame of the room camera (SPEC v1.4)."""
        result = await self._request_image(
            SOURCE_CAMERA, frame_id, proto.MSG_CAMERA_REQUEST, CAMERA_TIMEOUT_S
        )
        first = result[0] if isinstance(result, list) else result
        return self._remember_frame(first, SOURCE_CAMERA)

    async def _request_camera_frame_full(self, frame_id: str) -> ImageFrame | str:
        """Ask the client for one FULL-resolution camera frame (SPEC v1.6).

        ``find_object`` wants the object detector to see the frame at native
        camera resolution instead of the usual <=1280px presence downscale.
        """
        result = await self._request_image(
            SOURCE_CAMERA, frame_id, proto.MSG_CAMERA_REQUEST, CAMERA_TIMEOUT_S, full=True
        )
        return result[0] if isinstance(result, list) else result

    async def _request_camera_burst(self, frame_id: str, burst: int) -> list[ImageFrame] | str:
        """Ask the client for a burst of camera frames (SPEC v1.4 burst).

        Used by staged face enrollment, which needs several angles of the same
        person to pick the best one from — ``look_at_camera`` and
        ``find_object`` keep pulling a single frame via
        :meth:`_request_camera_frame`.
        """
        return await self._request_image(
            SOURCE_CAMERA, frame_id, proto.MSG_CAMERA_REQUEST, CAMERA_TIMEOUT_S, burst=burst
        )

    # ------------------------------------------------------------------ presence

    def _buffer_presence_frame(self, frame: ImageFrame) -> None:
        """Accumulate one frame of a presence mini-burst (SPEC v1.4 burst).

        The client now pushes presence as a burst of several frames sharing
        one id, headers carrying ``seq``/``of`` — dispatched to matching once
        the last one (``seq >= of``) arrives, or at once for a plain single
        frame (``of == 1``), so pre-burst behaviour is unchanged.
        """
        if frame.id and frame.id != self._presence_burst_id:
            # A new burst started (or the previous one was abandoned mid-way,
            # e.g. after a busy-matcher drop) — start collecting fresh.
            self._presence_burst_id = frame.id
            self._presence_burst_frames = []
        self._presence_burst_frames.append(frame)
        if frame.seq >= max(frame.of, 1) or len(self._presence_burst_frames) >= max(frame.of, 1):
            burst = self._presence_burst_frames
            self._presence_burst_frames = []
            self._presence_burst_id = ""
            self._on_presence_frame(burst)

    def _on_presence_frame(self, frames: list[ImageFrame]) -> None:
        """Match one completed presence burst against the face profiles (v1.4).

        Detection and embedding are heavy, so they run in a worker thread from
        a background task: the receive loop stays free for audio and action
        results. While one burst is being matched later ones are dropped —
        presence only needs the most recent picture, never a backlog.
        """
        engine = _face
        if not self.face_enabled or engine is None or not engine.available or not frames:
            return
        if self._presence_busy:
            log.debug("Dropping a presence burst — the previous one is still being matched")
            return
        self._presence_busy = True
        task = asyncio.create_task(self._match_presence(frames))
        self._presence_tasks.add(task)
        task.add_done_callback(self._presence_tasks.discard)

    async def _match_presence(self, frames: list[ImageFrame]) -> None:
        """Match every frame of a presence burst and update the tracker (v1.4 burst).

        Runs face matching over ALL frames of the burst instead of just one —
        a person's face may be a bad angle or blurry in any single frame — and
        keeps, per label, the BEST match score seen across the burst: a named
        person counts as present the moment ANY frame matches them above the
        threshold, deduplicated by name so several frames of the same person
        never inflate to several entries. Unrecognised faces have no identity
        to dedupe by, so instead the largest COUNT of unmatched faces seen in
        any single frame is kept — the best estimate of how many distinct
        strangers are actually in view (summing across frames would double
        count the same stranger appearing in more than one frame).
        """
        try:
            engine, registry = _face, _voices
            if engine is None or registry is None:
                return
            profiles = await asyncio.to_thread(registry.face_profiles)
            best_named: dict[str, float] = {}
            max_unknown = 0
            any_face = False
            for frame in frames:
                faces = await asyncio.to_thread(engine.detect_and_embed, frame.jpeg)
                if not faces:
                    continue
                any_face = True
                frame_unknown = 0
                for _area, embedding in faces:
                    name, score = engine.match(embedding, profiles)
                    if name:
                        if score > best_named.get(name, -1.0):
                            best_named[name] = score
                    else:
                        frame_unknown += 1
                max_unknown = max(max_unknown, frame_unknown)
            if not any_face:
                # Nobody recognisable in this burst; the ttl handles the rest.
                return
            labels = list(best_named.keys()) + [LABEL_UNKNOWN] * max_unknown
            self.presence.note_faces(labels)
            parts = [f"{name} ({score:.2f})" for name, score in best_named.items()]
            if max_unknown:
                parts.append(f"{max_unknown} unknown")
            log.info(
                "Presence in the room (burst of %d): %s",
                len(frames), ", ".join(parts) if parts else "nobody matched",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Could not match a presence burst")
        finally:
            self._presence_busy = False

    def presence_text(self) -> str:
        """The ``{presence}`` block of the system prompt (SPEC v1.4).

        For example ``"Present in the room: Anton (admin), 1 unknown person"``,
        or :data:`server.session.NO_PRESENCE_TEXT` when the camera is off,
        absent or sees nobody.
        """
        roles: dict[str, str] = {}
        if _voices is not None:
            try:
                roles = _voices.people()
            except Exception:
                log.exception("Could not read the people registry for the prompt")

        parts: list[str] = []
        unknown = 0
        for label in sorted(self.presence.present()):
            if label == LABEL_UNKNOWN:
                unknown = max(1, self.presence.unknown_count)
                continue
            role = roles.get(label) or speaker_mod.ROLE_USER
            parts.append(f"{label} ({role})")
        if unknown:
            parts.append(f"{unknown} unknown person" + ("s" if unknown > 1 else ""))
        if parts:
            return "Present in the room: " + ", ".join(parts)

        # No face matched yet — but YOLO may still have counted people.
        persons = int((self.camera_state or {}).get("persons") or 0)
        if persons > 0:
            plural = "s" if persons > 1 else ""
            return f"Present in the room: {persons} person{plural}, face not identified yet"
        return NO_PRESENCE_TEXT

    # ------------------------------------------------------------------ greeting

    def _start_greeting_task(self) -> None:
        """Start the background task that greets an unknown face (SPEC v1.4)."""
        if not self.face_enabled:
            return
        if self._greet_task is not None and not self._greet_task.done():
            return
        self._greet_task = asyncio.create_task(self._greeting_loop())

    def _greet_config(self) -> tuple[float, float]:
        face_cfg = getattr(getattr(self.cfg, "server", None), "face", None)
        try:
            after = float(getattr(face_cfg, "greet_after_s", 10.0))
        except (TypeError, ValueError):
            after = 10.0
        try:
            cooldown = float(getattr(face_cfg, "greeting_cooldown_s", 300.0))
        except (TypeError, ValueError):
            cooldown = 300.0
        return after, cooldown

    def _known_face_alone(self) -> bool:
        """True when a known face is present together with exactly one YOLO person (v1.6).

        That combination almost certainly means the "unknown" face the
        greeting task is looking at is really the same known person caught at
        a bad angle or not yet enrolled by face - not a second, genuine
        stranger - so the greeting must hold fire.
        """
        persons = int((self.camera_state or {}).get("persons") or 0)
        if persons != 1:
            return False
        present = self.presence.present()
        if not any(label != LABEL_UNKNOWN for label in present):
            return False
        # ... unless the last burst actually saw MORE faces than there are
        # known people: then one of them is a genuine stranger (a friend
        # leaning in, a face held up to the camera) and must be greeted.
        known = sum(1 for label in present if label != LABEL_UNKNOWN)
        return int(getattr(self.presence, "last_face_count", 0) or 0) <= known

    def _may_greet(self, greet_after_s: float, cooldown_s: float) -> bool:
        """True when an unknown face has waited long enough and the room is idle.

        BUG 2: gated STRICTLY on the presence tracker holding a
        CURRENTLY-FRESH unknown FACE match (:meth:`PresenceTracker.has_fresh_unknown_face`)
        — a face the face engine actually detected and could not match to any
        enrolled profile within the last ``presence_ttl_s`` — never on a bare
        YOLO person count (which never touches the tracker's unknown bucket,
        see :meth:`PresenceTracker.note_persons`) and never on a label that
        has already expired. If the face engine is unavailable or disabled
        this returns ``False`` unconditionally: no face engine means no face
        was ever really seen, so there is nothing to greet.
        """
        engine = _face
        if not self.face_enabled or engine is None or not engine.available:
            return False
        if self.session is None or _llm is None or _tts is None:
            return False
        if self.receiving:  # somebody is speaking to us right now
            return False
        if self._task is not None and not self._task.done():
            return False
        now = time.monotonic()
        if self._last_audio_at and now - self._last_audio_at < GREETING_QUIET_S:
            return False
        if self._last_greeting_at and now - self._last_greeting_at < cooldown_s:
            return False
        # v1.6: a KNOWN voice on THIS connection recently, or a known face
        # alone with exactly one YOLO person, means the "unknown" face is
        # almost certainly that same not-yet-enrolled/badly-angled person.
        if (
            self._last_known_voice_at
            and now - self._last_known_voice_at < KNOWN_VOICE_HOLDOFF_S
        ):
            return False
        if self._known_face_alone():
            return False
        if not self.presence.has_fresh_unknown_face():
            return False
        return self.presence.unknown_present_for() >= greet_after_s

    async def _greeting_loop(self) -> None:
        """Poll the presence tracker and greet an unknown face once (SPEC v1.4)."""
        greet_after_s, cooldown_s = self._greet_config()
        if greet_after_s <= 0.0:
            log.info("Proactive greetings are off (server.face.greet_after_s = 0)")
            return
        log.info(
            "Greeting task armed: unknown face for %.0f s, at most one per %.0f s",
            greet_after_s, cooldown_s,
        )
        while True:
            await asyncio.sleep(GREETING_POLL_S)
            try:
                if self._reply_lock.locked():
                    continue
                if not self._may_greet(greet_after_s, cooldown_s):
                    continue
                await self._greet_unknown(greet_after_s, cooldown_s)
            except asyncio.CancelledError:
                raise
            except (WebSocketDisconnect, RuntimeError):
                log.info("Client %s is gone — the greeting task stops", self.peer)
                return
            except Exception:
                log.exception("The greeting task failed — it keeps running")

    async def _greet_unknown(self, greet_after_s: float, cooldown_s: float) -> None:
        """Generate ONE greeting and push it as an unsolicited say + TTS block."""
        session, brain, voice = self.session, _llm, _tts
        if session is None or brain is None or voice is None:
            return
        async with self._reply_lock:
            # The room may have changed while we waited for the lock.
            if not self._may_greet(greet_after_s, cooldown_s):
                return
            # Start the cooldown before speaking: a failed greeting must not be
            # retried every second either.
            self._last_greeting_at = time.monotonic()
            engine = _face
            # BUG 2: log every input the gate decided on, so a wrong greeting
            # (or a missing one) can be diagnosed from the log alone.
            log.info(
                "Greeting decision: fresh_unknown_face=%s present_for=%.0fs "
                "greet_after_s=%.0f cooldown_s=%.0f face_enabled=%s "
                "face_available=%s known_face_alone=%s",
                self.presence.has_fresh_unknown_face(),
                self.presence.unknown_present_for(),
                greet_after_s,
                cooldown_s,
                self.face_enabled,
                bool(engine is not None and engine.available),
                self._known_face_alone(),
            )
            # Tell the model who the stranger is standing next to, by name.
            known_now = [
                label for label in self.presence.present() if label != LABEL_UNKNOWN
            ]
            request = GREETING_REQUEST
            if known_now:
                request += GREETING_COMPANY_HINT.format(names=", ".join(sorted(known_now)))
            try:
                result = await brain.generate(
                    session.messages(request), self._refuse_tools
                )
                text = result.text.strip()
            except (WebSocketDisconnect, RuntimeError):
                raise
            except Exception:
                log.exception("The greeting could not be generated")
                text = ""
            if not text:
                text = SAY_FALLBACK_GREETING
            session.remember(GREETING_REQUEST, text)
            await self.send_json({"type": proto.MSG_SAY, "text": text})
            await self._stream_tts(voice, text)
        log.info("Greeting spoken: %r", text)

    async def _refuse_tools(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Tool executor for the greeting: proactive speech may not act.

        Nobody asked for anything, and there is no identified speaker to check
        permissions against, so a greeting must never reach a real tool.
        """
        log.warning("The greeting tried to call %s — refused", name)
        return {
            "ok": False,
            "error": "tools are not available for a proactive greeting - just speak",
        }

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

    async def _run_look_at_camera(self, args: dict[str, Any]) -> dict[str, Any]:
        """Pull one camera frame and answer about the room (SPEC v1.4).

        Same pipeline and permissions as ``look_at_screen``, only the image
        comes from the room camera instead of the desktop.
        """
        query = " ".join(str(args.get("query") or "").split())
        frame_id = f"c{self._camera_seq}"
        self._camera_seq += 1
        record: dict[str, Any] = {"id": frame_id, "tool": "look_at_camera", "args": dict(args)}
        self._utterance_actions.append(record)

        if _vision is None:
            record["result"] = {"ok": False, "error": "vision model is not loaded"}
            return record["result"]

        log.info("look_at_camera (%s): %r", frame_id, query)
        captured = await self._request_camera_frame(frame_id)
        if isinstance(captured, str):
            result: dict[str, Any] = {"ok": False, "error": captured}
        else:
            answer = await _vision.describe_screenshot(
                captured.jpeg, f"{CAMERA_QUERY_PREFIX} Question: {query}".strip()
            )
            result = {"ok": True, "answer": answer}

        record["result"] = result
        return result

    async def _run_enroll_face(self, args: dict[str, Any]) -> dict[str, Any]:
        """Stage a face profile like voice enrollment (SPEC v1.4 burst).

        The FIRST burst is pulled and stored right away — the best face across
        it (by ``det_score * sqrt(bbox_area)``, see :meth:`server.face.FaceEngine.best_face`)
        becomes the person's first face sample, creating them like
        ``enroll_voice`` does. A background task then keeps sampling for a few
        more seconds so the tool call itself returns fast, mirroring how voice
        enrollment collects its extra samples from the utterances that follow
        instead of blocking the first reply.
        """
        name = " ".join(str(args.get("name") or "").split())
        frame_id = f"c{self._camera_seq}"
        self._camera_seq += 1
        record: dict[str, Any] = {"id": frame_id, "tool": "enroll_face", "args": dict(args)}
        self._utterance_actions.append(record)

        def fail(error: str) -> dict[str, Any]:
            record["result"] = {"ok": False, "error": error}
            return record["result"]

        if not name:
            return fail("enroll_face needs a non-empty name")
        if _voices is None:
            return fail("the people registry is not available")
        engine = _face
        if not self.face_enabled or engine is None or not engine.available:
            return fail(
                "face recognition is not available on the server - "
                "voice enrollment still works"
            )

        face_cfg = getattr(getattr(self.cfg, "server", None), "face", None)
        burst_size = max(1, _positive_int(getattr(face_cfg, "burst_size", 3)) or 3)
        enroll_bursts = max(0, _positive_int(getattr(face_cfg, "enroll_bursts", 3)) or 0)

        log.info("enroll_face (%s) for %s: first burst of %d frame(s)", frame_id, name, burst_size)
        captured = await self._request_camera_burst(frame_id, burst_size)
        if isinstance(captured, str):
            return fail(captured)

        best = await asyncio.to_thread(engine.best_face, [frame.jpeg for frame in captured])
        if best is None:
            return fail(
                "no face was visible in the camera frames - ask them to look "
                "straight at the camera and try once more"
            )
        embedding, score = best
        try:
            role, status = await asyncio.to_thread(
                _voices.add_face_embedding, name, embedding
            )
        except ValueError as exc:
            return fail(str(exc))
        except Exception as exc:
            log.exception("Could not store a face sample for %s", name)
            return fail(f"could not store the face sample: {exc}")

        result: dict[str, Any] = {
            "ok": True,
            "role": role,
            "status": status,
            "faces_seen": len(captured),
            "score": round(float(score), 3),
        }
        if enroll_bursts > 0:
            result["next"] = (
                "tell them to keep looking at the camera and slowly turn their "
                "head left and right for the next several seconds - more "
                "samples are collected automatically, no need to call this again"
            )
            self._start_enroll_face_task(name, enroll_bursts, burst_size)
        record["result"] = result
        return result

    def _start_enroll_face_task(self, name: str, enroll_bursts: int, burst_size: int) -> None:
        """Start the background sampler for ``name`` (SPEC v1.4 burst).

        Per-connection and singular: a second ``enroll_face`` call (same
        person or not) replaces whatever background sampling is still running
        rather than piling up several of them on one camera.
        """
        previous = self._enroll_face_task
        if previous is not None and not previous.done():
            previous.cancel()
        self._enroll_face_task = asyncio.create_task(
            self._enroll_face_background(name, enroll_bursts, burst_size)
        )

    async def _enroll_face_background(self, name: str, enroll_bursts: int, burst_size: int) -> None:
        """Collect extra face samples for ``name`` after the immediate one (v1.4 burst).

        Pulls ``enroll_bursts`` more bursts, :data:`ENROLL_FACE_INTERVAL_S`
        apart (so the whole run is roughly ``enroll_bursts *
        ENROLL_FACE_INTERVAL_S`` — about 9 s at the defaults), each burst
        contributing its own best-face embedding to the person (the per-person
        cap in :mod:`server.speaker` still applies). Never raises: a failed or
        empty burst is simply skipped — the person already has their first
        sample from the immediate call. Cancelled on disconnect
        (:meth:`Connection.close`).
        """
        added = 0
        for i in range(enroll_bursts):
            try:
                await asyncio.sleep(ENROLL_FACE_INTERVAL_S)
            except asyncio.CancelledError:
                raise
            engine, registry = _face, _voices
            if engine is None or registry is None or not self.face_enabled:
                continue
            frame_id = f"c{self._camera_seq}"
            self._camera_seq += 1
            try:
                captured = await self._request_camera_burst(frame_id, burst_size)
                if isinstance(captured, str):
                    log.debug(
                        "enroll_face background burst %d/%d failed for %s: %s",
                        i + 1, enroll_bursts, name, captured,
                    )
                    continue
                best = await asyncio.to_thread(engine.best_face, [frame.jpeg for frame in captured])
                if best is None:
                    continue
                embedding, _score = best
                await asyncio.to_thread(registry.add_face_embedding, name, embedding)
                added += 1
            except asyncio.CancelledError:
                raise
            except (WebSocketDisconnect, RuntimeError):
                log.info("Client %s is gone - enroll_face background sampling stops", self.peer)
                return
            except Exception:
                log.exception(
                    "enroll_face background sampling failed for %s (burst %d/%d)",
                    name, i + 1, enroll_bursts,
                )

        total = "?"
        try:
            if _voices is not None:
                total = str(len(_voices.face_profiles().get(name, [])))
        except Exception:
            log.debug("Could not read the final face sample count for %s", name, exc_info=True)
        log.info(
            "Face enrollment for %s finished: %d extra sample(s) added (%s total)",
            name, added, total,
        )

    async def _run_find_object(self, args: dict[str, Any]) -> dict[str, Any]:
        """Count and locate objects described by ``target`` with SAM3 (v1.5).

        Pulls one frame exactly like ``look_at_camera``/``look_at_screen``
        (same request/binary-frame machinery, chosen by ``source``), then runs
        it through :class:`server.segment.Sam3Engine` in a worker thread — SAM3
        inference is blocking CUDA work and must never touch the event loop.
        A camera pull asks for FULL resolution (SPEC v1.6): the detector
        should see the frame the way the C920 actually captured it, not the
        <=1280px size presence pulls use. The boxes are drawn on the frame and
        pushed to the room screen (v1.6) whenever :func:`_should_show_detections`
        says so — a successful match (count > 0), OR the user explicitly asked
        to see/show the result even when nothing was found (BUG 4).
        """
        target = " ".join(str(args.get("target") or "").split())
        source = str(args.get("source") or SOURCE_CAMERA).strip().lower()
        if source not in (SOURCE_CAMERA, SOURCE_SCREEN):
            source = SOURCE_CAMERA
        show_requested = bool(args.get("show"))

        if source == SOURCE_SCREEN:
            frame_id = f"s{self._screenshot_seq}"
            self._screenshot_seq += 1
        else:
            frame_id = f"c{self._camera_seq}"
            self._camera_seq += 1
        record: dict[str, Any] = {"id": frame_id, "tool": "find_object", "args": dict(args)}
        self._utterance_actions.append(record)

        def fail(error: str) -> dict[str, Any]:
            record["result"] = {"ok": False, "error": error}
            return record["result"]

        if not target:
            return fail("find_object needs a target: describe what to look for")
        if _segment is None or not _segment.enabled:
            return fail("the object finder is not available")

        log.info("find_object (%s, %s): %r", frame_id, source, target)
        if source == SOURCE_SCREEN:
            captured = await self._request_screenshot(frame_id)
        else:
            captured = await self._request_camera_frame_full(frame_id)
        if isinstance(captured, str):
            return fail(captured)

        result = await asyncio.to_thread(_segment.segment, captured.jpeg, target)
        if result.get("ok"):
            count = int(result.get("count") or 0)
            if count <= 0:
                result["summary"] = "nothing matching found"
            elif count == 1:
                result["summary"] = "found 1 match"
            else:
                result["summary"] = f"found {count} matches"
            if _should_show_detections(count, show_requested):
                await self._push_detections_photo(captured, target, result)
        record["result"] = result
        return result

    async def _run_show_photo(self, args: dict[str, Any]) -> dict[str, Any]:
        """Show the picture already taken, WITHOUT taking a new one (v1.6).

        "Show me the photo you just described" must display that exact frame:
        the last camera/screen frame and the last annotated detections photo
        are cached, so this costs no capture and no vision pass.
        """
        which = str(args.get("which") or "").strip().lower()
        record: dict[str, Any] = {
            "id": f"img{self._image_seq}",
            "tool": "show_photo",
            "args": dict(args),
        }
        self._utterance_actions.append(record)

        def done(result: dict[str, Any]) -> dict[str, Any]:
            record["result"] = result
            return result

        if which in ("hide", "close", "off"):
            # Dismiss whatever is on the screen; no frame follows the header.
            try:
                await self.send_json(
                    {"type": proto.MSG_IMAGE_SHOW, "id": f"img{self._image_seq}", "hide": True}
                )
                self._image_seq += 1
            except (WebSocketDisconnect, RuntimeError):
                raise
            except Exception as exc:  # noqa: BLE001
                return done({"ok": False, "error": f"could not close the photo: {exc}"})
            return done({"ok": True, "note": "the photo was closed"})

        # Prefer the annotated detections photo when it is the freshest thing
        # or explicitly asked for; otherwise the last plain frame.
        camera_ts = self._last_frame_ts.get(SOURCE_CAMERA, 0.0)
        # Default ("show me the photo"): the annotated one wins whenever it is
        # at least as fresh as the plain capture - its boxes and labels are the
        # whole reason the owner wants to see it.
        prefer_annotated = bool(
            self._last_annotated and self._last_annotated_ts >= camera_ts - 1.0
        )
        if which in ("detections", "objects") and self._last_annotated:
            jpeg, w, h, title = self._last_annotated
        elif which == "screen" and SOURCE_SCREEN in self._last_frames:
            frame = self._last_frames[SOURCE_SCREEN]
            jpeg, w, h, title = frame.jpeg, frame.w, frame.h, "the screen"
        elif which in ("camera", "room", "") and prefer_annotated:
            jpeg, w, h, title = self._last_annotated  # type: ignore[misc]
        elif which in ("camera", "room", "") and SOURCE_CAMERA in self._last_frames:
            frame = self._last_frames[SOURCE_CAMERA]
            jpeg, w, h, title = frame.jpeg, frame.w, frame.h, "the room"
        elif self._last_annotated:
            jpeg, w, h, title = self._last_annotated
        elif self._last_frames:
            frame = next(iter(self._last_frames.values()))
            jpeg, w, h, title = frame.jpeg, frame.w, frame.h, "the last photo"
        else:
            return done(
                {
                    "ok": False,
                    "error": "there is no photo yet - look at the camera or the "
                    "screen first, then show it",
                }
            )

        try:
            await self._send_image_show(jpeg, w, h, title, IMAGE_SHOW_TTL_S)
        except (WebSocketDisconnect, RuntimeError):
            raise
        except Exception as exc:  # noqa: BLE001 - showing is best effort
            log.exception("Could not show the cached photo")
            return done({"ok": False, "error": f"could not show the photo: {exc}"})
        return done(
            {
                "ok": True,
                "note": "the photo is now on the room screen - tell the user it is up",
            }
        )

    async def _push_detections_photo(
        self, frame: ImageFrame, target: str, result: dict[str, Any]
    ) -> None:
        """Draw find_object's boxes on the pulled frame and show it (SPEC v1.6).

        Best-effort: a failure here never touches ``result["ok"]`` — the tool
        call already succeeded, only the bonus photo on the TV is at risk.
        """
        try:
            count = int(result.get("count") or 0)
            annotated = await asyncio.to_thread(
                draw_boxes,
                frame.jpeg,
                result.get("boxes") or [],
                result.get("scores") or [],
                85,
                target,
            )
            title = f"{target} - {count} found"
            self._last_annotated = (annotated, frame.w, frame.h, title)
            self._last_annotated_ts = time.monotonic()
            await self._send_image_show(annotated, frame.w, frame.h, title, IMAGE_SHOW_TTL_S)
            # BUG 4: worded as a direct instruction (not just a fact) so the
            # model reliably says it out loud instead of only saying "Done".
            result["note"] = (
                "an annotated photo was just put on the room screen - tell the "
                "user you put the photo on the screen"
            )
        except (WebSocketDisconnect, RuntimeError):
            raise
        except Exception:
            log.exception("Could not push the find_object detections photo")

    async def _send_image_show(
        self, jpeg: bytes, w: int, h: int, title: str, ttl_s: float
    ) -> None:
        """Push one ``image_show`` header + its single binary JPEG (SPEC v1.6)."""
        image_id = f"img{self._image_seq}"
        self._image_seq += 1
        await self.send_json(
            {
                "type": proto.MSG_IMAGE_SHOW,
                "id": image_id,
                "w": int(w),
                "h": int(h),
                "title": str(title),
                "ttl_s": float(ttl_s),
            }
        )
        await self.ws.send_bytes(jpeg)

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
                presence=self.presence_text,
            )
            self._start_greeting_task()
        if self._task is not None and not self._task.done():
            # v1.6: a new utterance INTERRUPTS the one in flight (the user said
            # the wake word to stop a chain that was doing the wrong thing).
            log.info("New utterance from %s interrupts the reply in flight", self.peer)
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - the old turn must not break the new one
                log.debug("The cancelled turn ended noisily", exc_info=True)
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
        self._camera_seq = 1
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
            self._speaker_score = 0.0
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

        # 1b. Speaker identification + enrollment continuation (SPEC v1.3)
        self._current_pcm = pcm
        enroll_note = ""
        if _voices is not None and _voices.enabled:
            name, role, score = await asyncio.to_thread(
                _voices.identify, pcm, self.sample_rate
            )
            self._speaker_name, self._speaker_role, self._speaker_score = name, role, score
            if name != speaker_mod.ROLE_UNKNOWN:
                # v1.6: holds the greeting off while a known voice was just
                # heard on this connection (see _may_greet).
                self._last_known_voice_at = time.monotonic()
            pending = self._enroll_pending
            if pending:
                # v1.6: every attempt counts its seconds, even a failed one —
                # a too-short sample still tells us something about progress,
                # it just asks for a LONGER sentence instead of "keep talking".
                sample_s = speaker_mod.estimate_speech_seconds(pcm)
                try:
                    await asyncio.to_thread(
                        _voices.enroll, pending["name"], pcm, self.sample_rate
                    )
                except ValueError as exc:
                    pending["total_speech_s"] = pending.get("total_speech_s", 0.0) + sample_s
                    enroll_note = (
                        f" [enrollment: that sample was too short to use for "
                        f"{pending['name']} ({exc}) - ask for a LONGER sentence; "
                        f"{self._enroll_progress_note(pending)}]"
                    )
                else:
                    pending["samples"] = pending.get("samples", 0) + 1
                    pending["total_speech_s"] = pending.get("total_speech_s", 0.0) + sample_s
                    if speaker_mod.enrollment_complete(
                        pending["samples"], pending["total_speech_s"]
                    ):
                        self._enroll_pending = None
                        enroll_note = (
                            f" [enrollment: done - {pending['name']}'s voice profile "
                            "is complete, tell them so]"
                        )
                    else:
                        enroll_note = f" [enrollment: {self._enroll_progress_note(pending)}]"

        # The LLM sees who is talking and the live room view; permissions are
        # enforced server-side. Presence rides here (not in the system prompt)
        # so the prompt prefix stays byte-identical and Ollama's cache holds.
        try:
            room_text = self.presence_text()
        except Exception:  # noqa: BLE001 - presence must never break a reply
            room_text = ""
        room_part = f" [room: {room_text}]" if room_text else ""
        prefixed = (
            f"[speaker: {self._speaker_name} | role: {self._speaker_role}]"
            f"{room_part}{enroll_note} {text}"
        )

        # 2. LLM with the tool loop — tools are executed for real (SPEC §3, §5)
        # The lock keeps a proactive greeting (SPEC v1.4) from interleaving with
        # this reply: only one say + tts_start…tts_end block is ever in flight.
        async with self._reply_lock:
            t_llm = time.perf_counter()
            try:
                result = await brain.generate(session.messages(prefixed), self._execute_tool)
            except (WebSocketDisconnect, RuntimeError):
                raise
            except Exception:
                log.exception("LLM request failed")
                await self.send_error("llm failed")
                return
            # Self-check ("judge"): only after a turn that actually CHANGED
            # something (not plain chat, not a pure look) - re-prompt the model
            # to confirm it did everything asked/promised and finish any gap.
            if getattr(self.cfg.server.llm, "verify_actions", True) and self._turn_changed_state():
                try:
                    verified = await asyncio.wait_for(
                        brain.verify(result.history, result.text, self._execute_tool),
                        timeout=VERIFY_TIMEOUT_S,
                    )
                    if verified.text.strip():
                        log.info("Self-check produced the final reply (%d extra action(s))",
                                 len(verified.tool_calls))
                        result = verified
                except asyncio.TimeoutError:
                    log.warning("Self-check timed out - keeping the original reply")
                except (WebSocketDisconnect, RuntimeError):
                    raise
                except Exception:
                    log.exception("Self-check failed - keeping the original reply")

            llm_ms = int((time.perf_counter() - t_llm) * 1000)

            say_text = result.text.strip()
            if not say_text:
                say_text = SAY_AFTER_ACTIONS if self._utterance_actions else SAY_NOT_UNDERSTOOD
            session.remember(prefixed, say_text)

            # 3. say -> tts stream (order fixed by SPEC §4)
            say_payload: dict[str, Any] = {"type": proto.MSG_SAY, "text": say_text}
            if self._enroll_pending:
                # Voice enrollment expects the speaker to keep talking: tell the
                # client to hold the follow-up window open longer than usual.
                say_payload["listen_s"] = ENROLL_LISTEN_S
            await self.send_json(say_payload)
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
            "speaker": self._speaker_name,
            "speaker_score": round(self._speaker_score, 3),
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
        # v1.4: the greeting task waits out any audio the client is playing.
        self._last_audio_at = time.monotonic()

    async def close(self) -> None:
        """Cancel everything still in flight and fail every pending wait."""
        for background in (self._greet_task, self._enroll_face_task, *tuple(self._presence_tasks)):
            if background is not None and not background.done():
                background.cancel()
        self._greet_task = None
        self._enroll_face_task = None
        self._presence_tasks.clear()
        self._presence_burst_frames = []
        self._presence_burst_id = ""

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
        for future in list(self._image_futures.values()):
            if not future.done():
                future.set_result({"error": "client disconnected"})
        self._image_futures.clear()
        self._image_ids.clear()
        self._image_header = {}
        self._expect_image = None
        self.presence.clear()


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
