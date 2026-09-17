"""WebSocket protocol constants shared by server and client (SPEC section 4).

Control frames are JSON text frames, audio and screenshots are sent as raw binary
frames. Never inline the message-type strings anywhere else -- import them from
here.

Client -> Server
----------------
* ``{"type": MSG_HELLO, "client_id": str, "devices": [...]}``
* ``{"type": MSG_UTTERANCE_START, "sr": 16000, "format": "pcm_s16le", "channels": 1}``
* binary frames: raw PCM s16le mono 16 kHz
* ``{"type": MSG_UTTERANCE_END}``
* ``{"type": MSG_ACTION_RESULT, "id": str, "ok": bool, "error": str | None,
  "output": str | None}``
* ``{"type": MSG_SCREENSHOT, "id": str, "format": "jpeg"}`` followed by exactly ONE
  binary frame with the JPEG bytes, or ``{"type": MSG_SCREENSHOT_ERROR, "id": str,
  "error": str}`` with no binary frame.

Server -> Client
----------------
* ``{"type": MSG_READY}``
* ``{"type": MSG_TRANSCRIPT, "text": str, "language": str}``
* ``{"type": MSG_ACTIONS, "items": [{"id": str, "tool": str, "args": {...}}]}``
  -- may be sent several times per utterance (one per tool round).
* ``{"type": MSG_SCREENSHOT_REQUEST, "id": str}``
* ``{"type": MSG_SAY, "text": str}``
* ``{"type": MSG_TTS_START, "sr": int, "format": "pcm_s16le", "channels": 1}``
  followed by binary PCM frames and ``{"type": MSG_TTS_END}``
* ``{"type": MSG_ERROR, "message": str}``

Order per utterance: transcript -> zero or more rounds of actions and/or
screenshot_request (each awaited) -> say -> tts_start ... tts_end.

Binary-frame disambiguation: the client sends binary frames only between
``utterance_start``/``utterance_end`` and as the single frame announced by a
``screenshot`` header; the two never overlap.
"""

from __future__ import annotations

# --- client -> server -------------------------------------------------------
MSG_HELLO = "hello"
MSG_UTTERANCE_START = "utterance_start"
MSG_UTTERANCE_END = "utterance_end"
MSG_ACTION_RESULT = "action_result"
#: v1.1: header announcing the single binary frame with the JPEG screenshot.
MSG_SCREENSHOT = "screenshot"
#: v1.1: the client could not capture the screen; no binary frame follows.
MSG_SCREENSHOT_ERROR = "screenshot_error"

# --- server -> client -------------------------------------------------------
MSG_READY = "ready"
MSG_TRANSCRIPT = "transcript"
MSG_ACTIONS = "actions"
#: v1.1: ask the client for a screenshot of the room PC's screen.
MSG_SCREENSHOT_REQUEST = "screenshot_request"
MSG_SAY = "say"
MSG_TTS_START = "tts_start"
MSG_TTS_END = "tts_end"
MSG_ERROR = "error"

# --- shared literals used inside the frames ---------------------------------
#: WebSocket endpoint path served by the brain server.
WS_PATH = "/ws"

#: Audio wire format for both microphone and TTS streams.
AUDIO_FORMAT = "pcm_s16le"

#: Channel count for both directions (mono).
AUDIO_CHANNELS = 1

#: Microphone sample rate expected by the server (Whisper/VAD/Vosk all use it).
MIC_SAMPLE_RATE = 16000

#: Image format of the screenshot frame (SPEC section 4, client message 6).
SCREENSHOT_FORMAT = "jpeg"

#: Error message the server sends when STT produced nothing (false wake-word).
ERR_EMPTY_TRANSCRIPT = "empty transcript"

#: Tool result the server substitutes when the client does not answer in time.
ERR_CLIENT_TIMEOUT = "client timeout"

#: Every message type a client may send.
CLIENT_MESSAGE_TYPES = frozenset(
    {
        MSG_HELLO,
        MSG_UTTERANCE_START,
        MSG_UTTERANCE_END,
        MSG_ACTION_RESULT,
        MSG_SCREENSHOT,
        MSG_SCREENSHOT_ERROR,
    }
)

#: Every message type a server may send.
SERVER_MESSAGE_TYPES = frozenset(
    {
        MSG_READY,
        MSG_TRANSCRIPT,
        MSG_ACTIONS,
        MSG_SCREENSHOT_REQUEST,
        MSG_SAY,
        MSG_TTS_START,
        MSG_TTS_END,
        MSG_ERROR,
    }
)

__all__ = [
    "MSG_HELLO",
    "MSG_UTTERANCE_START",
    "MSG_UTTERANCE_END",
    "MSG_ACTION_RESULT",
    "MSG_SCREENSHOT",
    "MSG_SCREENSHOT_ERROR",
    "MSG_READY",
    "MSG_TRANSCRIPT",
    "MSG_ACTIONS",
    "MSG_SCREENSHOT_REQUEST",
    "MSG_SAY",
    "MSG_TTS_START",
    "MSG_TTS_END",
    "MSG_ERROR",
    "WS_PATH",
    "AUDIO_FORMAT",
    "AUDIO_CHANNELS",
    "MIC_SAMPLE_RATE",
    "SCREENSHOT_FORMAT",
    "ERR_EMPTY_TRANSCRIPT",
    "ERR_CLIENT_TIMEOUT",
    "CLIENT_MESSAGE_TYPES",
    "SERVER_MESSAGE_TYPES",
]
