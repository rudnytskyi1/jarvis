"""Loading and validation of the single YAML config shared by both machines.

Public API (SPEC section 6)::

    from common.config import load_config

    cfg = load_config("config.yaml")
    cfg.server.host                 # "0.0.0.0"
    cfg.server.port                 # 8765
    cfg.server.stt.model            # "large-v3"
    cfg.server.stt.allowed_languages  # ["en", "ru", "es"] (auto-detect whitelist)
    cfg.server.llm.base_url         # "http://127.0.0.1:11434/v1"
    cfg.server.llm.provider         # "ollama_native" | "openai"
    cfg.server.llm.think            # False (Qwen3 reasoning off for fast replies)
    cfg.server.llm.vision_model     # "qwen3-vl:8b" (look_at_screen)
    cfg.server.llm.max_tool_rounds  # 4
    cfg.server.speaker.threshold    # 0.72 (voice matching, v1.3)
    cfg.server.face.threshold       # 0.45 (face matching + presence, v1.4)
    cfg.server.segment.{enabled, checkpoint, confidence}  # v1.5: SAM3 find_object
    cfg.server.tts.speaker          # "en_0"
    cfg.client.server_url           # "ws://192.168.1.100:8765/ws"
    cfg.client.wakeword.phrases     # ["rowan", "roan", "rowen"]
    cfg.client.audio.input_device   # int | str | None
    cfg.client.vad.silence_ms       # 800
    cfg.client.camera.fps           # 5 (room camera -> YOLO presence, v1.4)
    cfg.client.followup_window_s    # 6.0
    cfg.client.apps                 # {"browser": "C:\\...\\chrome.exe"} (may be empty)
    cfg.client.devices              # [DeviceConfig, ...] (may be empty)

Defaults mirror ``config.example.yaml`` exactly. A missing required key (or a
key with a wrong type) raises a ``ValueError`` whose message names the key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

__all__ = [
    "Config",
    "ServerConfig",
    "STTConfig",
    "LLMConfig",
    "SpeakerConfig",
    "FaceConfig",
    "SegmentConfig",
    "TTSConfig",
    "ClientConfig",
    "WakewordConfig",
    "AudioConfig",
    "VADConfig",
    "CameraConfig",
    "DeviceConfig",
    "load_config",
    "DEFAULT_CONFIG_FILENAME",
    "EXAMPLE_CONFIG_FILENAME",
]

DEFAULT_CONFIG_FILENAME = "config.yaml"
EXAMPLE_CONFIG_FILENAME = "config.example.yaml"

#: Repo root (this file lives at <root>/common/config.py) — used to resolve a
#: relative ``server.segment.checkpoint`` (SAM3) regardless of the process's
#: current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]


class _Strict(BaseModel):
    """Base for config sections: unknown keys are an error (catches typos)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# ---------------------------------------------------------------------------
# server section
# ---------------------------------------------------------------------------


class STTConfig(_Strict):
    """faster-whisper settings (``server.stt``)."""

    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    #: ``None``/empty -> language auto-detection.
    language: Optional[str] = None
    #: Whitelist for auto-detection (the languages actually spoken in the room);
    #: empty list = any of Whisper's 99 languages. Ignored when ``language`` is set.
    allowed_languages: list[str] = Field(default_factory=lambda: ["en", "ru", "es"])

    @field_validator("language", mode="after")
    @classmethod
    def _empty_language_is_none(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            return None
        return value


class LLMConfig(_Strict):
    """LLM endpoint settings (``server.llm``).

    Two backends are supported (SPEC section 3): ``"ollama_native"`` talks to
    Ollama's own ``/api/chat`` (the only way to switch Qwen3 reasoning off) and
    ``"openai"`` uses the OpenAI-compatible ``/v1`` surface.
    """

    #: "ollama_native" (default) | "openai".
    provider: str = "ollama_native"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3:30b"
    api_key: str = "ollama"
    #: Qwen3 reasoning; False keeps voice replies fast (ollama_native only).
    think: bool = False
    #: Vision model used by the look_at_screen tool (8B fits in VRAM next to
    #: the 30B chat model, so screen questions do not trigger a model swap).
    vision_model: str = "qwen3-vl:8b"
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    #: How many tool-call rounds one utterance may take before a final answer.
    max_tool_rounds: int = Field(default=4, ge=1)
    #: How many last user/assistant exchanges are kept in the session history.
    history_turns: int = Field(default=12, ge=0)
    #: How long Ollama keeps the chat model loaded after a request
    #: (a duration string like "4h", or "-1" for forever; ollama_native only).
    #: Ollama's own default of 5 m makes the first command after a quiet spell
    #: pay a full model reload from disk.
    keep_alive: str = "4h"
    #: Context window requested from Ollama (ollama_native only). The model's
    #: own default (32k for qwen3:30b) wastes several GB of VRAM on KV cache
    #: that a voice assistant with a short history never uses.
    num_ctx: int = Field(default=8192, ge=1024)

    @field_validator("provider", mode="after")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        provider = (value or "").strip().lower()
        if provider not in {"ollama_native", "openai"}:
            raise ValueError('must be "ollama_native" or "openai"')
        return provider


class SpeakerConfig(_Strict):
    """Speaker recognition (``server.speaker``, SPEC v1.3)."""

    enabled: bool = True
    #: Cosine-similarity threshold for a voice to match an enrolled profile.
    threshold: float = Field(default=0.72, gt=0.0, le=1.0)
    #: Utterances shorter than this are not identified (too little voice).
    min_speech_s: float = Field(default=0.8, ge=0.0)


class FaceConfig(_Strict):
    """Face recognition and room presence (``server.face``, SPEC v1.4).

    The server matches every camera frame the client pushes against the
    ``face_embeddings`` of ``data/people.json`` and keeps a per-connection
    presence map that feeds the ``{presence}`` block of the system prompt.
    """

    enabled: bool = True
    #: Cosine-similarity threshold for a face to match an enrolled profile.
    threshold: float = Field(default=0.45, gt=0.0, le=1.0)
    #: Somebody is forgotten this long after the camera last saw them.
    presence_ttl_s: float = Field(default=30.0, gt=0.0)
    #: An unknown face present for this long triggers the proactive greeting.
    greet_after_s: float = Field(default=10.0, ge=0.0)
    #: At most one proactive greeting per this window (0 = no cooldown).
    greeting_cooldown_s: float = Field(default=300.0, ge=0.0)


class SegmentConfig(_Strict):
    """SAM3 open-vocabulary object detection (``server.segment``, v1.5).

    Backs the ``find_object`` tool: counts and locates objects described in
    free text in a camera or screen frame. Everything about SAM3 itself (the
    ``sys.path`` insertion, the imports, the checkpoint load) stays lazy in
    ``server/segment.py`` — this section only carries the settings.
    """

    enabled: bool = True
    #: Local checkpoint file; resolved below relative to the repo root when
    #: given as a relative path, regardless of the process's cwd.
    checkpoint: str = "third_party/sam3/server/model/sam3.pt"
    #: SAM3's own confidence threshold for keeping a match.
    confidence: float = Field(default=0.5, gt=0.0, le=1.0)

    @field_validator("checkpoint", mode="after")
    @classmethod
    def _resolve_checkpoint(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return str(path)


class TTSConfig(_Strict):
    """Silero TTS settings (``server.tts``)."""

    engine: str = "silero"
    language: str = "en"
    model_id: str = "v3_en"
    speaker: str = "en_0"
    sample_rate: int = Field(default=48000, ge=8000)


class ServerConfig(_Strict):
    """Everything the brain PC reads (``server``)."""

    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1, le=65535)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)


# ---------------------------------------------------------------------------
# client section
# ---------------------------------------------------------------------------


class WakewordConfig(_Strict):
    """Vosk wake-word settings (``client.wakeword``)."""

    word: str = "rowan"
    #: Recognition variants; empty list is normalised to ``[word]``.
    phrases: List[str] = Field(default_factory=list)
    vosk_model: str = "models/vosk-model-small-en-us-0.15"

    @model_validator(mode="after")
    def _default_phrases(self) -> "WakewordConfig":
        phrases = [p.strip() for p in self.phrases if isinstance(p, str) and p.strip()]
        if not phrases:
            phrases = [self.word.strip()]
        self.phrases = phrases
        return self


class AudioConfig(_Strict):
    """sounddevice settings (``client.audio``).

    ``input_device``/``output_device`` are either a device index (int), a
    substring of the device name (str) or ``None`` for the system default.
    """

    input_device: Union[int, str, None] = None
    output_device: Union[int, str, None] = None
    sample_rate: int = Field(default=16000, ge=8000)


class VADConfig(_Strict):
    """webrtcvad settings (``client.vad``)."""

    aggressiveness: int = Field(default=2, ge=0, le=3)
    silence_ms: int = Field(default=800, ge=0)
    max_utterance_s: float = Field(default=15.0, gt=0.0)
    pre_roll_ms: int = Field(default=300, ge=0)
    #: Minimum voiced audio for a recording to count as an utterance; anything
    #: shorter is a noise blip - discarded without contacting the server.
    min_speech_ms: int = Field(default=250, ge=0)


class CameraConfig(_Strict):
    """Room camera settings (``client.camera``, SPEC v1.4).

    The client runs YOLO on the capture and reports STATE (how many people,
    which objects), plus one JPEG every ``face_check_interval_s`` while
    somebody is visible so the server can recognise faces. Missing camera
    dependencies must never break the voice pipeline.
    """

    enabled: bool = True
    #: OpenCV capture device index.
    index: int = Field(default=0, ge=0)
    #: How many frames per second are pushed through YOLO.
    fps: int = Field(default=5, ge=1)
    #: Ultralytics model file (downloaded automatically on first run).
    model: str = "yolo11n.pt"
    #: One frame is sent to the server this often while a person is visible.
    face_check_interval_s: float = Field(default=5.0, gt=0.0)


class DeviceConfig(BaseModel):
    """One controllable device from ``client.devices``.

    ``name``/``type``/``area``/``description`` are first-class fields; every
    other key of the YAML mapping (``host``, ``dev_id``, ``local_key``,
    ``version``, ``mac``, ``mode``, ...) is collected into :attr:`params` and is
    also kept as an attribute of the model.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    name: str
    #: "magichome" | "tuya" | "switchbot_bot"
    type: str
    area: Optional[str] = None
    description: Optional[str] = None
    #: Type-specific fields taken from the same YAML mapping.
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_params(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        first_class = {"name", "type", "area", "description", "params"}
        explicit = data.get("params")
        params: Dict[str, Any] = dict(explicit) if isinstance(explicit, dict) else {}
        for key, value in data.items():
            if key not in first_class:
                params[str(key)] = value
        merged = dict(data)
        merged["params"] = params
        return merged


class ClientConfig(_Strict):
    """Everything the room PC reads (``client``)."""

    #: WebSocket URL of the brain PC, e.g. ``ws://192.168.1.100:8765/ws``.
    server_url: str
    client_id: str = "livingroom"
    wakeword: WakewordConfig = Field(default_factory=WakewordConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    #: Room camera: YOLO presence state + face frames for the server (v1.4).
    camera: CameraConfig = Field(default_factory=CameraConfig)
    #: Seconds to keep listening after a reply without the wake word (0 = off).
    followup_window_s: float = Field(default=6.0, ge=0.0)
    #: Soft repeating blips while the server is still working on a reply
    #: (vision, tool rounds) so silence never looks like a hang.
    thinking_sounds: bool = True
    #: Friendly app name -> executable path / command; OVERRIDES on top of the
    #: client's installed-app index. Empty by default.
    apps: Dict[str, str] = Field(default_factory=dict)
    #: Physical devices; empty by default (none are installed yet).
    devices: List[DeviceConfig] = Field(default_factory=list)

    @field_validator("apps", mode="after")
    @classmethod
    def _expand_app_paths(cls, value: Dict[str, str]) -> Dict[str, str]:
        # Allow %USERNAME%-style environment variables in the example config.
        return {name: os.path.expandvars(path) for name, path in value.items()}

    @model_validator(mode="after")
    def _unique_device_names(self) -> "ClientConfig":
        seen: set = set()
        for device in self.devices:
            key = device.name.strip().lower()
            if key in seen:
                raise ValueError(f"duplicate device name: {device.name!r}")
            seen.add(key)
        return self


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------


def _default_client() -> ClientConfig:
    return ClientConfig(server_url="ws://127.0.0.1:8765/ws")


class Config(_Strict):
    """Root of ``config.yaml`` — both sections live in one file."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    client: ClientConfig = Field(default_factory=_default_client)

    @model_validator(mode="before")
    @classmethod
    def _drop_empty_sections(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # "server:" / "client:" written without a body parses as None -> use
        # the defaults instead of failing on a null section.
        return {key: value for key, value in data.items() if value is not None}


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = [f"Invalid config {source}:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "<root>"
        kind = err.get("type", "")
        if kind == "missing":
            lines.append(f"  - {loc}: required key is missing")
        elif kind == "extra_forbidden":
            lines.append(f"  - {loc}: unknown key (typo? compare with {EXAMPLE_CONFIG_FILENAME})")
        else:
            lines.append(f"  - {loc}: {err.get('msg', kind)}")
    lines.append(f"Template with every key and its default: {EXAMPLE_CONFIG_FILENAME}")
    return "\n".join(lines)


def load_config(path: Union[str, os.PathLike, None] = None) -> Config:
    """Load, validate and return the config.

    :param path: path to the YAML file; ``None`` means ``config.yaml`` in the
        current working directory (entry points are started from the repo root).
    :raises ValueError: the file is missing, is not valid YAML, or a required
        key is missing / has a wrong type — the message names the key.
    """
    cfg_path = Path(path) if path is not None else Path(DEFAULT_CONFIG_FILENAME)
    if not cfg_path.is_file():
        raise ValueError(
            f"Config not found: {cfg_path}. "
            f"Copy {EXAMPLE_CONFIG_FILENAME} to {DEFAULT_CONFIG_FILENAME} and adjust it."
        )

    try:
        text = cfg_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - depends on the file
        raise ValueError(f"Config {cfg_path} must be UTF-8 encoded: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse YAML {cfg_path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {cfg_path} must be a mapping with server: and client: sections")

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc, str(cfg_path))) from exc
