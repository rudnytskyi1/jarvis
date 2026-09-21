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
    cfg.server.speaker.threshold    # 0.40 cosine, not a probability
    cfg.server.face.threshold       # 0.45 (face matching + presence, v1.4)
    cfg.server.face.{burst_size, enroll_bursts}  # v1.4 burst: 3, 3 (multi-frame camera pulls)
    cfg.server.segment.{enabled, checkpoint, confidence}  # v1.5: SAM3 find_object
    cfg.server.tts.speaker          # "en_0"
    cfg.client.server_url           # "ws://192.168.1.100:8765/ws"
    cfg.client.wakeword.phrases     # ["rowan ai"] unless variants are configured
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
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from common.client_config import (
    AudioConfig,
    CameraConfig,
    ClientConfig,
    ClientOTAConfig,
    DeviceConfig,
    OverlayConfig,
    RecordingConfig,
    VADConfig,
    WakewordConfig,
)
from common.openai_models import OPENAI_TEXT_RATES

__all__ = [
    "Config",
    "ServerConfig",
    "STTConfig",
    "DiarizationConfig",
    "LLMConfig",
    "SpeakerConfig",
    "FaceConfig",
    "SegmentConfig",
    "GpuQueueConfig",
    "SkillReloadConfig",
    "WebAdminConfig",
    "ModelsConfig",
    "ModelLevelConfig",
    "ModelRoutingConfig",
    "DEFAULT_LEVEL_NAMES",
    "TelegramConfig",
    "TTSConfig",
    "MediaConfig",
    "ClientConfig",
    "ClientOTAConfig",
    "WakewordConfig",
    "AudioConfig",
    "VADConfig",
    "CameraConfig",
    "OverlayConfig",
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
    language: str | None = None
    #: Whitelist for auto-detection (the languages actually spoken in the room);
    #: empty list = any of Whisper's 99 languages. Ignored when ``language`` is set.
    allowed_languages: list[str] = Field(default_factory=lambda: ["en", "ru", "es"])
    #: Spelling hints only; never include conversation history or command text.
    hotwords: list[str] = Field(default_factory=lambda: ["Rowan"], max_length=32)
    live_transcript: bool = True
    live_interval_s: float = Field(default=1.2, ge=.6, le=5)
    live_window_s: float = Field(default=12, ge=6, le=20)
    #: ТЗ 4.4/15.1: utterances that arrive together are decoded in one
    #: faster-whisper call (2-4 clips per batch). 1 switches batching off.
    batch_size: int = Field(default=4, ge=1, le=4)
    batch_window_ms: int = Field(default=40, ge=0, le=500)

    @field_validator("language", mode="after")
    @classmethod
    def _empty_language_is_none(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


class DiarizationConfig(_Strict):
    """Local multi-speaker processing, enabled after model setup."""

    enabled: bool = False
    model_path: str = "models/speaker-diarization-community-1"
    reject_mixed_speech: bool = True
    device: Literal["cuda", "cpu"] = "cuda"
    timeout_s: float = Field(default=45, ge=5, le=120)
    min_identity_s: float = Field(default=1.5, ge=.8, le=10)


class LLMConfig(_Strict):
    """LLM endpoint settings (``server.llm``).

    Four backends are supported (SPEC section 3): ``"ollama_native"`` talks to
    Ollama's own ``/api/chat`` (the only way to switch Qwen3 reasoning off),
    ``"openai"`` uses the OpenAI-compatible ``/v1`` surface, ``"vllm"`` is the
    same surface on a vLLM server (with schema-validated JSON, ТЗ F-402), and
    ``"openai_responses"`` is the official text-only API with persistent local
    budget accounting.
    """

    #: "ollama_native" (default) | "openai" | "vllm" | "openai_responses".
    provider: str = "ollama_native"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3:30b"
    api_key: str = "ollama"
    #: Budgeted official API reads the key only from this environment variable.
    api_key_env: str = "OPENAI_API_KEY"
    monthly_budget_usd: float = Field(default=18.0, gt=0, le=20)
    max_input_bytes: int = Field(default=64000, ge=4096, le=128000)
    #: None preserves the legacy Ollama URL. Required separately for cloud chat.
    vision_base_url: str | None = None
    prompt_file: str | None = None
    #: Qwen3 reasoning; False keeps voice replies fast (ollama_native only).
    think: bool = False
    #: Vision model used by the look_at_screen tool (8B fits in VRAM next to
    #: the 30B chat model, so screen questions do not trigger a model swap).
    vision_model: str = "qwen3-vl:8b"
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    #: How many tool-call rounds one utterance may take before a final answer.
    max_tool_rounds: int = Field(default=4, ge=1)
    #: v1.7: how long the VISION model stays in VRAM after a question.
    #: Separate from keep_alive because it is big (8.4 GB resident) and
    #: rarely used, and that memory is what SAM3 needs.
    vision_keep_alive: str = "10m"
    #: How many last user/assistant exchanges are kept in the session history.
    history_turns: int = Field(default=25, ge=0)
    #: How long Ollama keeps the chat model loaded after a request
    #: (a duration string like "4h", or "-1" for forever; ollama_native only).
    #: Ollama's own default of 5 m makes the first command after a quiet spell
    #: pay a full model reload from disk.
    keep_alive: str = "4h"
    #: Context window requested from Ollama (ollama_native only). The model's
    #: own default (32k for qwen3:30b) wastes several GB of VRAM on KV cache
    #: that a voice assistant with a short history never uses.
    #: v1.7: 16384 - the system prompt plus the tool schemas alone are ~7.6k
    #: tokens, so 8192 overflowed after a single exchange.
    num_ctx: int = Field(default=16384, ge=1024)
    #: After an action turn, re-prompt the model as a verifier that checks it
    #: actually did everything requested/promised and finishes anything missing.
    #: Runs only when a state-changing tool ran, so plain chat stays fast.
    verify_actions: bool = True
    #: Extra request fields for OpenAI-compatible servers (vLLM reads
    #: ``chat_template_kwargs`` and, on older builds, ``guided_json`` here).
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="after")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        provider = (value or "").strip().lower()
        if provider not in {"ollama_native", "openai", "openai_responses", "vllm"}:
            raise ValueError('must be "ollama_native", "openai", "openai_responses", or "vllm"')
        return provider

    @model_validator(mode="after")
    def _cloud_contract(self) -> LLMConfig:
        if self.provider == "openai_responses":
            if self.model not in OPENAI_TEXT_RATES:
                raise ValueError("openai_responses requires a model with reviewed pricing: " + ', '.join(OPENAI_TEXT_RATES))
            if self.max_tokens > 2048:
                raise ValueError("openai_responses max_tokens must not exceed 2048")
            if not self.api_key_env.strip():
                raise ValueError("api_key_env cannot be empty")
        return self


class ImageGenerationConfig(_Strict):
    """On-demand Google Nano Banana 2; shares the LLM's monthly allowance."""

    enabled: bool = False
    model: Literal['gemini-3.1-flash-image'] = 'gemini-3.1-flash-image'
    api_key_env: str = Field(default='GEMINI_API_KEY', min_length=1, pattern=r'^\w+$')
    image_size: Literal['1K'] = '1K'
    timeout_s: float = Field(default=120, ge=15, le=180)


class TelegramConfig(_Strict):
    """One group plus an optional account allowed to control Rowan; env token."""

    enabled: bool = False
    chat_id: int | None = Field(default=None, lt=0, gt=-(2 ** 63))
    control_user_id: int | None = Field(default=None, strict=True, gt=0, lt=2 ** 63)
    api_key_env: str = Field(default='TELEGRAM_BOT_TOKEN', min_length=1,
                             pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')
    timeout_s: float = Field(default=30, ge=5, le=120)
    respond_to_mentions: bool = False
    poll_timeout_s: int = Field(default=25, ge=1, le=50)


class SpeakerConfig(_Strict):
    """Speaker recognition (``server.speaker``, SPEC v1.3)."""

    enabled: bool = True
    #: Cosine-similarity threshold for a voice to match an enrolled profile.
    #: v1.7: on the ECAPA scale (equal error rate measured near 0.44); the old
    #: resemblyzer values do not transfer.
    threshold: float = Field(default=0.40, gt=0.0, le=1.0)
    #: v1.7: with two or more voices enrolled, the best match must lead the
    #: runner-up by this much, or the speaker is reported as unknown.
    margin: float = Field(default=0.15, ge=0.0, le=1.0)
    #: Higher bar for the most dangerous tools (run_command, set_role): the
    #: speaker must match this closely, not just the normal threshold, before
    #: those are allowed even to an admin profile. Guards against a lookalike
    #: voice slipping past the (lower) identification threshold.
    admin_threshold: float = Field(default=0.65, gt=0.0, le=1.0)
    #: Utterances shorter than this are not identified (too little voice).
    min_speech_s: float = Field(default=0.8, ge=0.0)


class FaceConfig(_Strict):
    """Face recognition and room presence (``server.face``, SPEC v1.4).

    The server matches every camera frame the client pushes against the
    ``face_embeddings`` of ``data/people.json`` and keeps a per-connection
    presence map that feeds the ``{presence}`` block of the system prompt.
    """

    enabled: bool = True
    #: Disable proactive speech while keeping detection/tracking/appearance active.
    greetings_enabled: bool = True
    #: Cosine-similarity threshold for a face to match an enrolled profile.
    threshold: float = Field(default=0.45, gt=0.0, le=1.0)
    #: Archive vetted portrait/body samples of enrolled people without expiry.
    appearance_enabled: bool = True
    #: Compare faces with curated auto-collected samples as well as enrollment.
    adaptive_recognition: bool = True
    #: Somebody is forgotten this long after the camera last saw them.
    presence_ttl_s: float = Field(default=30.0, gt=0.0)
    #: v1.7: greet with a fixed script instead of a model round. Going through
    #: the model cost 3-5 s (once 61 s) before a word was spoken, which is far
    #: too late for somebody standing in front of the camera. True restores it.
    greeting_llm: bool = False
    #: An unknown face present for this long triggers the proactive greeting.
    greet_after_s: float = Field(default=0.35, ge=0.0)
    #: v1.7: at most one greeting of the SAME STRANGER per this window
    #: (0 = no cooldown). Per person, not per room.
    greeting_cooldown_s: float = Field(default=300.0, ge=0.0)
    #: v1.7: the same, for somebody Rowan already knows by name. Longer,
    #: because a familiar face does not need introducing every few minutes.
    greeting_cooldown_known_s: float = Field(default=900.0, ge=0.0)
    #: Multi-frame bursts (v1.4 burst): frames pulled per camera_request when
    #: the server wants more than one shot (staged enroll_face's samples).
    burst_size: int = Field(default=3, ge=1, le=5)
    #: How many extra background bursts enroll_face pulls after its immediate
    #: first sample (spaced a few seconds apart while the person turns their
    #: head), on top of that first one.
    enroll_bursts: int = Field(default=3, ge=0)


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
    kokoro_model_path: str = 'models/kokoro/kokoro-v1.0.onnx'
    kokoro_voices_path: str = 'models/kokoro/voices-v1.0.bin'


class TrainingArchiveConfig(_Strict):
    """Permanent labeled source material for manual dataset preparation."""

    enabled: bool = False
    path: str = 'data/training_archive'
    min_free_gb: float = Field(default=5, ge=0)


class MediaConfig(_Strict):
    """Retention for room media stored under ``data/homes/<home_id>/media``.

    Frames and crops are short-lived privacy-sensitive captures; clips may be
    kept a little longer. Expired files are removed together with their row in
    the ``media`` table. Embeddings and presence events are deliberately not
    affected: those are controlled by the person's "forget me" request.
    """

    media_ttl_days: int = Field(default=3, ge=1)
    clip_ttl_days: int = Field(default=7, ge=1)


class VectorConfig(_Strict):
    """sqlite-vec index for stored embeddings (``server.vectors``, ТЗ 4.6).

    The metadata tables keep their float32 BLOB vectors; this section controls
    the search index mirrored into sqlite-vec virtual tables. ``extension_path``
    empty means auto-discovery (env var, the ``sqlite_vec`` package, then the
    copy vendored in ``hub/vendor``). ``dimensions`` overrides the default
    embedding size per kind and is normally left empty: the dimension is taken
    from the stored rows once they exist.
    """

    enabled: bool = True
    extension_path: str = ""
    dimensions: dict[str, int] = Field(default_factory=dict)

    @field_validator("dimensions")
    @classmethod
    def _known_kinds(cls, value: dict[str, int]) -> dict[str, int]:
        allowed = {"voice", "face", "body", "memory", "objects"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown vector kind(s) {unknown}; expected any of {sorted(allowed)}")
        for kind, dimension in value.items():
            if not 0 < dimension <= 8192:
                raise ValueError(f"vectors.dimensions.{kind} must be between 1 and 8192")
        return value


class OutboundConfig(_Strict):
    """Per-session send buffer (``server.outbound``, ТЗ 4.4 and 13).

    One slow room must not hold back the others: frames are written through a
    bounded queue per connection, and background frames (camera state, HUD
    captions, device state) are dropped before anything else once it fills.
    Replies and their PCM are never dropped.
    """

    queue_capacity: int = Field(default=32, ge=1, le=1024)


class GpuQueueConfig(_Strict):
    """The hub's single GPU queue (``server.gpu_queue``, ТЗ section 4.5).

    Every heavy job (STT, the LLM round, face embeddings, SAM3, the vision
    model) goes through one priority queue instead of hitting the card
    directly: classes are ordered utterance → face burst → background camera →
    nightly consolidation, and no single room may hold more than
    ``fair_share`` of the running slots while another room is waiting.

    The timeouts are safety nets, not UX limits: they only fire when a job
    waits behind a stuck worker for minutes, and hitting one is reported as a
    timeout the same way a missed STT deadline is.
    """

    #: A hub that cannot run the queue (or an operator debugging the pipeline)
    #: sets this to false; every call site then runs its job directly.
    enabled: bool = True
    #: How many GPU jobs may run at the same time across all rooms. The card
    #: itself serializes the big models, so this stays low on purpose.
    max_concurrent: int = Field(default=2, ge=1, le=16)
    fair_share: float = Field(default=0.5, gt=0.0, le=1.0)
    max_waiting: int = Field(default=256, ge=1, le=4096)
    utterance_timeout_s: float = Field(default=300.0, gt=1, le=3600)
    face_timeout_s: float = Field(default=120.0, gt=1, le=3600)
    background_timeout_s: float = Field(default=120.0, gt=1, le=3600)


class ModelLevelConfig(_Strict):
    """One model level (``models.levels.<name>``, ТЗ F-401).

    A level is an endpoint plus the settings a chat round needs. Levels are
    named after their role: ``local_fast`` and ``local_strong`` run on the
    hub's own GPU, ``cloud_cheap`` and ``cloud_strong`` are budgeted API
    levels used only when the owner allows them.
    """

    provider: Literal["vllm", "ollama_native", "openai", "openai_responses"] = "vllm"
    base_url: str = "http://127.0.0.1:8000/v1"
    #: Empty means "this level is not provisioned yet" - the router skips it.
    model: str = ""
    api_key: str = "vllm"
    #: Cloud levels keep their key in the environment, never in the config.
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    think: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """True when the level names a model, i.e. it can actually be used."""
        return bool(self.model.strip())


class ModelRoutingConfig(_Strict):
    """How one utterance picks a level (``models.routing``, ТЗ F-401/F-403)."""

    #: Utterances at most this long may use ``local_fast``.
    short_chars: int = Field(default=120, ge=1, le=2000)
    #: Utterances at least this long, or carrying a reasoning/code marker,
    #: need the strong level.
    strong_chars: int = Field(default=240, ge=1, le=8000)
    #: F-403: a class-0 job predicted to wait longer than this overflows.
    overflow_wait_s: float = Field(default=1.5, gt=0.0, le=60.0)
    overflow_level: str = "cloud_cheap"
    #: Off by default: the hub keeps everything local unless the owner opts
    #: into spending the API allowance on overflow.
    cloud_fallback: bool = False
    #: F-404: an image may go to a cloud vision model. Off by default.
    cloud_vision: bool = False


DEFAULT_LEVEL_NAMES = ("local_fast", "local_strong", "cloud_cheap", "cloud_strong")


class ModelsConfig(_Strict):
    """Model levels and routing (``models``, ТЗ section 9.1).

    Off by default: an existing single-model hub keeps using
    ``server.llm`` exactly as before until levels are provisioned and this
    section is switched on.
    """

    enabled: bool = False
    levels: dict[str, ModelLevelConfig] = Field(default_factory=dict)
    routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)

    @model_validator(mode="after")
    def _fill_and_check_levels(self) -> ModelsConfig:
        unknown = sorted(set(self.levels) - set(DEFAULT_LEVEL_NAMES))
        if unknown:
            raise ValueError(
                "unknown model level(s): " + ", ".join(unknown)
                + " (expected " + ", ".join(DEFAULT_LEVEL_NAMES) + ")"
            )
        for name in DEFAULT_LEVEL_NAMES:
            self.levels.setdefault(name, ModelLevelConfig())
        if self.enabled and self.routing.overflow_level not in self.levels:
            raise ValueError(f"routing.overflow_level is not a configured level: "
                             f"{self.routing.overflow_level!r}")
        if self.routing.overflow_level in {"local_fast", "local_strong"}:
            raise ValueError("routing.overflow_level must name a cloud level")
        return self


class QuietHoursConfig(_Strict):
    """Quiet hours of one room, ``HH:MM`` in the room's own time zone."""

    start: str = ""
    end: str = ""

    @field_validator("start", "end")
    @classmethod
    def _clock(cls, value: str) -> str:
        if value and not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value):
            raise ValueError("quiet hours must use HH:MM")
        return value

    @model_validator(mode="after")
    def _both_or_neither(self) -> QuietHoursConfig:
        if bool(self.start) != bool(self.end):
            raise ValueError("set both quiet-hour boundaries or leave both empty")
        if self.start and self.start == self.end:
            raise ValueError("quiet-hour start and end must differ")
        return self


class HomeConfig(_Strict):
    """One room (``home``) served by the hub (ТЗ section 4.2)."""

    home_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=80)
    tz: str = "America/Chicago"
    quiet_hours: QuietHoursConfig = Field(default_factory=QuietHoursConfig)
    owner_person_id: str = Field(default="", max_length=100)
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tz")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"tz must be a valid IANA time zone, got {value!r}") from None
        return value


class StageTimeouts(_Strict):
    """Per-stage budgets and the degradation rule (``server.timeouts``).

    The budgets are the ones the latency table (ТЗ 15.1) gives for the local
    models, plus a round budget for the model itself. Overrunning a budget must
    never leave the room in silence: the stage that ran late is dropped and the
    turn continues with what is available — a transcript without diarization,
    an answer without voice identity, a spoken apology instead of a hang. Every
    degradation is logged with the utterance id and counted in
    ``/health.utterances``.
    """

    #: False restores the pre-degradation behaviour: only the GPU-queue
    #: timeouts and ``server.diarization.timeout_s`` fire.
    enabled: bool = True
    #: VAD end → final transcript (without diarization).
    stt_ms: int = Field(default=700, ge=50, le=600000)
    #: The same deadline for the diarized path; overrunning it costs the
    #: speaker labels, not the transcript.
    diarization_ms: int = Field(default=700, ge=50, le=600000)
    #: Voice identification (the ReID step of a turn).
    speaker_ms: int = Field(default=700, ge=50, le=600000)
    #: The whole model round. This is a stuck-generation guard, not the
    #: end-to-end latency budget: it is deliberately far above 15.1's 1.2 s so
    #: a slow-but-working model is never cut off.
    reply_ms: int = Field(default=20000, ge=1000, le=600000)


class DeciderConfig(_Strict):
    """Which provider answers which decision, and how long it may think.

    ТЗ 5.2: the order of providers per decision type and the per-provider
    timeout are configuration, not code — swapping in the local model (or a
    later Jev provider) must not be an edit of ``hub/app.py``. A decision type
    missing from ``order`` keeps the built-in chain; a provider that is not
    available at runtime (the local model is switched off) is skipped by the
    chain itself, so a stale name is harmless.
    """

    #: 400 ms is the budget of ТЗ 15.1 for "transcript → generation starts".
    timeout_ms: int = Field(default=400, ge=50, le=10000)
    #: Decision type → providers, best first.
    order: dict[str, list[str]] = Field(default_factory=dict)
    #: ТЗ 5.4: the same question is answered once for this long (0 = no cache).
    cache_ttl_s: float = Field(default=60.0, ge=0.0, le=3600.0)
    cache_max_entries: int = Field(default=256, ge=1, le=10000)
    #: ТЗ 5.4: how far back the weekly calibration report in the panel looks.
    report_days: int = Field(default=7, ge=1, le=90)

    @field_validator("order")
    @classmethod
    def _known_providers(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        known = {"rules", "local_llm", "jev"}
        for decision_type, providers in value.items():
            if not providers:
                raise ValueError(f"decider.order.{decision_type} lists no provider")
            unknown = sorted(set(providers) - known)
            if unknown:
                raise ValueError(
                    f"decider.order.{decision_type} names unknown provider(s): "
                    + ", ".join(unknown) + " (expected " + ", ".join(sorted(known)) + ")"
                )
        return value


class SkillReloadConfig(_Strict):
    """Hot reload of skill files (ТЗ F-405).

    Reloading code while the hub serves rooms is a development convenience, not
    production behaviour, so it is off unless somebody asks for it: the shipped
    hub keeps running the skills it started with, and a room cannot be surprised
    by a file that was being edited.
    """

    dev_reload: bool = False
    interval_s: float = Field(default=2.0, ge=0.25, le=60.0)


class WebAdminConfig(_Strict):
    """The owner's web panel (ТЗ F-705).

    The panel is reachable from the overlay network only — a phone on the
    dorm's own Wi-Fi must not even see the login form. The password lives in
    the environment (ТЗ 15.4: secrets never in the config file or in git).
    """

    enabled: bool = False
    #: 127.0.0.1 by default: the overlay address belongs to the deployment.
    host: str = "127.0.0.1"
    port: int = Field(default=8099, ge=1, le=65535)
    #: Overlay ranges allowed to reach the panel (Tailscale CGNAT + loopback).
    allowed_networks: list[str] = Field(
        default_factory=lambda: ["127.0.0.0/8", "100.64.0.0/10"], max_length=8)
    password_env: str = Field(default="ROWAN_ADMIN_PASSWORD", min_length=4, max_length=60)
    session_minutes: int = Field(default=60, ge=5, le=1440)

    @field_validator("allowed_networks")
    @classmethod
    def _networks_are_cidrs(cls, value: list[str]) -> list[str]:
        import ipaddress

        for item in value:
            try:
                ipaddress.ip_network(str(item), strict=False)
            except ValueError as exc:
                raise ValueError(f"allowed_networks: {item!r} is not a network: {exc}") from exc
        return value


class StreamingReplyConfig(_Strict):
    """Streaming answers (``server.streaming_reply``, ТЗ F-101).

    The first sound of a reply has a budget of its own in the latency table
    (ТЗ 15.1): 1.2 s after the end of speech. The budget only lives in the
    config so a stand whose models are slower can be calibrated without
    touching the code; the measurement itself is always on.
    """

    enabled: bool = True
    #: End of speech -> the first audio frame on the wire (ТЗ 15.1).
    first_audio_budget_ms: int = Field(default=1200, ge=100, le=10000)
    #: Стартовать ход по промежуточному транскрипту, не дожидаясь финального
    #: (P2-41). Выключено по умолчанию: сначала должен быть замер.
    early_start: bool = False


class ServerConfig(_Strict):
    """Everything the brain PC reads (``server``)."""

    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1, le=65535)
    #: False temporarily grants every speaker access to all tools, including guests.
    #: Voice recognition still selects personal history; no stored roles are changed.
    permissions_enabled: bool = True
    audio_recording: RecordingConfig = Field(default_factory=RecordingConfig)
    camera_request_recording: RecordingConfig = Field(default_factory=RecordingConfig)
    training_archive: TrainingArchiveConfig = Field(default_factory=TrainingArchiveConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image_generation: ImageGenerationConfig = Field(default_factory=ImageGenerationConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    gpu_queue: GpuQueueConfig = Field(default_factory=GpuQueueConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    vectors: VectorConfig = Field(default_factory=VectorConfig)
    outbound: OutboundConfig = Field(default_factory=OutboundConfig)
    timeouts: StageTimeouts = Field(default_factory=StageTimeouts)
    decider: DeciderConfig = Field(default_factory=DeciderConfig)
    skills: SkillReloadConfig = Field(default_factory=SkillReloadConfig)
    streaming_reply: StreamingReplyConfig = Field(default_factory=StreamingReplyConfig)
    web_admin: WebAdminConfig = Field(default_factory=WebAdminConfig)
    #: Release tag the room clients should run (ТЗ 4.9 OTA). Empty = не трогать.
    client_release: str = Field(default="", max_length=60)


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------


def _default_client() -> ClientConfig:
    return ClientConfig(server_url="ws://127.0.0.1:8765/ws")


class Config(_Strict):
    """Root of ``config.yaml`` — both sections live in one file."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    client: ClientConfig = Field(default_factory=_default_client)
    #: Rooms served by this hub (ТЗ section 4.2). Empty keeps the classic
    #: single-room behaviour, so an existing config.yaml keeps working.
    homes: list[HomeConfig] = Field(default_factory=list)
    #: Model levels and the router that picks one (ТЗ section 9.1). Disabled
    #: by default: until then every round uses ``server.llm``.
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    @model_validator(mode="after")
    def _unique_homes(self) -> Config:
        seen: set[str] = set()
        for home in self.homes:
            if home.home_id in seen:
                raise ValueError(f"duplicate home_id: {home.home_id!r}")
            seen.add(home.home_id)
        return self

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


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
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
