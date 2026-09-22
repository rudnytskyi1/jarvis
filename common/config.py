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
    cfg.server.identity.reid.threshold  # 0.5 (same-day body match, F-203/F-206)
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

#: ТЗ F-113: the default list of dangerous calls the room has to confirm with a
#: spoken "yes". It lives here (not in the hub) because the config is validated
#: on both machines and the client ships without the hub.
DEFAULT_DANGEROUS_TOOLS: tuple[str, ...] = ("run_command",)
DEFAULT_DANGEROUS_PC_COMMANDS: tuple[str, ...] = (
    "sleep", "suspend", "hibernate", "shutdown", "power_off", "reboot",
    "restart", "logoff", "logout", "close_app",
)

__all__ = [
    "Config",
    "ServerConfig",
    "STTConfig",
    "DiarizationConfig",
    "LLMConfig",
    "SpeakerConfig",
    "FaceConfig",
    "IdentityConfig",
    "GuestConfig",
    "LearningConfig",
    "ReidConfig",
    "MIN_GUEST_FRAMES",
    "MAX_GUEST_FRAMES",
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


class HallucinationConfig(_Strict):
    """Фильтр галлюцинаций Whisper v2 (``server.stt.hallucination``, ТЗ F-105).

    D-03 already dropped an impossible speech rate. These are the two shapes
    that arrive with confident scores: a transcript made only of stop phrases
    ("продолжение следует", "thanks for watching") and a phrase the decoder got
    stuck repeating. Both rules need the WHOLE transcript, so a normal request
    that merely contains such a phrase survives.
    """

    stop_phrases: bool = True
    repetition: bool = True
    #: Room-specific phrases (a TV show's outro, a podcast intro, ...).
    extra_stop_phrases: list[str] = Field(default_factory=list, max_length=64)
    #: Above this many characters per second of audio the transcript cannot be
    #: speech at all (D-03). Fast speech reaches ~20 characters per second.
    max_chars_per_second: float = Field(default=60.0, gt=1.0, le=1000.0)


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
    hallucination: HallucinationConfig = Field(default_factory=HallucinationConfig)

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
    #: ТЗ F-108: more than this share of the utterance spoken by two voices at
    #: once and the turn is not executed - the room is asked to repeat one at a
    #: time. 0 switches the rule off (the measurement stays in the trace).
    overlap_limit: float = Field(default=0.40, ge=0.0, le=1.0)


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
    #: Extra accounts with the hub admin's own rights (see DECISIONS.md TG-01):
    #: they may open ``/tools``, control the rooms and be written to in private.
    #: The owner stays the only account that cannot be removed.
    admin_user_ids: list[int] = Field(default_factory=list)
    api_key_env: str = Field(default='TELEGRAM_BOT_TOKEN', min_length=1,
                             pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')
    timeout_s: float = Field(default=30, ge=5, le=120)
    respond_to_mentions: bool = False
    poll_timeout_s: int = Field(default=25, ge=1, le=50)

    @field_validator("admin_user_ids", mode="after")
    @classmethod
    def _check_admin_ids(cls, value: list[int]) -> list[int]:
        for user_id in value:
            if type(user_id) is not int or not 0 < user_id < 2 ** 63:
                raise ValueError("every admin_user_ids entry must be a positive Telegram user id")
        return sorted(set(value))


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


class GreetingConfig(_Strict):
    """Приветствия и прощания по событию входа (``server.greeting``, ТЗ F-302).

    Приветствие произносит сам хаб по событию `person_entered` (F-301), пока
    человек ещё стоит перед камерой, поэтому строки фиксированные и берутся из
    TTS-кэша. Число ТЗ — «не чаще одного раза в 20 минут на человека» — это
    ``cooldown_s``. Тихие часы — окно ``HH:MM``; пустые строки означают, что
    хаб говорит в любое время суток (прежнее поведение).
    """

    enabled: bool = True
    #: ТЗ F-302: приветствие одного человека не чаще раза в 20 минут.
    cooldown_s: float = Field(default=1200.0, ge=0.0, le=86400.0)
    #: Окно тишины в местном времени дома («23:00» … «08:00»); пусто = нет.
    quiet_start: str = ""
    quiet_end: str = ""
    #: Прощаться ли с человеком, когда он вышел из комнаты (событие F-301).
    farewell: bool = True

    @field_validator("quiet_start", "quiet_end")
    @classmethod
    def _clock(cls, value: str) -> str:
        if value and not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value):
            raise ValueError("quiet hours must use HH:MM")
        return value

    @model_validator(mode="after")
    def _both_or_neither(self) -> GreetingConfig:
        if bool(self.quiet_start) != bool(self.quiet_end):
            raise ValueError("set both quiet-hour boundaries or leave both empty")
        if self.quiet_start and self.quiet_start == self.quiet_end:
            raise ValueError("quiet-hour start and end must differ")
        return self


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

    ``cleanup_interval_s`` is the period of the F-304 scheduler task (the hub
    also expires media once while booting); ``0`` turns the periodic pass off
    and keeps only the startup one.
    """

    media_ttl_days: int = Field(default=3, ge=1)
    clip_ttl_days: int = Field(default=7, ge=1)
    cleanup_interval_s: int = Field(default=3600, ge=0)


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


class MemoryConfig(_Strict):
    """Memory retrieval (``server.memory``, ТЗ F-414, 9.4).

    Facts live in the ``memories`` table; the prompt carries the ``top_k`` most
    relevant facts for what is being said, found by BM25 and by the embedding
    model of ТЗ 9.4 running on the CPU. ``embedding_model`` is a path relative
    to the repo (an operator drops the model into ``models/``) or an HF id, and
    an id reaches the network only when ``allow_download`` says so: a hub whose
    model is missing refuses honestly and searches by words
    (``hub/embeddings.py`` says the same). ``vector_weight`` is the share of
    the vector half in the combined score, ``bm25_k1``/``bm25_b`` are the BM25
    parameters, and ``embed_on_remember`` decides whether a new fact is
    embedded as it is stored (off means vectors arrive with the nightly
    consolidation instead).
    """

    retrieval_enabled: bool = True
    top_k: int = Field(default=8, ge=1, le=64)
    embedding_model: str = "models/multilingual-e5-small"
    allow_download: bool = False
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_k1: float = Field(default=1.5, ge=0.0, le=10.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    embed_on_remember: bool = True
    #: ТЗ 9.4 (F-414, P3-17): read dialogues from ``dialog_turns`` instead of
    #: the archive store. Off by default, because a working hub is not swapped
    #: in the same step (ТЗ section 1); the flag turns the table into the source
    #: of history, and the archive keeps being written as its readable twin.
    dialogs_from_db: bool = False
    #: F-416: the nightly consolidation. It is a scheduler job that wakes every
    #: ``consolidation_check_interval_s`` and works on a home once that home's
    #: OWN clock has passed ``consolidation_hour``; a hub that was off at 04:00
    #: consolidates at the next check instead of skipping the night.
    consolidation_enabled: bool = True
    consolidation_hour: int = Field(default=4, ge=0, le=23)
    consolidation_minute: int = Field(default=0, ge=0, le=59)
    consolidation_check_interval_s: float = Field(default=900.0, gt=0, le=86400)
    #: How far back "the day" reaches for the digest and for the decay.
    consolidation_window_hours: float = Field(default=24.0, gt=0, le=168)
    #: ТЗ F-416: "день в 5–10 фактов". Outside this range the digest is refused.
    consolidation_min_facts: int = Field(default=5, ge=1, le=64)
    consolidation_max_facts: int = Field(default=10, ge=1, le=64)
    #: "Понижение веса старых": the share of weight a fact keeps per full day
    #: of age, and the floor it never falls below while it is still alive.
    decay_per_day: float = Field(default=0.9, gt=0.0, le=1.0)
    decay_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Two facts of one owner, kind and scope are the same fact when their
    #: normalized texts match or their embeddings are at least this close.
    duplicate_similarity: float = Field(default=0.93, ge=0.5, le=1.0)
    #: ТЗ F-416 compresses the day into the digest; a fact that went into it is
    #: removed (its content lives on in the digest). Off keeps the raw facts
    #: with their decayed weight - for an owner who wants the full trail.
    fold_sources: bool = True
    #: How many missing embeddings one pass computes at most (a CPU embedder
    #: needs ~10 ms per fact; the rest waits for the next night).
    embed_batch: int = Field(default=256, ge=0, le=4096)

    @model_validator(mode="after")
    def _digest_bounds(self) -> MemoryConfig:
        if self.consolidation_min_facts > self.consolidation_max_facts:
            raise ValueError(
                "consolidation_min_facts must not be larger than consolidation_max_facts")
        return self


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
    #: F-404: the level that looks at images (screenshots, camera frames). It is
    #: a level like any other, so its model, endpoint and budget have one home.
    vision_level: str = "local_vision"
    #: F-404: where a hard image goes when the home allows the cloud at all.
    vision_cloud_level: str = "cloud_strong"
    #: F-404: an image may go to a cloud vision model. Off by default, and the
    #: home has to allow it too (``homes[].cloud_vision``).
    cloud_vision: bool = False


DEFAULT_LEVEL_NAMES = ("local_fast", "local_strong", "local_vision",
                       "cloud_cheap", "cloud_strong")


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
        # ТЗ F-404: зрение — такой же уровень, и назван он должен быть по имени
        # из схемы, иначе «модель для картинок» снова разъедется с уровнями.
        if self.routing.vision_level not in self.levels:
            raise ValueError(f"routing.vision_level is not a configured level: "
                             f"{self.routing.vision_level!r}")
        if self.routing.vision_cloud_level not in self.levels:
            raise ValueError(f"routing.vision_cloud_level is not a configured level: "
                             f"{self.routing.vision_cloud_level!r}")
        if self.routing.vision_cloud_level not in {"cloud_cheap", "cloud_strong"}:
            raise ValueError("routing.vision_cloud_level must name a cloud level")
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
    #: ТЗ F-701: the Telegram account of this home's owner. Their private chat
    #: with the bot opens the panel for THEIR homes only; ``0`` means nobody
    #: yet, and the hub admin can grant a home from the panel at run time.
    telegram_user_id: int = Field(default=0, ge=0)
    #: ТЗ F-702: the minimum notification cooldown of this home. A rule that
    #: asks for a shorter one still waits this long; ``0`` means the rule's own
    #: number is used.
    alert_cooldown_s: int = Field(default=0, ge=0)
    #: ТЗ F-404: this home allows hard images to be looked at by a cloud vision
    #: model. Off by default: a picture of the room leaving the house is the
    #: owner's decision, not a default.
    cloud_vision: bool = False
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


class PresenceConfig(_Strict):
    """Состояние присутствия дома (``server.presence``, ТЗ F-301).

    ТЗ не называет числа: сколько секунд без кадров считать выходом человека
    из комнаты, и сколько событий отдавать вопросам. Оба числа — настройки, а
    не константы в коде, потому что зависят от комнаты и камеры.
    """

    enabled: bool = True
    #: Столько секунд без новых кадров — и человек считается вышедшим.
    absence_s: float = Field(default=30.0, ge=1.0, le=3600.0)
    #: Сколько последних событий дома читают вопросы F-301.
    history_limit: int = Field(default=1000, ge=10, le=100000)


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


class ConfirmationsConfig(_Strict):
    """Подтверждение опасных действий (``server.confirmations``, ТЗ F-113).

    The list is the point of the feature, so it lives in the config where a
    room can extend it: a whole tool (a shell command) and the ``pc_control``
    commands that can lose work. An action from either list is spoken back and
    waits for an oral "yes"; anything else cancels it, and the answer goes to
    the audit table either way.
    """

    enabled: bool = True
    #: How long the spoken "yes" has to arrive (ТЗ F-113: 8 s).
    window_s: float = Field(default=8.0, ge=1.0, le=60.0)
    tools: list[str] = Field(default_factory=lambda: list(DEFAULT_DANGEROUS_TOOLS))
    pc_commands: list[str] = Field(default_factory=lambda: list(DEFAULT_DANGEROUS_PC_COMMANDS))


class ReidConfig(_Strict):
    """ReID тела по кропам (``server.identity.reid``, ТЗ F-203/F-206).

    The hub turns every stored body crop into a 512-d OSNet vector and keeps it
    in ``body_embeddings`` for the day it was recorded. ``threshold`` is the
    cosine a same-day body match has to clear before a track is linked to a
    person (ТЗ F-206 says "cos ≥ порога" without a number; the default is the
    executor's, see ``DECISIONS.md`` P2-13). ``margin`` keeps a near-tie
    between two people from turning into a guess.
    """

    enabled: bool = True
    #: ``osnet_x1_0`` (default) or ``osnet_ain_x1_0`` - both of the ТЗ.
    model: str = Field(default="osnet_x1_0", min_length=1, max_length=64)
    #: Cosine threshold of a same-day body match (F-206).
    threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    #: How far the winner must beat the runner-up.
    match_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Optional path to already-downloaded weights (empty = torchreid's own).
    weights: str = Field(default="", max_length=512)


#: ТЗ F-210: "лицо, 5-10 кадров с разных ракурсов" - the frame counts of a
#: guest's face registration (``server.identity.guest``).
MIN_GUEST_FRAMES = 5
MAX_GUEST_FRAMES = 10


class GuestConfig(_Strict):
    """Регистрация гостя (``server.identity.guest``, ТЗ F-210).

    The numbers the flow of ``hub/guest_registration.py`` runs on: how many
    camera frames of the guest may be stored (F-210 says 5-10, from different
    angles), how much clean speech the phrase has to last, and how long a
    half-finished registration and the owner's Telegram question stay open.
    """

    enabled: bool = True
    #: F-210: "5-10 кадров с разных ракурсов".
    min_frames: int = Field(default=MIN_GUEST_FRAMES, ge=1, le=10)
    max_frames: int = Field(default=MAX_GUEST_FRAMES, ge=1, le=10)
    #: "просит произнести фразу": this much clean speech makes a usable vector.
    min_voice_seconds: float = Field(default=3.0, ge=0.5, le=30.0)
    #: The whole flow, and the owner's confirmation window, in seconds.
    flow_ttl_s: float = Field(default=300.0, ge=30.0, le=3600.0)
    confirm_ttl_s: float = Field(default=900.0, ge=60.0, le=86400.0)

    @model_validator(mode="after")
    def _frames_in_order(self) -> GuestConfig:
        if self.max_frames < self.min_frames:
            raise ValueError("guest.max_frames must be at least guest.min_frames")
        return self


class LearningConfig(_Strict):
    """Адаптивное дообучение профиля (``server.identity.learning``, ТЗ F-211).

    ТЗ F-211: до 12 векторов лица, до 8 голоса, тело — по дням; новые векторы
    принимаются только при ``p ≥ 0,9`` и без конфликта с другими людьми. Здесь
    живут эти числа: ``conflict_similarity`` — насколько вектор должен быть
    похож на другого человека, чтобы его вообще нельзя было учить, а
    ``duplicate_similarity`` — насколько он должен быть похож на свой же, чтобы
    не считаться новым ракурсом.
    """

    enabled: bool = True
    #: ТЗ F-211: сильнее порога «узнан» (0,8) из F-207 - в профиль навсегда
    #: попадает только то, в чём хаб уверен сильнее.
    min_p: float = Field(default=0.9, gt=0.0, le=1.0)
    face_max_vectors: int = Field(default=12, ge=1, le=64)
    voice_max_vectors: int = Field(default=8, ge=1, le=64)
    #: Сколько векторов тела одного человека хранится за ОДИН день.
    body_per_day: int = Field(default=4, ge=1, le=64)
    conflict_similarity: float = Field(default=0.6, gt=0.0, le=1.0)
    duplicate_similarity: float = Field(default=0.98, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _many_not_conflicting(self) -> LearningConfig:
        if self.conflict_similarity >= self.duplicate_similarity:
            raise ValueError("learning.conflict_similarity must be below "
                             "learning.duplicate_similarity")
        return self


class AntiSpoofingConfig(_Strict):
    """Anti-spoofing (``server.identity.anti_spoofing``, ТЗ F-214).

    Две половины ТЗ живут здесь. Лицо: на бёрсте кадров хаб ищет муар экрана,
    отсутствие микродвижений внутри лица и движение, которое целиком
    объясняется одной плоскостью (``face`` и пороги признаков); нейросетевая
    liveness-модель ТЗ подключается через ``model``, и если она нужна
    (``require_model``), то её отсутствие — отказ, а не «наверное, живой».
    Голос: ``challenge`` решает, когда привилегированное действие ждёт
    случайное слово, ``challenge_window_s`` — сколько хаб его ждёт, а
    ``challenge_voice_threshold`` — насколько произнесённое должно быть
    похоже на голос САМОГО человека (ECAPA).
    """

    #: Ловля фотографии и экрана перед камерой.
    face: bool = True
    #: Путь к весам anti-spoof модели; пусто = модели нет (см. ``require_model``).
    model: str = ""
    #: True — без модели ни один бёрст не считается живым (по умолчанию False:
    #: в сборке модели нет, и признаки ТЗ работают сами).
    require_model: bool = False
    #: Сколько последних кадров трека составляют бёрст, и сколько нужно, чтобы
    #: вообще судить о живости.
    window_frames: int = Field(default=8, ge=2, le=60)
    min_frames: int = Field(default=5, ge=2, le=30)
    #: Пороги признаков: решётка экрана, «застывшее» лицо и «плоское» движение.
    moire_threshold: float = Field(default=0.35, gt=0.0, le=1.0)
    motion_min: float = Field(default=0.004, ge=0.0, le=0.5)
    planar_residual_max: float = Field(default=0.02, gt=0.0, le=1.0)
    planar_motion_min: float = Field(default=0.02, ge=0.0, le=1.0)
    #: "off" — поведение фазы 2 (F-208 решает один); "on_missing_witness" —
    #: спросить слово, когда свидетелей F-208 нет; "always" — спрашивать всегда.
    challenge: Literal["off", "on_missing_witness", "always"] = "on_missing_witness"
    challenge_window_s: float = Field(default=20.0, ge=5.0, le=120.0)
    challenge_voice_threshold: float = Field(default=0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _enough_frames(self) -> AntiSpoofingConfig:
        if self.min_frames > self.window_frames:
            raise ValueError("anti_spoofing.window_frames must be at least "
                             "anti_spoofing.min_frames")
        return self


class IdentityConfig(_Strict):
    """Идентичность человека: голос + лицо + тело (``server.identity``, ТЗ 7).

    The section owns the thresholds of the fusion (F-206-F-208) and the body
    appearance signal F-203 produces. It is separate from ``server.face``
    because the face engine has its own lifecycle (insightface) while the
    identity rules are what turns several noisy signals into one person.
    """

    enabled: bool = True
    reid: ReidConfig = Field(default_factory=ReidConfig)
    guest: GuestConfig = Field(default_factory=GuestConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    anti_spoofing: AntiSpoofingConfig = Field(default_factory=AntiSpoofingConfig)
    #: ТЗ F-208: a privileged action needs this confident a voice...
    admin_voice_threshold: float = Field(default=0.65, gt=0.0, le=1.0)
    #: ...plus a face at least this confident, or the body of the same day.
    admin_face_threshold: float = Field(default=0.55, gt=0.0, le=1.0)
    #: A phone has no camera: its own voice bar and the spoken PIN instead.
    phone_admin_threshold: float = Field(default=0.65, gt=0.0, le=1.0)
    #: False lets a phone ask for admin actions without a PIN (ТЗ wants it on).
    phone_pin_required: bool = True
    #: How long the room waits for the spoken PIN (per question).
    pin_window_s: float = Field(default=20.0, ge=5.0, le=120.0)
    #: Wrong tries before the person is locked out, and for how long.
    pin_max_failures: int = Field(default=3, ge=1, le=10)
    pin_lockout_s: float = Field(default=300.0, ge=0.0, le=3600.0)
    #: ТЗ F-209: how long the "appearance of the day" of a person is kept.
    appearance_retention_days: int = Field(default=7, ge=1, le=90)


class FollowupConfig(_Strict):
    """Follow-up window addressing (``server.followup``, ТЗ F-103).

    The window itself is the client's: after a reply it keeps the microphone
    open for ``client.followup_window_s`` seconds without the wake word. What
    the hub owns is the question the ТЗ asks next - was that speech addressed
    to Rowan? D-02 answers it from the wake word (stage 2: from the decider
    chain when one is configured) and D-11 says whether the turn continues the
    open dialogue.

    ``gate_unaddressed`` is what turns that answer into behaviour: a client
    that declares its window (``utterance_start.followup``) gets its
    out-of-window speech without a wake word *ignored* instead of answered.
    Off by default, because a client that predates the window must keep the
    behaviour it has always had.
    """

    gate_unaddressed: bool = False


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
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    gpu_queue: GpuQueueConfig = Field(default_factory=GpuQueueConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    vectors: VectorConfig = Field(default_factory=VectorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    outbound: OutboundConfig = Field(default_factory=OutboundConfig)
    timeouts: StageTimeouts = Field(default_factory=StageTimeouts)
    decider: DeciderConfig = Field(default_factory=DeciderConfig)
    skills: SkillReloadConfig = Field(default_factory=SkillReloadConfig)
    streaming_reply: StreamingReplyConfig = Field(default_factory=StreamingReplyConfig)
    followup: FollowupConfig = Field(default_factory=FollowupConfig)
    confirmations: ConfirmationsConfig = Field(default_factory=ConfirmationsConfig)
    presence: PresenceConfig = Field(default_factory=PresenceConfig)
    greeting: GreetingConfig = Field(default_factory=GreetingConfig)
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
