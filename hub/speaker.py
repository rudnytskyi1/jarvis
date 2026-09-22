"""The people registry: voices, faces and role-based permissions (SPEC v1.3/v1.4).

Every utterance is embedded with SpeechBrain's ECAPA-TDNN (192-d, v1.7) and
compared by cosine similarity against the profiles enrolled in
``data/people.json``. Roles: ``admin`` > ``trusted`` > ``user``; a voice that
matches nobody is ``unknown``. Permissions are enforced here, server-side — the
prompt only explains refusals.

v1.7 — why not resemblyzer any more: measured on the room's own webcam mic, two
different people (Anton against Drew) scored 0.659 on average and the same
person against himself 0.666. The distributions sat on top of each other, so no
threshold could ever tell them apart, and the configured 0.62 accepted almost
anybody. On the same degraded speech ECAPA separates voices twice as widely
(same 0.67 / different 0.17, equal error rate 2.0% against 4.1%). Embeddings
from the two models are not comparable, so the file records which model made
its voice vectors and stale ones are dropped on load.

v1.4: one file holds both modalities —
``{"people": {name: {"role": str, "voice_embeddings": [[…]],
"face_embeddings": [[…]]}}}``. This module owns the file and the roles; the
512-d face vectors themselves are produced and matched by ``server/face.py``.
A pre-v1.4 ``data/voices.json`` (whose per-person key was ``embeddings``) is
migrated to ``data/people.json`` transparently on first load.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("jarvis.server.speaker")

ROLE_ADMIN = "admin"
ROLE_TRUSTED = "trusted"
ROLE_USER = "user"
#: ТЗ F-210: an owner-registered guest of one room. Below ``user`` in the
#: ordering of section 14 (``admin > trusted > user > guest``), and the role a
#: confirmed guest registration of ``hub/guest_registration.py`` writes.
ROLE_GUEST = "guest"
ROLE_UNKNOWN = "unknown"
ROLES = (ROLE_ADMIN, ROLE_TRUSTED, ROLE_USER, ROLE_GUEST)
#: Rank used to pick the winning role when ``rename_person`` merges two
#: profiles (v1.6) - a merge must never demote whichever identity was admin.
_ROLE_RANK = {ROLE_GUEST: -1, ROLE_USER: 0, ROLE_TRUSTED: 1, ROLE_ADMIN: 2}


def _higher_role(a: str, b: str) -> str:
    """The more privileged of two roles (ties keep ``a``)."""
    return a if _ROLE_RANK.get(a, 0) >= _ROLE_RANK.get(b, 0) else b

DEFAULT_DATA_DIR = Path("data")
#: v1.4: one registry file for voices, faces and roles.
PEOPLE_FILENAME = "people.json"
#: Pre-v1.4 file, migrated into :data:`PEOPLE_FILENAME` on first load.
LEGACY_VOICES_FILENAME = "voices.json"
#: Kept as the old name of the legacy file.
VOICES_FILENAME = LEGACY_VOICES_FILENAME

#: Per-person keys inside the registry file.
VOICE_KEY = "voice_embeddings"
FACE_KEY = "face_embeddings"
#: What :data:`VOICE_KEY` was called before v1.4.
LEGACY_VOICE_KEY = "embeddings"

#: Each guided recording session collects six separate samples.
ENROLL_EXTRA_SAMPLES = 5
#: v1.6: total accepted samples (first + extra) an enrollment needs at minimum.
ENROLL_MIN_SAMPLES = 1 + ENROLL_EXTRA_SAMPLES
#: v1.6: total voiced seconds (summed across every accepted sample) an
#: enrollment needs at minimum, alongside :data:`ENROLL_MIN_SAMPLES`. A handful
#: of one-word samples used to "complete" an enrollment on a profile too thin
#: to actually recognize the person later.
MIN_ENROLL_SPEECH_S = 20.0
#: Separate caps keep voice additions from changing face-gallery behavior.
#: ТЗ F-211 states the numbers: "как сейчас для лиц (до 12 векторов)" and
#: "расширить на голос (до 8 векторов)". The phase-1 defaults (10 and 30) were
#: the executor's; the profile of the hub database is capped by the same numbers
#: in ``hub/adaptive_learning.py``, so the two stores cannot disagree.
MAX_SAMPLES_PER_PERSON = 12
MAX_VOICE_SAMPLES_PER_PERSON = 8

#: v1.7: identifies the model that produced the stored voice vectors. Written
#: at the top of people.json; vectors from any other model are meaningless to
#: this one and are dropped on load (faces and roles are kept).
VOICE_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
#: What a file written before v1.7 (no voice_model key) was embedded with.
LEGACY_VOICE_MODEL_ID = "resemblyzer"
#: Where the ECAPA checkpoint is cached, relative to the repo root.
VOICE_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "spkrec-ecapa-voxceleb"
#: v1.7: defaults calibrated for ECAPA on noisy room speech (equal error rate
#: point measured at 0.44). The old resemblyzer defaults (0.72 / 0.70) mean
#: nothing on this model's scale.
#: v1.7.1: calibrated on the room's REAL webcam mic, not clean synthetic TTS.
#: The synthetic benchmark put same-speaker at 0.67, but measured on Anton's
#: own recording two 1-second windows of the SAME utterance scored 0.19 - real
#: mic speech of one person, cross-utterance, sits around 0.2-0.45. A 0.40 bar
#: therefore rejected the owner's own voice every time; 0.28 clears real
#: same-speaker while a different person (0.0-0.2) still falls short. Re-tune
#: from the per-utterance scores now printed in the log once profiles are built.
DEFAULT_THRESHOLD = 0.40
#: With two or more voices enrolled the winner must also lead the runner-up by
#: this much - the RELATIVE gap is far more reliable than the absolute score.
DEFAULT_MARGIN = 0.15
#: An enrollment sample is refused only when it sounds clearly MORE like an
#: already-enrolled OTHER person than like the enrollee - a relative test, not
#: an absolute floor. The old absolute floor (0.35) sat above real same-speaker
#: similarity and deadlocked enrollment: the owner's own voice was rejected as
#: "somebody else" against his single short sample, so he could never add more.
ENROLL_REJECT_MARGIN = 0.10
#: A sample scoring below this against the enrollee's own samples is treated as
#: a DIFFERENT voice (or non-speech): the last absolute bar, set below real
#: same-speaker full-utterance similarity but above typical different-speaker,
#: so it catches somebody else speaking during enrollment without deadlocking
#: on the enrollee's own natural variation.
ENROLL_MIN_SELF = 0.12
#: v1.7: enrollment audio is kept so a future, better model can be calibrated
#: on the room's real voices instead of on guesses.
VOICE_AUDIO_DIRNAME = "voices"


class VoiceMismatch(ValueError):
    """An enrollment sample that does not sound like the person being enrolled."""


class DuplicateVoice(ValueError):
    """A new profile would duplicate a strongly matching existing voice."""

    def __init__(self, person: str):
        self.person = person
        super().__init__(f"This voice is already very similar to {person}")


class EnrollmentConfirmationRequired(ValueError):
    """Recorded samples need a physical confirmation before changing a profile."""

#: v1.6: names nobody should actually be enrolled under - the model must ask
#: for the real name instead (enroll_voice and rename_person both reject them).
PLACEHOLDER_NAMES = frozenset({"guest", "user", "friend", "unknown", "none", "null", "speaker", "me", "you", "someone", "неизвестный", "гость"})

_INT16_SCALE = 32768.0
#: Mic wire format (SPEC §4): 16 kHz mono s16le, 2 bytes per sample.
_PCM_SECOND_BYTES = 16000 * 2

# ---------------------------------------------------------------------------
# permissions
# ---------------------------------------------------------------------------

#: pc_control commands anyone may use (shared-room basics).
SAFE_PC_COMMANDS = frozenset(
    {
        "volume_set",
        "volume_up",
        "volume_down",
        "mute",
        "unmute",
        "media_play_pause",
        "media_next",
        "media_prev",
        "display_off",
        "display_on",
    }
)

#: Tools anyone may use, including unknown voices. ``enroll_face`` is here for
#: the same reason as ``enroll_voice``: a guest may always introduce themselves.
_EVERYONE_TOOLS = frozenset({"set_light", "set_switch", "enroll_voice", "enroll_face", "look_at_camera"})
#: Tools that need admin or trusted. Looking through the room camera is as
#: sensitive as looking at the screen, so it sits in the same tier.
#: ``find_object`` (v1.5) pulls the same kind of frame as ``look_at_camera``/
#: ``look_at_screen``, so it joins them here.
_TRUSTED_TOOLS = frozenset(
    {
        "browser_control",
        "click_screen",
        "look_at_screen",
        "look_at_camera",
        "remember",
        "find_object",
        "show_photo",
        "save_photo",
        "generate_image",
        "telegram_send",
        "set_wallpaper",
        # Who holds admin is security-relevant: a stranger must not be able to
        # enumerate the room's people and find out whom to imitate.
        "list_people",
        "forget_fact",
    }
)
#: Tools that need admin.
_ADMIN_TOOLS = frozenset({"run_command", "set_role"})
#: ``rename_person`` (v1.6) is NOT in any tier above: its own admin-or-self
#: rule lives directly in :func:`check_permission` because, unlike every other
#: tool, it depends on WHO is speaking, not just their role.


#: Tools dangerous enough to demand a higher voice-match confidence, not just
#: an admin role — a lookalike voice must not run a command or change roles.
HIGH_CONFIDENCE_TOOLS = frozenset({"run_command", "set_role", "rename_person"})


def check_permission(
    role: str,
    tool: str,
    args: dict[str, Any] | None,
    speaker_name: str | None = None,
    speaker_score: float | None = None,
    admin_threshold: float | None = None,
    permissions_enabled: bool = True,
) -> str | None:
    """Return None when allowed, or the denial message for the LLM.

    ``speaker_name`` (v1.6) is only used by ``rename_person``: unlike every
    other tool, its permission depends on WHO is talking, not just their role
    - the speaker may always rename their own profile, whatever their role.

    ``speaker_score``/``admin_threshold`` (v1.6): the most dangerous tools
    (:data:`HIGH_CONFIDENCE_TOOLS`) additionally require the voice match to be
    at least ``admin_threshold`` confident, so a lookalike voice that merely
    cleared the (lower) identification bar cannot run commands.
    """
    if not permissions_enabled:
        return None
    role = role if role in ROLES else ROLE_UNKNOWN

    args = args or {}
    global_memory = tool == 'remember' and (args.get('scope') == 'global' or str(args.get('about', '')).casefold() in {'room', 'everyone', 'all', 'everybody', 'general'})
    if (
        (tool in HIGH_CONFIDENCE_TOOLS or global_memory)
        and role == ROLE_ADMIN
        and admin_threshold is not None
        and speaker_score is not None
        and speaker_score < float(admin_threshold)
    ):
        return (
            f"permission denied: {tool} needs a confident voice match "
            f"(at least {float(admin_threshold):.2f}), but this voice matched "
            f"only {float(speaker_score):.2f} - ask the person to say it again "
            "clearly, or from closer to the microphone. Also say: If I often fail to recognize "
            "your voice, say Rowan, update my voice, to add more voice samples."
        )

    def deny(needed: str) -> str:
        message = (
            f"permission denied: {tool} requires {needed}, but the current "
            f"speaker's role is {role} - politely refuse and suggest asking "
            "an authorized person"
        )
        if role == ROLE_UNKNOWN:
            message += ('. If you have a saved profile, repeat clearly closer to the microphone. '
                        'If I often fail to recognize your voice, say Rowan, update my voice, to add more samples.')
        return message

    if tool in _EVERYONE_TOOLS:
        return None
    if tool == 'find_object' and str(args.get('source') or 'camera').lower() == 'camera':
        return None
    if tool == 'remember':
        if global_memory:
            return None if role == ROLE_ADMIN else deny('admin for shared memory')
        current = str(speaker_name or '').strip()
        about = str(args.get('about') or 'me').strip().casefold()
        if current and current.casefold() != ROLE_UNKNOWN and about in {'me', 'myself', 'speaker', 'user', current.casefold()}:
            return None
        return 'permission denied: personal memory belongs only to the recognized current speaker; you cannot write another person\'s memory'
    if tool == "rename_person":
        if role == ROLE_ADMIN:
            return None
        old_name = " ".join(str((args or {}).get("old_name") or "").split()).lower()
        current = " ".join(str(speaker_name or "").split()).lower()
        if old_name and current and old_name == current:
            return None
        return (
            f"permission denied: rename_person requires admin, or the speaker "
            f"being the person renamed, but the current speaker's role is "
            f"{role} - politely refuse and suggest asking an authorized person "
            "or the person themselves"
        )
    if tool in _ADMIN_TOOLS:
        return None if role == ROLE_ADMIN else deny("admin")
    if tool in _TRUSTED_TOOLS:
        return None if role in (ROLE_ADMIN, ROLE_TRUSTED) else deny("admin or trusted")
    if tool == "pc_control":
        command = str((args or {}).get("command") or "").strip().lower()
        if command in SAFE_PC_COMMANDS:
            return None
        if role in (ROLE_ADMIN, ROLE_TRUSTED):
            return None
        return deny("admin or trusted")
    # Unknown tools fall through to the executor's own error handling.
    return None


# ---------------------------------------------------------------------------
# people registry
# ---------------------------------------------------------------------------


def pcm_to_float32(pcm_s16le: bytes) -> np.ndarray:
    usable = len(pcm_s16le) - (len(pcm_s16le) % 2)
    samples = np.frombuffer(memoryview(pcm_s16le)[:usable], dtype="<i2")
    return samples.astype(np.float32) / _INT16_SCALE


def is_placeholder_name(name: Any) -> bool:
    """True when ``name`` is a reserved placeholder (v1.6, e.g. "Guest").

    ``enroll_voice`` and ``rename_person`` both reject these - the model must
    ask for the person's real name instead of enrolling them under one.
    """
    cleaned = " ".join(str(name or "").split()).strip().lower()
    return cleaned in PLACEHOLDER_NAMES


def estimate_speech_seconds(pcm_s16le: bytes | None) -> float:
    """Rough voiced-seconds estimate for one accepted sample (v1.6).

    The mic wire format is fixed (16 kHz mono s16le, SPEC §4), so this is a
    plain byte-count conversion - not a VAD measurement, just the same
    estimate :data:`MIN_ENROLL_SPEECH_S` is defined against.
    """
    if not pcm_s16le:
        return 0.0
    return max(0.0, len(pcm_s16le) / float(_PCM_SECOND_BYTES))


def enrollment_complete(samples: int, total_speech_s: float) -> bool:
    """True once an in-progress enrollment has both enough samples and speech.

    v1.6: a handful of very short "yes"/"okay" samples used to complete
    enrollment in 3 turns while producing a profile too thin to recognize the
    person later, so BOTH :data:`ENROLL_MIN_SAMPLES` and
    :data:`MIN_ENROLL_SPEECH_S` must be met.
    """
    return samples >= ENROLL_MIN_SAMPLES and total_speech_s >= MIN_ENROLL_SPEECH_S


def _vector_list(raw: Any) -> list[list[float]]:
    """Keep only the well-formed embedding rows of a stored profile."""
    vectors: list[list[float]] = []
    if not isinstance(raw, list):
        return vectors
    for row in raw:
        if not isinstance(row, (list, tuple)):
            continue
        try:
            vectors.append([float(value) for value in row])
        except (TypeError, ValueError):
            log.warning("Dropping an embedding row that is not numeric")
    return vectors


def normalize_people(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Bring a loaded registry into the v1.4 shape.

    Accepts both the v1.4 layout and the v1.3 one (per-person key
    ``embeddings``), so a ``data/voices.json`` written before the rename loads
    as voice profiles instead of being silently dropped.
    """
    people: dict[str, dict[str, Any]] = {}
    for name, person in (raw or {}).items():
        if not isinstance(person, dict):
            log.warning("Skipping a malformed registry entry for %r", name)
            continue
        voices = person.get(VOICE_KEY)
        if not isinstance(voices, list):
            voices = person.get(LEGACY_VOICE_KEY)
        people[str(name)] = {
            "role": str(person.get("role") or ROLE_USER),
            VOICE_KEY: _vector_list(voices),
            FACE_KEY: _vector_list(person.get(FACE_KEY)),
            #: ТЗ F-106: what language this person wants to be answered in.
            "preferred_language": person.get("preferred_language"),
        }
    return people


class VoiceRegistry:
    """Enrolled people: roles, voice profiles and face profiles.

    Blocking (the encoder is a small torch model on CPU, and every change is
    written to disk): call the public methods through ``asyncio.to_thread``.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_speech_s: float = 0.8,
        enabled: bool = True,
        margin: float = DEFAULT_MARGIN,
        save_audio: bool = True,
    ) -> None:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.path = base / PEOPLE_FILENAME
        #: Pre-v1.4 file; migrated into :attr:`path` when that one is absent.
        self.legacy_path = base / LEGACY_VOICES_FILENAME
        self.audio_dir = base / VOICE_AUDIO_DIRNAME
        self.threshold = float(threshold)
        #: v1.7: with two or more people enrolled, the best match must beat the
        #: runner-up by at least this much - otherwise the voice is ambiguous and
        #: nobody is named, rather than a coin toss deciding who holds admin.
        self.margin = max(0.0, float(margin))
        self.min_speech_s = float(min_speech_s)
        self.enabled = bool(enabled)
        self.save_audio = bool(save_audio)
        self._lock = threading.Lock()
        self._encoder: Any = None
        self._people: dict[str, dict[str, Any]] = {}
        self._load()
        log.info(
            "People registry: %s (%d profile(s), voice model %s, threshold %.2f, margin %.2f)",
            self.path,
            len(self._people),
            VOICE_MODEL_ID,
            self.threshold,
            self.margin,
        )

    # -- storage ---------------------------------------------------------

    def _load(self) -> None:
        """Read the registry, migrating a pre-v1.4 ``voices.json`` if needed."""
        source = self.path
        migrating = False
        if not self.path.is_file() and self.legacy_path.is_file():
            source = self.legacy_path
            migrating = True
        stale_voices = False
        try:
            if source.is_file():
                data = json.loads(source.read_text(encoding="utf-8"))
                people = data.get("people") if isinstance(data, dict) else None
                if isinstance(people, dict):
                    self._people = normalize_people(people)
                model = (
                    str(data.get("voice_model") or LEGACY_VOICE_MODEL_ID)
                    if isinstance(data, dict)
                    else LEGACY_VOICE_MODEL_ID
                )
                if model != VOICE_MODEL_ID:
                    stale_voices = self._drop_voice_vectors(model)
        except Exception:
            log.exception("Could not read %s - starting empty", source)
            self._people = {}
            return
        if stale_voices and not migrating:
            try:
                with self._lock:
                    self._save_locked()
            except Exception:
                log.exception("Could not rewrite %s without the stale voices", self.path)
        if not migrating:
            return
        try:
            with self._lock:
                self._save_locked()
        except Exception:
            log.exception("Could not write the migrated registry %s", self.path)
            return
        log.info(
            "Migrated %s -> %s (%d person(s); %r is now %r)",
            self.legacy_path,
            self.path,
            len(self._people),
            LEGACY_VOICE_KEY,
            VOICE_KEY,
        )

    def _drop_voice_vectors(self, model: str) -> bool:
        """Forget voice vectors made by ``model``; faces and roles survive.

        Returns True when anything was actually dropped.
        """
        dropped = {
            name: len(person.get(VOICE_KEY) or [])
            for name, person in self._people.items()
            if person.get(VOICE_KEY)
        }
        for person in self._people.values():
            person[VOICE_KEY] = []
        if dropped:
            log.warning(
                "Voice samples in %s were made by %s, which %s cannot compare "
                "against - dropped them, so these people must enroll their voice "
                "again (faces and roles are kept): %s",
                self.path,
                model,
                VOICE_MODEL_ID,
                ", ".join(f"{name} ({count})" for name, count in sorted(dropped.items())),
            )
        return bool(dropped)

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"voice_model": VOICE_MODEL_ID, "people": self._people}, ensure_ascii=False
        )
        # Write-then-replace: a crash mid-write must not leave half a registry.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)

    def _person_locked(self, name: str) -> dict[str, Any]:
        """Return the person's record, creating it on first enrollment.

        The very first person enrolled in an empty registry becomes admin (the
        owner); everybody after that starts as user.
        """
        person = self._people.get(name)
        if person is None:
            role = ROLE_ADMIN if not self._people else ROLE_USER
            person = {"role": role, VOICE_KEY: [], FACE_KEY: []}
            self._people[name] = person
        person.setdefault(VOICE_KEY, [])
        person.setdefault(FACE_KEY, [])
        return person

    # -- encoder ---------------------------------------------------------

    def _get_encoder(self) -> Any:
        """Load ECAPA once, on the CPU.

        CPU on purpose: the 5090 is already full (chat model, Whisper, and
        other programs of the owner's), and a 20 M-parameter encoder embeds
        three seconds of speech in well under 100 ms without it.
        """
        if self._encoder is None:
            import torch  # noqa: PLC0415 - heavy, kept lazy
            from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415
            from speechbrain.utils.fetching import LocalStrategy  # noqa: PLC0415

            torch.set_grad_enabled(False)
            self._encoder = EncoderClassifier.from_hparams(
                source=VOICE_MODEL_ID,
                savedir=str(VOICE_MODEL_DIR),
                run_opts={"device": "cpu"},
                # Windows refuses symlinks without admin rights or developer mode.
                local_strategy=LocalStrategy.COPY,
            )
            log.info("ECAPA voice encoder loaded (%s)", VOICE_MODEL_ID)
        return self._encoder

    def _embed(self, pcm_s16le: bytes, sample_rate: int) -> np.ndarray | None:
        wav = pcm_to_float32(pcm_s16le)
        if sample_rate != 16000 and sample_rate > 0 and wav.size:
            duration = wav.size / float(sample_rate)
            positions = np.linspace(0.0, wav.size - 1, max(1, int(duration * 16000)))
            wav = np.interp(positions, np.arange(wav.size), wav).astype(np.float32)
        if wav.size / 16000.0 < self.min_speech_s:
            return None
        import torch  # noqa: PLC0415 - already imported by _get_encoder

        encoder = self._get_encoder()
        with torch.no_grad():
            embedding = encoder.encode_batch(torch.from_numpy(wav)[None])
        vector = embedding.squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None

    def warm_up(self) -> None:
        """Load the encoder now, so the first utterance does not wait ~4 s for it."""
        if not self.enabled:
            return
        try:
            self._get_encoder()
        except Exception:  # noqa: BLE001 - identify() retries and logs properly
            log.exception("Could not pre-load the voice encoder")

    @staticmethod
    def _centroid(vectors: list[list[float]]) -> np.ndarray | None:
        """The unit-length mean of a person's voice samples.

        Scoring against the centre of a profile rather than its single closest
        sample is what keeps one bad sample from deciding everything: under the
        old max-over-samples rule a single stray sentence filed under the wrong
        name made its real owner match that profile forever.
        """
        if not vectors:
            return None
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mean = (matrix / norms).mean(axis=0)
        length = float(np.linalg.norm(mean))
        return mean / length if length > 0 else None

    def _scores_locked(self, embedding: np.ndarray) -> list[tuple[float, str]]:
        """``(score, name)`` for every person with a voice profile, best first."""
        scores: list[tuple[float, str]] = []
        for name, person in self._people.items():
            if is_placeholder_name(name):
                continue
            centre = self._centroid(person.get(VOICE_KEY) or [])
            if centre is None or centre.shape != embedding.shape:
                continue
            scores.append((float(np.dot(centre, embedding)), name))
        scores.sort(reverse=True)
        return scores

    def _save_audio(self, name: str, pcm_s16le: bytes, sample_rate: int) -> None:
        """Keep an enrollment clip as a WAV next to the registry (best-effort)."""
        if not self.save_audio or not pcm_s16le:
            return
        try:
            import wave  # noqa: PLC0415 - stdlib, only needed here
            from datetime import datetime  # noqa: PLC0415

            folder = self.audio_dir / "".join(
                ch if ch.isalnum() or ch in "-_ " else "_" for ch in name
            ).strip()
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{datetime.now():%Y%m%d-%H%M%S-%f}.wav"
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(int(sample_rate) or 16000)
                handle.writeframes(pcm_s16le)
            clips = sorted(folder.glob("*.wav"))
            for old in clips[:-MAX_VOICE_SAMPLES_PER_PERSON]:
                old.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - keeping audio is a convenience
            log.debug("Could not keep the enrollment clip for %s", name, exc_info=True)

    # -- public API ------------------------------------------------------

    def people(self) -> dict[str, str]:
        """name -> role for every enrolled person."""
        with self._lock:
            return {name: str(p.get("role") or ROLE_USER) for name, p in self._people.items()}

    def profile_snapshot(self, name: str) -> dict[str, Any]:
        """Copy the local enrollment record for the explicitly enabled dataset archive."""
        with self._lock:
            person = self._people.get(name)
            return ({'name': name, 'voice_model': VOICE_MODEL_ID,
                     **json.loads(json.dumps(person))} if person is not None else {'name': name})

    def vectors_of(self, name: str) -> list[np.ndarray]:
        """The person's own voice vectors (ТЗ F-214: the challenge compares them).

        The challenge word of a privileged call has to prove that the SAME
        person said the random word, so the utterance's embedding is compared
        against the person's own profile - not against the room's population,
        where the best match would always be somebody. A person without a
        profile has no vectors, and ``[]`` is the honest answer.
        """
        with self._lock:
            person = self._people.get(str(name or "").strip())
            if person is None:
                return []
            stored = person.get(VOICE_KEY) or person.get(LEGACY_VOICE_KEY) or []
            return [np.asarray(vector, dtype="float32") for vector in stored]

    def role_of(self, name: str) -> str | None:
        with self._lock:
            person = self._people.get(name)
        return str(person.get("role") or ROLE_USER) if person else None

    def language_of(self, name: str) -> str | None:
        """The person's preferred language, or ``None`` (ТЗ F-106)."""
        cleaned = " ".join(str(name or "").split())
        with self._lock:
            person = self._people.get(cleaned)
            value = person.get("preferred_language") if person else None
        text = str(value or "").strip().lower()
        return text or None

    def set_language(self, name: str, code: str | None) -> str:
        """Store the language whisper and the model should use (ТЗ F-106).

        The registry keeps a copy of the canonical ``persons.preferred_language``
        so the voice pipeline never needs a database round-trip mid-turn; the
        caller (:func:`hub.languages.set_preferred_language`) writes both.
        """
        cleaned = " ".join(str(name or "").split())
        stored = str(code or "").strip().lower() or None
        with self._lock:
            person = self._people.get(cleaned)
            if person is None:
                known = ", ".join(sorted(self._people)) or "(nobody enrolled yet)"
                raise ValueError(f"no profile for {cleaned!r}; enrolled: {known}")
            person["preferred_language"] = stored
            self._save_locked()
        log.info("Preferred language of %s set to %s", cleaned, stored or "(auto)")
        return stored or ""

    def admin_profile(self, action, name, *, role=ROLE_USER):
        """Explicit owner-panel changes to active enrollment; archives remain."""
        import copy
        import shutil
        import uuid
        cleaned = ' '.join(str(name or '').split())
        if (not cleaned or len(cleaned) > 80 or is_placeholder_name(cleaned)
                or any(ord(char) < 32 for char in cleaned)):
            raise ValueError('Invalid profile name')
        if action not in {'create', 'reset_voice', 'reset_face', 'delete'} or role not in ROLES:
            raise ValueError('Invalid profile operation')
        with self._lock:
            existing = next((key for key in self._people if key.casefold() == cleaned.casefold()), None)
            if (action == 'create') == (existing is not None):
                raise ValueError('Profile already exists' if existing else 'Profile does not exist')
            if self.path.is_file():
                backup = self.path.parent / 'profile_backups'
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.path, backup / ('people-' + uuid.uuid4().hex + '.json'))
            before = copy.deepcopy(self._people)
            try:
                if action == 'create':
                    self._people[cleaned] = {'role': role, VOICE_KEY: [], FACE_KEY: []}
                elif action == 'delete':
                    del self._people[existing]
                else:
                    self._people[existing][VOICE_KEY if action == 'reset_voice' else FACE_KEY] = []
                self._save_locked()
            except Exception:
                self._people = before
                raise
        return cleaned

    def identify(self, pcm_s16le: bytes, sample_rate: int) -> tuple[str, str, float]:
        """Return ``(name, role, score)``; ``("unknown", "unknown", score)`` on no match."""
        name, role, score, _embedding = self.identify_ex(pcm_s16le, sample_rate)
        return name, role, score

    def identify_ex(self, pcm_s16le: bytes, sample_rate: int,
                    ) -> tuple[str, str, float, np.ndarray | None]:
        """``identify`` plus the embedding it scored (ТЗ F-205).

        Binding a voice to the right body (F-205) needs the vector that was
        matched, and computing it a second time would cost another ECAPA pass
        in the middle of a turn. The fourth element is ``None`` whenever no
        embedding could be made (disabled registry, unusable speech, failure),
        in which case the result is exactly what :meth:`identify` returns.
        """
        if not self.enabled:
            return ROLE_UNKNOWN, ROLE_UNKNOWN, 0.0, None
        try:
            embedding = self._embed(pcm_s16le, sample_rate)
        except Exception:
            log.exception("Voice embedding failed")
            return ROLE_UNKNOWN, ROLE_UNKNOWN, 0.0, None
        if embedding is None:
            return ROLE_UNKNOWN, ROLE_UNKNOWN, 0.0, None

        with self._lock:
            scores = self._scores_locked(embedding)
            best_score, best_name = scores[0] if scores else (0.0, None)
            runner_up = scores[1][0] if len(scores) > 1 else None
            matched = None
            reason = ""
            if best_name is None:
                reason = "no voice profiles"
            elif best_score < self.threshold:
                reason = f"below threshold {self.threshold:.2f}"
            elif runner_up is not None and best_score - runner_up < self.margin:
                reason = (
                    f"too close to {scores[1][1]} ({runner_up:.2f}), "
                    f"needs a {self.margin:.2f} lead"
                )
            else:
                matched = best_name
            role = (
                str(self._people[matched].get("role") or ROLE_USER)
                if matched
                else ROLE_UNKNOWN
            )
        name = matched or ROLE_UNKNOWN
        # Every candidate's score, so a wrong match can be diagnosed - and the
        # thresholds recalibrated - from the log alone.
        board = ", ".join(f"{who} {score:.2f}" for score, who in scores[:4]) or "-"
        log.info(
            "Speaker: %s (score %.2f)%s [%s]",
            name,
            max(best_score, 0.0),
            f" - unknown: {reason}" if reason else "",
            board,
        )
        return name, role, max(best_score, 0.0), embedding

    def prepare_enrollment_sample(self, pcm: bytes, sample_rate: int, previous: list) -> dict:
        """Validate a recording without changing any profile or permissions."""
        duration = estimate_speech_seconds(pcm)
        if duration < 2.5:
            raise ValueError('sample too short')
        vector = self._embed(pcm, sample_rate)
        if vector is None or not np.isfinite(vector).all():
            raise ValueError('unusable speech')
        # Compare with THIS recording session, not the old broken profile.
        # Room-mic samples of one enrolled person currently range as low as
        # .15-.23. The identity threshold would deadlock learning again. This
        # is only a coarse recording check; finish_enrollment still verifies
        # EVERY sample at the owner threshold or requires physical approval.
        if any(float(np.dot(vector, sample['embedding'])) < ENROLL_MIN_SELF for sample in previous):
            raise VoiceMismatch('sample does not match the voice that started this recording')
        return {'pcm': pcm, 'embedding': vector, 'speech_s': duration}

    def finish_enrollment(self, name: str, samples: list, sample_rate: int,
                          owner_threshold: float, *, confirmed: bool = False) -> tuple[str, str]:
        """Commit a complete recording atomically; old identity alone is never guessed.

        A short enrollment command need not match anyone. Existing profiles
        require strong matches from the long samples or a physical room-PC
        confirmation. Incomplete/cancelled recordings cannot poison a profile.
        """
        cleaned = ' '.join(str(name or '').split())
        if not cleaned or is_placeholder_name(cleaned):
            raise ValueError('a real name is required')
        if not enrollment_complete(len(samples), sum(s['speech_s'] for s in samples)):
            raise ValueError('recording incomplete')
        vectors = [s['embedding'] for s in samples]
        with self._lock:
            cleaned = next((n for n in self._people if n.casefold() == cleaned.casefold()), cleaned)
            existing = self._people.get(cleaned)
            for vector in vectors:
                scores = self._scores_locked(vector)
                best, who = scores[0] if scores else (0.0, None)
                runner = scores[1][0] if len(scores) > 1 else -1.0
                if existing and not confirmed:
                    if who != cleaned or best < owner_threshold or best - runner < self.margin:
                        raise EnrollmentConfirmationRequired(cleaned)
                elif not existing and who and best >= max(.78, self.threshold + self.margin):
                    raise DuplicateVoice(who)
            # Save once after all validations. Restore RAM too if the disk write fails.
            prior = list(existing.get(VOICE_KEY) or []) if existing else []
            person = self._person_locked(cleaned)
            person[VOICE_KEY] = (prior + [[round(float(v), 6) for v in vector.tolist()]
                                         for vector in vectors])[-MAX_VOICE_SAMPLES_PER_PERSON:]
            try:
                self._save_locked()
            except Exception:
                if existing:
                    person[VOICE_KEY] = prior
                else:
                    self._people.pop(cleaned, None)
                raise
            role = str(person.get('role') or ROLE_USER)
        for sample in samples:
            self._save_audio(cleaned, sample['pcm'], sample_rate)
        log.info('Voice enrollment completed: %s, %d samples, local_confirmation=%s',
                 cleaned, len(samples), confirmed)
        return role, f'{len(samples)} samples stored'

    def enroll(self, name: str, pcm_s16le: bytes, sample_rate: int) -> tuple[str, str]:
        """Store one voice sample for ``name``; returns ``(role, status)``.

        Creates the person on first use: the very first person enrolled in an
        empty registry becomes admin (the owner), later ones start as user.
        """
        cleaned = " ".join(str(name or "").split())
        if not cleaned:
            raise ValueError("enroll_voice needs a non-empty name")
        if is_placeholder_name(cleaned):
            raise ValueError(
                f"{cleaned!r} is a placeholder name, not a real one - ask for "
                "their actual name before enrolling them"
            )
        embedding = self._embed(pcm_s16le, sample_rate)
        if embedding is None:
            raise ValueError(
                "the utterance was too short to make a voice sample - "
                "ask the speaker to say a full sentence"
            )
        note = ""
        with self._lock:
            cleaned = next((name for name in self._people if name.casefold() == cleaned.casefold()), cleaned)
            existing = self._people.get(cleaned)
            own = self._centroid((existing or {}).get(VOICE_KEY) or [])
            best_other, best_other_name = -1.0, None
            for score, who in self._scores_locked(embedding):
                if who != cleaned:
                    best_other, best_other_name = score, who
                    break
            self_sim = float(np.dot(own, embedding)) if own is not None else None

            # Reject only when the sample sounds clearly MORE like an ALREADY
            # ENROLLED other person than like the enrollee. This catches a
            # second person speaking during enrollment, but - unlike an
            # absolute floor - can never reject the enrollee's own natural
            # variation, which is what deadlocked the owner.
            if (
                self_sim is not None
                and best_other_name is not None
                and best_other - self_sim > ENROLL_REJECT_MARGIN
            ):
                log.info(
                    "Rejected a voice sample for %s: it sounds more like %s "
                    "(%.2f) than like %s (%.2f)",
                    cleaned, best_other_name, best_other, cleaned, self_sim,
                )
                raise VoiceMismatch(
                    f"that sounded more like {best_other_name} than {cleaned} "
                    f"- it was probably somebody else speaking. Ask {cleaned} to "
                    "say the next sentence themselves while nobody else talks"
                )
            if self_sim is not None and self_sim < ENROLL_MIN_SELF:
                log.info(
                    "Rejected a voice sample for %s: %.2f against their own "
                    "samples is too low to be the same speech", cleaned, self_sim,
                )
                raise VoiceMismatch(
                    f"that did not sound like usable speech for {cleaned} - ask "
                    "them to say a full sentence clearly"
                )
            if (
                own is None
                and best_other_name is not None
                and best_other >= max(0.78, self.threshold + self.margin)
            ):
                raise DuplicateVoice(best_other_name)
            if (
                own is None
                and best_other_name is not None
                and best_other >= self.threshold + self.margin
            ):
                # Brand-new person whose first sample already matches somebody.
                note = (
                    f"this voice already sounds a lot like {best_other_name} "
                    f"({best_other:.2f}) - make sure it really is "
                    f"{cleaned} speaking"
                )
            person = self._person_locked(cleaned)
            vectors = person[VOICE_KEY]
            vectors.append([round(float(v), 6) for v in embedding.tolist()])
            del vectors[:-MAX_VOICE_SAMPLES_PER_PERSON]
            self._save_locked()
            role = str(person.get("role") or ROLE_USER)
            count = len(vectors)
        self._save_audio(cleaned, pcm_s16le, sample_rate)
        log.info("Enrolled voice sample %d for %s (%s)", count, cleaned, role)
        status = f"sample {count} stored"
        return role, f"{status} - {note}" if note else status

    # -- faces (SPEC v1.4) -----------------------------------------------

    def add_face_embedding(self, name: str, vector: Any) -> tuple[str, str]:
        """Store one face embedding for ``name``; returns ``(role, status)``.

        Creates the person exactly like :meth:`enroll` does — the first person
        ever added to an empty registry becomes admin — so somebody may be
        enrolled by face before their voice is known.
        """
        cleaned = " ".join(str(name or "").split())
        if not cleaned:
            raise ValueError("enroll_face needs a non-empty name")
        if is_placeholder_name(cleaned):
            raise ValueError("Please give your real name before saving your face")
        try:
            values = np.asarray(vector, dtype=np.float32).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"the face embedding is not numeric: {exc}") from exc
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("the face embedding is empty or not finite")
        with self._lock:
            person = self._person_locked(cleaned)
            vectors = person[FACE_KEY]
            vectors.append([round(float(v), 6) for v in values.tolist()])
            del vectors[:-MAX_SAMPLES_PER_PERSON]
            self._save_locked()
            role = str(person.get("role") or ROLE_USER)
            count = len(vectors)
        log.info("Stored face sample %d for %s (%s)", count, cleaned, role)
        return role, f"face sample {count} stored"

    def voice_profiles(self) -> dict[str, int]:
        """name -> how many voice samples they have (people with none are left out)."""
        with self._lock:
            return {
                name: len(person.get(VOICE_KEY) or [])
                for name, person in self._people.items()
                if person.get(VOICE_KEY)
            }

    def face_profiles(self) -> dict[str, list[list[float]]]:
        """name -> face embeddings, for :meth:`server.face.FaceEngine.match`.

        People without a face sample are left out, so an empty dict means
        nobody can be recognised by sight yet.
        """
        with self._lock:
            return {
                name: [list(vector) for vector in (person.get(FACE_KEY) or [])]
                for name, person in self._people.items()
                if person.get(FACE_KEY) and not is_placeholder_name(name)
            }

    def set_role(self, name: str, role: str) -> str:
        """Change a person's role; returns the applied role. Raises on bad input."""
        cleaned = " ".join(str(name or "").split())
        role = str(role or "").strip().lower()
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; valid roles: {', '.join(ROLES)}")
        with self._lock:
            person = self._people.get(cleaned)
            if person is None:
                known = ", ".join(sorted(self._people)) or "(nobody enrolled yet)"
                raise ValueError(f"no profile for {cleaned!r}; enrolled: {known}")
            person["role"] = role
            self._save_locked()
        log.info("Role of %s set to %s", cleaned, role)
        return role

    def rename_person(self, old_name: str, new_name: str, *, allow_merge: bool = True) -> tuple[str, str]:
        """Rename ``old_name`` to ``new_name`` (v1.6); returns ``(role, status)``.

        Renames in place when ``new_name`` is not enrolled yet. When it
        already is, the two profiles MERGE: voice and face embeddings are
        concatenated (still capped at :data:`MAX_SAMPLES_PER_PERSON` each) and
        the higher of the two roles wins, so neither a self-rename nor an
        admin merging two profiles can ever demote an admin. Works mid
        enrollment too - the caller (``server/app.py``) is the one that also
        updates any in-progress enrollment's pending name.

        :raises ValueError: either name is empty, ``new_name`` is a reserved
            placeholder, or ``old_name`` has no profile.
        :returns: ``(role, "renamed same name"|"renamed"|"merged")``.
        """
        old_clean = " ".join(str(old_name or "").split())
        new_clean = " ".join(str(new_name or "").split())
        if not old_clean:
            raise ValueError("rename_person needs a non-empty old_name")
        if not new_clean:
            raise ValueError("rename_person needs a non-empty new_name")
        if is_placeholder_name(new_clean):
            raise ValueError(
                f"{new_clean!r} is a placeholder name, not a real one - ask "
                "for their actual name"
            )
        with self._lock:
            old_clean = next((n for n in self._people if n.casefold() == old_clean.casefold()), old_clean)
            new_clean = next((n for n in self._people if n.casefold() == new_clean.casefold()), new_clean)
            person = self._people.get(old_clean)
            if person is None:
                known = ", ".join(sorted(self._people)) or "(nobody enrolled yet)"
                raise ValueError(f"no profile for {old_clean!r}; enrolled: {known}")
            role = str(person.get("role") or ROLE_USER)
            if old_clean.lower() == new_clean.lower():
                return role, "renamed same name"

            target = self._people.get(new_clean)
            if target is not None and not allow_merge:
                raise ValueError('That name already belongs to a profile. Combining profiles needs confirmation on this PC.')
            if target is None:
                self._people[new_clean] = self._people.pop(old_clean)
                self._save_locked()
                self._copy_renamed_audio(old_clean, new_clean)
                log.info("Renamed %s -> %s (%s)", old_clean, new_clean, role)
                return role, "renamed"

            # Merge into the existing target profile.
            merged_role = _higher_role(str(target.get("role") or ROLE_USER), role)
            target[VOICE_KEY] = list(target.get(VOICE_KEY) or []) + list(person.get(VOICE_KEY) or [])
            del target[VOICE_KEY][:-MAX_VOICE_SAMPLES_PER_PERSON]
            target[FACE_KEY] = list(target.get(FACE_KEY) or []) + list(person.get(FACE_KEY) or [])
            del target[FACE_KEY][:-MAX_SAMPLES_PER_PERSON]
            target["role"] = merged_role
            del self._people[old_clean]
            self._save_locked()
            self._copy_renamed_audio(old_clean, new_clean)
            log.info("Merged %s into existing profile %s (%s)", old_clean, new_clean, merged_role)
            return merged_role, "merged"

    def _copy_renamed_audio(self, old, new):
        """Keep source clips as a backup, and make them available under the new name."""
        import shutil
        clean = lambda name: ''.join(c if c.isalnum() or c in '-_ ' else '_' for c in name).strip()
        source, target = self.audio_dir / clean(old), self.audio_dir / clean(new)
        if source == target or not source.is_dir():
            return
        target.mkdir(parents=True, exist_ok=True)
        for clip in source.glob('*.wav'):
            dest = target / ('renamed-' + clean(old) + '-' + clip.name)
            if not dest.exists():
                shutil.copy2(clip, dest)


#: v1.4 name of the same class — it owns voices, faces and roles alike.
PeopleRegistry = VoiceRegistry


__all__ = [
    "ROLE_ADMIN",
    "ROLE_TRUSTED",
    "ROLE_USER",
    "ROLE_GUEST",
    "ROLE_UNKNOWN",
    "ROLES",
    "PEOPLE_FILENAME",
    "LEGACY_VOICES_FILENAME",
    "VOICE_KEY",
    "FACE_KEY",
    "LEGACY_VOICE_KEY",
    "ENROLL_EXTRA_SAMPLES",
    "ENROLL_MIN_SAMPLES",
    "MIN_ENROLL_SPEECH_S",
    "MAX_SAMPLES_PER_PERSON",
    "MAX_VOICE_SAMPLES_PER_PERSON",
    "PLACEHOLDER_NAMES",
    "VOICE_MODEL_ID",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MARGIN",
    "ENROLL_REJECT_MARGIN",
    "ENROLL_MIN_SELF",
    "VoiceMismatch",
    "SAFE_PC_COMMANDS",
    "check_permission",
    "normalize_people",
    "pcm_to_float32",
    "is_placeholder_name",
    "estimate_speech_seconds",
    "enrollment_complete",
    "VoiceRegistry",
    "PeopleRegistry",
]
