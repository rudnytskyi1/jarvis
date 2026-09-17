"""The people registry: voices, faces and role-based permissions (SPEC v1.3/v1.4).

Every utterance is embedded with resemblyzer (256-d d-vector) and compared by
cosine similarity against the profiles enrolled in ``data/people.json``. Roles:
``admin`` > ``trusted`` > ``user``; a voice that matches nobody is ``unknown``.
Permissions are enforced here, server-side — the prompt only explains refusals.

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
ROLE_UNKNOWN = "unknown"
ROLES = (ROLE_ADMIN, ROLE_TRUSTED, ROLE_USER)

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

#: Extra samples collected after ``enroll_voice`` stored the first one.
ENROLL_EXTRA_SAMPLES = 2
#: Cap per person so the profile file cannot grow without bound.
MAX_SAMPLES_PER_PERSON = 10

_INT16_SCALE = 32768.0

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
_EVERYONE_TOOLS = frozenset({"set_light", "set_switch", "enroll_voice", "enroll_face"})
#: Tools that need admin or trusted. Looking through the room camera is as
#: sensitive as looking at the screen, so it sits in the same tier.
_TRUSTED_TOOLS = frozenset(
    {"click_screen", "look_at_screen", "look_at_camera", "remember"}
)
#: Tools that need admin.
_ADMIN_TOOLS = frozenset({"run_command", "set_role"})


def check_permission(role: str, tool: str, args: dict[str, Any] | None) -> str | None:
    """Return None when allowed, or the denial message for the LLM."""
    role = role if role in ROLES else ROLE_UNKNOWN

    def deny(needed: str) -> str:
        return (
            f"permission denied: {tool} requires {needed}, but the current "
            f"speaker's role is {role} - politely refuse and suggest asking "
            "an authorized person"
        )

    if tool in _EVERYONE_TOOLS:
        return None
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
        threshold: float = 0.72,
        min_speech_s: float = 0.8,
        enabled: bool = True,
    ) -> None:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.path = base / PEOPLE_FILENAME
        #: Pre-v1.4 file; migrated into :attr:`path` when that one is absent.
        self.legacy_path = base / LEGACY_VOICES_FILENAME
        self.threshold = float(threshold)
        self.min_speech_s = float(min_speech_s)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._encoder: Any = None
        self._people: dict[str, dict[str, Any]] = {}
        self._load()
        log.info(
            "People registry: %s (%d profile(s), voice threshold %.2f)",
            self.path,
            len(self._people),
            self.threshold,
        )

    # -- storage ---------------------------------------------------------

    def _load(self) -> None:
        """Read the registry, migrating a pre-v1.4 ``voices.json`` if needed."""
        source = self.path
        migrating = False
        if not self.path.is_file() and self.legacy_path.is_file():
            source = self.legacy_path
            migrating = True
        try:
            if source.is_file():
                data = json.loads(source.read_text(encoding="utf-8"))
                people = data.get("people") if isinstance(data, dict) else None
                if isinstance(people, dict):
                    self._people = normalize_people(people)
        except Exception:
            log.exception("Could not read %s - starting empty", source)
            self._people = {}
            return
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

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"people": self._people}, ensure_ascii=False)
        self.path.write_text(payload, encoding="utf-8")

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
        if self._encoder is None:
            from resemblyzer import VoiceEncoder  # heavy import, kept lazy

            self._encoder = VoiceEncoder("cpu", verbose=False)
            log.info("resemblyzer voice encoder loaded")
        return self._encoder

    def _embed(self, pcm_s16le: bytes, sample_rate: int) -> np.ndarray | None:
        wav = pcm_to_float32(pcm_s16le)
        if sample_rate != 16000 and sample_rate > 0 and wav.size:
            duration = wav.size / float(sample_rate)
            positions = np.linspace(0.0, wav.size - 1, max(1, int(duration * 16000)))
            wav = np.interp(positions, np.arange(wav.size), wav).astype(np.float32)
        if wav.size / 16000.0 < self.min_speech_s:
            return None
        embedding = self._get_encoder().embed_utterance(wav)
        return np.asarray(embedding, dtype=np.float32)

    # -- public API ------------------------------------------------------

    def people(self) -> dict[str, str]:
        """name -> role for every enrolled person."""
        with self._lock:
            return {name: str(p.get("role") or ROLE_USER) for name, p in self._people.items()}

    def role_of(self, name: str) -> str | None:
        with self._lock:
            person = self._people.get(name)
        return str(person.get("role") or ROLE_USER) if person else None

    def identify(self, pcm_s16le: bytes, sample_rate: int) -> tuple[str, str, float]:
        """Return ``(name, role, score)``; ``("unknown", "unknown", score)`` on no match."""
        if not self.enabled:
            return ROLE_UNKNOWN, ROLE_UNKNOWN, 0.0
        try:
            embedding = self._embed(pcm_s16le, sample_rate)
        except Exception:
            log.exception("Voice embedding failed")
            return ROLE_UNKNOWN, ROLE_UNKNOWN, 0.0
        if embedding is None:
            return ROLE_UNKNOWN, ROLE_UNKNOWN, 0.0

        best_name, best_score = None, -1.0
        with self._lock:
            for name, person in self._people.items():
                vectors = person.get(VOICE_KEY) or []
                for raw in vectors:
                    vector = np.asarray(raw, dtype=np.float32)
                    denom = float(np.linalg.norm(vector) * np.linalg.norm(embedding))
                    if denom <= 0.0:
                        continue
                    score = float(np.dot(vector, embedding) / denom)
                    if score > best_score:
                        best_name, best_score = name, score
            matched = best_name if best_score >= self.threshold else None
            role = (
                str(self._people[matched].get("role") or ROLE_USER)
                if matched
                else ROLE_UNKNOWN
            )
        name = matched or ROLE_UNKNOWN
        log.info("Speaker: %s (score %.2f)", name, max(best_score, 0.0))
        return name, role, max(best_score, 0.0)

    def enroll(self, name: str, pcm_s16le: bytes, sample_rate: int) -> tuple[str, str]:
        """Store one voice sample for ``name``; returns ``(role, status)``.

        Creates the person on first use: the very first person enrolled in an
        empty registry becomes admin (the owner), later ones start as user.
        """
        cleaned = " ".join(str(name or "").split())
        if not cleaned:
            raise ValueError("enroll_voice needs a non-empty name")
        embedding = self._embed(pcm_s16le, sample_rate)
        if embedding is None:
            raise ValueError(
                "the utterance was too short to make a voice sample - "
                "ask the speaker to say a full sentence"
            )
        with self._lock:
            person = self._person_locked(cleaned)
            vectors = person[VOICE_KEY]
            vectors.append([round(float(v), 6) for v in embedding.tolist()])
            del vectors[:-MAX_SAMPLES_PER_PERSON]
            self._save_locked()
            role = str(person.get("role") or ROLE_USER)
            count = len(vectors)
        log.info("Enrolled voice sample %d for %s (%s)", count, cleaned, role)
        return role, f"sample {count} stored"

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

    def face_profiles(self) -> dict[str, list[list[float]]]:
        """name -> face embeddings, for :meth:`server.face.FaceEngine.match`.

        People without a face sample are left out, so an empty dict means
        nobody can be recognised by sight yet.
        """
        with self._lock:
            return {
                name: [list(vector) for vector in (person.get(FACE_KEY) or [])]
                for name, person in self._people.items()
                if person.get(FACE_KEY)
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


#: v1.4 name of the same class — it owns voices, faces and roles alike.
PeopleRegistry = VoiceRegistry


__all__ = [
    "ROLE_ADMIN",
    "ROLE_TRUSTED",
    "ROLE_USER",
    "ROLE_UNKNOWN",
    "ROLES",
    "PEOPLE_FILENAME",
    "LEGACY_VOICES_FILENAME",
    "VOICE_KEY",
    "FACE_KEY",
    "LEGACY_VOICE_KEY",
    "ENROLL_EXTRA_SAMPLES",
    "MAX_SAMPLES_PER_PERSON",
    "SAFE_PC_COMMANDS",
    "check_permission",
    "normalize_people",
    "pcm_to_float32",
    "VoiceRegistry",
    "PeopleRegistry",
]
