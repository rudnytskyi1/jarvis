"""Регистрация гостя голосом или в Telegram (ТЗ F-210).

A stranger cannot become a known person by accident. The ТЗ puts four things
between "somebody new walked in" and a ``guest`` profile, and this module owns
all four: the owner's own words ("это Макс, друг"), a phrase the guest reads
aloud, 5-10 camera frames of the guest from different angles with a quality
check, and the body of the current day. Nothing is written before the OWNER
confirms in Telegram - that confirmation is mandatory in F-210, so the collected
vectors wait in memory until the button is pressed.

Three layers, so each can be tested without a microphone, a camera or Telegram:

* the WORDS - the owner's declaration, the phrase the guest reads, and the short
  notice of what is stored that the guest confirms out loud (ТЗ 15.4: "человек
  слышит короткое уведомление о том, что хранится, и подтверждает голосом").
  The wording is the executor's default for the open question of section 17
  ("текст согласия при регистрации (F-210) и в каком виде?") - see
  ``DECISIONS.md`` P2-20;
* the GATES - how many frames may be stored and which of them are usable at all
  (:class:`Shot`, :func:`assess_burst`);
* the RECORD - :func:`commit_guest`, one transaction that creates the ``persons``
  row, its ``guest`` membership, the voice/face/body vectors and the naming of
  the live track (through :func:`hub.identity_link.link_track_to_person`).

The flow itself (:class:`GuestRegistration`) is a small state machine with an
expiry, so a half-finished registration of somebody who walked away does not
stay open forever.
"""
from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from hub import confirmations
from hub.identity_link import link_track_to_person
from hub.speaker import ROLE_GUEST, is_placeholder_name
from hub.vectors import pack_vector

log = logging.getLogger("jarvis.server.guest_registration")

#: ТЗ F-210: "лицо, 5-10 кадров с разных ракурсов, проверка качества".
MIN_FRAMES = 5
MAX_FRAMES = 10
#: The face has to be at least this fraction of the frame height (closer than
#: a face at the far end of the room).
MIN_FACE_HEIGHT = 0.12
#: Sharpness of the face crop (variance of the Laplacian of its grayscale
#: pixels): below this the frame is a blur and teaches the profile nothing.
MIN_SHARPNESS = 40.0
#: The frame has to be no more than this far off the camera. 55 degrees is the
#: same window F-205 calls "лицом к камере" (nose at 0.19-0.81 of the eye span,
#: see :func:`yaw_degrees` and ``hub/voice_tracks.py``).
MAX_ANGLE_DEG = 55.0
#: A turn of at least this much counts as another angle (ТЗ F-210 wants
#: "с разных ракурсов", not ten copies of the same shot).
TURN_DEG = 20.0
MIN_ANGLES = 2
#: F-210 asks for "фразу" (one phrase), not for the 20 s of an owner's
#: enrollment: this much clean speech already makes a usable ECAPA vector.
MIN_VOICE_SECONDS = 3.0
#: One repeated try per step, then the flow is cancelled. A guest who cannot be
#: seen or heard does not become a profile by insisting.
VOICE_ATTEMPTS = 2
FACE_ATTEMPTS = 2
#: A half-finished flow expires; the owner has longer to find their phone.
FLOW_TTL_S = 300.0
CONFIRM_TTL_S = 900.0
#: Callback-data namespace of the owner's buttons; ``guest:<token>:yes``.
TOKEN_PREFIX = "guest:"

#: The flow's steps. ``declined``/``cancelled`` are terminal.
STEP_VOICE = "voice"
STEP_FACE = "face"
STEP_BODY = "body"
STEP_CONSENT = "consent"
STEP_OWNER = "owner"
STEP_DONE = "done"
STEP_DECLINED = "declined"
STEP_CANCELLED = "cancelled"


def language_of(value: Any, *, default: str = "ru") -> str:
    """Which of the three languages of the house a value names."""
    code = str(value or "").strip().casefold()[:2]
    return code if code in PHRASES else default


# ---------------------------------------------------------------------------
# the words
# ---------------------------------------------------------------------------

#: ТЗ F-210: "Rowan просит произнести фразу". Deterministic, so the same room
#: asks for the same sentence and a test can check that it was read.
PHRASES: dict[str, tuple[str, ...]] = {
    "ru": (
        "Роуэн, послушай мой голос. Сегодня я гость в этой комнате.",
        "Роуэн, это моя обычная речь. Я стою у камеры.",
    ),
    "en": (
        "Rowan, listen to my voice. I am a guest in this room today.",
        "Rowan, this is how I normally speak. I am standing by the camera.",
    ),
    "es": (
        "Rowan, escucha mi voz. Hoy soy invitado en esta habitación.",
        "Rowan, así hablo normalmente. Estoy delante de la cámara.",
    ),
}

#: ТЗ 15.4: what the guest hears before anything is stored. Short on purpose -
#: it is spoken in the room, not posted - and it names the three things F-210
#: collects and the way out ("забудь меня", F-213).
CONSENT: dict[str, str] = {
    "ru": (
        "{name}, я сохраню о тебе три вещи: отпечаток голоса, фото лица за сегодня "
        "и «внешность дня» — как выглядит твоя одежда, чтобы узнавать тебя в этой комнате. "
        "Скажи «да», если согласен. Скажи «забудь меня», и я всё удалю."
    ),
    "en": (
        "{name}, I will keep three things about you: a voice print, today's photos of "
        "your face, and how your clothes look today, so I can recognise you in this room. "
        "Say yes if you agree. Say forget me, and I will delete everything."
    ),
    "es": (
        "{name}, guardaré tres cosas sobre ti: tu huella de voz, fotos de tu cara de hoy "
        "y cómo se ve tu ropa hoy, para reconocerte en esta habitación. "
        "Di sí si estás de acuerdo. Di olvídame y lo borraré todo."
    ),
}

_FACE_PROMPT: dict[str, str] = {
    "ru": "{name}, посмотри в камеру и медленно поверни голову влево, потом вправо. Я сделаю несколько снимков.",
    "en": "{name}, look at the camera and slowly turn your head left, then right. I will take a few photos.",
    "es": "{name}, mira a la cámara y gira la cabeza despacio a la izquierda y luego a la derecha. Haré varias fotos.",
}

_RETRY: dict[str, dict[str, str]] = {
    "ru": {
        "no_frames": "{name}, я тебя не вижу. Встань перед камерой.",
        "too_few_frames": "{name}, я не успел тебя снять. Встань ближе к камере и поверни голову медленнее.",
        "one_angle": "{name}, все снимки с одного ракурса. Поверни голову влево, потом вправо.",
        "done": "Спасибо, {name}. Снимки получились.",
    },
    "en": {
        "no_frames": "{name}, I cannot see you. Stand in front of the camera.",
        "too_few_frames": "{name}, I could not take the photos. Come closer and turn your head more slowly.",
        "one_angle": "{name}, every photo is from the same angle. Turn your head left, then right.",
        "done": "Thank you, {name}. The photos came out.",
    },
    "es": {
        "no_frames": "{name}, no te veo. Ponte delante de la cámara.",
        "too_few_frames": "{name}, no pude hacer las fotos. Acércate y gira la cabeza más despacio.",
        "one_angle": "{name}, todas las fotos son del mismo ángulo. Gira la cabeza a la izquierda y luego a la derecha.",
        "done": "Gracias, {name}. Las fotos salieron bien.",
    },
}

#: What the room says while the owner's phone is the only thing left.
_WAITING_OWNER: dict[str, str] = {
    "ru": "Спасибо. Теперь владелец должен подтвердить это в Telegram — до подтверждения я ничего о тебе не сохраняю.",
    "en": "Thank you. The owner now has to confirm this in Telegram — nothing about you is stored until then.",
    "es": "Gracias. Ahora el dueño debe confirmarlo en Telegram: no guardo nada sobre ti hasta entonces.",
}

_OWNER_ASK: dict[str, str] = {
    "ru": "{name} ({relation}) просит сохранить биометрию в комнате. Подтвердите или отклоните.",
    "en": "{name} ({relation}) asks to store biometrics in the room. Confirm or decline.",
    "es": "{name} ({relation}) pide guardar su biometría en la habitación. Confirma o rechaza.",
}

_RELATION: dict[str, dict[str, str]] = {
    "friend": {"ru": "друг", "en": "friend", "es": "amigo"},
    "guest": {"ru": "гость", "en": "guest", "es": "invitado"},
}

_NAME = r"([^\W\d_]+(?:[ '\-][^\W\d_]+)?)"
_FRIEND = r"(?:friend|buddy|mate|друг|подруга|amigo|amiga)"
_GUEST = r"(?:guest|visitor|гость|гостья|invitado|invitada)"
_CALL = r"(?:" + _FRIEND + r"|" + _GUEST + r")"

#: "это Макс, друг" / "this is Max, a friend" / "este es Max, un amigo".
_DECLARATION = re.compile(
    r"\bthis is " + _NAME + r",?\s+(?:a |my |our )?" + _CALL
    + r"|\bmeet " + _NAME + r",?\s+(?:a |my |our )?" + _CALL
    + r"|\b(?:это|знакомься,? это)\s+" + _NAME + r",?\s+(?:(?:он|она|это)\s+)?(?:мой |наш |его |её )?" + _CALL
    + r"|\beste es " + _NAME + r",?\s+(?:un |mi |nuestro )?" + _CALL,
    re.IGNORECASE,
)

#: A registration asked for without naming anybody yet: the hub then asks who.
_REQUEST = re.compile(
    r"\b(?:register|add|remember)\s+(?:a |this |my |the )?(?:guest|visitor|friend)\b"
    r"|\b(?:добавь|зарегистрируй|запомни)\s+(?:этого\s+|нового\s+)?(?:гостя|гостью|друга)\b"
    r"|\bregistra\s+(?:un |a |el )?(?:invitado|invitada|amigo|amiga)\b",
    re.IGNORECASE,
)

_FRIEND_WORDS = re.compile(_FRIEND, re.IGNORECASE)

#: "отмена" / "cancel the registration": the flow of F-210 is dropped and
#: nothing collected for it is stored.
_CANCEL = re.compile(
    r"\b(?:cancel|stop|abort)\b[^.]{0,24}\b(?:guest|registration|this)\b"
    r"|\bforget (?:it|this|the guest)\b"
    r"|\bне надо\b|\bотмен\w*\b[^.]{0,24}\b(?:регистрац|гост)\w*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Declaration:
    """One owner's introduction of somebody else (ТЗ F-210)."""

    name: str | None
    relation: str = "guest"
    text: str = ""

    @property
    def named(self) -> bool:
        return bool(self.name)


def clean_name(value: Any) -> str | None:
    """A usable display name, or ``None`` (never a placeholder like "Guest").

    ТЗ F-210 registers a person the owner NAMED. ``Guest``/``User``/``Friend``
    are the placeholders ``hub/speaker.py`` already refuses, and a one-letter
    name is a misheard word rather than a name.
    """
    name = " ".join(str(value or "").split()).strip(" ,.!?;:")
    name = re.sub(r"['’]s$", "", name)
    if len(name) < 2 or len(name) > 40 or is_placeholder_name(name):
        return None
    return name


def declaration(text: Any) -> Declaration | None:
    """Read the owner's own words, or ``None`` when this is something else.

    F-210 gives one sentence - "это Макс, друг" - and this is what recognises
    it. A request without a name ("добавь гостя") is a declaration with
    ``name=None``: the hub then asks who, instead of inviting the speaker to
    enroll themselves (which is what "register" used to mean).
    """
    raw = " ".join(str(text or "").split())
    if not raw:
        return None
    match = _DECLARATION.search(raw)
    if match is not None:
        name = clean_name(next((group for group in match.groups() if group), None))
        if name is None:
            return None
        relation = "friend" if _FRIEND_WORDS.search(match.group(0)) else "guest"
        return Declaration(name=name, relation=relation, text=raw)
    if _REQUEST.search(raw):
        return Declaration(name=None, relation="guest", text=raw)
    return None


def cancel_requested(text: Any) -> bool:
    """Whether the room asked to stop the open guest registration (F-210)."""
    return bool(_CANCEL.search(" ".join(str(text or "").split())))


def phrase(language: Any = "ru", index: int = 0) -> str:
    """The sentence the guest reads aloud (ТЗ F-210)."""
    lines = PHRASES[language_of(language)]
    return lines[max(0, min(int(index), len(lines) - 1))]


def voice_prompt(name: str, language: Any = "ru", *, index: int = 0) -> str:
    """What the room says to the guest before the voice sample."""
    line = phrase(language, index)
    prefix = {"ru": "{name}, произнесите, пожалуйста", "en": "{name}, please read",
              "es": "{name}, lee por favor"}[language_of(language)]
    return f"{prefix}: «{line}».".replace("{name}", str(name or ""))


def face_prompt(name: str, language: Any = "ru") -> str:
    """What the room says before the frames of F-210 are taken."""
    return _FACE_PROMPT[language_of(language)].replace("{name}", str(name or ""))


def consent_request(name: str, language: Any = "ru") -> str:
    """The short notice of what is stored, spoken before the guest's "yes"."""
    return CONSENT[language_of(language)].replace("{name}", str(name or ""))


def consent_given(text: Any) -> bool:
    """Whether the guest answered "yes" (the same words as F-113's "да")."""
    return confirmations.answer(text) is True


def consent_refused(text: Any) -> bool:
    """Whether the guest said no - then nothing is stored at all."""
    return confirmations.answer(text) is False


def waiting_for_owner(language: Any = "ru") -> str:
    return _WAITING_OWNER[language_of(language)]


def relation_word(relation: str, language: Any = "ru") -> str:
    table = _RELATION.get(str(relation or "guest"), _RELATION["guest"])
    return table[language_of(language)]


def owner_question(name: str, relation: str, language: Any = "ru") -> str:
    """The line of the owner's Telegram confirmation (F-210)."""
    return (_OWNER_ASK[language_of(language)]
            .replace("{name}", str(name or ""))
            .replace("{relation}", relation_word(relation, language)))


def retry_hint(quality: BurstQuality, name: str, language: Any = "ru") -> str:
    """What to tell the guest after a failed camera step."""
    table = _RETRY[language_of(language)]
    reason = quality.reasons[0] if quality.reasons else "done"
    return table.get(reason, table["done"]).replace("{name}", str(name or ""))


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------


def yaw_degrees(landmarks: Any) -> float | None:
    """How far the head is turned, in degrees (0 = straight at the camera).

    The five insightface kps of ``hub/face.py`` are eye corners, nose and mouth
    corners. Facing the camera, the nose sits in the middle of the eye span and
    walks to one side as the head turns - the same measure F-205 uses to ask
    whether somebody is "лицом к камере" (``hub/voice_tracks.py``). A half
    eye-span of offset is 90 degrees, so the 0.19-0.81 window of F-205 lands on
    :data:`MAX_ANGLE_DEG`. ``None`` when the frame carries no usable kps: this
    is a measurement, and no landmarks means no measurement, not "zero".
    """
    points = landmarks
    if points is None:
        return None
    try:
        array = [[float(x), float(y)] for x, y in points]
    except (TypeError, ValueError):
        return None
    if len(array) < 3:
        return None
    left, right, nose = array[0], array[1], array[2]
    span = right[0] - left[0]
    if span <= 0.0:
        return None
    ratio = (nose[0] - left[0]) / span
    return max(-89.9, min(89.9, (ratio - 0.5) * 180.0))


@dataclass(frozen=True)
class Shot:
    """One camera frame of the guest's face, as the gates measure it."""

    box: tuple[float, float, float, float] = ()
    sharpness: float = 0.0
    angle: float | None = None
    score: float = 0.0

    @property
    def height(self) -> float:
        """Face height as a fraction of the frame (0 when there is no box)."""
        if len(self.box) != 4:
            return 0.0
        return max(0.0, float(self.box[3]) - float(self.box[1]))

    def bucket(self, *, turn: float = TURN_DEG) -> str:
        """``left``/``front``/``right``, or ``unknown`` without landmarks."""
        if self.angle is None:
            return "unknown"
        if self.angle <= -float(turn):
            return "left"
        if self.angle >= float(turn):
            return "right"
        return "front"


@dataclass(frozen=True)
class BurstQuality:
    """The answer of the camera gate of F-210."""

    ok: bool
    kept: tuple[Shot, ...] = ()
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def angles(self) -> tuple[str, ...]:
        """The distinct angles actually kept, in a stable order."""
        order = {"front": 0, "left": 1, "right": 2, "unknown": 3}
        return tuple(sorted({shot.bucket() for shot in self.kept}, key=lambda name: order[name]))


def shot_usable(shot: Shot, *, min_height: float = MIN_FACE_HEIGHT,
                min_sharpness: float = MIN_SHARPNESS, max_angle: float = MAX_ANGLE_DEG,
                min_score: float = 0.0) -> bool:
    """Whether one frame may teach the profile anything (ТЗ F-210)."""
    if shot.height < float(min_height) or shot.sharpness < float(min_sharpness):
        return False
    if shot.score < float(min_score):
        return False
    return shot.angle is None or abs(float(shot.angle)) <= float(max_angle)


def assess_burst(shots: Iterable[Shot], *, min_frames: int = MIN_FRAMES,
                 max_frames: int = MAX_FRAMES, min_angles: int = MIN_ANGLES,
                 min_height: float = MIN_FACE_HEIGHT, min_sharpness: float = MIN_SHARPNESS,
                 max_angle: float = MAX_ANGLE_DEG, min_score: float = 0.0) -> BurstQuality:
    """Judge one burst of the guest's frames (ТЗ F-210: 5-10, different angles).

    The gate keeps at most :data:`MAX_FRAMES` usable frames and asks for at
    least :data:`MIN_FRAMES` of them from at least two angles. Angles that could
    not be measured (a detector without kps) are reported as a note instead of
    failing the burst: what cannot be measured must not be invented, but it also
    must not block a registration on a hub whose model has no landmarks.
    """
    frames = [shot for shot in shots or () if isinstance(shot, Shot)]
    if not frames:
        return BurstQuality(False, reasons=("no_frames",))
    usable = [shot for shot in frames
              if shot_usable(shot, min_height=min_height, min_sharpness=min_sharpness,
                             max_angle=max_angle, min_score=min_score)]
    reasons: list[str] = []
    if len(usable) < int(min_frames):
        reasons.append("too_few_frames")
    kept = tuple(usable[:int(max_frames)])
    buckets = {shot.bucket() for shot in usable if shot.angle is not None}
    notes: list[str] = []
    if kept and not buckets:
        notes.append("angles_unknown")
    elif len(kept) >= int(min_frames) and len(buckets) < int(min_angles):
        reasons.append("one_angle")
    return BurstQuality(not reasons, kept=kept, reasons=tuple(reasons), notes=tuple(notes))


def phrase_matches(text: Any, expected: str, *, ratio: float = 0.5) -> bool:
    """Whether the guest actually read the sentence of F-210.

    Whisper rewrites punctuation and drops the odd word, so this asks for half
    of the sentence's words rather than an exact transcript: enough to know the
    person spoke the phrase instead of answering something else.
    """
    words = [word for word in re.findall(r"\w+", str(expected or "").casefold()) if len(word) > 2]
    if not words:
        return False
    heard = set(re.findall(r"\w+", str(text or "").casefold()))
    return sum(1 for word in words if word in heard) / len(words) >= float(ratio)


# ---------------------------------------------------------------------------
# the flow
# ---------------------------------------------------------------------------


@dataclass
class PendingGuest:
    """One registration in progress - everything before the owner's button."""

    name: str
    home_id: str
    relation: str = "guest"
    source: str = "voice"
    requested_by: str = ""
    language: str = "ru"
    step: str = STEP_VOICE
    created_at: float = 0.0
    expires_at: float = 0.0
    voice_attempts: int = 0
    face_attempts: int = 0
    voice_seconds: float = 0.0
    frames: int = 0
    angles: tuple[str, ...] = ()
    body_saved: bool = False
    owner_decision: str = "pending"
    track_id: str = ""
    day: str = ""
    note: str = ""

    def expired(self, *, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else float(now)
        return moment >= float(self.expires_at)

    @property
    def waiting_owner(self) -> bool:
        return self.step == STEP_OWNER

    def summary(self) -> dict[str, Any]:
        """What is known about this registration, without any vectors."""
        return {"name": self.name, "relation": self.relation, "home_id": self.home_id,
                "source": self.source, "requested_by": self.requested_by,
                "step": self.step, "frames": int(self.frames),
                "angles": list(self.angles), "body_saved": bool(self.body_saved),
                "owner_decision": self.owner_decision}


class GuestRegistration:
    """The single open guest registration of one room (ТЗ F-210).

    Every step reports the step the flow stands on now, so the hub can answer
    the room without keeping a second copy of the state. A step called out of
    order changes nothing - the same rule as the rest of the pipeline.
    """

    def __init__(self, *, ttl_s: float = FLOW_TTL_S,
                 clock: Any = time.monotonic) -> None:
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self._pending: PendingGuest | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def pending(self) -> PendingGuest | None:
        """The open registration, expired ones dropped."""
        if self._pending is not None and self._pending.expired(now=self.clock()):
            log.info("Guest registration of %s expired at %s",
                     self._pending.name, self._pending.step)
            self._pending = None
        return self._pending

    def start(self, invite: Declaration | str, *, home_id: str, requested_by: str = "",
              source: str = "voice", language: Any = "ru", track_id: str = "",
              day: str = "", now: float | None = None) -> PendingGuest | None:
        """Open the flow for a named person; ``None`` when there is no name yet."""
        name = invite.name if isinstance(invite, Declaration) else clean_name(invite)
        if not name:
            return None
        relation = invite.relation if isinstance(invite, Declaration) else "guest"
        moment = self.clock() if now is None else float(now)
        self._pending = PendingGuest(
            name=name, home_id=str(home_id), relation=str(relation or "guest"),
            source=str(source or "voice"), requested_by=str(requested_by or ""),
            language=language_of(language), created_at=moment,
            expires_at=moment + self.ttl_s, track_id=str(track_id or ""), day=str(day or ""),
        )
        log.info("Guest registration started for %s in %s (%s)", name, home_id, self._pending.source)
        return self._pending

    def cancel(self, note: str = "") -> PendingGuest | None:
        """Drop the flow; nothing collected for it is stored."""
        pending = self._pending
        self._pending = None
        if pending is not None and note:
            pending.note = str(note)
        return pending

    # -- steps -------------------------------------------------------------

    def voice_step(self, *, seconds: float, text: str = "", phrase_index: int = 0,
                   now: float | None = None) -> str:
        """The guest's phrase: enough clean speech, and the sentence itself."""
        pending = self.pending
        if pending is None or pending.step != STEP_VOICE:
            return pending.step if pending else STEP_CANCELLED
        pending.voice_attempts += 1
        spoke = float(seconds or 0.0)
        expected = phrase(pending.language, phrase_index)
        if spoke >= MIN_VOICE_SECONDS and phrase_matches(text, expected):
            pending.voice_seconds = round(spoke, 2)
            pending.step = STEP_FACE
            return pending.step
        if pending.voice_attempts >= VOICE_ATTEMPTS:
            return self._terminate(pending, STEP_CANCELLED, "voice_not_usable")
        pending.note = "voice_not_usable"
        return pending.step

    def face_step(self, shots: Iterable[Shot], *, now: float | None = None) -> BurstQuality:
        """The frames of F-210: their quality decides whether the flow moves on."""
        pending = self.pending
        if pending is None or pending.step != STEP_FACE:
            return BurstQuality(False, reasons=("wrong_step",))
        pending.face_attempts += 1
        quality = assess_burst(shots)
        if quality.ok:
            pending.frames = len(quality.kept)
            pending.angles = quality.angles
            pending.step = STEP_BODY
            return quality
        if pending.face_attempts >= FACE_ATTEMPTS:
            self._terminate(pending, STEP_CANCELLED, "face_not_usable")
        else:
            pending.note = quality.reasons[0] if quality.reasons else "face_not_usable"
        return quality

    def body_step(self, *, saved: bool, now: float | None = None) -> str:
        """The body of the current day is what F-210 stores second to last."""
        pending = self.pending
        if pending is None or pending.step != STEP_BODY:
            return pending.step if pending else STEP_CANCELLED
        pending.body_saved = bool(saved)
        pending.step = STEP_CONSENT
        return pending.step

    def consent_step(self, text: str, *, now: float | None = None) -> bool:
        """ТЗ 15.4: the guest confirms out loud what is stored (or refuses)."""
        pending = self.pending
        if pending is None or pending.step != STEP_CONSENT:
            return False
        if consent_given(text):
            pending.step = STEP_OWNER
            return True
        if consent_refused(text):
            self._terminate(pending, STEP_CANCELLED, "consent_refused")
            return False
        pending.note = "consent_not_given"
        return False

    def owner_step(self, approved: bool, *, now: float | None = None) -> str:
        """The owner's Telegram answer. Nothing has been stored before it."""
        pending = self.pending
        if pending is None or pending.step != STEP_OWNER:
            return pending.step if pending else STEP_CANCELLED
        if not approved:
            pending.owner_decision = "declined"
            return self._terminate(pending, STEP_DECLINED, "owner_declined")
        pending.owner_decision = "approved"
        return self._terminate(pending, STEP_DONE, "")

    def _terminate(self, pending: PendingGuest, step: str, note: str) -> str:
        pending.step = step
        if note:
            pending.note = note
        if self._pending is pending:
            self._pending = None
        return step


# ---------------------------------------------------------------------------
# the owner's confirmation (ТЗ F-210: обязательна)
# ---------------------------------------------------------------------------


@dataclass
class GuestConfirmation:
    """One question waiting for the owner's button in Telegram."""

    token: str
    name: str
    home_id: str
    relation: str = "guest"
    requested_by: str = ""
    language: str = "ru"
    created_at: float = 0.0
    expires_at: float = 0.0
    decision: str = "pending"
    message: str = ""

    def expired(self, *, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else float(now)
        return moment >= float(self.expires_at)

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "relation": self.relation, "home_id": self.home_id,
                "requested_by": self.requested_by, "decision": self.decision,
                "expires_at": self.expires_at}


class OwnerConfirmations:
    """Expiring, single-use owner confirmations for guest registration.

    A guest is not registered by the hub deciding it was probably fine: F-210
    says the owner confirms in Telegram, so the flow waits in :data:`STEP_OWNER`
    while the question sits here. The buttons carry a random token, the token is
    dropped the moment it is used, and an expired question can no longer
    register anybody (a stale button in a chat is not consent).
    """

    def __init__(self, *, provider: Any = None, ttl_s: float = CONFIRM_TTL_S,
                 clock: Any = time.monotonic) -> None:
        self.provider = provider
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self._open: dict[str, GuestConfirmation] = {}

    def open(self, pending: PendingGuest, *, message: str = "",
             now: float | None = None) -> GuestConfirmation:
        """Ask the owner about ``pending``; returns the question with its token."""
        moment = self.clock() if now is None else float(now)
        request = GuestConfirmation(
            token=uuid.uuid4().hex[:16], name=pending.name, home_id=pending.home_id,
            relation=pending.relation, requested_by=pending.requested_by,
            language=pending.language, created_at=moment, expires_at=moment + self.ttl_s,
            message=message or owner_question(pending.name, pending.relation, pending.language),
        )
        self._open = {token: item for token, item in self._open.items()
                      if not item.expired(now=moment) and item.decision == "pending"}
        self._open[request.token] = request
        return request

    def get(self, token: Any) -> GuestConfirmation | None:
        """The open question of ``token``, or ``None`` (expired ones dropped)."""
        request = self._open.get(str(token or ""))
        if request is None:
            return None
        if request.expired(now=self.clock()) or request.decision != "pending":
            self._open.pop(request.token, None)
            return None
        return request

    def resolve(self, token: Any, approved: bool, *,
                now: float | None = None) -> GuestConfirmation | None:
        """Consume the token; ``None`` when it is unknown, used or expired."""
        request = self.get(token)
        if request is None:
            return None
        self._open.pop(request.token, None)
        request.decision = "approved" if approved else "declined"
        return request

    def cancel_home(self, home_id: str) -> list[GuestConfirmation]:
        """Drop every open question of one room (its client disconnected)."""
        dropped = [item for item in self._open.values() if item.home_id == str(home_id)]
        for item in dropped:
            self._open.pop(item.token, None)
        return dropped

    @staticmethod
    def keyboard(token: str, language: Any = "ru") -> dict[str, Any]:
        """The inline keyboard of the question (``guest:<token>:yes|no``)."""
        labels = {"ru": ("Подтвердить", "Отклонить"), "en": ("Confirm", "Decline"),
                  "es": ("Confirmar", "Rechazar")}[language_of(language)]
        return {"inline_keyboard": [[
            {"text": labels[0], "callback_data": f"{TOKEN_PREFIX}{token}:yes"},
            {"text": labels[1], "callback_data": f"{TOKEN_PREFIX}{token}:no"},
        ]]}

    async def handle_update(self, update: Any, *, is_owner: Any = None,
                            on_decision: Any = None) -> bool:
        """Answer the owner's button press; ``True`` when this update was ours.

        ``is_owner`` is the hub's own access check (``hub/telegram_admin.py``
        asks the same question before any panel action exists). A press from
        anybody else is acked with an alert and changes nothing.
        """
        callback = update.get("callback_query") if isinstance(update, dict) else None
        if not isinstance(callback, dict):
            return False
        data = callback.get("data")
        if not isinstance(data, str) or not data.startswith(TOKEN_PREFIX):
            return False
        sender = callback.get("from")
        owner_ok = callable(is_owner) and bool(is_owner(sender))
        token, _, choice = data[len(TOKEN_PREFIX):].partition(":")
        if not owner_ok:
            await self._answer(callback, "Only the owner can confirm a guest.", alert=True)
            return True
        request = self.resolve(token, choice.casefold() == "yes")
        if request is None:
            await self._answer(callback, "This request has expired.", alert=True)
            return True
        await self._answer(callback, "Confirmed" if request.decision == "approved" else "Declined")
        await self._edit(callback, request)
        log.info("Owner %s guest registration of %s in %s",
                 request.decision, request.name, request.home_id)
        if callable(on_decision):
            try:
                await on_decision(request)
            except Exception:  # noqa: BLE001 - the decision is already recorded
                log.exception("Guest confirmation callback failed for %s", request.name)
        return True

    async def _answer(self, callback: dict[str, Any], text: str, *, alert: bool = False) -> None:
        provider, callback_id = self.provider, callback.get("id")
        if provider is None or not isinstance(callback_id, str):
            return
        try:
            await provider.answer_callback(callback_id, text, show_alert=alert)
        except Exception:  # noqa: BLE001 - an unanswered toast is not a failure
            log.debug("Could not answer the guest callback", exc_info=True)

    async def _edit(self, callback: dict[str, Any], request: GuestConfirmation) -> None:
        provider, message = self.provider, callback.get("message")
        if provider is None or not isinstance(message, dict):
            return
        message_id = message.get("message_id")
        if not isinstance(message_id, int):
            return
        note = {"approved": "Подтверждено", "declined": "Отклонено"}[request.decision]
        try:
            await provider.edit_text(f"{request.message}\n\n{note}", message_id=message_id)
        except Exception:  # noqa: BLE001 - the button was already spent
            log.debug("Could not edit the guest confirmation message", exc_info=True)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuestRecord:
    """What one confirmed guest registration wrote into the hub database."""

    person_id: str
    display_name: str
    home_id: str
    role: str = ROLE_GUEST
    voice_samples: int = 0
    face_samples: int = 0
    body_samples: int = 0
    faces_linked: int = 0
    bodies_linked: int = 0
    track_id: str = ""
    existed: bool = False

    def summary(self) -> dict[str, Any]:
        return {"person_id": self.person_id, "name": self.display_name, "home_id": self.home_id,
                "role": self.role, "voice_samples": self.voice_samples,
                "face_samples": self.face_samples, "body_samples": self.body_samples,
                "track_id": self.track_id, "existed": self.existed}


def membership_role(conn: sqlite3.Connection, person_id: str, home_id: str) -> str | None:
    """The role this person already has in this room, or ``None``."""
    if not person_id or not home_id:
        return None
    row = conn.execute("SELECT role FROM memberships WHERE person_id=? AND home_id=?",
                       (str(person_id), str(home_id))).fetchone()
    return str(row[0]) if row else None


def ensure_person(conn: sqlite3.Connection, display_name: str, *, home_id: str,
                  role: str = ROLE_GUEST, language: str | None = None,
                  settings: dict[str, Any] | None = None,
                  commit: bool = True) -> tuple[str, bool]:
    """Create or reuse the ``persons`` row and its membership of this room.

    An existing membership is never demoted: a person who is already a member
    stays one, and the caller is told the person existed (``True``) so it can
    decide what to do instead of silently overwriting a role.

    Names are matched with Python's ``casefold`` rather than SQL ``lower()``:
    SQLite only lowercases ASCII, so "Макс" and "макс" would otherwise become
    two different people. ``commit=False`` lets :func:`commit_guest` write the
    person and their vectors as one transaction.

    :returns: ``(person_id, existed)``.
    """
    name = clean_name(display_name)
    if name is None:
        raise ValueError("a guest needs a real name, not a placeholder")
    if conn.execute("SELECT 1 FROM homes WHERE home_id=?", (str(home_id),)).fetchone() is None:
        raise ValueError(f"unknown home {home_id!r}")
    wanted = name.casefold()
    row = next((str(person_id) for person_id, display_name
                in conn.execute("SELECT person_id, display_name FROM persons").fetchall()
                if str(display_name or "").casefold() == wanted), None)
    existed = row is not None
    person_id = str(row) if row else uuid.uuid4().hex
    if not existed:
        conn.execute("INSERT INTO persons(person_id, display_name, preferred_language,"
                     " settings_json) VALUES (?,?,?,?)",
                     (person_id, name, language or None,
                      _json(settings) if settings else "{}"))
    if membership_role(conn, person_id, str(home_id)) is None:
        conn.execute("INSERT INTO memberships(person_id, home_id, role, share_identity,"
                     " share_presence) VALUES (?,?,?,0,0)",
                     (person_id, str(home_id), str(role)))
    if commit:
        conn.commit()
    return person_id, existed


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def ensure_track(conn: sqlite3.Connection, track_id: str, *, home_id: str,
                 client_id: str | None = None, now: float | None = None,
                 commit: bool = True) -> bool:
    """Make sure ``track_id`` exists so a person can be attached to it."""
    if not track_id:
        return False
    if conn.execute("SELECT 1 FROM tracks WHERE track_id=?", (str(track_id),)).fetchone() is not None:
        return True
    if conn.execute("SELECT 1 FROM homes WHERE home_id=?", (str(home_id),)).fetchone() is None:
        raise ValueError(f"unknown home {home_id!r}")
    moment = time.monotonic() if now is None else float(now)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(moment))
    conn.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen,"
                 " sources_json) VALUES (?,?,?,?,?,'{}')",
                 (str(track_id), str(home_id), client_id, stamp, stamp))
    if commit:
        conn.commit()
    return True


def _store_vectors(conn: sqlite3.Connection, table: str, column: str, person_id: str,
                   vectors: Sequence[Any], *, track_id: str | None = None,
                   day: str | None = None, quality: float | None = None) -> int:
    """Insert the vectors of one modality; returns how many rows were written."""
    stored = 0
    for vector in vectors or ():
        values = [float(value) for value in _flatten(vector)]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"a {table} vector has to be a non-empty finite vector")
        columns = ["id", "person_id", "track_id", column, "dim", "quality"]
        params: list[Any] = [uuid.uuid4().hex, str(person_id), str(track_id or "") or None,
                             pack_vector(values), len(values), quality]
        if table == "body_embeddings":
            columns.insert(3, "session_day")
            params.insert(3, day)
        conn.execute(f"INSERT INTO {table}({','.join(columns)})"
                     f" VALUES ({','.join('?' * len(columns))})", params)
        stored += 1
    return stored


def _flatten(vector: Any) -> list[Any]:
    values = getattr(vector, "tolist", None)
    if callable(values):
        vector = values()
    if isinstance(vector, (list, tuple)) and vector and isinstance(vector[0], (list, tuple)):
        try:
            return [float(value) for row in vector for value in row]
        except (TypeError, ValueError) as exc:
            raise ValueError("a vector has to be numeric") from exc
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError("a vector has to be numeric") from exc


def commit_guest(conn: sqlite3.Connection, pending: PendingGuest, *,
                 voice_vectors: Sequence[Any] = (), face_vectors: Sequence[Any] = (),
                 body_vectors: Sequence[Any] = (), day: str | None = None,
                 track_id: str | None = None, home_id: str | None = None,
                 client_id: str | None = None, bodies: Any = None,
                 audit: Any = None, actor: str = "") -> GuestRecord:
    """Write one confirmed guest (ТЗ F-210) - the only place that stores them.

    Called exactly once, after the owner's button: it creates the person, the
    ``guest`` membership of this room, the voice/face/body vectors collected by
    the flow, and names the live track through the shared path of F-204/F-205
    (``hub/identity_link.py``), which carries the person over the track's faces
    and the body vectors of ``day``.
    """
    room = str(home_id or pending.home_id)
    room_track = str(track_id or pending.track_id or "")
    stored_day = str(day or pending.day or "") or None
    try:
        # The hub's own connection runs in autocommit mode, so the person, the
        # membership and every vector are wrapped in one explicit transaction:
        # a guest is written whole or not at all.
        if not getattr(conn, "in_transaction", False):
            conn.execute("BEGIN IMMEDIATE")
        person_id, existed = ensure_person(conn, pending.name, home_id=room,
                                           role=ROLE_GUEST, language=pending.language,
                                           commit=False)
        if room_track:
            ensure_track(conn, room_track, home_id=room, client_id=client_id, commit=False)
        voice_samples = _store_vectors(conn, "voice_embeddings", "vector", person_id,
                                       voice_vectors, track_id=room_track or None)
        face_samples = _store_vectors(conn, "face_embeddings", "vector", person_id,
                                      face_vectors, track_id=room_track or None)
        body_samples = _store_vectors(conn, "body_embeddings", "vector", person_id,
                                      body_vectors, track_id=room_track or None, day=stored_day)
        conn.execute("COMMIT")
    except (sqlite3.Error, ValueError):
        try:
            conn.rollback()
        except sqlite3.Error:  # noqa: PERF203 - a broken socket is not the error to report
            log.debug("Could not roll back the guest registration", exc_info=True)
        raise
    spread = None
    if room_track:
        spread = link_track_to_person(conn, track_id=room_track, person_id=person_id,
                                      day=stored_day, bodies=bodies, faces=True)
    record = GuestRecord(
        person_id=person_id, display_name=pending.name, home_id=room, role=ROLE_GUEST,
        voice_samples=voice_samples, face_samples=face_samples, body_samples=body_samples,
        faces_linked=int(getattr(spread, "faces_linked", 0) or 0),
        bodies_linked=int(getattr(spread, "bodies_linked", 0) or 0),
        track_id=room_track, existed=existed,
    )
    log.info("Guest %s (%s) registered in %s: %d voice, %d face, %d body vector(s)",
             pending.name, person_id, room, voice_samples, face_samples, body_samples)
    if audit is not None:
        try:
            audit.record(action="guest.register", actor=str(actor or pending.requested_by or ""),
                         target=person_id, home_id=room, result="ok",
                         detail={**record.summary(), "source": pending.source})
        except Exception:  # noqa: BLE001 - the record stands even if auditing fails
            log.warning("Could not audit the guest registration of %s", pending.name)
    return record


def revoke_guest(conn: sqlite3.Connection, person_id: str, *, home_id: str,
                 audit: Any = None, actor: str = "") -> bool:
    """Take the guest membership back (used when a registration is undone).

    Only the membership of one room is dropped - the person, their vectors and
    the other rooms they are a member of are F-213's "забудь меня", not this.
    """
    if not person_id or not home_id:
        return False
    removed = conn.execute("DELETE FROM memberships WHERE person_id=? AND home_id=?",
                           (str(person_id), str(home_id))).rowcount
    conn.commit()
    gone = bool(removed)
    if gone and audit is not None:
        try:
            audit.record(action="guest.revoke", actor=str(actor or ""), target=str(person_id),
                         home_id=str(home_id), result="ok", detail={"role": ROLE_GUEST})
        except Exception:  # noqa: BLE001
            log.warning("Could not audit revoking the guest %s", person_id)
    return gone


__all__ = [
    "CONFIRM_TTL_S",
    "CONSENT",
    "FACE_ATTEMPTS",
    "FLOW_TTL_S",
    "MAX_ANGLE_DEG",
    "MAX_FRAMES",
    "MIN_ANGLES",
    "MIN_FACE_HEIGHT",
    "MIN_FRAMES",
    "MIN_SHARPNESS",
    "MIN_VOICE_SECONDS",
    "PHRASES",
    "STEP_BODY",
    "STEP_CANCELLED",
    "STEP_CONSENT",
    "STEP_DECLINED",
    "STEP_DONE",
    "STEP_FACE",
    "STEP_OWNER",
    "STEP_VOICE",
    "TOKEN_PREFIX",
    "TURN_DEG",
    "VOICE_ATTEMPTS",
    "BurstQuality",
    "Declaration",
    "GuestConfirmation",
    "GuestRecord",
    "GuestRegistration",
    "OwnerConfirmations",
    "PendingGuest",
    "Shot",
    "assess_burst",
    "cancel_requested",
    "clean_name",
    "commit_guest",
    "consent_given",
    "consent_refused",
    "consent_request",
    "declaration",
    "ensure_person",
    "ensure_track",
    "face_prompt",
    "language_of",
    "membership_role",
    "owner_question",
    "phrase",
    "phrase_matches",
    "relation_word",
    "retry_hint",
    "revoke_guest",
    "shot_usable",
    "voice_prompt",
    "waiting_for_owner",
    "yaw_degrees",
]
