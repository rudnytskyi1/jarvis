"""Слияние голоса, лица и тела в одного человека (ТЗ F-206).

Every second the hub asks one question per live track: WHO is this body? Three
signals can answer it, each with its own bar from the ТЗ:

* **face** - cosine ≥ 0.45 (``server.face.threshold``);
* **voice** - cosine ≥ 0.40 AND at least 0.15 ahead of the runner-up, because
  a voice that is only slightly closer to one profile than another is not
  evidence at all;
* **body** (ReID) - cosine ≥ ``server.identity.reid.threshold``, and only
  within the SAME day: clothes change overnight (ТЗ F-206 says so outright,
  and :func:`body_signal` therefore never looks at another day).

Each signal becomes a confidence ``c = (score - bar) / (1 - bar)`` in ``[0, 1]``
and the confidences of one person combine with ``p = 1 - Π(1 - c)``: one good
face is enough to believe (0.45-threshold face at 0.90 is p ≈ 0.82), two weak
signals reinforce each other, and nothing ever exceeds 1. When the two best
people are within :data:`AMBIGUITY_MARGIN` the signals are a coin toss, and the
question goes to D-06 (``hub/decision_points.py::identity_heuristic``): the
context of who lives here and who is already in the room decides, or nobody is
named.

The result is one row per track in ``identity_belief`` - ``(track_id,
person_id, p, sources)`` in the words of the ТЗ - which is what F-207's
hysteresis reads and what the explainability of F-215 quotes.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from hub.decision_points import identity_heuristic

log = logging.getLogger("jarvis.server.identity_fusion")

#: The bars of ТЗ F-206, per signal kind.
FACE_THRESHOLD = 0.45
VOICE_THRESHOLD = 0.40
BODY_THRESHOLD = 0.5
#: A voice has to beat the next-best PERSON by this much (ТЗ F-206).
VOICE_MARGIN = 0.15
#: Two candidates closer than this are a coin toss for the signals alone.
AMBIGUITY_MARGIN = 0.2
#: ТЗ F-206: the fusion runs for every track once a second.
FUSION_INTERVAL_S = 1.0

KINDS = ("face", "voice", "body")

#: ТЗ F-207: "узнан" at p ≥ 0.8 for two passes in a row, "потерян" at p < 0.5
#: for three in a row. Two runs of a strong claim are what keeps one lucky
#: frame from naming a stranger; three weak ones are what keeps a single bad
#: pass from losing somebody who is still standing there.
RECOGNIZE_P = 0.8
LOSE_P = 0.5
RECOGNIZE_RUNS = 2
LOSE_RUNS = 3

#: ТЗ F-208: a privileged action needs the voice at least this confident AND a
#: second factor - the face at ≥ 0.55, or the body of the same day already
#: linked to that face. A phone has no camera, so it carries its own bar and a
#: spoken PIN instead.
ADMIN_VOICE_THRESHOLD = 0.65
ADMIN_FACE_THRESHOLD = 0.55
PHONE_ADMIN_THRESHOLD = 0.65

#: The spoken PIN: 4-8 digits, three tries, then a five-minute lockout.
PIN_MIN_DIGITS = 4
PIN_MAX_DIGITS = 8
PIN_MAX_FAILURES = 3
PIN_LOCKOUT_S = 300.0


@dataclass(frozen=True)
class Signal:
    """One piece of evidence that a track is a person (ТЗ F-206)."""

    kind: str
    person_id: str
    score: float

    def confidence(self, threshold: float) -> float:
        """How far past its bar this signal is, as ``0…1``."""
        span = 1.0 - float(threshold)
        if span <= 0.0:
            return 1.0 if self.score >= threshold else 0.0
        return max(0.0, min(1.0, (float(self.score) - float(threshold)) / span))


@dataclass(frozen=True)
class Belief:
    """The fusion's answer for one track (ТЗ F-206)."""

    track_id: str
    person_id: str | None
    p: float
    sources: dict[str, Any] = field(default_factory=dict)
    at: float = 0.0

    def explains(self) -> str:
        """The sentence F-215 wants: "почему ты решил, что это Макс"."""
        parts = [f"{kind} {self.sources[kind]:.2f}" for kind in KINDS
                 if isinstance(self.sources.get(kind), (int, float))]
        if self.sources.get("ambiguous"):
            parts.append("ambiguous")
        if self.sources.get("context"):
            parts.append(f"context: {self.sources['context']}")
        if not parts:
            parts.append("no signal")
        return "; ".join(parts)


def thresholds_for(cfg: Any = None) -> dict[str, float]:
    """The three bars, taken from the config when it is there (ТЗ F-206)."""
    face_cfg = getattr(getattr(cfg, "face", None), "face", None) or getattr(cfg, "face", None)
    reid_cfg = getattr(getattr(cfg, "identity", None), "reid", None)
    return {
        "face": float(getattr(face_cfg, "threshold", FACE_THRESHOLD)),
        "voice": float(getattr(cfg, "speaker_threshold", VOICE_THRESHOLD)),
        "body": float(getattr(reid_cfg, "threshold", BODY_THRESHOLD)),
    }


def _voice_kept(signals: Sequence[Signal], *, threshold: float, margin: float) -> list[Signal]:
    """Drop voices that do not beat the runner-up PERSON by ``margin`` (F-206)."""
    by_person: dict[str, float] = {}
    for signal in signals:
        if signal.score >= threshold:
            by_person[signal.person_id] = max(by_person.get(signal.person_id, -1.0), signal.score)
    if not by_person:
        return []
    ranked = sorted(by_person.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
        return []
    return [Signal("voice", ranked[0][0], ranked[0][1])]


def fuse(*, track_id: str, signals: Iterable[Signal] | None, expected: Sequence[str] | None = None,
         present: Sequence[str] | None = None, thresholds: dict[str, float] | None = None,
         ambiguity: float = AMBIGUITY_MARGIN,
         decide: Callable[..., tuple[str | None, str]] = identity_heuristic,
         now: float | None = None) -> Belief:
    """Turn the signals of one track into one belief (ТЗ F-206).

    ``decide`` is D-06: by default the context rule of
    ``hub/decision_points.py``, and the hub may hand in the decider chain's
    answer instead. The belief carries every number it used, so F-215 can
    explain it without guessing.
    """
    bars = dict(thresholds_for() if thresholds is None else thresholds)
    at = time.time() if now is None else float(now)
    raw = [signal for signal in (signals or ())
           if str(signal.kind) in KINDS and str(signal.person_id or "")]
    # The voice bar has a second half - the lead over the runner-up PERSON
    # (ТЗ F-206) - and that can only be judged with every voice in hand.
    kept = _voice_kept([signal for signal in raw if str(signal.kind) == "voice"],
                       threshold=bars["voice"], margin=VOICE_MARGIN)
    for signal in raw:
        kind = str(signal.kind)
        if kind != "voice" and float(signal.score) >= float(bars[kind]):
            kept.append(Signal(kind, str(signal.person_id), float(signal.score)))
    if not kept:
        return Belief(track_id=str(track_id), person_id=None, p=0.0,
                      sources={"reason": "no signal past its threshold"}, at=at)
    per_person: dict[str, dict[str, float]] = {}
    best_score: dict[str, dict[str, float]] = {}
    for signal in kept:
        person = str(signal.person_id)
        confidence = signal.confidence(float(bars[signal.kind]))
        kinds = per_person.setdefault(person, {})
        kinds[signal.kind] = max(kinds.get(signal.kind, 0.0), confidence)
        scores = best_score.setdefault(person, {})
        scores[signal.kind] = max(scores.get(signal.kind, 0.0), float(signal.score))
    probabilities = {}
    for person, kinds in per_person.items():
        keep_out = 1.0
        for confidence in kinds.values():
            keep_out *= 1.0 - confidence
        probabilities[person] = 1.0 - keep_out
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    best_person, best_p = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    sources: dict[str, Any] = {kind: round(score, 3)
                               for kind, score in best_score[best_person].items()}
    sources["p"] = round(best_p, 3)
    sources["lead"] = round(best_p - runner_up, 3)
    ambiguous = len(ranked) > 1 and (best_p - runner_up) < float(ambiguity)
    if ambiguous:
        # The signals cannot separate these people; D-06 decides with the
        # context (ТЗ F-206), and "nobody" is a perfectly good answer.
        chosen, reason = decide(ranked, expected=expected, present=present,
                                margin=float(ambiguity))
        sources["ambiguous"] = [{"person_id": person, "p": round(p, 3)}
                                for person, p in ranked[:3]]
        if chosen is None:
            sources["reason"] = reason
            return Belief(track_id=str(track_id), person_id=None, p=float(best_p),
                          sources=sources, at=at)
        if str(chosen) != best_person:
            best_person, best_p = str(chosen), float(dict(ranked)[str(chosen)])
            sources.update({kind: round(score, 3)
                            for kind, score in best_score[best_person].items()})
            sources["p"] = round(best_p, 3)
        sources["context"] = "expected here / already in the room"
    return Belief(track_id=str(track_id), person_id=best_person, p=float(best_p),
                  sources=sources, at=at)


def body_signal(bodies: Any, *, track_id: str, day: str | None = None) -> Signal | None:
    """The ReID signal of a track, from its stored body vectors of THAT day.

    The day is not a detail: ТЗ F-206 allows a body match only inside the day
    the vector was recorded, because the same person wears different clothes
    tomorrow. ``bodies`` is a ``hub.reid.BodyEmbeddingStore``; a track with no
    vector yet, or with nobody else looking like it, yields no signal.
    """
    if bodies is None:
        return None
    try:
        vectors = bodies.for_track(str(track_id), day=day, limit=1)
        if not vectors:
            return None
        person_id, score = bodies.match(vectors[0].as_array(), day=day)
    except Exception as exc:  # noqa: BLE001 - a missing signal is never a failure
        log.debug("No body signal for track %s (%s)", track_id, exc)
        return None
    if not person_id:
        return None
    return Signal("body", str(person_id), float(score))


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


@dataclass
class _TrackState:
    """What the hysteresis remembers about one track (ТЗ F-207)."""

    candidate: str | None = None
    runs: int = 0
    recognized: str | None = None
    lost_runs: int = 0
    greeted: bool = False


class IdentityHysteresis:
    """«Узнан» дважды подряд, «потерян» трижды (ТЗ F-207).

    The fusion of F-206 answers every second, and a single pass is not a
    decision: one lucky frame must not name a stranger, and one bad frame must
    not lose somebody who is still standing there. This class keeps the run
    counts per track, so the confirmed identity changes only when the evidence
    has held for long enough. It also owns the visit flag behind "приветствие —
    один раз на трек за визит": the greeting itself is F-302's job (the ТЗ
    puts it there), and it asks this class before speaking.
    """

    def __init__(self, *, recognize_p: float = RECOGNIZE_P, lose_p: float = LOSE_P,
                 recognize_runs: int = RECOGNIZE_RUNS, lose_runs: int = LOSE_RUNS) -> None:
        self.recognize_p = float(recognize_p)
        self.lose_p = float(lose_p)
        self.recognize_runs = max(1, int(recognize_runs))
        self.lose_runs = max(1, int(lose_runs))
        self._tracks: dict[str, _TrackState] = {}

    def observe(self, track_id: str, belief: Belief) -> str:
        """Feed one belief; returns what the hysteresis decided.

        ``recognized`` - this track IS that person now (the run of strong
        passes completed); ``lost`` - the person is gone; ``pending`` - the
        claim is being counted; ``fading`` - the losing run is being counted;
        ``unchanged`` - nothing to decide from this pass.
        """
        key = str(track_id)
        state = self._tracks.setdefault(key, _TrackState())
        person = None if belief.person_id is None else str(belief.person_id)
        p = float(belief.p)
        if person is not None and p >= self.recognize_p:
            if state.candidate == person:
                state.runs += 1
            else:
                state.candidate, state.runs = person, 1
            state.lost_runs = 0
            if state.runs >= self.recognize_runs and state.recognized != person:
                if state.recognized is not None and state.recognized != person:
                    # Somebody else's greeting does not cover this person.
                    state.greeted = False
                state.recognized = person
                return "recognized"
            return "pending" if state.recognized != person else "unchanged"
        state.candidate, state.runs = None, 0
        if state.recognized is not None and p < self.lose_p:
            state.lost_runs += 1
            if state.lost_runs >= self.lose_runs:
                state.recognized = None
                state.lost_runs = 0
                return "lost"
            return "fading"
        state.lost_runs = 0
        return "unchanged"

    def recognized(self, track_id: str) -> str | None:
        """The person the hysteresis has confirmed for this track, if any."""
        state = self._tracks.get(str(track_id))
        return state.recognized if state is not None else None

    def should_greet(self, track_id: str) -> bool:
        """True once per track per visit (ТЗ F-207), and marks it as greeted.

        Only a track that already has a confirmed person asks this; the answer
        is consumed here, so two callers can never greet the same visit twice.
        """
        state = self._tracks.get(str(track_id))
        if state is None or state.recognized is None or state.greeted:
            return False
        state.greeted = True
        return True

    def greeted(self, track_id: str) -> bool:
        state = self._tracks.get(str(track_id))
        return bool(state.greeted) if state is not None else False

    def forget(self, track_id: str) -> None:
        """Drop a track that left: a new visit starts its own counts."""
        self._tracks.pop(str(track_id), None)

    def live(self) -> list[str]:
        return list(self._tracks)


@dataclass(frozen=True)
class AdminDecision:
    """ТЗ F-208's answer about one privileged call."""

    allowed: bool
    code: str
    detail: str = ""
    factors: dict[str, Any] = field(default_factory=dict)


def admin_gate(*, person_id: str | None, voice_score: float | None,
               face_score: float | None = None, body_linked_today: bool = False,
               channel: str = "room", pin_ok: bool = False,
               voice_threshold: float = ADMIN_VOICE_THRESHOLD,
               face_threshold: float = ADMIN_FACE_THRESHOLD,
               phone_threshold: float = PHONE_ADMIN_THRESHOLD) -> AdminDecision:
    """ТЗ F-208: may this person run a privileged call right now?

    The room asks for the voice at ``voice_threshold`` AND one more witness: a
    face at least ``face_threshold`` confident, or the body of the same day
    already linked to that person (F-204/F-205 did that linking, so "this body
    is theirs" is not an assumption made here). A phone has no camera, so it
    asks for its own bar plus the spoken PIN.

    The answer carries a short code (``ok``, ``no_person``, ``voice``, ``pin``,
    ``second_factor``) so the caller can act on it - ask for the PIN, say the
    denial - instead of parsing a sentence.
    """
    factors: dict[str, Any] = {
        "channel": str(channel),
        "voice": None if voice_score is None else round(float(voice_score), 3),
        "face": None if face_score is None else round(float(face_score), 3),
        "body_of_day": bool(body_linked_today),
        "pin": bool(pin_ok),
    }
    if not person_id:
        return AdminDecision(False, "no_person", "nobody is identified for this action", factors)
    bar = float(phone_threshold) if str(channel) == "phone" else float(voice_threshold)
    factors["voice_threshold"] = bar
    if voice_score is None or float(voice_score) < bar:
        heard = 0.0 if voice_score is None else float(voice_score)
        return AdminDecision(
            False, "voice",
            f"this action needs a confident voice match (at least {bar:.2f}), "
            f"but the voice matched only {heard:.2f}",
            factors)
    if str(channel) == "phone":
        if not pin_ok:
            return AdminDecision(False, "pin", "this action needs the spoken PIN", factors)
        return AdminDecision(True, "ok", "voice and PIN", factors)
    if face_score is not None and float(face_score) >= float(face_threshold):
        return AdminDecision(True, "ok", "voice and face", factors)
    if body_linked_today:
        return AdminDecision(True, "ok", "voice and the body of today", factors)
    return AdminDecision(
        False, "second_factor",
        f"this action needs the voice at {bar:.2f} AND the face (at least "
        f"{float(face_threshold):.2f}) or the body of today - ask the person to "
        "look at the camera for a moment",
        factors)


#: The digits of the three languages of the house, for a spoken PIN.
DIGIT_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ноль": "0", "нуль": "0", "один": "1", "одна": "1", "два": "2", "две": "2",
    "три": "3", "четыре": "4", "пять": "5", "шесть": "6", "семь": "7",
    "восемь": "8", "девять": "9",
    "cero": "0", "uno": "1", "una": "1", "dos": "2", "tres": "3", "cuatro": "4",
    "cinco": "5", "seis": "6", "siete": "7", "ocho": "8", "nueve": "9",
}


@dataclass
class _PinState:
    failures: int = 0
    locked_until: float = 0.0


class VoicePin:
    """The spoken PIN of ТЗ F-208, stored as a salted hash in ``persons.settings_json``.

    A PIN is a secret, so only a PBKDF2 hash with a random salt is written down
    - never the digits themselves, and never into a log. Three wrong tries lock
    the person out for five minutes; a correct one clears the counter. The
    lockout lives in memory: a restart forgets it, which is the honest trade
    for not writing a security counter into a dorm hub's database.
    """

    def __init__(self, *, max_failures: int = PIN_MAX_FAILURES,
                 lockout_s: float = PIN_LOCKOUT_S, min_digits: int = PIN_MIN_DIGITS,
                 max_digits: int = PIN_MAX_DIGITS) -> None:
        self.max_failures = max(1, int(max_failures))
        self.lockout_s = max(0.0, float(lockout_s))
        self.min_digits = max(1, int(min_digits))
        self.max_digits = max(self.min_digits, int(max_digits))
        self._state: dict[str, _PinState] = {}

    # ---------------------------------------------------------------- digits

    def digits(self, text: Any) -> str:
        """The PIN spoken in ``text``: digits and digit words, in order.

        Whisper usually writes numbers as digits; when somebody says "один два
        три четыре" it may write the words instead, so the three languages of
        the house are understood too. Everything else is dropped.
        """
        found: list[str] = []
        for word in str(text or "").replace(",", " ").split():
            token = word.strip(".,!?;:\"'()[]").casefold()
            if not token:
                continue
            if token.isdigit():
                found.append(token)
                continue
            digit = DIGIT_WORDS.get(token)
            if digit is not None:
                found.append(digit)
        return "".join(found)

    def valid(self, pin: Any) -> bool:
        text = str(pin or "")
        return text.isdigit() and self.min_digits <= len(text) <= self.max_digits

    # ---------------------------------------------------------------- storage

    def hash(self, pin: str) -> str:
        """A salted PBKDF2 hash of ``pin`` (the only form ever written down)."""
        text = str(pin or "")
        if not self.valid(text):
            raise ValueError(f"a PIN is {self.min_digits} to {self.max_digits} digits")
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", text.encode("utf-8"), salt, 120_000)
        return f"pbkdf2${salt.hex()}${digest.hex()}"

    def verify_hash(self, pin: str, stored: str | None) -> bool:
        """True when ``pin`` matches the stored hash (constant-time compare)."""
        if not stored or not str(stored).startswith("pbkdf2$"):
            return False
        try:
            _prefix, salt_hex, digest_hex = str(stored).split("$", 2)
            expected = bytes.fromhex(digest_hex)
            digest = hashlib.pbkdf2_hmac("sha256", str(pin or "").encode("utf-8"),
                                         bytes.fromhex(salt_hex), 120_000)
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(digest, expected)

    def stored_for(self, conn: sqlite3.Connection, person_id: str) -> str | None:
        """The stored hash of a person's PIN, or ``None`` when none is set."""
        row = conn.execute("SELECT settings_json FROM persons WHERE person_id=?",
                           (str(person_id),)).fetchone()
        if row is None:
            return None
        try:
            settings = json.loads(row[0] or "{}")
        except (TypeError, ValueError):
            return None
        value = settings.get("admin_pin") if isinstance(settings, dict) else None
        return str(value) if value else None

    def set_pin(self, conn: sqlite3.Connection, person_id: str, pin: Any) -> bool:
        """Store a new PIN for a person; ``False`` when it is not a PIN."""
        text = str(pin or "")
        if not self.valid(text):
            return False
        row = conn.execute("SELECT settings_json FROM persons WHERE person_id=?",
                           (str(person_id),)).fetchone()
        if row is None:
            return False
        try:
            settings = json.loads(row[0] or "{}")
        except (TypeError, ValueError):
            settings = {}
        if not isinstance(settings, dict):
            settings = {}
        settings["admin_pin"] = self.hash(text)
        conn.execute("UPDATE persons SET settings_json=? WHERE person_id=?",
                     (json.dumps(settings, ensure_ascii=False), str(person_id)))
        conn.commit()
        self._state.pop(str(person_id), None)
        return True

    # ---------------------------------------------------------------- checking

    def failures(self, person_id: str) -> int:
        state = self._state.get(str(person_id))
        return state.failures if state is not None else 0

    def locked(self, person_id: str, *, now: float | None = None) -> bool:
        state = self._state.get(str(person_id))
        if state is None:
            return False
        moment = time.monotonic() if now is None else float(now)
        return state.locked_until > moment

    def verify(self, conn: sqlite3.Connection, person_id: str, said: Any, *,
               now: float | None = None) -> tuple[bool, str]:
        """``(ok, reason)`` for one spoken PIN.

        Reasons: ``ok``, ``locked``, ``no_pin`` (nobody ever set one, so a phone
        cannot run admin actions at all), ``not_a_pin``, ``wrong``.
        """
        key = str(person_id)
        moment = time.monotonic() if now is None else float(now)
        if self.locked(key, now=moment):
            return False, "locked"
        stored = self.stored_for(conn, key)
        if stored is None:
            return False, "no_pin"
        pin = self.digits(said)
        if not self.valid(pin):
            self._fail(key, moment)
            return False, "not_a_pin"
        if self.verify_hash(pin, stored):
            self._state.pop(key, None)
            return True, "ok"
        self._fail(key, moment)
        return False, "wrong"

    def _fail(self, person_id: str, now: float) -> None:
        state = self._state.setdefault(str(person_id), _PinState())
        state.failures += 1
        if state.failures >= self.max_failures:
            state.locked_until = now + self.lockout_s
            log.warning("The spoken PIN of %s is locked for %.0f s after %d tries",
                        person_id, self.lockout_s, state.failures)


class BeliefStore:
    """``identity_belief`` rows: one current belief per track (ТЗ F-206)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, belief: Belief, *, home_id: str = "", client_id: str = "") -> bool:
        """Store (or replace) the belief of one track."""
        try:
            self._ensure_track(str(belief.track_id), home_id=home_id, client_id=client_id,
                               ts=belief.at)
            person = self._existing_person(belief.person_id)
            self._conn.execute(
                "INSERT INTO identity_belief(track_id, home_id, person_id, p, sources_json, at)"
                " VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(track_id) DO UPDATE SET person_id=excluded.person_id,"
                " p=excluded.p, sources_json=excluded.sources_json, at=excluded.at,"
                " home_id=COALESCE(NULLIF(excluded.home_id, ''), identity_belief.home_id)",
                (str(belief.track_id), str(home_id), person, float(belief.p),
                 json.dumps(belief.sources or {}, ensure_ascii=False, allow_nan=False), float(belief.at)),
            )
            self._conn.commit()
        except (sqlite3.Error, ValueError) as exc:
            log.warning("Could not store the belief of track %s (%s)", belief.track_id, exc)
            return False
        return True

    def get(self, track_id: str) -> Belief | None:
        """The current belief of one track, or ``None`` when it has none."""
        row = self._conn.execute(
            "SELECT track_id, person_id, p, sources_json, at FROM identity_belief"
            " WHERE track_id=?", (str(track_id),)).fetchone()
        return self._row(row)

    def for_home(self, home_id: str, *, min_p: float = 0.0) -> list[Belief]:
        """Every belief of one home, strongest first (the presence view)."""
        rows = self._conn.execute(
            "SELECT track_id, person_id, p, sources_json, at FROM identity_belief"
            " WHERE home_id=? AND p>=? ORDER BY p DESC", (str(home_id), float(min_p))).fetchall()
        return [belief for belief in (self._row(row) for row in rows) if belief is not None]

    def forget(self, track_id: str) -> int:
        """Drop the belief of a track that left the room."""
        cursor = self._conn.execute("DELETE FROM identity_belief WHERE track_id=?",
                                    (str(track_id),))
        self._conn.commit()
        return int(cursor.rowcount or 0)

    def count(self, *, home_id: str | None = None) -> int:
        if home_id is None:
            row = self._conn.execute("SELECT COUNT(*) FROM identity_belief").fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM identity_belief WHERE home_id=?",
                                     (str(home_id),)).fetchone()
        return int(row[0]) if row else 0

    def _row(self, row: tuple[Any, ...] | None) -> Belief | None:
        if row is None:
            return None
        try:
            sources = json.loads(row[3] or "{}")
        except (TypeError, ValueError):
            sources = {}
        return Belief(track_id=str(row[0]), person_id=None if row[1] is None else str(row[1]),
                      p=float(row[2]), sources=sources if isinstance(sources, dict) else {},
                      at=float(row[4]))

    def _existing_person(self, person_id: str | None) -> str | None:
        if not person_id:
            return None
        row = self._conn.execute("SELECT 1 FROM persons WHERE person_id=?",
                                 (str(person_id),)).fetchone()
        if row is None:
            log.info("Belief of a track: person %s is not in persons - stored without a link",
                     person_id)
            return None
        return str(person_id)

    def _ensure_track(self, track_id: str, *, home_id: str, client_id: str,
                      ts: float | None = None) -> None:
        """The ``tracks`` row a belief hangs off; raises when its home is unknown."""
        if self._conn.execute("SELECT 1 FROM tracks WHERE track_id=?",
                              (str(track_id),)).fetchone() is not None:
            return
        if not home_id or self._conn.execute("SELECT 1 FROM homes WHERE home_id=?",
                                             (str(home_id),)).fetchone() is None:
            raise ValueError(f"track {track_id} is new and home {home_id!r} is unknown")
        moment = datetime.fromtimestamp(float(ts), tz=UTC).isoformat(timespec="seconds") \
            if ts is not None else datetime.now(UTC).isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen)"
            " VALUES (?,?,?,?,?)",
            (str(track_id), str(home_id), str(client_id), moment, moment),
        )
        self._conn.commit()


__all__ = [
    "ADMIN_FACE_THRESHOLD",
    "ADMIN_VOICE_THRESHOLD",
    "AMBIGUITY_MARGIN",
    "BODY_THRESHOLD",
    "DIGIT_WORDS",
    "FACE_THRESHOLD",
    "FUSION_INTERVAL_S",
    "KINDS",
    "LOSE_P",
    "LOSE_RUNS",
    "PHONE_ADMIN_THRESHOLD",
    "PIN_LOCKOUT_S",
    "PIN_MAX_FAILURES",
    "PIN_MIN_DIGITS",
    "RECOGNIZE_P",
    "RECOGNIZE_RUNS",
    "VOICE_MARGIN",
    "VOICE_THRESHOLD",
    "AdminDecision",
    "Belief",
    "BeliefStore",
    "IdentityHysteresis",
    "Signal",
    "VoicePin",
    "admin_gate",
    "body_signal",
    "fuse",
    "thresholds_for",
]
