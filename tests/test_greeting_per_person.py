"""server/app.py: per-person greeting cooldowns (SPEC v1.7).

Rowan greets EVERY person the camera sees, not only strangers, and each
person carries their own clock: a familiar face is greeted again after
``greeting_cooldown_known_s``, a stranger after ``greeting_cooldown_s``.

Pure-python: a Connection is built with ``__new__`` and only the handful of
attributes the greeting gate reads are filled in, so none of this needs a
WebSocket, a camera, a face engine or a model.
"""
import time

from server.app import LABEL_UNKNOWN, Connection, PresenceTracker

GREET_AFTER_S = 10.0
UNKNOWN_COOLDOWN_S = 300.0  # 5 minutes
KNOWN_COOLDOWN_S = 900.0  # 15 minutes


def _conn(persons: int = 1) -> Connection:
    """A Connection carrying only what _greet_target reads."""
    conn = Connection.__new__(Connection)
    conn.presence = PresenceTracker(ttl_s=30.0)
    conn._greeted_at = {}
    conn._last_known_voice_at = 0.0
    conn.camera_state = {"persons": persons}
    return conn


def _stranger_waiting(conn: Connection, seconds: float) -> None:
    """Backdate the unknown face's first sighting so it has 'waited' that long."""
    conn.presence._seen[LABEL_UNKNOWN].first_seen -= seconds


def _target(conn: Connection) -> str | None:
    return conn._greet_target(GREET_AFTER_S, UNKNOWN_COOLDOWN_S, KNOWN_COOLDOWN_S)


# -- a familiar face ----------------------------------------------------------

def test_a_known_face_never_greeted_is_due():
    conn = _conn()
    conn.presence.note_faces(["Anton"])
    assert _target(conn) == "Anton"


def test_a_known_face_just_greeted_is_not_due_again():
    conn = _conn()
    conn.presence.note_faces(["Anton"])
    conn._greeted_at["Anton"] = time.monotonic()
    assert _target(conn) is None


def test_a_known_face_is_due_again_after_the_known_cooldown():
    conn = _conn()
    conn.presence.note_faces(["Anton"])
    conn._greeted_at["Anton"] = time.monotonic() - (KNOWN_COOLDOWN_S + 1.0)
    assert _target(conn) == "Anton"


def test_a_known_face_is_still_held_just_before_the_cooldown_expires():
    conn = _conn()
    conn.presence.note_faces(["Anton"])
    conn._greeted_at["Anton"] = time.monotonic() - (KNOWN_COOLDOWN_S - 30.0)
    assert _target(conn) is None


def test_the_person_greeted_longest_ago_goes_first():
    conn = _conn(persons=2)
    conn.presence.note_faces(["Anton", "Bob"])
    now = time.monotonic()
    conn._greeted_at["Anton"] = now - (KNOWN_COOLDOWN_S + 60.0)
    conn._greeted_at["Bob"] = now - (KNOWN_COOLDOWN_S + 600.0)
    assert _target(conn) == "Bob"


def test_a_person_who_was_never_greeted_outranks_one_who_was():
    conn = _conn(persons=2)
    conn.presence.note_faces(["Anton", "Bob"])
    conn._greeted_at["Anton"] = time.monotonic() - (KNOWN_COOLDOWN_S + 60.0)
    assert _target(conn) == "Bob"


def test_a_zero_cooldown_means_always_due():
    conn = _conn()
    conn.presence.note_faces(["Anton"])
    conn._greeted_at["Anton"] = time.monotonic()
    assert conn._greet_target(GREET_AFTER_S, UNKNOWN_COOLDOWN_S, 0.0) == "Anton"


# -- a stranger ---------------------------------------------------------------

def test_a_stranger_who_has_waited_long_enough_outranks_a_due_known_face():
    conn = _conn()
    conn.presence.note_faces(["Anton", LABEL_UNKNOWN])
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    assert _target(conn) == LABEL_UNKNOWN


def test_a_stranger_who_just_arrived_does_not_jump_the_queue():
    # Not waited greet_after_s yet: Anton, who is due, is greeted instead.
    conn = _conn()
    conn.presence.note_faces(["Anton", LABEL_UNKNOWN])
    assert _target(conn) == "Anton"


def test_the_same_stranger_is_not_greeted_twice_inside_the_cooldown():
    conn = _conn()
    conn.presence.note_faces([LABEL_UNKNOWN])
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    conn._greeted_at[LABEL_UNKNOWN] = time.monotonic()
    assert _target(conn) is None


def test_a_stranger_is_due_again_after_the_shorter_unknown_cooldown():
    conn = _conn()
    conn.presence.note_faces([LABEL_UNKNOWN])
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    conn._greeted_at[LABEL_UNKNOWN] = time.monotonic() - (UNKNOWN_COOLDOWN_S + 1.0)
    assert _target(conn) == LABEL_UNKNOWN


def test_a_known_voice_moments_ago_still_suppresses_a_lone_unknown_face():
    # The bad-angle case: one body, one face, nobody extra - that "stranger"
    # is almost certainly the man who just spoke.
    conn = _conn()
    conn.presence.note_faces(["Anton"])
    conn.presence.note_faces([LABEL_UNKNOWN])
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    conn._last_known_voice_at = time.monotonic()
    conn._greeted_at["Anton"] = time.monotonic()
    assert _target(conn) is None


def test_a_known_voice_does_not_suppress_a_second_face_in_the_same_frame():
    # One body, two faces: the owner holding a stranger's photo to the camera.
    conn = _conn()
    conn.presence.note_faces(["Anton", LABEL_UNKNOWN])
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    conn._last_known_voice_at = time.monotonic()
    conn._greeted_at["Anton"] = time.monotonic()
    assert _target(conn) == LABEL_UNKNOWN


def test_nobody_in_the_room_means_nobody_to_greet():
    conn = _conn(persons=0)
    assert _target(conn) is None
