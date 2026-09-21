"""server/app.py: per-person greetings driven by ABSENCE (SPEC v1.7).

Rowan greets every person the camera sees, not only strangers, and the rule
is how long since he LAST SAW them: a familiar face gone for
``greeting_cooldown_known_s`` and a stranger gone for ``greeting_cooldown_s``
are welcomed back when they turn up again. Somebody who never left is never
greeted twice.

Pure-python: a Connection is built with ``__new__`` and only the handful of
attributes the greeting gate reads are filled in, so none of this needs a
WebSocket, a camera, a face engine or a model.
"""
import time
from types import SimpleNamespace

from hub.app import LABEL_UNKNOWN, Connection, PresenceTracker

GREET_AFTER_S = 10.0
UNKNOWN_GAP_S = 300.0  # 5 minutes away
KNOWN_GAP_S = 900.0  # 15 minutes away


def _conn(
    persons: int = 1,
    unknown_gap_s: float = UNKNOWN_GAP_S,
    known_gap_s: float = KNOWN_GAP_S,
) -> Connection:
    """A Connection carrying only what the greeting gate reads.

    The config is real enough for _greet_config() to read the thresholds back
    out of it, so these tests also cover that plumbing.
    """
    conn = Connection.__new__(Connection)
    conn.presence = PresenceTracker(ttl_s=30.0)
    conn._last_seen_at = {}
    conn._due_greeting = set()
    conn._last_known_voice_at = 0.0
    conn.camera_state = {"persons": persons}
    conn.cfg = SimpleNamespace(
        server=SimpleNamespace(
            face=SimpleNamespace(
                greet_after_s=GREET_AFTER_S,
                greeting_cooldown_s=unknown_gap_s,
                greeting_cooldown_known_s=known_gap_s,
            )
        )
    )
    return conn


def _see(conn: Connection, *labels: str) -> None:
    """One presence burst, exactly as _match_presence feeds it."""
    conn.note_sightings(list(labels))
    conn.presence.note_faces(list(labels))


def _went_away(conn: Connection, label: str, seconds: float) -> None:
    """Backdate the last sighting of ``label`` so it reads as an absence."""
    conn._last_seen_at[label] -= seconds


def _stranger_waiting(conn: Connection, seconds: float) -> None:
    """Backdate the unknown face's arrival so it has 'waited' that long in frame."""
    conn.presence._seen[LABEL_UNKNOWN].first_seen -= seconds


def _target(conn: Connection) -> str | None:
    return conn._greet_target(GREET_AFTER_S, UNKNOWN_GAP_S, KNOWN_GAP_S)


# -- a familiar face ----------------------------------------------------------

def test_a_face_seen_for_the_first_time_is_greeted():
    conn = _conn()
    _see(conn, "Anton")
    assert _target(conn) == "Anton"


def test_somebody_who_never_left_is_not_greeted_again():
    conn = _conn()
    _see(conn, "Anton")
    conn._due_greeting.discard("Anton")  # the first hello was spoken
    for _ in range(5):  # he just keeps sitting there
        _see(conn, "Anton")
    assert _target(conn) is None


def test_coming_back_after_the_known_gap_earns_a_hello():
    conn = _conn()
    _see(conn, "Anton")
    conn._due_greeting.discard("Anton")
    _went_away(conn, "Anton", KNOWN_GAP_S + 1.0)
    _see(conn, "Anton")
    assert _target(conn) == "Anton"


def test_a_short_absence_does_not_earn_a_hello():
    conn = _conn()
    _see(conn, "Anton")
    conn._due_greeting.discard("Anton")
    _went_away(conn, "Anton", KNOWN_GAP_S - 60.0)
    _see(conn, "Anton")
    assert _target(conn) is None


def test_a_stranger_uses_the_shorter_gap_than_a_known_face():
    # 6 minutes away: past the 5-minute stranger gap, short of the 15-minute one.
    conn = _conn()
    _see(conn, "Anton", LABEL_UNKNOWN)
    conn._due_greeting.clear()
    for label in ("Anton", LABEL_UNKNOWN):
        _went_away(conn, label, 360.0)
    _see(conn, "Anton", LABEL_UNKNOWN)
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    assert LABEL_UNKNOWN in conn._due_greeting
    assert "Anton" not in conn._due_greeting


def test_the_person_away_longest_is_welcomed_first():
    conn = _conn(persons=2)
    _see(conn, "Anton", "Bob")
    conn._due_greeting.clear()
    _went_away(conn, "Anton", KNOWN_GAP_S + 60.0)
    _went_away(conn, "Bob", KNOWN_GAP_S + 600.0)
    _see(conn, "Anton", "Bob")
    # Both are due; Bob's previous sighting is the older one.
    conn._last_seen_at["Bob"] -= 500.0
    assert _target(conn) == "Bob"


def test_a_zero_gap_means_greet_on_every_sighting():
    conn = _conn(unknown_gap_s=0.0, known_gap_s=0.0)
    _see(conn, "Anton")
    conn._due_greeting.discard("Anton")
    conn.note_sightings(["Anton"])
    assert "Anton" in conn._due_greeting


# -- a stranger ---------------------------------------------------------------

def test_a_stranger_must_still_linger_before_being_greeted():
    conn = _conn()
    _see(conn, LABEL_UNKNOWN)
    assert LABEL_UNKNOWN in conn._due_greeting  # latched on arrival
    assert _target(conn) is None  # but has not been in frame long enough


def test_a_stranger_who_has_lingered_outranks_a_due_known_face():
    conn = _conn()
    _see(conn, "Anton", LABEL_UNKNOWN)
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    assert _target(conn) == LABEL_UNKNOWN


def test_a_known_voice_moments_ago_still_suppresses_a_lone_unknown_face():
    # The bad-angle case: one body, one face, nobody extra - that "stranger"
    # is almost certainly the man who just spoke.
    conn = _conn()
    _see(conn, "Anton")
    _see(conn, LABEL_UNKNOWN)
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    conn._last_known_voice_at = time.monotonic()
    conn._due_greeting.discard("Anton")
    assert _target(conn) is None


def test_a_known_voice_does_not_suppress_a_second_face_in_the_same_frame():
    # One body, two faces: the owner holding a stranger's photo to the camera.
    conn = _conn()
    _see(conn, "Anton", LABEL_UNKNOWN)
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    conn._last_known_voice_at = time.monotonic()
    conn._due_greeting.discard("Anton")
    assert _target(conn) == LABEL_UNKNOWN


def test_nobody_in_the_room_means_nobody_to_greet():
    conn = _conn(persons=0)
    assert _target(conn) is None


# -- the return value is a NAME, and it gets spoken ---------------------------

def test_a_blocked_gate_returns_none_and_never_a_falsy_stand_in():
    # The caller reads this value as the name to say out loud, so a bare False
    # is not "no": it once made Rowan greet somebody called "False".
    conn = _conn()
    conn._greet_block_key = ""
    conn._greet_block_logged_at = 0.0
    _see(conn, "Anton")
    assert conn._greet_blocked("some_gate", "because of a reason") is None


def test_every_greeting_target_is_a_non_empty_string_or_none():
    conn = _conn()
    _see(conn, "Anton", LABEL_UNKNOWN)
    _stranger_waiting(conn, GREET_AFTER_S + 1.0)
    for gaps in ((0.0, 0.0), (UNKNOWN_GAP_S, KNOWN_GAP_S), (1e9, 1e9)):
        target = conn._greet_target(GREET_AFTER_S, *gaps)
        assert target is None or (isinstance(target, str) and target)
