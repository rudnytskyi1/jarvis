"""server/app.py: PresenceTracker.has_fresh_unknown_face and the BUG 2 greeting gate.

Pure-python, fake tracker states only: no camera, no face engine, no
WebSocket. The proactive greeting must fire ONLY on a CURRENTLY-FRESH unknown
FACE match (something :meth:`PresenceTracker.note_faces` actually recorded),
never on a bare YOLO person count (:meth:`PresenceTracker.note_persons` never
touches the unknown bucket) and never on a label that has already expired.
"""
import time

from server.app import LABEL_UNKNOWN, PresenceTracker


def test_no_unknown_face_ever_seen_is_not_fresh():
    tracker = PresenceTracker(ttl_s=30.0)
    assert not tracker.has_fresh_unknown_face()


def test_bare_yolo_person_count_never_makes_an_unknown_face_fresh():
    # note_persons() is fed straight from the client's camera_state - a bare
    # YOLO count with no face match behind it must never satisfy the gate.
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_persons(2)
    assert not tracker.has_fresh_unknown_face()
    assert tracker.unknown_present_for() == 0.0


def test_a_real_unknown_face_match_is_fresh():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces([LABEL_UNKNOWN])
    assert tracker.has_fresh_unknown_face()


def test_a_named_face_only_is_not_an_unknown_face():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton"])
    assert not tracker.has_fresh_unknown_face()


def test_an_expired_unknown_label_is_no_longer_fresh():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.ttl_s = 0.05  # PresenceTracker.__init__ clamps ttl_s to >= 1.0
    tracker.note_faces([LABEL_UNKNOWN])
    assert tracker.has_fresh_unknown_face()
    time.sleep(0.08)
    assert not tracker.has_fresh_unknown_face()
    # unknown_present_for() must agree - the same stale label must not read as
    # "still present" either.
    assert tracker.unknown_present_for() == 0.0


def test_reconcile_dropping_the_unknown_bucket_also_drops_freshness():
    # A stale/misattributed unknown gets reconciled away by named labels
    # covering YOLO's count (SPEC v1.6) - the greeting must not see it as fresh.
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", LABEL_UNKNOWN])
    assert tracker.has_fresh_unknown_face()
    tracker.reconcile(1)  # 1 named label already covers YOLO's 1 person
    assert not tracker.has_fresh_unknown_face()


def test_clear_removes_freshness():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces([LABEL_UNKNOWN])
    assert tracker.has_fresh_unknown_face()
    tracker.clear()
    assert not tracker.has_fresh_unknown_face()


def test_re_arrival_after_expiry_is_fresh_again_with_a_new_first_seen():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.ttl_s = 0.05  # PresenceTracker.__init__ clamps ttl_s to >= 1.0
    tracker.note_faces([LABEL_UNKNOWN])
    time.sleep(0.08)
    assert not tracker.has_fresh_unknown_face()
    tracker.note_faces([LABEL_UNKNOWN])
    assert tracker.has_fresh_unknown_face()
    # A fresh arrival must reset the "present for" clock, not carry over the
    # gap while it was gone.
    assert tracker.unknown_present_for() < 0.05
