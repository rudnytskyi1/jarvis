"""server/app.py: PresenceTracker.reconcile (SPEC v1.6).

Pure-python, fake tracker states only: no camera, no face engine, no
WebSocket. YOLO's own person count and the face-matched labels are two
independent signals that can disagree when the same not-yet-enrolled (or
badly-angled) person is seen as both a named face in one frame and an
unmatched face in another - reconcile() is what stops that from being
reported as "you and a stranger" while the owner is alone.
"""
from server.app import LABEL_UNKNOWN, PresenceTracker


def test_reconcile_drops_unknown_when_named_covers_persons():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", LABEL_UNKNOWN])
    assert LABEL_UNKNOWN in tracker.present()

    tracker.reconcile(1)  # YOLO: only 1 person - Anton alone accounts for it
    present = tracker.present()
    assert LABEL_UNKNOWN not in present
    assert "Anton" in present


def test_reconcile_keeps_unknown_when_named_does_not_cover_persons():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", LABEL_UNKNOWN])

    tracker.reconcile(2)  # YOLO: 2 people - the unknown one may be real
    assert LABEL_UNKNOWN in tracker.present()


def test_reconcile_drops_unknown_when_several_named_labels_cover_persons():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", "Bob", LABEL_UNKNOWN])

    tracker.reconcile(2)  # 2 named labels already cover YOLO's 2 people
    present = tracker.present()
    assert LABEL_UNKNOWN not in present
    assert set(present) == {"Anton", "Bob"}


def test_reconcile_ignores_zero_or_negative_persons():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces([LABEL_UNKNOWN])
    tracker.reconcile(0)
    # No YOLO signal at all - leave the tracker exactly as it was.
    assert LABEL_UNKNOWN in tracker.present()
    assert tracker.unknown_count == 1


def test_reconcile_is_noop_without_an_unknown_bucket():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton"])
    tracker.reconcile(1)  # nothing to drop - must not raise or touch Anton
    assert set(tracker.present()) == {"Anton"}


def test_reconcile_noop_when_named_labels_are_too_few():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", LABEL_UNKNOWN])

    tracker.reconcile(3)  # YOLO sees 3 people, only 1 named - keep the unknown
    assert LABEL_UNKNOWN in tracker.present()
