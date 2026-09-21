"""server/app.py: PresenceTracker.reconcile (SPEC v1.6).

Pure-python, fake tracker states only: no camera, no face engine, no
WebSocket. YOLO's own person count and the face-matched labels are two
independent signals that can disagree when the same not-yet-enrolled (or
badly-angled) person is seen as both a named face in one frame and an
unmatched face in another - reconcile() is what stops that from being
reported as "you and a stranger" while the owner is alone.
"""
from hub.app import LABEL_UNKNOWN, PresenceTracker


def test_reconcile_drops_unknown_seen_in_a_later_worse_angle_burst():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton"])  # a good-angle frame matched him
    tracker.note_faces([LABEL_UNKNOWN])  # the next frame caught him badly
    assert LABEL_UNKNOWN in tracker.present()

    tracker.reconcile(1)  # YOLO: only 1 person - Anton alone accounts for it
    present = tracker.present()
    assert LABEL_UNKNOWN not in present
    assert "Anton" in present


def test_reconcile_keeps_an_unknown_face_seen_beside_a_named_one():
    # One burst, two faces, only one of them named: the face engine really did
    # see a second face where YOLO counts one body - a photo held up to the
    # camera, a face on the TV, somebody leaning in. That is a real stranger
    # and dropping it is what used to keep Rowan silent.
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", LABEL_UNKNOWN])

    tracker.reconcile(1)  # YOLO sees one body, the camera saw two faces
    present = tracker.present()
    assert LABEL_UNKNOWN in present
    assert "Anton" in present


def test_reconcile_keeps_unknown_when_named_does_not_cover_persons():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", LABEL_UNKNOWN])

    tracker.reconcile(2)  # YOLO: 2 people - the unknown one may be real
    assert LABEL_UNKNOWN in tracker.present()


def test_reconcile_drops_unknown_when_several_named_labels_cover_persons():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", "Bob"])
    tracker.note_faces(["Anton", LABEL_UNKNOWN])  # Bob caught at a bad angle

    tracker.reconcile(2)  # 2 named labels already cover YOLO's 2 people
    present = tracker.present()
    assert LABEL_UNKNOWN not in present
    assert set(present) == {"Anton", "Bob"}


def test_reconcile_keeps_a_third_face_two_named_people_cannot_account_for():
    tracker = PresenceTracker(ttl_s=30.0)
    tracker.note_faces(["Anton", "Bob", LABEL_UNKNOWN])

    # Three faces in ONE frame but only two names: whoever the third face
    # belongs to, it is not Anton and not Bob.
    tracker.reconcile(2)
    assert LABEL_UNKNOWN in tracker.present()


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
