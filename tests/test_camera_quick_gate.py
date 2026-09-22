"""Owner's report (2026-09-22): a person walking quickly past the camera was
not caught, even though the FPS cap was already lifted while somebody is
visible.

The FPS cap was never the whole story: ``yolo11x`` needs ~300 ms per frame on
the owner's PC, so the accurate detector sees the room about three times a
second. A person crossing it in half a second can be gone before the next
frame, and no later stage can report somebody nobody saw. The light guard
(``quick_model``) closes exactly that gap: it runs between the heavy frames and
releases the presence burst on the first sighting.
"""
from __future__ import annotations

from types import SimpleNamespace

from client import camera as client_camera


class _Boxes:
    """The slice of an ultralytics ``Boxes`` object the guard reads."""

    def __init__(self, rows):
        self.cls = _List([row[0] for row in rows])
        self.conf = _List([row[1] for row in rows])
        self.xyxyn = _List([row[2] for row in rows])


class _List(list):
    def tolist(self):
        return list(self)


class _Result:
    names = {0: "person", 56: "chair"}

    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class _Model:
    """A YOLO stand-in that answers with the boxes it was built with."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        return [_Result(self.rows)]


def _service(**camera_cfg):
    cfg = {"enabled": True, "model": "yolo11x.pt", **camera_cfg}
    return client_camera.CameraService(SimpleNamespace(**cfg))


def test_the_guard_is_on_with_the_light_model_by_default():
    service = _service()
    assert service.quick_model_name == client_camera.QUICK_PASS_MODEL
    assert service._quick_model() == "yolo11n.pt"
    assert service.quick_fps == client_camera.QUICK_PASS_FPS


def test_an_empty_quick_model_turns_the_guard_off():
    service = _service(quick_model="")
    assert service._quick_model() == ""


def test_a_room_whose_own_detector_is_the_light_model_needs_no_guard():
    """Nothing to add when the accurate detector is already the light one."""
    service = _service(model="yolo11n.pt")
    assert service._quick_model() == ""


def test_a_person_box_becomes_a_track_the_hub_can_read():
    service = _service()
    model = _Model([(0, 0.9, [0.1, 0.2, 0.4, 0.8])])
    boxes = service._quick_person_boxes(model, frame=object(), frame_ts=1000.5)
    assert boxes == [{"id": "quick:1000500:0", "box": [0.1, 0.2, 0.4, 0.8]}]


def test_other_classes_and_weak_boxes_are_not_people():
    service = _service()
    model = _Model([
        (56, 0.9, [0.1, 0.2, 0.4, 0.8]),                      # a chair
        (0, client_camera.QUICK_PASS_CONF - 0.01, [0.1, 0.2, 0.4, 0.8]),
        (0, 0.9, [0.4, 0.2, 0.4, 0.8]),                        # no width
    ])
    assert service._quick_person_boxes(model, frame=object(), frame_ts=1.0) == []


def test_a_guard_sighting_takes_the_burst_when_no_track_exists():
    service = _service()
    service._tracks = []
    assert service._quick_burst_due(now=10.0) is True


def test_a_guard_sighting_waits_while_the_heavy_detector_owns_a_track():
    service = _service()
    service._tracks = [{"id": "room:1", "box": [0.1, 0.1, 0.2, 0.2]}]
    assert service._quick_burst_due(now=10.0) is False


def test_the_guard_does_not_re_burst_the_same_pass_by():
    service = _service()
    service._tracks = []
    service._quick_burst_at = 10.0
    assert service._quick_burst_due(now=10.0 + client_camera.QUICK_PASS_COOLDOWN_S / 2) is False
    assert service._quick_burst_due(now=10.0 + client_camera.QUICK_PASS_COOLDOWN_S) is True


def test_a_guard_sighting_marks_the_room_active_and_marks_a_burst_due():
    """The burst itself is the heavy loop's job; the guard only asks for it."""
    service = _service()
    service._tracks = []
    service._last_person_seen = 0.0
    pushed: list = []
    # The guard pushes with frame_prefix='q' so its burst id cannot collide
    # with the heavy detector's own 'p' ids (see _maybe_push_presence).
    service._maybe_push_presence = lambda frame=None, frame_prefix='p': pushed.append(frame)
    frame = object()
    service._note_quick_persons(frame, [{"id": "quick:1:0", "box": [0.1, 0.2, 0.4, 0.8]}])
    assert pushed == [frame]
    assert service._burst_due is True
    assert service._quick_tracks == [{"id": "quick:1:0", "box": [0.1, 0.2, 0.4, 0.8]}]
    assert service._detection_budget(1 / 3) == 0.0, "the heavy detector stops waiting too"


def test_the_guard_is_silent_while_the_socket_is_busy():
    """A pending presence push means one is already going out."""
    service = _service()
    service._tracks = []
    service._presence_pending.set()
    assert service._quick_burst_due(now=10.0) is False
