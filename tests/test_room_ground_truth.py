"""server/app.py: the measured facts that ride with a camera description (v1.7).

The vision model writes fluent prose and invents things inside it — a bowl and
scattered papers in a room that has neither, "one person... I don't recognise
them" about the owner whose face had just matched at 0.85. So its answer never
travels alone: the room camera's own detector says what is really there, and
the face matcher is the only thing that can put a name to anybody.
"""
from hub.app import LABEL_UNKNOWN, Connection, PresenceTracker


def _conn(camera_state=None) -> Connection:
    conn = Connection.__new__(Connection)
    conn.presence = PresenceTracker(ttl_s=30.0)
    conn.camera_state = camera_state
    return conn


def test_detected_objects_are_reported_verbatim_from_the_detector():
    conn = _conn({"persons": 1, "objects": {"couch": 1, "chair": 2, "bottle": 1}})
    truth = conn._room_ground_truth()
    assert truth["objects_detected"] == "bottle x1, chair x2, couch x1"
    assert truth["persons_detected"] == 1


def test_an_empty_detector_says_so_rather_than_going_silent():
    # A missing field must not read as "the detector agrees with the model".
    truth = _conn({"persons": 0, "objects": {}})._room_ground_truth()
    assert "no known objects" in truth["objects_detected"]


def test_no_camera_state_at_all_is_still_explicit():
    truth = _conn(None)._room_ground_truth()
    assert "no known objects" in truth["objects_detected"]
    assert truth["persons_detected"] is None


def test_only_matched_faces_become_names():
    conn = _conn({"persons": 2, "objects": {}})
    conn.presence.note_faces(["Anton", LABEL_UNKNOWN])
    truth = conn._room_ground_truth()
    assert "Anton" in truth["people_recognised"]
    # The stranger is counted, never named.
    assert "1 person(s) whose face you do not recognise" in truth["people_recognised"]


def test_nobody_recognised_is_said_out_loud():
    truth = _conn({"persons": 1, "objects": {}})._room_ground_truth()
    assert truth["people_recognised"] == "(nobody recognised)"


def test_an_unknown_face_alone_yields_no_name():
    conn = _conn({"persons": 1, "objects": {}})
    conn.presence.note_faces([LABEL_UNKNOWN])
    truth = conn._room_ground_truth()
    assert "Anton" not in truth["people_recognised"]
    assert "do not recognise" in truth["people_recognised"]


def test_the_note_tells_the_model_which_source_to_trust():
    truth = _conn({"persons": 1, "objects": {"couch": 1}})._room_ground_truth()
    note = truth["note"].lower()
    # The whole point of the field: the fluent text is the least reliable part.
    assert "invents" in note
    assert "only source of names" in note
    assert "detector does not list it" in note
