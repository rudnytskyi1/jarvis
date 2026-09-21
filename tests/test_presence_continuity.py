"""Room evidence must govern greetings even when an old unknown latch exists."""
from types import SimpleNamespace

from hub import app
from hub.room_state import RoomState

BODY = {'id': 'room:1', 'box': [.1, .1, .5, .9]}
FACE = {'box': [.2, .12, .3, .25], 'embedding': [1, 0], 'score': .95, 'area': 3000}


def connection():
    conn = app.Connection.__new__(app.Connection)
    conn.cfg = SimpleNamespace(server=SimpleNamespace(face=SimpleNamespace(
        greet_after_s=.35, greeting_cooldown_s=300, greeting_cooldown_known_s=900)))
    conn.presence = app.PresenceTracker(ttl_s=30)
    conn.room = RoomState()
    conn._presence_has_tracks = True
    conn.camera_state = {'persons': 1}
    conn._last_seen_at = {}
    conn._due_greeting = set()
    conn._last_known_voice_at = 0
    return conn


def test_legacy_unknown_greeting_cannot_bypass_track_stability(monkeypatch):
    clock = [20.0]
    monkeypatch.setattr(app.time, 'monotonic', lambda: clock[0])
    conn = connection()
    conn.room.update([BODY])
    conn.room.resolve_faces([FACE], [BODY], lambda *_: (None, .1), {})
    conn.note_sightings([app.LABEL_UNKNOWN])
    conn.presence.note_faces([app.LABEL_UNKNOWN])
    clock[0] = 20.4
    assert conn._greet_target(.35, 300, 900) is None
    clock[0] = 20.8
    conn.room.update([BODY])
    conn.room.resolve_faces([FACE], [BODY], lambda *_: (None, .1), {})
    assert conn._greet_target(.35, 300, 900) == app.LABEL_UNKNOWN


def test_empty_tracked_frame_does_not_reenable_old_unknown_bucket(monkeypatch):
    clock = [20.0]
    monkeypatch.setattr(app.time, 'monotonic', lambda: clock[0])
    conn = connection()
    conn.note_sightings([app.LABEL_UNKNOWN])
    conn.presence.note_faces([app.LABEL_UNKNOWN])
    clock[0] = 24
    conn.room.update([])
    assert conn._greet_target(.35, 300, 900) is None


def test_track_stability_gate_does_not_delay_confirmed_known_person(monkeypatch):
    monkeypatch.setattr(app.time, 'monotonic', lambda: 20.0)
    conn = connection()
    conn.room.update([BODY])
    conn.room.resolve_faces([FACE], [BODY], lambda *_: ('Anton', .8), {})
    conn.note_sightings(['Anton'])
    conn.presence.note_faces(['Anton'])
    assert conn._greet_target(.35, 300, 900) == 'Anton'


def test_disabled_camera_greetings_do_not_wake_but_keep_identity(monkeypatch):
    monkeypatch.setattr(app.time, 'monotonic', lambda: 20.0)
    conn = connection()
    conn.cfg.server.face.greetings_enabled = False
    conn.face_enabled = True
    conn._greet_task = None
    conn.room.update([BODY])
    conn.room.resolve_faces([FACE], [BODY], lambda *_: ('Anton', .8), {})
    conn.note_sightings(['Anton'])
    conn.presence.note_faces(['Anton'])
    conn._start_greeting_task()
    assert conn._greet_task is None
    assert conn._greet_target(.35, 300, 900) is None
    assert 'Anton' in conn.presence.present()
