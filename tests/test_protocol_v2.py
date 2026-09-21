"""Protocol v2: strict typed frames with a working v1 fallback (ТЗ section 13)."""
from __future__ import annotations

import pytest

from common import protocol as proto

BASE = {"home_id": "home-a", "client_id": "client-1"}

CLIENT_SAMPLES = [
    proto.Hello(**BASE, token="t0ken"),
    proto.UtteranceStart(**BASE, utterance_id="u1", sample_rate=16000, pre_roll_ms=300),
    proto.UtteranceEnd(**BASE, utterance_id="u1"),
    proto.ActionResult(**BASE, action_id="act-1", ok=True, detail="volume 30"),
    proto.Tracks(**BASE, tracks=[proto.Track(track_id="tr-1", conf=0.9, zone="desk")]),
    proto.BodyCropHeader(**BASE, track_id="tr-1", kind="face", w=640, h=640),
    proto.FaceBurstHeader(**BASE, track_id="tr-1"),
    proto.ScreenshotHeader(**BASE),
    proto.CameraFrameHeader(**BASE, reason="presence", index=2, of=3),
    proto.CameraClipHeader(**BASE, bytes=1024, seconds=5.0, fps=8),
    proto.SoundEvent(**BASE, label="knock", conf=0.8, at_ms=1234),
    proto.DeviceState(**BASE, device_id="desk-lamp", capability="on_off", value=True),
    proto.BargeIn(**BASE, say_id="say-1", at_ms=42),
    proto.Ping(**BASE),
    proto.Pong(**BASE),
]

SERVER_SAMPLES = [
    proto.HelloOk(**BASE, session_id="sess-1", server_version="0.2.0", config_rev=3),
    proto.HelloErr(**BASE, code=4401, message="bad token"),
    proto.Transcript(**BASE, utterance_id="u1", text="привет", language="ru", speaker="Anton"),
    proto.Actions(**BASE, utterance_id="u1", actions=[proto.ActionItem(id="act-1", kind="volume_set", args={"value": 30})]),
    proto.Say(**BASE, say_id="say-1", text="Done", voice="am_michael", volume=0.6, interruptible=True),
    proto.TtsStart(**BASE, say_id="say-1", sample_rate=48000),
    proto.TtsEnd(**BASE, say_id="say-1"),
    proto.ListenFollowup(**BASE, window_ms=6000),
    proto.Identity(**BASE, track_id="tr-1", person_id="p-1", name="Anton", p=0.91, role=proto.Role.ADMIN),
    proto.CameraRequest(**BASE, kind="frame", burst=3, full=True),
    proto.ScreenshotRequest(**BASE, event_id="ev-1"),
    proto.CameraClipRequest(**BASE, seconds=5, fps=8),
    proto.DeviceSet(**BASE, device_id="desk-lamp", capability="brightness", value=40, action_id="act-1"),
    proto.Hud(**BASE, kind="card", payload={"title": "Reminder"}),
    proto.ConfigUpdate(**BASE, config_rev=4, patch={"quiet_hours": {"start": "23:00"}}),
    proto.OfflineHint(**BASE, reason="maintenance", eta_s=30.0),
    proto.Intercom(**BASE, from_person="Max", text="I am coming"),
    proto.ErrorMessage(**BASE, code=1001, message="unknown type", ref_seq=7),
]


@pytest.mark.parametrize("sample", CLIENT_SAMPLES, ids=lambda item: type(item).__name__)
def test_client_frames_round_trip(sample):
    parsed = proto.parse_message(sample.model_dump(mode="json"), direction="client")
    assert type(parsed) is type(sample)
    assert parsed == sample


@pytest.mark.parametrize("sample", SERVER_SAMPLES, ids=lambda item: type(item).__name__)
def test_server_frames_round_trip(sample):
    parsed = proto.parse_message(sample.model_dump(mode="json"), direction="server")
    assert type(parsed) is type(sample)
    assert parsed == sample


def test_envelope_carries_the_version_and_identity():
    frame = proto.Ping(**BASE, seq=5)
    payload = frame.model_dump(mode="json")
    assert payload["proto"] == proto.PROTOCOL_VERSION == 2
    assert payload["home_id"] == "home-a" and payload["client_id"] == "client-1"
    assert payload["seq"] == 5


def test_v1_frames_are_left_to_the_legacy_path():
    assert proto.parse_message({"type": "hello", "client_id": "old-client"}) is None
    assert proto.parse_message({"type": "say", "text": "hi"}, direction="server") is None
    assert proto.protocol_version({"proto": 1}) == 1


def test_protocol_version_helper_rejects_bad_values():
    assert proto.protocol_version({}) == proto.LEGACY_PROTOCOL_VERSION
    assert proto.protocol_version({"proto": "2"}) == proto.LEGACY_PROTOCOL_VERSION
    assert proto.protocol_version({"proto": True}) == proto.LEGACY_PROTOCOL_VERSION


def test_unknown_type_is_reported_not_crashed():
    with pytest.raises(proto.ProtocolError):
        proto.parse_message({"type": "not_a_frame", "proto": 2, **BASE})


def test_unknown_fields_are_rejected():
    with pytest.raises(proto.ProtocolError):
        proto.parse_message({"type": "ping", "proto": 2, **BASE, "bogus": 1})


def test_out_of_range_values_are_rejected():
    with pytest.raises(proto.ProtocolError):
        proto.parse_message({"type": "say", "proto": 2, **BASE, "volume": 5})


def test_unknown_direction_is_rejected():
    with pytest.raises(proto.ProtocolError):
        proto.parse_message({"type": "ping", "proto": 2, **BASE}, direction="sideways")
