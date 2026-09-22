"""Регистрация гостя (ТЗ F-210): слова, кадры, поток и одна запись в БД.

The tests follow the module's own three layers: what the owner and the guest
say, which camera frames are usable at all, and what (if anything) reaches the
database once the owner pressed the button in Telegram.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from starlette.websockets import WebSocketState

from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub.guest_registration import (
    MAX_FRAMES,
    MIN_FRAMES,
    STEP_BODY,
    STEP_CANCELLED,
    STEP_CONSENT,
    STEP_DECLINED,
    STEP_DONE,
    STEP_FACE,
    STEP_OWNER,
    STEP_VOICE,
    BurstQuality,
    Declaration,
    GuestRegistration,
    OwnerConfirmations,
    PendingGuest,
    Shot,
    assess_burst,
    clean_name,
    commit_guest,
    consent_given,
    consent_refused,
    consent_request,
    declaration,
    ensure_person,
    ensure_track,
    face_prompt,
    membership_role,
    owner_question,
    phrase,
    phrase_matches,
    retry_hint,
    revoke_guest,
    voice_prompt,
    waiting_for_owner,
    yaw_degrees,
)
from hub.speaker import ROLE_GUEST


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('legacy-drew', 'Drew')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES ('legacy-drew',"
                 " 'livingroom', 'admin')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


class _Audit:
    """A recording stand-in for the hub's ``audit`` table (F-706)."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


class _Provider:
    """The two Telegram calls the owner's buttons need, recorded."""

    def __init__(self) -> None:
        self.answered: list[tuple[str, str, bool]] = []
        self.edited: list[tuple[str, int]] = []

    async def answer_callback(self, callback_id: str, text: str = "", show_alert: bool = False) -> None:
        self.answered.append((callback_id, text, show_alert))

    async def edit_text(self, text: str, *, message_id: int) -> None:
        self.edited.append((text, message_id))


def _shot(height: float = 0.3, sharpness: float = 120.0, angle: float | None = 0.0,
          score: float = 0.8) -> Shot:
    return Shot(box=(0.3, 0.2, 0.6, 0.2 + height), sharpness=sharpness, angle=angle, score=score)


def _burst(size: int = 6, angles: tuple[float | None, ...] = (-25.0, -12.0, 0.0, 12.0, 30.0, 5.0)) -> list[Shot]:
    return [_shot(angle=angles[index % len(angles)]) for index in range(size)]


def _vector(seed: int = 0, *, dim: int = 192) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


# --- the words the owner and the guest use ---------------------------------


def test_the_owner_introduces_a_friend_in_three_languages():
    russian = declaration("Rowan, это Макс, друг")
    assert russian is not None and russian.name == "Макс" and russian.relation == "friend"
    english = declaration("Rowan AI, this is Max, a friend")
    assert english is not None and english.name == "Max" and english.relation == "friend"
    spanish = declaration("Rowan, este es Ana, mi amiga")
    assert spanish is not None and spanish.name == "Ana" and spanish.relation == "friend"


def test_an_introduction_without_a_name_asks_who():
    invite = declaration("Rowan, добавь гостя")
    assert invite is not None and invite.name is None
    assert not invite.named


def test_ordinary_speech_is_not_an_introduction():
    for text in ("Rowan, turn off the light", "как дела", "this is my room", "",
                 "Rowan, remember my voice"):
        assert declaration(text) is None, text


def test_a_placeholder_is_not_a_name():
    assert declaration("Rowan, это гость, гость") is None
    assert clean_name("Guest") is None
    assert clean_name("Макс") == "Макс"
    assert clean_name("Max's") == "Max"
    assert clean_name("M") is None


def test_the_guest_reads_a_phrase_and_the_hub_says_what_it_stores():
    assert phrase("ru").startswith("Роуэн")
    assert "Max" in voice_prompt("Max", "en")
    assert "Max" in face_prompt("Max", "en")
    notice = consent_request("Макс", "ru")
    assert "Макс" in notice and "голос" in notice and "забудь меня" in notice
    # An unknown language falls back to the room's default instead of failing.
    assert consent_request("Max", "uk") == consent_request("Max", "ru")


def test_the_phrase_has_to_be_read_not_merely_answered():
    expected = phrase("ru")
    assert phrase_matches(expected, expected)
    assert phrase_matches("Роуэн послушай мой голос сегодня я гость", expected)
    assert not phrase_matches("да", expected)
    assert not phrase_matches("", expected)


def test_the_guests_own_yes_is_the_same_word_as_f113():
    assert consent_given("да") and consent_given("Yes, please")
    assert consent_refused("нет") and consent_refused("no thanks")
    assert not consent_given("maybe") and not consent_refused("maybe")


def test_the_owner_question_names_the_person_and_the_relation():
    assert "Макс" in owner_question("Макс", "friend", "ru")
    assert "friend" in owner_question("Max", "friend", "en")
    assert "Telegram" in waiting_for_owner("ru") and "Telegram" in waiting_for_owner("en")


# --- the camera gate --------------------------------------------------------


def test_a_frame_is_measured_by_height_blur_and_turn():
    assert _shot().height == pytest.approx(0.3)
    assert Shot().height == 0.0
    assert _shot(height=0.05).height == pytest.approx(0.05)


def test_the_head_turn_is_measured_from_the_landmarks():
    facing = [[100.0, 100.0], [200.0, 100.0], [150.0, 130.0], [130.0, 160.0], [170.0, 160.0]]
    assert yaw_degrees(facing) == pytest.approx(0.0)
    # The nose sliding towards the left eye is a turn to the left.
    left = [[100.0, 100.0], [200.0, 100.0], [130.0, 130.0], [0.0, 0.0], [0.0, 0.0]]
    assert yaw_degrees(left) == pytest.approx(-36.0)
    assert yaw_degrees(None) is None
    assert yaw_degrees([[10.0, 10.0]]) is None
    assert yaw_degrees([[100.0, 0.0], [100.0, 0.0], [120.0, 0.0]]) is None
    # A profile view is clipped, not reported as an impossible angle.
    assert abs(yaw_degrees([[100.0, 0.0], [200.0, 0.0], [500.0, 0.0]])) < 90.0


def test_a_burst_of_five_to_ten_frames_from_two_angles_is_accepted():
    quality = assess_burst(_burst())
    assert quality.ok and MIN_FRAMES <= len(quality.kept) <= MAX_FRAMES
    assert set(quality.angles) >= {"front", "left"} and not quality.reasons


def test_no_frames_at_all_is_a_clear_refusal():
    quality = assess_burst([])
    assert not quality.ok and quality.reasons == ("no_frames",) and quality.kept == ()


def test_too_few_usable_frames_are_refused():
    frames = [_shot(), _shot(), _shot(angle=25.0)]
    quality = assess_burst(frames)
    assert not quality.ok and quality.reasons == ("too_few_frames",)


def test_a_dark_blurred_or_far_face_is_not_a_frame():
    frames = [_shot(height=0.05) for _ in range(6)]
    assert assess_burst(frames).reasons == ("too_few_frames",)
    assert assess_burst([_shot(sharpness=5.0) for _ in range(6)]).reasons == ("too_few_frames",)
    assert assess_burst([_shot(angle=70.0) for _ in range(6)]).reasons == ("too_few_frames",)


def test_ten_copies_of_one_angle_are_not_a_burst():
    quality = assess_burst([_shot(angle=0.0) for _ in range(8)])
    assert not quality.ok and quality.reasons == ("one_angle",)


def test_at_most_ten_frames_are_kept():
    quality = assess_burst(_burst(14))
    assert quality.ok and len(quality.kept) == MAX_FRAMES


def test_a_detector_without_landmarks_is_reported_not_invented():
    quality = assess_burst([_shot(angle=None) for _ in range(6)])
    assert quality.ok and quality.notes == ("angles_unknown",)
    assert quality.angles == ("unknown",)


def test_the_retry_hint_says_what_to_fix():
    assert "Макс" in retry_hint(BurstQuality(False, reasons=("no_frames",)), "Макс", "ru")
    assert "камер" in retry_hint(BurstQuality(False, reasons=("no_frames",)), "Макс", "ru")
    assert "same angle" in retry_hint(BurstQuality(False, reasons=("one_angle",)), "Max", "en")


# --- the flow ---------------------------------------------------------------


def _flow(*, ttl_s: float = 300.0, now: list[float] | None = None) -> GuestRegistration:
    clock = (lambda: now[0]) if now is not None else None
    return GuestRegistration(ttl_s=ttl_s, **({"clock": clock} if clock else {}))


def _started(flow: GuestRegistration, **kwargs: Any) -> PendingGuest | None:
    return flow.start(Declaration(name="Макс", relation="friend"), home_id="livingroom",
                      requested_by="legacy-drew", language="ru", track_id="track-1",
                      day="2026-09-21", **kwargs)


def test_the_flow_starts_only_with_a_name():
    flow = _flow()
    assert flow.start(Declaration(name=None), home_id="livingroom") is None
    assert flow.pending is None
    started = _started(flow)
    assert started is not None and started.step == STEP_VOICE
    assert flow.pending is started


def test_the_flow_walks_voice_face_body_consent_owner():
    flow = _flow()
    _started(flow)
    assert flow.voice_step(seconds=4.0, text=phrase("ru")) == STEP_FACE
    assert flow.pending is not None and flow.pending.voice_seconds == 4.0
    assert flow.face_step(_burst()).ok and flow.pending.step == STEP_BODY
    assert flow.pending.frames == 6 and set(flow.pending.angles) >= {"front", "left"}
    assert flow.body_step(saved=True) == STEP_CONSENT
    assert flow.consent_step("да") and flow.pending.step == STEP_OWNER
    assert flow.owner_step(True) == STEP_DONE
    assert flow.pending is None


def test_a_short_or_wrong_sample_is_asked_for_again_then_cancelled():
    flow = _flow()
    _started(flow)
    assert flow.voice_step(seconds=1.0, text=phrase("ru")) == STEP_VOICE
    assert flow.pending is not None and flow.pending.note == "voice_not_usable"
    assert flow.voice_step(seconds=1.0, text=phrase("ru")) == STEP_CANCELLED
    assert flow.pending is None
    # The phrase itself is checked too, not just the length of the recording.
    flow = _flow()
    _started(flow)
    assert flow.voice_step(seconds=9.0, text="включи свет") == STEP_VOICE


def test_a_bad_burst_is_asked_for_again_then_cancelled():
    flow = _flow()
    _started(flow)
    flow.voice_step(seconds=4.0, text=phrase("ru"))
    first = flow.face_step([])
    assert not first.ok and flow.pending is not None and flow.pending.step == STEP_FACE
    second = flow.face_step([])
    assert not second.ok and flow.owner_step(True) == STEP_CANCELLED
    assert flow.pending is None


def test_steps_called_out_of_order_change_nothing():
    flow = _flow()
    _started(flow)
    assert flow.face_step(_burst()).reasons == ("wrong_step",)
    assert flow.body_step(saved=True) == STEP_VOICE
    assert not flow.consent_step("да")
    assert flow.owner_step(True) == STEP_VOICE
    assert flow.pending is not None and flow.pending.step == STEP_VOICE


def test_the_guest_can_refuse_the_consent():
    flow = _flow()
    _started(flow)
    flow.voice_step(seconds=4.0, text=phrase("ru"))
    flow.face_step(_burst())
    flow.body_step(saved=True)
    assert not flow.consent_step("нет")
    assert flow.pending is None


def test_a_declined_owner_leaves_the_flow_declined():
    flow = _flow()
    pending = _started(flow)
    assert pending is not None
    flow.voice_step(seconds=4.0, text=phrase("ru"))
    flow.face_step(_burst())
    flow.body_step(saved=True)
    flow.consent_step("да")
    assert flow.owner_step(False) == STEP_DECLINED
    assert pending.owner_decision == "declined" and flow.pending is None


def test_a_walked_away_guest_does_not_leave_the_flow_open():
    now = [0.0]
    flow = _flow(ttl_s=60.0, now=now)
    _started(flow)
    now[0] = 61.0
    assert flow.pending is None
    assert flow.voice_step(seconds=9.0, text=phrase("ru")) == STEP_CANCELLED


def test_the_summary_carries_no_vectors():
    flow = _flow()
    pending = _started(flow)
    assert pending is not None
    summary = pending.summary()
    assert summary["name"] == "Макс" and summary["step"] == STEP_VOICE
    assert all(not isinstance(value, np.ndarray) for value in summary.values())


# --- the owner's confirmation ----------------------------------------------


def _pending(name: str = "Макс", **kwargs: Any) -> PendingGuest:
    return PendingGuest(name=name, home_id="livingroom", relation="friend",
                        requested_by="legacy-drew", **kwargs)


def test_a_token_is_single_use():
    confirmations = OwnerConfirmations(ttl_s=100.0)
    request = confirmations.open(_pending())
    assert confirmations.get(request.token) is request
    resolved = confirmations.resolve(request.token, True)
    assert resolved is request and resolved.decision == "approved"
    # The same button cannot register a second person.
    assert confirmations.resolve(request.token, True) is None
    assert confirmations.get(request.token) is None


def test_an_expired_question_can_no_longer_register_anybody():
    now = [0.0]
    confirmations = OwnerConfirmations(ttl_s=10.0, clock=lambda: now[0])
    request = confirmations.open(_pending(), now=now[0])
    now[0] = 11.0
    assert confirmations.get(request.token) is None
    assert confirmations.resolve(request.token, True) is None


def test_the_buttons_carry_the_token_and_the_decision():
    keyboard = OwnerConfirmations.keyboard("abc123", "ru")
    data = [button["callback_data"] for row in keyboard["inline_keyboard"] for button in row]
    assert data == ["guest:abc123:yes", "guest:abc123:no"]


def test_the_owner_press_resolves_the_question_and_runs_the_callback():
    provider = _Provider()
    confirmations = OwnerConfirmations(provider=provider)
    request = confirmations.open(_pending())
    seen: list[Any] = []

    async def on_decision(resolved: Any) -> None:
        seen.append(resolved)

    update = {"callback_query": {"id": "cb-1", "data": f"guest:{request.token}:yes",
                                "from": {"id": 7},
                                "message": {"message_id": 42}}}
    handled = asyncio.run(confirmations.handle_update(
        update, is_owner=lambda sender: sender.get("id") == 7, on_decision=on_decision))
    assert handled and seen and seen[0].decision == "approved"
    assert provider.answered == [("cb-1", "Confirmed", False)]
    assert provider.edited and provider.edited[0][1] == 42
    assert confirmations.get(request.token) is None


def test_a_press_from_somebody_else_registers_nobody():
    provider = _Provider()
    confirmations = OwnerConfirmations(provider=provider)
    request = confirmations.open(_pending())
    update = {"callback_query": {"id": "cb-2", "data": f"guest:{request.token}:yes",
                                "from": {"id": 9}, "message": {"message_id": 42}}}
    assert asyncio.run(confirmations.handle_update(update, is_owner=lambda sender: sender.get("id") == 7))
    assert provider.answered[0][2] is True and "owner" in provider.answered[0][1]
    assert confirmations.get(request.token) is request


def test_an_update_of_another_handler_is_left_alone():
    confirmations = OwnerConfirmations()
    assert not asyncio.run(confirmations.handle_update({"message": {"text": "/tools"}}))
    assert not asyncio.run(confirmations.handle_update(
        {"callback_query": {"id": "cb", "data": "adm:token", "from": {"id": 7}}}))


def test_an_unknown_or_spent_token_is_answered_but_changes_nothing():
    provider = _Provider()
    confirmations = OwnerConfirmations(provider=provider)
    update = {"callback_query": {"id": "cb-3", "data": "guest:nope:yes", "from": {"id": 7},
                                "message": {"message_id": 1}}}
    assert asyncio.run(confirmations.handle_update(update, is_owner=lambda sender: True))
    assert provider.answered[0][2] is True and "expired" in provider.answered[0][1]


def test_a_room_that_disconnects_takes_its_question_with_it():
    confirmations = OwnerConfirmations()
    request = confirmations.open(_pending())
    assert confirmations.cancel_home("kitchen") == []
    assert confirmations.get(request.token) is request
    assert confirmations.cancel_home("livingroom") == [request]
    assert confirmations.get(request.token) is None


# --- the record -------------------------------------------------------------


def test_a_guest_is_a_person_with_a_guest_membership(hub_db):
    person_id, existed = ensure_person(hub_db, "Макс", home_id="livingroom", language="ru")
    assert not existed and membership_role(hub_db, person_id, "livingroom") == ROLE_GUEST
    assert hub_db.execute("SELECT display_name, preferred_language FROM persons WHERE person_id=?",
                          (person_id,)).fetchone() == ("Макс", "ru")
    # Asking twice reuses the person instead of creating a second row.
    again, existed_again = ensure_person(hub_db, "макс", home_id="livingroom")
    assert again == person_id and existed_again
    assert hub_db.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 2


def test_an_existing_member_is_never_demoted_to_guest(hub_db):
    person_id, existed = ensure_person(hub_db, "Drew", home_id="livingroom")
    assert existed and person_id == "legacy-drew"
    assert membership_role(hub_db, person_id, "livingroom") == "admin"


def test_a_placeholder_or_a_foreign_room_is_refused(hub_db):
    with pytest.raises(ValueError):
        ensure_person(hub_db, "Guest", home_id="livingroom")
    with pytest.raises(ValueError):
        ensure_person(hub_db, "Макс", home_id="nowhere")


def test_a_confirmed_guest_is_written_in_one_go(hub_db):
    audit = _Audit()
    pending = _pending(day="2026-09-21", track_id="track-1")
    ensure_track(hub_db, "track-1", home_id="livingroom")
    hub_db.execute("INSERT INTO face_embeddings(id, track_id, vector, dim) VALUES ('f1','track-1',?,512)",
                   (np.zeros(512, dtype=np.float32).tobytes(),))
    hub_db.commit()

    record = commit_guest(
        hub_db, pending,
        voice_vectors=[_vector(1)],
        face_vectors=[_vector(2, dim=512), _vector(3, dim=512)],
        body_vectors=[_vector(4, dim=512)],
        day="2026-09-21", track_id="track-1", audit=audit, actor="legacy-drew",
    )
    assert record.person_id and record.role == ROLE_GUEST and not record.existed
    assert (record.voice_samples, record.face_samples, record.body_samples) == (1, 2, 1)
    assert membership_role(hub_db, record.person_id, "livingroom") == ROLE_GUEST
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='track-1'").fetchone()[0] == record.person_id
    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings WHERE person_id=?",
                          (record.person_id,)).fetchone()[0] == 1
    # The two stored frames plus the track's own face, which follows the person.
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings WHERE person_id=?",
                          (record.person_id,)).fetchone()[0] == 3
    assert hub_db.execute("SELECT session_day FROM body_embeddings WHERE person_id=?",
                          (record.person_id,)).fetchone()[0] == "2026-09-21"
    assert audit.rows and audit.rows[0]["action"] == "guest.register"


def test_a_body_of_another_day_is_not_carried_over(hub_db):
    class Bodies:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str | None]] = []

        def link_track(self, track_id: str, person_id: str, *, day: str | None = None) -> int:
            self.calls.append((track_id, person_id, day))
            return 2

    bodies = Bodies()
    ensure_track(hub_db, "track-2", home_id="livingroom")
    record = commit_guest(hub_db, _pending(day="2026-09-21", track_id="track-2"),
                          body_vectors=[_vector(5, dim=512)], day="2026-09-21",
                          track_id="track-2", bodies=bodies)
    assert bodies.calls == [("track-2", record.person_id, "2026-09-21")]
    assert record.bodies_linked == 2


def test_a_broken_vector_writes_nothing_at_all(hub_db):
    with pytest.raises(ValueError):
        commit_guest(hub_db, _pending(), voice_vectors=[[]])
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE display_name='Макс'").fetchone()[0] == 0
    assert hub_db.execute("SELECT COUNT(*) FROM memberships").fetchone()[0] == 1


def test_taking_the_guest_membership_back_is_audited(hub_db):
    audit = _Audit()
    person_id, _ = ensure_person(hub_db, "Макс", home_id="livingroom")
    assert revoke_guest(hub_db, person_id, home_id="livingroom", audit=audit)
    assert membership_role(hub_db, person_id, "livingroom") is None
    assert audit.rows and audit.rows[0]["action"] == "guest.revoke"
    assert not revoke_guest(hub_db, person_id, home_id="livingroom")


def test_an_unknown_track_of_an_unknown_room_is_not_invented(hub_db):
    with pytest.raises(ValueError):
        ensure_track(hub_db, "track-9", home_id="nowhere")
    assert not ensure_track(hub_db, "", home_id="livingroom")
    assert hub_db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0


def test_nothing_is_stored_while_the_owner_has_not_answered(hub_db):
    """The point of F-210: the flow may finish its steps, the hub writes nothing."""
    flow = _flow()
    pending = _started(flow)
    assert pending is not None
    flow.voice_step(seconds=4.0, text=phrase("ru"))
    flow.face_step(_burst())
    flow.body_step(saved=True)
    flow.consent_step("да")
    assert pending.step == STEP_OWNER
    assert hub_db.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 1
    assert hub_db.execute("SELECT COUNT(*) FROM memberships").fetchone()[0] == 1
    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0
    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == 0


def test_the_guest_membership_of_a_second_room_does_not_touch_the_first(hub_db):
    hub_db.execute("INSERT INTO homes(home_id, name) VALUES ('kitchen', 'Kitchen')")
    hub_db.commit()
    person_id, _ = ensure_person(hub_db, "Макс", home_id="livingroom")
    ensure_person(hub_db, "Макс", home_id="kitchen")
    assert revoke_guest(hub_db, person_id, home_id="kitchen")
    assert membership_role(hub_db, person_id, "livingroom") == ROLE_GUEST
    assert membership_role(hub_db, person_id, "kitchen") is None


class _FlakyConnection:
    """A real connection whose face INSERT fails, so the rollback is visible."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.failures = 0

    def execute(self, sql: str, *args: Any) -> Any:
        if "INSERT INTO face_embeddings" in str(sql):
            self.failures += 1
            raise sqlite3.OperationalError("disk I/O error")
        return self._real.execute(sql, *args)

    def commit(self) -> None:
        self._real.commit()

    def rollback(self) -> None:
        self._real.rollback()


def test_a_database_error_rolls_the_vectors_back(hub_db):
    broken = _FlakyConnection(hub_db)
    with pytest.raises(sqlite3.OperationalError):
        commit_guest(broken, _pending(), voice_vectors=[_vector(6)],
                     face_vectors=[_vector(7, dim=512)])
    assert broken.failures == 1, "the failing INSERT was reached"
    # The voice vector, the person and the membership of the same transaction
    # are gone: a guest is either written whole or not at all.
    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0
    assert hub_db.execute("SELECT COUNT(*) FROM persons WHERE display_name='Макс'").fetchone()[0] == 0
    assert hub_db.execute("SELECT COUNT(*) FROM memberships").fetchone()[0] == 1


# --- the room (ТЗ F-210 end to end, everything but the camera) --------------


def _noise_jpeg(seed: int = 0) -> bytes:
    """A real JPEG: the sharpness gate measures the actual pixels."""
    import cv2

    image = np.random.default_rng(seed).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return encoded.tobytes() if ok else b""


def _kps(angle: float) -> list[list[float]]:
    """Five insightface-style keypoints of a head turned by ``angle`` degrees."""
    ratio = 0.5 + float(angle) / 180.0
    return [[0.2, 0.4], [0.8, 0.4], [0.2 + ratio * 0.6, 0.6], [0.35, 0.8], [0.65, 0.8]]


class _FakeFaces:
    """One face per frame, at the angle the flow is being tested with."""

    available = True

    def __init__(self, angles: tuple[float, ...] = (-25.0, 0.0, 25.0, 5.0, -12.0)) -> None:
        self.angles, self.calls = list(angles), 0

    def located_faces(self, jpeg: bytes) -> list[dict[str, Any]]:
        angle = self.angles[self.calls % len(self.angles)]
        self.calls += 1
        return [{"score": 1.0, "box": [0.3, 0.2, 0.6, 0.6],
                 "embedding": _vector(self.calls, dim=512), "landmarks": _kps(angle)}]


class _FakeFrames:
    """The camera of the room: a burst of real JPEGs, from the same shot."""

    def __init__(self, size: int = 5) -> None:
        self.size, self.calls = size, 0
        self.jpeg = _noise_jpeg(3)

    async def __call__(self, frame_id: str, burst: int) -> list[Any]:
        self.calls += 1
        return [SimpleNamespace(id=frame_id, jpeg=self.jpeg, w=320, h=240)
                for _ in range(min(int(burst), self.size))]


class _FakeTelegram:
    """The hub's Telegram provider, ready to talk to the owner."""

    ready = True

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send_text(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.sent.append((text, kwargs))
        return {"ok": True, "message_id": 1}


class _FakeRegistry:
    """The people registry of the room (``data/people.json``)."""

    enabled = True

    def __init__(self, roles: dict[str, str] | None = None) -> None:
        self.roles = dict(roles or {})
        self.enrolled: list[str] = []
        self.faces: list[str] = []

    def role_of(self, name: str) -> str | None:
        return self.roles.get(name)

    def enroll(self, name: str, pcm: bytes, sample_rate: int) -> tuple[str, str]:
        self.enrolled.append(name)
        self.roles.setdefault(name, "user")
        return self.roles[name], "stored"

    def add_face_embedding(self, name: str, vector: Any) -> tuple[str, str]:
        self.faces.append(name)
        self.roles.setdefault(name, "user")
        return self.roles[name], "stored"

    def set_role(self, name: str, role: str) -> str:
        self.roles[name] = role
        return role

    def identify_ex(self, pcm: bytes, sample_rate: int) -> tuple[str, str, float, Any]:
        return "Drew", "admin", 0.9, _vector(11, dim=192)


class _FakeSocket:
    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)
        self.client_state = WebSocketState.CONNECTED
        self.frames: list[dict[str, Any]] = []

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover - TTS is stubbed
        return None

    async def close(self, code: int = 1000) -> None:
        self.client_state = WebSocketState.DISCONNECTED

    def texts(self) -> list[str]:
        """Everything the room was told, in order, whatever frame carried it."""
        return [str(frame.get("text") or "") for frame in self.frames if frame.get("text")]


@pytest.fixture()
def guest_room(hub_db, tmp_path, monkeypatch):
    """One real ``Connection`` whose whole F-210 machinery is stubbed but real."""
    from hub.auth import ClientTokenStore
    from hub.gateway import Gateway
    from hub.room_state import RoomState

    audit = _Audit()
    provider = _FakeTelegram()
    confirmations = OwnerConfirmations(provider=provider, ttl_s=300.0)
    engine, frames, registry = _FakeFaces(), _FakeFrames(), _FakeRegistry()
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    # The hub builds its database lazily, so a test that leaves the gateway
    # unset would get a second, empty ``hub.db`` and a different connection.
    monkeypatch.setattr(hub_app, "_gateway", Gateway(ClientTokenStore(hub_db)))
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    monkeypatch.setattr(hub_app, "_tts", None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    monkeypatch.setattr(hub_app, "_body_embedding_store", lambda: None)
    monkeypatch.setattr(hub_app, "_face", engine)
    monkeypatch.setattr(hub_app, "_voices", registry)
    monkeypatch.setattr(hub_app, "_telegram", provider)
    monkeypatch.setattr(hub_app, "_guest_confirmations", confirmations)
    monkeypatch.setattr(hub_app, "GUEST_FACE_DELAY_S", 0.0)

    cfg = Config()
    cfg.server.permissions_enabled = True
    socket = _FakeSocket()
    connection = hub_app.Connection(socket, cfg)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.room = RoomState()
    connection.room.update([{"id": "track-1", "box": [0.3, 0.2, 0.6, 0.9]}])
    connection._speaker_name = "Drew"
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection.sample_rate = 16000
    connection._request_camera_burst = frames
    return SimpleNamespace(connection=connection, socket=socket, provider=provider,
                           confirmations=confirmations, registry=registry, frames=frames,
                           audit=audit, db=hub_db)


_PHRASE_PCM = b"\x01\x02" * (16000 * 4)  # four seconds of the guest's voice


def test_the_hub_starts_the_flow_from_the_owners_own_words(guest_room):
    turn = asyncio.run(guest_room.connection._guest_turn("Rowan, это Макс, друг", "ru"))
    assert turn is not None and "Макс" in turn
    pending = guest_room.connection._guest_flow.pending
    assert pending is not None and pending.step == STEP_VOICE
    assert pending.requested_by == "legacy-drew" and pending.track_id == "track-1"
    assert any("Макс" in text for text in guest_room.socket.texts())


def test_a_stranger_or_a_guest_cannot_register_anybody(guest_room):
    for role in ("unknown", ROLE_GUEST, "user"):
        guest_room.connection._speaker_role = role
        turn = asyncio.run(guest_room.connection._guest_turn("Rowan, это Макс, друг", "ru"))
        assert turn is not None and "owner or a trusted" in turn
        assert guest_room.connection._guest_flow.pending is None


def test_an_introduction_without_a_name_is_answered_not_guessed(guest_room):
    turn = asyncio.run(guest_room.connection._guest_turn("Rowan, добавь гостя", "ru"))
    assert turn is not None and turn.startswith("Who is it?")
    assert guest_room.connection._guest_flow.pending is None


def test_a_guest_phrase_moves_the_flow_to_the_camera(guest_room):
    connection = guest_room.connection
    asyncio.run(connection._guest_turn("Rowan, это Макс, друг", "ru"))
    connection._current_pcm = _PHRASE_PCM
    turn = asyncio.run(connection._guest_turn(phrase("ru"), "ru"))
    assert turn is not None and "камер" in turn
    pending = connection._guest_flow.pending
    assert pending is not None and pending.step == STEP_FACE
    assert connection._guest_task is not None, "the burst collection was started"
    connection._guest_task.cancel()


def test_a_sample_that_is_not_the_phrase_is_asked_for_again(guest_room):
    connection = guest_room.connection
    asyncio.run(connection._guest_turn("Rowan, это Макс, друг", "ru"))
    connection._current_pcm = _PHRASE_PCM
    turn = asyncio.run(connection._guest_turn("включи свет", "ru"))
    assert turn is not None and "please read it again" in turn
    assert connection._guest_flow.pending is not None
    assert connection._guest_flow.pending.step == STEP_VOICE


def test_the_whole_flow_asks_the_owner_and_stores_nothing_yet(guest_room):
    connection = guest_room.connection

    async def scenario() -> str:
        await connection._guest_turn("Rowan, это Макс, друг", "ru")
        connection._current_pcm = _PHRASE_PCM
        await connection._guest_turn(phrase("ru"), "ru")
        await connection._guest_face_capture(connection._guest_flow.pending)
        return await connection._guest_turn("да", "ru")

    answer = asyncio.run(scenario())
    assert "Telegram" in answer
    pending = connection._guest_flow.pending
    assert pending is not None and pending.step == STEP_OWNER
    assert len(connection._guest_face_vectors) >= MIN_FRAMES
    assert connection._guest_voice_vectors and connection._guest_voice_pcm
    # The owner got exactly one question, with the two buttons of F-210.
    assert len(guest_room.provider.sent) == 1
    _text, kwargs = guest_room.provider.sent[0]
    data = [button["callback_data"]
            for row in kwargs["reply_markup"]["inline_keyboard"] for button in row]
    token = connection._guest_confirmation_token
    assert data == [f"guest:{token}:yes", f"guest:{token}:no"]
    # Nothing is stored before the owner answers.
    assert guest_room.db.execute("SELECT COUNT(*) FROM persons WHERE display_name='Макс'").fetchone()[0] == 0
    assert guest_room.db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0
    assert connection._guest_flow.pending is pending


def test_the_owners_yes_registers_the_guest(guest_room):
    connection, db = guest_room.connection, guest_room.db

    async def scenario() -> Any:
        await connection._guest_turn("Rowan, это Макс, друг", "ru")
        connection._current_pcm = _PHRASE_PCM
        await connection._guest_turn(phrase("ru"), "ru")
        await connection._guest_face_capture(connection._guest_flow.pending)
        await connection._guest_turn("да", "ru")
        request = guest_room.confirmations.resolve(connection._guest_confirmation_token, True)
        assert request is not None
        await connection._guest_owner_decision(request)
        return request

    request = asyncio.run(scenario())
    assert request.name == "Макс" and request.decision == "approved"
    row = db.execute("SELECT person_id FROM persons WHERE display_name='Макс'").fetchone()
    assert row is not None
    person_id = row[0]
    assert membership_role(db, person_id, "livingroom") == ROLE_GUEST
    assert db.execute("SELECT COUNT(*) FROM voice_embeddings WHERE person_id=?",
                      (person_id,)).fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM face_embeddings WHERE person_id=?",
                      (person_id,)).fetchone()[0] >= MIN_FRAMES
    assert db.execute("SELECT person_id FROM tracks WHERE track_id='track-1'").fetchone()[0] == person_id
    assert guest_room.audit.rows and guest_room.audit.rows[0]["action"] == "guest.register"
    # The room is told what happened, and the registry knows the guest.
    assert any("Макс" in text for text in guest_room.socket.texts())
    assert guest_room.registry.enrolled == ["Макс"]
    assert guest_room.registry.roles["Макс"] == ROLE_GUEST
    assert connection._guest_flow.pending is None
    assert not connection._guest_face_vectors and not connection._guest_voice_vectors


def test_the_owners_no_stores_nothing(guest_room):
    connection, db = guest_room.connection, guest_room.db

    async def scenario() -> None:
        await connection._guest_turn("Rowan, это Макс, друг", "ru")
        connection._current_pcm = _PHRASE_PCM
        await connection._guest_turn(phrase("ru"), "ru")
        await connection._guest_face_capture(connection._guest_flow.pending)
        await connection._guest_turn("да", "ru")
        request = guest_room.confirmations.resolve(connection._guest_confirmation_token, False)
        assert request is not None
        await connection._guest_owner_decision(request)

    asyncio.run(scenario())
    assert db.execute("SELECT COUNT(*) FROM persons WHERE display_name='Макс'").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0
    assert connection._guest_flow.pending is None
    assert not connection._guest_face_vectors


def test_without_a_telegram_owner_nothing_is_saved(guest_room, monkeypatch):
    connection, db = guest_room.connection, guest_room.db
    monkeypatch.setattr(hub_app, "_telegram", None)
    monkeypatch.setattr(hub_app, "_guest_confirmations", None)

    async def scenario() -> str:
        await connection._guest_turn("Rowan, это Макс, друг", "ru")
        connection._current_pcm = _PHRASE_PCM
        await connection._guest_turn(phrase("ru"), "ru")
        await connection._guest_face_capture(connection._guest_flow.pending)
        return await connection._guest_turn("да", "ru")

    answer = asyncio.run(scenario())
    assert "no Telegram owner" in answer and "Nothing about Макс" in answer
    assert connection._guest_flow.pending is None
    assert db.execute("SELECT COUNT(*) FROM memberships").fetchone()[0] == 1


def test_the_guest_can_cancel_the_registration_by_voice(guest_room):
    connection, db = guest_room.connection, guest_room.db
    asyncio.run(connection._guest_turn("Rowan, это Макс, друг", "ru"))
    turn = asyncio.run(connection._guest_turn("отмена регистрации", "ru"))
    assert turn is not None and "Cancelled" in turn
    assert connection._guest_flow.pending is None
    assert db.execute("SELECT COUNT(*) FROM persons WHERE display_name='Макс'").fetchone()[0] == 0


def test_frames_from_one_angle_are_retried_and_then_cancelled(guest_room, monkeypatch):
    connection, db = guest_room.connection, guest_room.db
    monkeypatch.setattr(hub_app, "_face", _FakeFaces(angles=(0.0,)))

    async def scenario() -> None:
        await connection._guest_turn("Rowan, это Макс, друг", "ru")
        connection._current_pcm = _PHRASE_PCM
        await connection._guest_turn(phrase("ru"), "ru")
        await connection._guest_face_capture(connection._guest_flow.pending)

    asyncio.run(scenario())
    assert connection._guest_flow.pending is None
    assert any("Nothing was saved" in text or "nothing was saved" in text
               for text in guest_room.socket.texts())
    assert db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == 0


def test_the_registry_gets_a_guest_profile_unless_the_name_was_known(guest_room, monkeypatch):
    connection, registry = guest_room.connection, guest_room.registry
    pending = _pending()
    connection._guest_voice_pcm = _PHRASE_PCM
    connection._guest_face_vectors = [_vector(12, dim=512)]
    connection._guest_registry_write(pending, False)
    assert registry.enrolled == ["Макс"] and registry.faces == ["Макс"]
    assert registry.roles["Макс"] == ROLE_GUEST
    # Somebody who already had a profile keeps their role - the owner's words
    # introduce a NEW guest, they do not demote an existing member.
    known = _FakeRegistry({"Макс": "admin"})
    monkeypatch.setattr(hub_app, "_voices", known)
    connection._guest_registry_write(pending, True)
    assert known.roles["Макс"] == "admin"
    assert known.enrolled == ["Макс"]


def test_the_declaration_reaches_the_room_through_the_utterance_pipeline(guest_room, monkeypatch):
    """The hook is in ``_handle_utterance``, not only in the method."""
    connection, socket = guest_room.connection, guest_room.socket
    from unittest.mock import AsyncMock

    from hub.session import Session

    connection.utterance_id = "01J00000000000000000000000"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    monkeypatch.setattr(hub_app, "_stt",
                        SimpleNamespace(transcribe_pcm=lambda *args: ("Rowan, это Макс, друг", "ru")))
    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(
        generate=AsyncMock(side_effect=AssertionError("the guest flow must not need the model")),
        verify=AsyncMock(side_effect=AssertionError("no verifier")),
    ))
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000))
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    connection._stream_tts = AsyncMock()

    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert any("Макс" in text for text in socket.texts())
    pending = connection._guest_flow.pending
    assert pending is not None and pending.requested_by == "legacy-drew"
