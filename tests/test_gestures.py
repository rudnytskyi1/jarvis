"""Жесты руки: ладонь дольше секунды останавливает TTS (ТЗ F-306, P5-08)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from client.gestures import (
    DEFAULT_HOLD_S,
    GESTURES,
    GestureHold,
    GestureService,
    GestureUnavailable,
    MediaPipeHands,
    recognize,
    truthy,
)
from client.main import JarvisClient


def _hand(*, index=False, middle=False, ring=False, pinky=False, thumb=False,
          thumb_up=False):
    """21 точка MediaPipe: запястье внизу, выпрямленный палец — далеко от него."""
    points = [(0.5, 0.9, 0.0)] * 21
    points[0] = (0.5, 0.9, 0.0)                      # wrist
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    bases = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    states = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, tip in tips.items():
        base = bases[name]
        if states[name]:
            points[base] = (0.5, 0.80, 0.0)
            points[tip] = (0.5, 0.30, 0.0)
        else:
            points[base] = (0.5, 0.80, 0.0)
            points[tip] = (0.5, 0.88, 0.0)      # палец сжат: кончик ближе к запястью
    points[2] = (0.5, 0.80, 0.0)                     # THUMB_MCP
    points[4] = (0.5, 0.35 if thumb else 0.86, 0.0)  # THUMB_TIP
    if thumb_up:
        points[4] = (0.5, 0.35, 0.0)
    return points


# ---------------------------------------------------------------------------
# распознавание
# ---------------------------------------------------------------------------


def test_the_three_gestures_of_the_tz_are_told_apart():
    assert recognize(_hand(index=True, middle=True, ring=True, pinky=True, thumb=True)) == "palm"
    assert recognize(_hand(index=True)) == "point"
    assert recognize(_hand(thumb_up=True)) == "thumb_up"
    # Сжатая рука — не жест: «наверное, он хотел» тут недопустимо.
    assert recognize(_hand()) == ""
    assert recognize([]) == ""
    assert recognize(None) == ""
    assert recognize([None] * 21) == ""


def test_a_gesture_does_not_depend_on_how_the_hand_is_turned():
    # Ладонь вбок: «выпрямлен» считается по расстоянию до запястья, а не по
    # высоте пальца на картинке.
    side = [(x, y, 0.0) for x, y, _ in _hand(index=True, middle=True, ring=True, pinky=True)]
    for index, (_x, y, _z) in enumerate(list(side)):
        side[index] = (0.9 - (y - 0.2), 0.5, 0.0)
    assert recognize(side) == "palm"


def test_the_hold_is_one_shock_not_a_tick_per_frame():
    hold = GestureHold(1.0)
    assert hold.observe("palm", 100.0) is False        # жест только начался
    assert hold.observe("palm", 100.5) is False        # полсекунды — мало
    assert hold.observe("palm", 101.0) is True         # секунда прошла: стоп
    assert hold.observe("palm", 101.2) is False        # пока держат — тишина
    # Руку убрали — и вернули: это снова «стоп», а не «уже сработало».
    assert hold.observe("", 102.0) is False
    assert hold.observe("palm", 103.0) is False
    assert hold.observe("palm", 104.1) is True
    # Другой жест вместо ладони сбрасывает отсчёт.
    hold.observe("", 105.0)
    assert hold.observe("point", 106.0) is False
    assert hold.observe("point", 107.5) is True


# ---------------------------------------------------------------------------
# сервис жестов
# ---------------------------------------------------------------------------


class _Detector:
    def __init__(self, *hands):
        self.hands = list(hands)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self.hands)


def test_the_service_fires_only_after_the_hold_and_only_when_enabled():
    detector = _Detector(_hand(index=True, middle=True, ring=True, pinky=True))
    fired: list[str] = []
    service = GestureService(SimpleNamespace(enabled=True, hold_s=1.0, interval_s=0.0),
                             detector=detector, on_event=fired.append)
    assert service.enabled is True
    assert service.submit(object(), now=10.0) == []
    assert service.submit(object(), now=10.5) == []
    assert service.submit(object(), now=11.0) == ["palm"]
    assert fired == ["palm"]
    assert service.events == 1
    # Выключенный флаг дома гасит всё, даже если рука в кадре.
    assert service.set_enabled(False) is True
    assert service.set_enabled(False) is False
    before = detector.calls
    assert service.submit(object(), now=20.0) == []
    assert detector.calls == before


def test_a_default_service_does_nothing_until_the_home_turns_it_on():
    detector = _Detector(_hand(index=True, middle=True, ring=True, pinky=True))
    service = GestureService(SimpleNamespace(), detector=detector)
    assert service.enabled is False and DEFAULT_HOLD_S == 1.0
    assert service.submit(object(), now=1.0) == []
    # Дом включает жесты — и тот же сервис начинает работать.
    assert service.set_enabled(True) is True
    service.submit(object(), now=2.0)
    assert service.submit(object(), now=3.1) == ["palm"]
    # Строка "false" из yaml — это НЕТ, а не непустая строка.
    assert service.set_enabled("false") is True
    assert service.enabled is False
    assert truthy("on") is True and truthy("") is False and truthy(None) is False


def test_the_camera_frame_rate_is_capped_and_errors_do_not_escape():
    detector = _Detector(_hand(index=True, middle=True, ring=True, pinky=True))
    service = GestureService(SimpleNamespace(enabled=True, hold_s=0.1, interval_s=0.2),
                             detector=detector)
    service.submit(object(), now=1.0)
    service.submit(object(), now=1.1)       # слишком часто: кадр пропущен
    assert detector.calls == 1
    service.submit(object(), now=1.3)
    assert detector.calls == 2

    class _Broken:
        def detect(self, frame):
            raise RuntimeError("nope")

    broken = GestureService(SimpleNamespace(enabled=True), detector=_Broken())
    assert broken.submit(object()) == []    # ход комнаты не падает
    assert broken._errors == 1
    unavailable = GestureService(SimpleNamespace(enabled=True), detector=MediaPipeHands())
    assert unavailable.submit(object()) == []
    assert "mediapipe" in unavailable._warned
    # Вторая попытка не переимпортирует сломанный пакет на каждом кадре.
    assert unavailable.submit(object()) == []


def test_no_mediapipe_is_named_not_hidden():
    hands = MediaPipeHands()
    try:
        import mediapipe  # noqa: F401

        pytest.skip("mediapipe установлен в этом окружении")
    except ImportError:
        pass
    with pytest.raises(GestureUnavailable) as error:
        hands.detect(object())
    assert "mediapipe" in str(error.value)
    assert "mediapipe" in hands._error          # причина названа один раз
    assert set(GESTURES) == {"palm", "thumb_up", "point"}


# ---------------------------------------------------------------------------
# клиент: ладонь останавливает речь
# ---------------------------------------------------------------------------


class _Audio:
    def __init__(self):
        self.dropped = 0

    def cancel_pending(self):
        self.dropped += 1
        return 3


def test_the_palm_stops_the_voice_and_the_rest_of_the_stream():
    assistant = JarvisClient.__new__(JarvisClient)
    assistant.audio_out = _Audio()
    assistant._idle_tts_active = False
    assistant._idle_stream_active = False
    assistant._idle_playing = False
    assistant._idle_interrupted = False
    assistant._on_gesture("palm")
    assert assistant.audio_out.dropped == 1
    assert assistant._stopped_by_gesture is True
    # Остальные кадры синтеза глотаются, а не звучат дальше.
    assistant._tts_active = True
    assistant._tts_bytes = 0
    assistant._barged = False
    assistant.sample_rate = 22050
    assistant._reply_pcm = b""
    assistant._remember_reply_audio = lambda data: None
    import asyncio

    asyncio.run(assistant._on_tts_chunk(b"pcm"))
    assert assistant._tts_bytes == 0
    # Чужой жест ничего не делает: подтверждение — своя задача (P5-09).
    assistant._stopped_by_gesture = False
    assistant._on_gesture("thumb_up")
    assert assistant.audio_out.dropped == 1


def test_the_home_flag_turns_the_gestures_on_from_the_room_patch():
    from common.protocol import MSG_CONFIG_UPDATE

    class _Gestures:
        def __init__(self):
            self.states = []

        def set_enabled(self, value):
            self.states.append(value)

    class _Overlay:
        def set_status(self, text):
            pass

    assistant = JarvisClient.__new__(JarvisClient)
    assistant.overlay = _Overlay()
    assistant.room_config_rev = 0
    assistant.room_config = {}
    assistant.gestures = _Gestures()
    assistant._apply_room_config(
        {"type": MSG_CONFIG_UPDATE, "config_rev": 3,
         "patch": {"settings": {"gestures": "true"}}}
    )
    assert assistant.gestures.states == ["true"]
    # Патч без флага жестов ничего не меняет: чужой дом живёт как жил.
    assistant._apply_room_config(
        {"type": MSG_CONFIG_UPDATE, "config_rev": 4, "patch": {"name": "Other"}}
    )
    assert assistant.gestures.states == ["true"]


def test_the_camera_hands_every_frame_to_the_gesture_service():
    camera = SimpleNamespace(frames=[], listener=None)
    assistant = JarvisClient.__new__(JarvisClient)
    submitted: list = []
    assistant.gestures = SimpleNamespace(submit=submitted.append)
    assistant.camera = camera
    assistant._on_camera_frame("кадр")
    assert submitted == ["кадр"]
    # Клиент без жестов (тесты, старое железо) молчит, а не падает.
    assistant.gestures = None
    assistant._on_camera_frame("кадр")


class _OverlayConfirm:
    """HUD подтверждения: кнопка, которую в этих тестах нажимает жест."""

    def __init__(self):
        self.callback = None
        self.shown = []
        self.cancelled = 0

    def confirm_voice(self, details, callback):
        self.shown.append(details)
        self.callback = callback

    def cancel_voice_confirmation(self):
        self.cancelled += 1


class _WS:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


def _confirming_client():
    assistant = JarvisClient.__new__(JarvisClient)
    assistant.overlay = _OverlayConfirm()
    assistant.ws = _WS()
    assistant._quiet_turn = False

    async def _await_actions():
        return None

    assistant._await_actions = _await_actions
    return assistant


def test_a_thumb_up_answers_the_pending_confirmation_as_yes():
    import asyncio

    assistant = _confirming_client()

    async def scenario():
        task = asyncio.get_running_loop().create_task(
            assistant._handle_voice_confirmation({"id": "c1", "kind": "log", "name": "Блокнот"})
        )
        # Ждём, пока подтверждение появится на HUD, и отвечаем жестом —
        # ровно так же, как это сделал бы человек пальцем.
        for _ in range(100):
            await asyncio.sleep(0.01)
            if assistant._confirmation_resolve is not None:
                break
        assert assistant.overlay.shown == ["Блокнот"]
        assistant._on_gesture("thumb_up")
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert assistant.ws.sent == [{"type": "voice_confirmation_result",
                                  "id": "c1", "approved": True}]
    assert assistant.overlay.cancelled == 1
    # Жест больше ни за что не отвечает: подтверждение уже закрыто.
    assert assistant._confirmation_resolve is None


def test_a_thumb_up_without_a_question_stays_silent():
    assistant = _confirming_client()
    assistant._confirmation_resolve = None
    assistant._on_gesture("thumb_up")            # ничего не ждёт ответа
    assert assistant.overlay.cancelled == 0
    # Ладонь по-прежнему обрабатывается как «стоп».
    assistant.audio_out = _Audio()
    assistant._idle_tts_active = False
    assistant._idle_stream_active = False
    assistant._idle_playing = False
    assistant._idle_interrupted = False
    assistant._on_gesture("palm")
    assert assistant.audio_out.dropped == 1
