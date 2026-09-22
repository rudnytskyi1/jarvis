"""Поза, режим сна и утренний подъём (ТЗ F-307/F-420, задачи P5-11 и P5-12)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from client.posture import (
    LYING,
    SITTING,
    STANDING,
    PoseDetector,
    PoseUnavailable,
    PostureService,
    StillnessWatch,
    body_centre,
    posture_of,
)
from common.protocol import CLIENT_MESSAGE_TYPES, MSG_POSTURE_EVENT, PHONE_FORBIDDEN_INPUTS
from hub import app as hub_app
from hub import migrations_runner
from hub.audit import AuditLog
from hub.home_modes import ASLEEP, AWAKE, HomeModes
from hub.homes import ensure_home

HOME = "bedroom"


def _skeleton(*, shoulders=(0.45, 0.40), hips=(0.45, 0.60), knees=None, ankles=None):
    """17 COCO-точек: достаточно плеч и бёдер, чтобы поза была понятной."""
    points = [(0.5, 0.5, 0.0)] * 17
    points[5] = (shoulders[0], shoulders[1], 0.9)
    points[6] = (shoulders[0] + 0.02, shoulders[1], 0.9)
    points[11] = (hips[0], hips[1], 0.9)
    points[12] = (hips[0] + 0.02, hips[1], 0.9)
    if knees is not None:
        points[13] = (knees[0], knees[1], 0.9)
        points[14] = (knees[0] + 0.02, knees[1], 0.9)
    if ankles is not None:
        points[15] = (ankles[0], ankles[1], 0.9)
        points[16] = (ankles[0] + 0.02, ankles[1], 0.9)
    return points


# ---------------------------------------------------------------------------
# поза
# ---------------------------------------------------------------------------


def test_the_body_says_whether_the_person_lies_or_stands():
    # Туловище почти горизонтально — человек лежит, как его ни поверни.
    assert posture_of(_skeleton(shoulders=(0.20, 0.50), hips=(0.70, 0.52))) == LYING
    # Туловище вертикально, ноги видны и не сложены — стоит.
    assert posture_of(_skeleton(knees=(0.45, 0.75), ankles=(0.45, 0.95))) == STANDING
    # Ноги сложены — сидит.
    assert posture_of(_skeleton(knees=(0.75, 0.70), ankles=(0.50, 0.95))) == SITTING
    # Ног не видно (человек за столом) — «сидит», а не «встал».
    assert posture_of(_skeleton()) == SITTING
    # Точки потерялись — позу не выдумываем.
    assert posture_of([(0.5, 0.5, 0.0)] * 17) == ""
    assert posture_of(None) == "" and posture_of([]) == ""
    # Центр тела — середина плеч и бёдер.
    centre = body_centre(_skeleton(shoulders=(0.40, 0.40), hips=(0.60, 0.60)))
    assert centre is not None
    assert abs(centre[0] - 0.51) < 1e-9 and abs(centre[1] - 0.5) < 1e-9


def test_the_stillness_watch_fires_once_and_wakes_on_sitting_up():
    watch = StillnessWatch(still_s=600.0)
    centre = (0.5, 0.5)
    assert watch.observe(LYING, centre, 1000.0) == ""
    assert watch.observe(LYING, centre, 1300.0) == ""
    assert watch.observe(LYING, centre, 1600.0) == "sleep"      # 10 минут лежит
    assert watch.observe(LYING, centre, 2000.0) == ""           # повторно не будим
    assert watch.asleep is True
    # Встал — и это повод для утренней рутины, один раз.
    assert watch.observe(STANDING, (0.5, 0.4), 2600.0) == "awake"
    assert watch.observe(STANDING, (0.5, 0.4), 2700.0) == ""
    # Пропажа из кадра подъёмом не считается: человек мог укрыться одеялом.
    assert watch.observe(LYING, centre, 3000.0) == ""
    assert watch.observe("", None, 3600.0) == ""
    assert watch.asleep is False


def test_moving_in_bed_postpones_the_sleep_mode():
    watch = StillnessWatch(still_s=600.0)
    watch.observe(LYING, (0.50, 0.50), 0.0)
    # Человек повернулся на 20-й минуте — отсчёт начинается заново.
    assert watch.observe(LYING, (0.80, 0.50), 1200.0) == ""
    assert watch.observe(LYING, (0.80, 0.50), 1500.0) == ""
    assert watch.observe(LYING, (0.80, 0.50), 1800.0) == "sleep"


class _Detector:
    def __init__(self, *people):
        self.people = list(people)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [{"keypoints": person} for person in self.people]


def test_sleep_mode_is_born_only_in_quiet_hours():
    lying = _skeleton(shoulders=(0.20, 0.50), hips=(0.70, 0.52))
    service = PostureService(SimpleNamespace(enabled=True, interval_s=0.0, still_s=60.0),
                             detector=_Detector(lying))
    events: list[str] = []
    service._hook = events.append
    assert service.submit(object(), now=1000.0, quiet=True) == []
    # Днём «лежит неподвижно» — отдых, а не сон (ТЗ F-307).
    assert service.submit(object(), now=1300.0, quiet=False) == []
    assert events == [] and service.watch.asleep is False
    # В тихие часы тот же кадр становится режимом сна.
    assert service.submit(object(), now=1600.0, quiet=True) == []
    assert service.submit(object(), now=1900.0, quiet=True) == ["sleep"]
    assert events == ["sleep"]
    # Выключенный флаг дома гасит всё.
    assert service.set_enabled(False) is True
    assert service.submit(object(), now=2200.0, quiet=True) == []


def test_the_pose_model_is_lazy_and_named_when_missing():
    detector = PoseDetector(model="нет-таких-весов.pt")
    try:
        import ultralytics  # noqa: F401

        pytest.skip("ultralytics установлен в этом окружении")
    except ImportError:
        pass
    with pytest.raises(PoseUnavailable):
        detector.detect(object())
    assert "ultralytics" in detector._error


# ---------------------------------------------------------------------------
# режим сна дома
# ---------------------------------------------------------------------------


def _conn(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, HOME, name="Bedroom", tz="America/Chicago")
    return conn


def test_the_mode_and_the_wakeups_survive_a_reconnect(tmp_path):
    conn = _conn(tmp_path)
    try:
        modes = HomeModes(conn)
        assert modes.mode(HOME) == "" and modes.asleep(HOME) is False
        assert modes.set_mode(HOME, ASLEEP, at=100.0) == ASLEEP
        assert modes.asleep(HOME) is True and modes.asleep_since(HOME) == 100.0
        assert modes.set_mode(HOME, AWAKE, at=200.0) == AWAKE
        assert modes.asleep(HOME) is False and modes.asleep_since(HOME) == 0.0
        # Неизвестный режим не записывается, чужой дом не падает.
        assert modes.set_mode(HOME, "обед") == "" and modes.set_mode("", ASLEEP) == ""
        # Подъём помнится по дням.
        assert modes.note_wakeup(HOME, "person-max", at=300.0) is True
        assert modes.note_wakeup(HOME, "", at=300.0) is False
        assert modes.wakeups(HOME, start=0.0, end=1000.0) == {"person-max": 300.0}
        assert modes.wakeups(HOME, start=400.0, end=1000.0) == {}
        assert modes.snapshot()["wakeups"] == 1
        assert modes.forget_before(400.0) == 1
        assert modes.wakeups(HOME, start=0.0, end=1000.0) == {}
    finally:
        conn.close()


def _connection(home=HOME):
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.peer = "bedroom-pc"
    conn.home_id = home
    conn.session = SimpleNamespace(client_id="bedroom")
    conn._speaker_name = ""
    return conn


def test_the_room_going_to_sleep_silences_the_home(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    modes = HomeModes(conn)
    monkeypatch.setattr(hub_app, "_home_modes", modes)
    monkeypatch.setattr(hub_app, "_audit", AuditLog(conn))
    monkeypatch.setattr(hub_app, "_home_settings_of", lambda home: {"sleep_scene": "ночь"})
    applied: list = []

    async def scene(self, home, name):
        applied.append((home, name))
        return "ночь (2/2)"

    monkeypatch.setattr(hub_app.Connection, "_apply_home_scene", scene)
    try:
        connection = _connection()
        asyncio.run(connection._on_posture_event({"state": "sleep"}))
        assert modes.asleep(HOME) is True
        assert applied == [(HOME, "ночь")]
        row = conn.execute("SELECT action, result FROM audit WHERE action='home.sleep'").fetchone()
        assert row == ("home.sleep", "ok")
        # Уведомления этого дома становятся беззвучными.
        monkeypatch.setattr(hub_app, "get_config",
                            lambda: SimpleNamespace(homes=[SimpleNamespace(
                                home_id=HOME, quiet_hours=None, tz="America/Chicago",
                                alert_cooldown_s=0)]))
        assert hub_app._alert_home(HOME)["asleep"] is True
        assert hub_app._home_asleep(HOME) is True

        # «Встал» гасит режим и запоминает подъём (повод для F-420).
        connection._speaker_name = ""
        asyncio.run(connection._on_posture_event({"state": "awake"}))
        assert modes.asleep(HOME) is False
        assert conn.execute("SELECT action FROM audit WHERE action='home.wake'").fetchone() \
            is not None
        # Событие без дома или с чужим состоянием ничего не делает.
        asyncio.run(_connection("")._on_posture_event({"state": "sleep"}))
        asyncio.run(connection._on_posture_event({"state": "обед"}))
        assert modes.mode("") == ""
    finally:
        monkeypatch.setattr(hub_app, "_home_modes", None)
        conn.close()


def test_the_wakeup_becomes_the_morning_briefing_trigger(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    modes = HomeModes(conn)
    monkeypatch.setattr(hub_app, "_home_modes", modes)
    monkeypatch.setattr(hub_app, "_presence_log", lambda: None)
    monkeypatch.setattr(hub_app, "_home_timezone_of", lambda home: "America/Chicago")
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        moment = datetime(2026, 9, 22, 8, 30, tzinfo=ZoneInfo("America/Chicago"))
        modes.note_wakeup(HOME, "person-max", at=moment.timestamp())
        assert hub_app._briefing_entries(HOME, moment) == {"person-max": moment.timestamp()}
        # Вчерашний подъём сегодняшним поводом не считается.
        assert hub_app._briefing_entries(
            HOME, datetime(2026, 9, 23, 8, 30, tzinfo=ZoneInfo("America/Chicago"))) == {}
    finally:
        monkeypatch.setattr(hub_app, "_home_modes", None)
        conn.close()


def test_the_posture_event_is_a_known_client_message():
    assert MSG_POSTURE_EVENT in CLIENT_MESSAGE_TYPES
    assert MSG_POSTURE_EVENT in PHONE_FORBIDDEN_INPUTS


def test_a_sleeping_home_gets_a_silent_caption_instead_of_telegram(tmp_path):
    """ТЗ F-307: пока дом спит, уведомление не звонит, а показывается подписью."""
    from hub.presence_alerts import PresenceAlerts

    class _Provider:
        ready = True

        def __init__(self):
            self.sent = []

        async def send_image(self, *args, **kwargs):
            self.sent.append(args)
            return {"ok": True, "chat_id": 1, "message_id": 1}

        async def send_video(self, *args, **kwargs):
            self.sent.append(args)
            return {"ok": True, "chat_id": 1, "message_id": 1}

    class _Room:
        workplace_name = "Спальня"

        def __init__(self):
            self.captions = []

        async def send_json(self, payload):
            self.captions.append(payload)

    async def scenario():
        provider, room = _Provider(), _Room()
        alerts = PresenceAlerts(tmp_path, lambda: provider, lambda client_id=None: room, 7, -100,
                                get_home=lambda home_id: {"asleep": True,
                                                          "timezone": "America/Chicago"})
        alerts.start()
        try:
            alerts.save_rule(dict(enabled=True, event="presence", target="any",
                                  media="photo", home_id=HOME, min_frames=1))
            alerts.observe(persons=1, source_id="bedroom", home_id=HOME, jpeg=b"jpeg-bytes")
            await alerts.drain()
            assert provider.sent == [], "спящий дом не звонит в Telegram"
            assert room.captions and room.captions[0]["type"] == "status"
            assert alerts.status()["deliveries"][0]["status"] == "sent"
        finally:
            await alerts.close()

    asyncio.run(scenario())
