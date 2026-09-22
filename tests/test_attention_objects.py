"""Объекты внимания F-311: классы, зона и уведомление по правилу дома (P5-07).

Проверяется настоящий путь: клиентский YOLO-детект (подставная модель) даёт
объект внимания вместе с зоной дома, событие уходит хабу, а правило владельца —
написанное на его языке и с его зоной — решает, слать ли уведомление
«посылка у двери».
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from client.camera import CameraService
from common.attention_objects import (
    ATTENTION_GROUPS,
    attention_group,
    attention_line,
    attention_word,
)
from common.frame_zones import FrameZone
from common.protocol import CLIENT_MESSAGE_TYPES, MSG_OBJECT_EVENT, PHONE_FORBIDDEN_INPUTS
from hub import app as hub_app
from hub.presence_alerts import PresenceAlerts, validate_rule

DOOR = FrameZone(name="дверь", points=[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]])
DESK = FrameZone(name="стол", points=[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])
SECRET = FrameZone(name="маска", mask=True,
                   points=[[0.0, 0.75], [1.0, 0.75], [1.0, 1.0], [0.0, 1.0]])


# ---------------------------------------------------------------------------
# слова и группы
# ---------------------------------------------------------------------------


def test_only_objects_of_attention_have_a_group():
    for word in ("cat", "Cat", "cats", "кошка", "кошки", "кот", "gato", "kitten"):
        assert attention_group(word) == "cat"
    assert attention_group("dog") == "dog" and attention_group("Perro") == "dog"
    for word in ("box", "package", "посылка", "коробка", "paquete", "parcel", "suitcase"):
        assert attention_group(word) == "package"
    # Всё остальное — не объект внимания: правило не должно срабатывать на мебель.
    for word in ("chair", "стул", "cup", "чашка", "", "   ", None, "person"):
        assert attention_group(word) == ""


def test_the_object_is_named_in_the_language_of_the_owner():
    assert attention_word("package", "ru") == "посылка"
    assert attention_word("package", "en") == "package"
    assert attention_word("package", "es") == "paquete"
    assert attention_word("package", "de") == "package"      # неизвестный язык — английский
    assert attention_word("unicorn", "ru") == ""             # выдумывать слово нельзя
    assert attention_line("package", "дверь", "ru") == "посылка у дверь"
    # Имя зоны — слова владельца: оно не переводится, только чистится.
    assert attention_line("cat", "  у  окна ", "ru") == "кошка у у окна"
    assert attention_line("dog", "door", "en") == "dog at door"
    assert attention_line("dog", "", "en") == "dog"          # без зоны — просто слово


def test_the_groups_are_the_ones_the_tz_names():
    assert set(ATTENTION_GROUPS) == {"cat", "dog", "package"}


# ---------------------------------------------------------------------------
# клиент: объект внимания вместе с зоной
# ---------------------------------------------------------------------------


class _List:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _Boxes:
    def __init__(self, rows):
        self.cls = _List(row[0] for row in rows)
        self.conf = _List(row[1] for row in rows)
        self.id = _List(row[2] for row in rows) if any(row[2] is not None for row in rows) else None
        self.xyxyn = _List([row[3:] for row in rows])


class _Result:
    names = {0: "person", 1: "cat", 2: "chair", 3: "dog", 4: "box"}

    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class _Model:
    def __init__(self, rows):
        self._rows = rows

    def track(self, **kwargs):
        return [_Result(self._rows)]


def _camera(*zones):
    camera = CameraService(SimpleNamespace(enabled=False))
    if zones:
        camera.set_zones([zone.model_dump() for zone in zones])
    return camera


def test_the_client_signs_the_object_with_its_zone():
    camera = _camera(DOOR, DESK, SECRET)
    rows = [
        (1, 0.9, 7, 0.60, 0.10, 0.80, 0.60),   # cat справа — «дверь»
        (2, 0.8, None, 0.10, 0.10, 0.30, 0.40),  # chair — не объект внимания
        (3, 0.7, None, 0.10, 0.85, 0.30, 0.95),  # dog в маске — молчим
        (4, 0.6, None, 0.10, 0.10, 0.30, 0.30),  # box слева — «стол»
        (0, 0.95, 3, 0.40, 0.10, 0.50, 0.50),   # person — не объект внимания
    ]
    persons, objects = camera._detect(_Model(rows), object())
    attention = camera._attention_found
    assert persons == 1 and objects == {"cat": 1, "chair": 1, "dog": 1, "box": 1}
    assert attention == [
        {"label": "cat", "zone": "дверь", "conf": 0.9},
        {"label": "package", "zone": "стол", "conf": 0.6},
    ]
    assert camera._tracks and camera._tracks[0]["id"] == f"{camera._track_epoch}:3"


def test_an_object_without_a_zone_is_still_a_finding():
    camera = _camera()
    persons, _objects = camera._detect(
        _Model([(3, 0.9, None, 0.10, 0.10, 0.30, 0.30)]), object())
    attention = camera._attention_found
    assert persons == 0
    assert attention == [{"label": "dog", "zone": "", "conf": 0.9}]


def _publish(camera, attention):
    """Собрать кадры, которые комната отправила бы по этому детекту."""
    sent: list[dict] = []

    async def send_json(payload):
        sent.append(payload)

    camera._send_json = send_json
    camera._submit = lambda coro: asyncio.run(coro)
    camera._publish_attention(attention)
    return sent


def test_the_event_is_born_on_the_transition_not_every_frame():
    camera = _camera(DOOR)
    found = [{"label": "package", "zone": "дверь", "conf": 0.8}]
    first = _publish(camera, found)
    assert len(first) == 1
    assert first[0]["type"] == MSG_OBJECT_EVENT
    assert first[0]["label"] == "package" and first[0]["zone"] == "дверь"
    assert first[0]["conf"] == 0.8 and first[0]["event_id"]
    assert isinstance(first[0]["at_ms"], int) and first[0]["at_ms"] > 1_600_000_000_000
    # Пока посылка стоит на месте, правило не получает событие каждую секунду.
    assert _publish(camera, found) == []
    # Уехала — и вернулась: это снова событие.
    assert _publish(camera, []) == []
    assert len(_publish(camera, found)) == 1
    # Другая зона — другое событие: правило «у двери» должно отличить стол.
    assert len(_publish(camera, [{"label": "package", "zone": "стол", "conf": 0.7}])) == 1


def test_the_room_stays_silent_in_privacy_mode_and_on_a_bare_socket():
    camera = _camera(DOOR)
    camera.privacy.set(True, reason="voice")
    assert _publish(camera, [{"label": "cat", "zone": "дверь", "conf": 0.9}]) == []
    # Камера уже что-то сказала, но socket мёртв: это не исключение наружу.
    camera.privacy.set(False)
    sent: list[dict] = []

    async def broken(payload):
        raise ConnectionResetError("down")

    camera._send_json = broken
    camera._submit = lambda coro: asyncio.run(coro)
    camera._publish_attention([{"label": "cat", "zone": "дверь", "conf": 0.9}])
    assert sent == [] and camera._send_errors == 1


# ---------------------------------------------------------------------------
# хаб: событие уходит правилам с зоной
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.events = []

    def observe_event(self, kind, **payload):
        self.events.append((kind, payload))
        return True


def _connection(home_id="livingroom"):
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.peer = "room-pc"
    conn.session = SimpleNamespace(client_id="client-1")
    conn.home_id = home_id
    return conn


def test_the_hub_turns_the_object_event_into_a_rule_event(monkeypatch):
    alerts = _Recorder()
    monkeypatch.setattr(hub_app, "_presence_alerts", alerts)
    conn = _connection()
    conn._on_object_event({"label": "box", "zone": " дверь ", "conf": 0.8})
    assert alerts.events == [("object", {"label": "package", "zone": "дверь",
                                         "confidence": 0.8, "source_id": "client-1",
                                         "home_id": "livingroom"})]
    # Метка вне объектов внимания событием не становится.
    conn._on_object_event({"label": "chair", "zone": "дверь", "conf": 0.9})
    assert len(alerts.events) == 1
    # Без сервиса правил ход комнаты не ломается.
    monkeypatch.setattr(hub_app, "_presence_alerts", None)
    conn._on_object_event({"label": "cat", "zone": "", "conf": 0.5})


def test_the_object_event_is_a_known_client_message():
    assert MSG_OBJECT_EVENT in CLIENT_MESSAGE_TYPES
    # Телефон без камеры не может «увидеть» объект (ТЗ F-711).
    assert MSG_OBJECT_EVENT in PHONE_FORBIDDEN_INPUTS


def test_the_profile_of_the_rule_matches_the_objects_of_the_camera():
    # Правило владельца написано словами его языка, а событие приходит группой.
    rule = validate_rule({"event": "object", "name": "посылка", "zone": "дверь"})
    assert PresenceAlerts._event_match(rule, {"kind": "object", "label": "package",
                                              "zone": "дверь"}) is True
    assert PresenceAlerts._event_match(rule, {"kind": "object", "label": "package",
                                              "zone": "стол"}) is False
    assert PresenceAlerts._event_match(rule, {"kind": "object", "label": "cat",
                                              "zone": "дверь"}) is False
    # «Кошка» и «кошки» — одна и та же группа, регистр не важен.
    cat_rule = validate_rule({"event": "object", "name": "Кошки"})
    assert PresenceAlerts._event_match(cat_rule, {"kind": "object", "label": "cat"}) is True


def test_the_notification_names_the_object_and_the_zone():
    rule = validate_rule({"event": "object", "name": "посылка", "zone": "дверь"})
    text = PresenceAlerts._event_text(None, rule, {"kind": "object", "label": "package",
                                                  "zone": "дверь"})
    assert text == "the camera sees: package at дверь."
    # Без зоны строка честно называет только объект.
    text = PresenceAlerts._event_text(None, rule, {"kind": "object", "label": "package"})
    assert text == "the camera sees: package."
    # Чужой ярлык от старого клиента показывается как есть, а не теряется.
    text = PresenceAlerts._event_text(None, rule, {"kind": "object", "label": "хрень",
                                                   "zone": ""})
    assert text == "the camera sees: хрень."
