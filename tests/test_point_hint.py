"""Указание пальцем: «что это?» смотрит туда, куда показали (ТЗ F-306, P5-10)."""
from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from PIL import Image

from client.gestures import GestureService, point_hint, recognize
from common.protocol import CLIENT_MESSAGE_TYPES, MSG_POINT_EVENT, PHONE_FORBIDDEN_INPUTS
from hub import app as hub_app


def _hand(pointer=(0.50, 0.50), base=(0.50, 0.60)):
    """21 точка MediaPipe: указательный палец указывает из ``base`` в ``pointer``."""
    points = [(0.5, 0.9, 0.0)] * 21
    points[0] = (0.5, 0.9, 0.0)                     # wrist
    points[5] = (base[0], base[1], 0.0)             # index MCP
    points[8] = (pointer[0], pointer[1], 0.0)       # index tip (выпрямлен)
    points[6] = (pointer[0], (base[1] + pointer[1]) / 2, 0.0)
    for tip, joint in ((12, 9), (16, 13), (20, 17)):
        points[joint] = (0.5, 0.80, 0.0)
        points[tip] = (0.5, 0.88, 0.0)
    points[2], points[4] = (0.5, 0.80, 0.0), (0.5, 0.86, 0.0)
    return points


def test_the_point_is_beyond_the_finger_not_on_it():
    # Палец смотрит вверх: предмет ЗА кончиком, а не на самом кончике.
    hint = point_hint(_hand(pointer=(0.50, 0.40), base=(0.50, 0.60)))
    assert hint is not None
    x, y = hint
    assert abs(x - 0.5) < 1e-6
    assert y < 0.40, "точка указывает за кончиком пальца"
    # Указание за пределы кадра не выдумывается.
    assert point_hint(_hand(pointer=(0.99, 0.10), base=(0.60, 0.60))) is None
    assert point_hint(None) is None
    assert point_hint([None] * 21) is None


def test_the_service_remembers_where_the_person_pointed():
    service = GestureService(SimpleNamespace(enabled=True, interval_s=0.0),
                             detector=SimpleNamespace(
                                 detect=lambda frame: [_hand(pointer=(0.5, 0.4), base=(0.5, 0.6))]))
    assert service.last_point is None
    service.submit(object(), now=5.0)
    assert service.last_point is not None
    assert service.last_point["at"] == 5.0
    assert service.last_point["y"] < 0.4
    # Жест указания распознан именно как указание.
    assert recognize(_hand(pointer=(0.5, 0.4), base=(0.5, 0.6))) == "point"


class _Gestures:
    def __init__(self, hint):
        self.last_point = hint


def _client_with(hint):
    from client.main import JarvisClient

    assistant = JarvisClient.__new__(JarvisClient)
    assistant.gestures = _Gestures(hint) if hint is not None else None
    assistant.ws = SimpleNamespace(sent=[], send_json=None)

    async def send_json(payload):
        assistant.ws.sent.append(payload)

    assistant.ws.send_json = send_json
    assistant._loop = None
    return assistant


def test_the_client_sends_the_direction_and_nothing_else():
    assistant = _client_with({"x": 0.25, "y": 0.75, "at": 1.0})

    async def scenario():
        assistant._loop = asyncio.get_running_loop()
        assistant._on_gesture("point")
        for _ in range(50):
            await asyncio.sleep(0.01)
            if assistant.ws.sent:
                break

    asyncio.run(scenario())
    assert len(assistant.ws.sent) == 1
    payload = assistant.ws.sent[0]
    assert payload["type"] == MSG_POINT_EVENT
    assert (payload["x"], payload["y"]) == (0.25, 0.75)
    assert payload["event_id"] and isinstance(payload["at_ms"], int)
    # За кадром (или без жестов) молчим: «наверное, вон туда» хаб не поймёт.
    empty = _client_with(None)
    empty._on_gesture("point")
    assert empty.ws.sent == []
    off_frame = _client_with({"x": 1.5, "y": 0.5})
    off_frame._on_gesture("point")
    assert off_frame.ws.sent == []


def _connection():
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.peer = "room-pc"
    conn.home_id = "livingroom"
    conn.session = SimpleNamespace(client_id="livingroom")
    conn._utterance_actions = []
    conn._camera_seq = 1
    return conn


def test_the_hub_keeps_a_fresh_point_and_forgets_a_stale_one(monkeypatch):
    conn = _connection()
    assert conn._point_hint() is None
    conn._on_point_event({"x": 0.25, "y": 0.75})
    assert conn._point_hint() == (0.25, 0.75)
    # Мусор и точка вне кадра не становятся «направлением».
    conn._on_point_event({"x": "нет", "y": 0.5})
    assert conn._point_hint() == (0.25, 0.75)
    conn._on_point_event({"x": 2.0, "y": 0.5})
    assert conn._point_hint() == (0.25, 0.75)
    # Указание живёт до вопроса, но не вечно.
    conn._pointed = (0.25, 0.75, 0.0)
    assert conn._point_hint(max_age_s=10) is None


def _jpeg(width=200, height=100) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_the_crop_around_the_point_is_real_and_clipped():
    crop = hub_app._crop_around_point(_jpeg(200, 100), 0.5, 0.5)
    assert crop is not None
    assert (crop.w, crop.h) == (45, 45)          # 45% от меньшей стороны
    assert crop.jpeg.startswith(b"\xff\xd8")     # настоящий JPEG
    # Точка у самого края: прямоугольник обрезается по кадру, а не вылезает.
    edge = hub_app._crop_around_point(_jpeg(200, 100), 0.0, 0.0)
    assert edge is not None and edge.w <= 200 and edge.h <= 100
    assert hub_app._crop_around_point(b"not a jpeg", 0.5, 0.5) is None


def test_what_is_this_looks_where_the_person_pointed(monkeypatch):
    monkeypatch.setattr(hub_app, "_vision", object())   # модель есть (подставная)
    conn = _connection()
    captured = hub_app.ImageFrame(jpeg=_jpeg(200, 100), w=200, h=100,
                                  screen_w=200, screen_h=100,
                                  source=hub_app.SOURCE_CAMERA)
    seen: list = []

    async def request(frame_id):
        return captured

    async def describe(jpeg, prompt, **kwargs):
        seen.append((jpeg, prompt))
        return "кружка", "local"

    conn._request_camera_frame_full = request
    conn._describe_image = describe
    conn._room_ground_truth = lambda: {"note": ""}
    face_calls: list = []

    async def people(frame):
        face_calls.append(frame)
        return {"faces_in_frame": [], "face_positions_available": False, "people_detected": 0}

    conn._camera_frame_people = people
    conn.gallery = SimpleNamespace(list_people=lambda profiles=None: [])

    # Без указания описывается весь кадр.
    goal = asyncio.run(conn._run_look_at_camera({"query": "что это?"}))
    assert goal["ok"] is True and len(seen[-1][0]) > 0
    assert seen[-1][0] == captured.jpeg
    assert "pointed" not in seen[-1][1]

    # С указанием — вырезанная часть, и модель об этом знает.
    conn._on_point_event({"x": 0.5, "y": 0.5})
    goal = asyncio.run(conn._run_look_at_camera({"query": "что это?"}))
    assert goal["ok"] is True
    # Лица ищут, когда смотрят на всю комнату, и не ищут в вырезанной части:
    # вопрос про предмет, а не про людей.
    assert len(face_calls) == 1
    crop_jpeg, prompt = seen[-1]
    assert crop_jpeg != captured.jpeg and "pointed at THIS part" in prompt
    assert goal["faces_in_frame"] == []
    assert conn._utterance_actions[-1]["point_hint"] == {"x": 0.5, "y": 0.5}

def test_the_point_event_is_a_known_client_message():
    assert MSG_POINT_EVENT in CLIENT_MESSAGE_TYPES
    assert MSG_POINT_EVENT in PHONE_FORBIDDEN_INPUTS
