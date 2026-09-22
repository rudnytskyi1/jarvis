"""Маска зон кадра: клиент закрашивает её до JPEG, хаб это проверяет (ТЗ F-309, P5-06).

Проверяется настоящий путь: настоящий JPEG (Pillow/OpenCV) с настоящей чёрной
областью вместо маски, настоящие заголовки ``camera_frame`` и настоящий отказ
хаба анализировать кадр, пришедший без маски.
"""
from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from client.camera import CameraService
from common.frame_zones import FrameZone, mask_polygons, masks_rev, parse_zones, zones_rev
from hub import app as hub_app
from hub import migrations_runner
from hub.audit import AuditLog
from hub.homes import ensure_home

HOME = "livingroom"

#: Нижняя полоса кадра — «не анализировать».
MASK = FrameZone(name="маска", mask=True,
                 points=[[0.0, 0.75], [1.0, 0.75], [1.0, 1.0], [0.0, 1.0]])
DOOR = FrameZone(name="дверь", points=[[0.6, 0.0], [1.0, 0.0], [1.0, 0.5], [0.6, 0.5]])


def _cv2():
    return pytest.importorskip("cv2")


def _numpy():
    return pytest.importorskip("numpy")


def _frame(width: int = 80, height: int = 60):
    """Настоящая картинка OpenCV: белая, а в области маски — красная."""
    np = _numpy()
    frame = np.zeros((height, width, 3), dtype="uint8")
    frame[:, :] = (255, 255, 255)
    frame[int(height * 0.8):, :] = (0, 0, 255)
    return frame


def _jpeg_bytes(width: int = 80, height: int = 60) -> bytes:
    """Настоящий JPEG для пути доставки хаба."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 200, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _pixels(jpeg: bytes):
    np = _numpy()
    return np.array(Image.open(io.BytesIO(jpeg)).convert("RGB"))


# ---------------------------------------------------------------------------
# клиент: маска закрашивается ДО JPEG
# ---------------------------------------------------------------------------


def test_the_client_paints_the_mask_before_the_jpeg():
    camera = CameraService(SimpleNamespace(enabled=False, zones=[MASK.model_dump()]))
    camera._cv2 = _cv2()
    jpeg, width, height = camera._encode(_frame())
    assert (width, height) == (80, 60)
    pixels = _pixels(jpeg)
    # Верх кадра остался белым, а область маски стала чёрной — красного нет.
    assert tuple(pixels[5, 5]) == (255, 255, 255)
    assert tuple(pixels[-2, 5]) == (0, 0, 0)
    assert tuple(pixels[55, 40]) == (0, 0, 0)
    assert camera.masked_rev == masks_rev([MASK])


def test_the_mask_does_not_touch_the_source_frame():
    camera = CameraService(SimpleNamespace(enabled=False, zones=[MASK.model_dump()]))
    camera._cv2 = _cv2()
    frame = _frame()
    camera._encode(frame)
    # Кадр из кэша камеры не испорчен: следующая картинка останется целой.
    assert tuple(frame[-2, 5]) == (0, 0, 255)


def test_a_room_without_masks_sends_the_frame_untouched():
    camera = CameraService(SimpleNamespace(enabled=False))
    frame = _frame()
    assert camera._mask_frame(frame, _cv2()) is frame
    assert camera.masked_rev == ""


def test_zones_from_the_hub_apply_and_replace_the_old_ones():
    camera = CameraService(SimpleNamespace(enabled=False))
    assert camera.set_zones([MASK.model_dump()]) is True
    assert camera.masked_rev == masks_rev([MASK])
    # Тот же набор — ничего не меняется.
    assert camera.set_zones([MASK.model_dump()]) is False
    # Другая маска — новый отпечаток (иначе хаб отклонил бы кадр за старую).
    moved = FrameZone(name="маска", mask=True, points=[[0.0, 0.0], [0.2, 0.0], [0.2, 0.2]])
    assert camera.set_zones([moved.model_dump()]) is True
    assert camera.masked_rev == masks_rev([moved])
    # Порядок зон на смысл не влияет: перестановка — не новый набор.
    assert camera.set_zones([moved.model_dump(), DOOR.model_dump()]) is True
    assert camera.set_zones([DOOR.model_dump(), moved.model_dump()]) is False
    # Зона без маски кадр не маскирует, но хранится (её читает хаб).
    assert camera.set_zones([DOOR.model_dump()]) is True
    assert camera.masked_rev == ""
    assert [zone.name for zone in camera.zones] == ["дверь"]


def test_a_broken_zone_in_the_patch_is_skipped_not_invented():
    camera = CameraService(SimpleNamespace(enabled=False))
    camera.set_zones([{"name": "маска", "mask": True, "points": [[0, 0], [1, 1]]},   # 2 точки
                      {"name": "", "mask": True, "points": [[0, 0], [1, 0], [1, 1]]},
                      {"name": "чужая", "mask": True},          # нет точек
                      MASK.model_dump()])
    assert [zone.name for zone in camera.zones] == ["маска"]
    assert camera.masked_rev == masks_rev([MASK])


def test_the_header_carries_the_mask_fingerprint():
    sent_json: list[dict] = []
    sent_bytes: list[bytes] = []

    async def send_json(payload):
        sent_json.append(payload)

    async def send_bytes(data):
        sent_bytes.append(data)

    camera = CameraService(SimpleNamespace(enabled=True, zones=[MASK.model_dump()]))
    camera._send_json = send_json
    camera._send_bytes = send_bytes
    camera._send_lock = asyncio.Lock()
    asyncio.run(camera._send_burst("p1", "presence", [(b"jpeg", 80, 60)]))
    assert sent_bytes == [b"jpeg"]
    header = sent_json[0]
    assert header["masked"] is True
    assert header["zones_rev"] == masks_rev([MASK])
    # Без масок заголовок честно говорит «маски нет».
    plain = CameraService(SimpleNamespace(enabled=True))
    plain._send_json = send_json
    plain._send_bytes = send_bytes
    plain._send_lock = asyncio.Lock()
    asyncio.run(plain._send_burst("p2", "presence", [(b"jpeg", 80, 60)]))
    assert sent_json[-1]["masked"] is False and sent_json[-1]["zones_rev"] == ""


# ---------------------------------------------------------------------------
# общие функции
# ---------------------------------------------------------------------------


def test_the_fingerprint_ignores_the_order_of_the_zones():
    first = [MASK.model_dump(), DOOR.model_dump()]
    second = [DOOR.model_dump(), MASK.model_dump()]
    assert masks_rev(first) == masks_rev(second)
    assert zones_rev(first) == zones_rev(second)
    # Зона без маски в отпечаток масок не входит, а в общий — входит.
    assert masks_rev([DOOR]) == ""
    assert zones_rev([DOOR]) == zones_rev([DOOR.model_dump()])
    assert masks_rev([]) == "" and zones_rev([]) == ""


def test_the_pixel_polygon_matches_the_normalized_points():
    assert mask_polygons([MASK], 200, 100) == [[(0, 75), (200, 75), (200, 100), (0, 100)]]
    assert mask_polygons([MASK], 0, 100) == []
    assert mask_polygons([DOOR], 200, 100) == []
    assert parse_zones(MASK) == [MASK]
    assert parse_zones("не список зон") == []


# ---------------------------------------------------------------------------
# хаб: кадр без маски не анализируется
# ---------------------------------------------------------------------------


def _config(*zones):
    from common.config import Config

    return Config(homes=[{"home_id": HOME, "name": "Living room", "tz": "America/Chicago",
                          "zones": [zone.model_dump() for zone in zones]}])


def test_the_hub_refuses_a_camera_frame_that_arrived_without_the_mask():
    hub_app._refresh_frame_zones(_config(MASK, DOOR))
    try:
        assert hub_app._masks_rev_of_home(HOME) == masks_rev([MASK])
        problem = hub_app._frame_mask_problem(HOME, {"masked": False, "zones_rev": ""})
        assert "without the mask" in problem
        # Чужой отпечаток — это «клиент закрасил другие области», а не «почти то же».
        stale = masks_rev([FrameZone(name="маска", mask=True,
                                     points=[[0.0, 0.5], [1.0, 0.5], [1.0, 1.0]])])
        problem = hub_app._frame_mask_problem(HOME, {"masked": True, "zones_rev": stale})
        assert "different mask revision" in problem
        # Правильный отпечаток — кадр принимается.
        assert hub_app._frame_mask_problem(
            HOME, {"masked": True, "zones_rev": masks_rev([MASK])}) == ""
        # Дом без масок не проверяется вообще: работающее поведение не меняется.
        assert hub_app._frame_mask_problem("unknown-home", {}) == ""
    finally:
        hub_app._refresh_frame_zones(SimpleNamespace(homes=[]))


def _connection(tmp_path, home_id: str = HOME):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, home_id, name="Living room", tz="America/Chicago")
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.home_id = home_id
    connection.peer = "room-pc"
    connection.session = SimpleNamespace(client_id="client-1")
    connection.presence = SimpleNamespace(note_persons=lambda count: None,
                                          reconcile=lambda count: None)
    connection.camera_state = None
    connection._image_futures = {}
    connection._image_incoming = {}
    connection._image_ids = {}
    connection._image_event_ids = {}
    connection._image_recording_context = {}
    connection._image_recording_tasks = set()
    connection._expect_image = None
    connection._image_header = {}
    connection._last_camera_seen = None
    connection._buffer_presence_frame = lambda frame: None
    return connection, conn


def test_the_delivery_path_refuses_the_unmasked_frame_and_answers_the_waiter(tmp_path, monkeypatch):
    hub_app._refresh_frame_zones(_config(MASK, DOOR))
    connection, conn = _connection(tmp_path)
    monkeypatch.setattr(hub_app, "_audit", AuditLog(conn))
    try:
        future = asyncio.new_event_loop().create_future()
        connection._image_futures = {"camera": future}
        connection._expect_image = "camera"
        connection._image_header = {"source": "camera", "reason": "request", "id": "r1",
                                    "event_id": "e1", "w": 80, "h": 60,
                                    "masked": False, "zones_rev": ""}
        connection._deliver_image(_jpeg_bytes())
        assert future.result()["error"].startswith(
            "the frame from livingroom arrived without the mask")
        # Кадр не попал ни в кэш индексатора, ни в события комнаты.
        assert connection._last_camera_seen is None
        row = conn.execute("SELECT action, result FROM audit "
                           "WHERE action='camera.frame_unmasked'").fetchone()
        assert row == ("camera.frame_unmasked", "failed")
        assert hub_app._unmasked_frame_count() >= 1
    finally:
        conn.close()
        hub_app._refresh_frame_zones(SimpleNamespace(homes=[]))


def test_a_masked_frame_travels_the_normal_path(tmp_path):
    hub_app._refresh_frame_zones(_config(MASK, DOOR))
    connection, conn = _connection(tmp_path)
    try:
        seen: list = []
        connection._buffer_presence_frame = seen.append
        connection._expect_image = "camera"
        connection._image_header = {"source": "camera", "reason": "presence", "id": "p1",
                                    "event_id": "e1", "w": 80, "h": 60,
                                    "masked": True, "zones_rev": masks_rev([MASK])}
        connection._deliver_image(_jpeg_bytes())
        assert connection._last_camera_seen is not None
        assert len(seen) == 1
    finally:
        conn.close()
        hub_app._refresh_frame_zones(SimpleNamespace(homes=[]))


def test_a_body_crop_without_the_mask_is_not_stored(tmp_path, monkeypatch):
    hub_app._refresh_frame_zones(_config(MASK))
    connection, conn = _connection(tmp_path)
    try:
        connection._expect_body_crop = {"track_id": "t1", "w": 40, "h": 40,
                                        "masked": False, "zones_rev": ""}
        connection._deliver_body_crop(_jpeg_bytes(40, 40))
        assert not list(conn.execute("SELECT * FROM body_crops"))
    finally:
        conn.close()
        hub_app._refresh_frame_zones(SimpleNamespace(homes=[]))


def test_the_home_patch_and_the_hello_frame_carry_the_zones(tmp_path):
    from hub.config_reload import current_room_frame, home_patch

    cfg = _config(MASK, DOOR)
    assert home_patch(cfg.homes[0])["zones"] == [MASK.model_dump(), DOOR.model_dump()]
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    try:
        ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
        frame = current_room_frame(conn, HOME, zones=[MASK.model_dump()])
        assert frame is not None
        assert frame["patch"]["zones"] == [MASK.model_dump()]
        # Патч должен пережить провод: клиент читает тот же JSON.
        assert json.loads(json.dumps(frame))["patch"]["zones"][0]["mask"] is True
    finally:
        conn.close()


def test_changing_only_the_zones_bumps_the_room_revision(tmp_path):
    from common.config import Config
    from hub.homes import sync_homes_from_config

    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    try:
        migrations_runner.migrate(conn)
        base = {"home_id": HOME, "name": "Living room", "tz": "America/Chicago"}
        sync_homes_from_config(conn, Config(homes=[base]).homes)
        unchanged = sync_homes_from_config(conn, Config(homes=[base]).homes)
        assert unchanged == [], "дом без изменений не переобъявляется"
        before = conn.execute("SELECT config_rev, zones_rev FROM homes").fetchone()
        # Меняются ТОЛЬКО зоны — комната всё равно должна узнать (ТЗ F-309).
        changed = sync_homes_from_config(
            conn, Config(homes=[{**base, "zones": [MASK.model_dump()]}]).homes)
        assert changed == [HOME]
        after = conn.execute("SELECT config_rev, zones_rev FROM homes").fetchone()
        assert after[1] == zones_rev([MASK])
        assert after[0] != before[0]
    finally:
        conn.close()
