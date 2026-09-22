"""Зоны кадра и маска «не анализировать» (ТЗ F-309, P5-05).

Проверяется настоящая геометрия (точка в полигоне), настоящая связка с
индексатором объектов (находка получает зону «стол», а находка в маске не
записывается вовсе) и то, что владелец видит настроенные зоны в админке.
"""
from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from common.config import Config, FrameZone, HomeConfig
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.object_index import DetectedObject, SceneIndexer, jpeg_size
from hub.object_memory import ObjectMemoryStore
from hub.zones import FrameZones, zones_for_homes

HOME = "livingroom"
SIZE = (100, 100)


def _jpeg(width: int = 100, height: int = 100) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (5, 5, 5)).save(buffer, format="JPEG")
    return buffer.getvalue()


JPEG = _jpeg()

#: «Стол» — левая половина кадра, «дверь» — правая, маска — нижняя полоса.
DESK = FrameZone(name="стол", points=[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])
DOOR = FrameZone(name="дверь", points=[[0.6, 0.0], [1.0, 0.0], [1.0, 0.5], [0.6, 0.5]])
SECRET = FrameZone(name="маска", points=[[0.0, 0.8], [1.0, 0.8], [1.0, 1.0], [0.0, 1.0]],
                   mask=True)


@pytest.fixture()
def memory(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    yield ObjectMemoryStore(conn), conn
    conn.close()


class _Detector:
    def __init__(self, found):
        self.found = list(found)

    def detect(self, frame):
        return list(self.found)


def test_the_polygon_answers_where_the_object_is():
    zones = FrameZones([DESK, DOOR, SECRET])
    # Центр бокса слева-сверху — «стол».
    assert zones.zone_at([10, 10, 40, 40], SIZE) == "стол"
    # Справа-сверху — «дверь».
    assert zones.zone_at([70, 10, 90, 40], SIZE) == "дверь"
    # Между зонами — никакой выдуманной комнаты.
    assert zones.zone_at([55, 55, 58, 58], SIZE) == ""
    # Маска отвечает отдельно и в зону не попадает.
    assert zones.masked_at([40, 90, 60, 99], SIZE) is True
    assert zones.zone_at([40, 90, 60, 99], SIZE) == ""
    assert zones.masked_at([10, 10, 40, 40], SIZE) is False
    # Без размера кадра спросить нечего: честный пустой ответ.
    assert zones.zone_at([10, 10, 40, 40], (0, 0)) == ""
    assert zones.masked_at([1, 2, 3, 4], None) is False


def test_the_zones_come_from_the_home_config():
    cfg = Config(homes=[{"home_id": HOME, "name": "Living room",
                         "tz": "America/Chicago",
                         "zones": [{"name": "дверь", "points": [[0, 0], [1, 0], [1, 1]]},
                                   {"name": "не смотреть", "mask": True,
                                    "points": [[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]]}]}])
    zones = zones_for_homes(cfg.homes)
    assert list(zones) == [HOME]
    described = zones[HOME].describe()
    assert described[0]["name"] == "дверь" and described[0]["mask"] is False
    assert described[1]["mask"] is True
    assert zones["livingroom"].zone_at([0, 0, 10, 10], SIZE) == ""
    assert zones["livingroom"].masked_at([0, 0, 10, 10], SIZE) is True
    # Дом без зон — пустой набор, а не выдуманная зона.
    assert zones_for_homes([HomeConfig(home_id="kyivflat", name="Kyiv")])["kyivflat"].describe() == []


def test_a_bad_polygon_is_a_config_error_not_a_silent_zone():
    with pytest.raises(ValidationError):
        FrameZone(name="дверь", points=[[0, 0], [1, 1]])           # меньше трёх точек
    with pytest.raises(ValidationError):
        FrameZone(name="дверь", points=[[0, 0], [2, 0], [1, 1]])   # точка вне кадра
    with pytest.raises(ValidationError):
        FrameZone(name="дверь", points=[[0, 0], [1, 0], [1]])      # точка не [x, y]
    with pytest.raises(ValidationError):
        HomeConfig(home_id="a", name="A", zones=[DESK, FrameZone(name="стол", points=DESK.points)])


def test_the_indexer_signs_the_zone_and_skips_the_mask(memory):
    store, _conn = memory
    zones = FrameZones([DESK, DOOR, SECRET])
    found = [DetectedObject(label="keys", bbox=[10, 10, 40, 40]),     # стол
             DetectedObject(label="keys", bbox=[70, 10, 90, 40]),     # дверь
             DetectedObject(label="wallet", bbox=[40, 90, 60, 99])]   # маска
    indexer = SceneIndexer(_Detector(found), store,
                           zone_of=lambda home, bbox, size: zones.zone_at(bbox, size),
                           masked=lambda home, bbox, size: zones.masked_at(bbox, size))
    result = indexer.index(HOME, JPEG)
    assert result.ok is True
    assert result.labels == ["keys", "keys"] and result.masked == 1
    rows = store.sightings(HOME)
    assert [row.zone for row in rows] == ["дверь", "стол"] or \
           [row.zone for row in rows] == ["стол", "дверь"]
    assert "wallet" not in [row.label for row in rows], \
        "в области «не анализировать» объект не записывается"
    assert memory[0].sightings(HOME, since_hours=1)[0].zone in {"стол", "дверь"}


def test_a_broken_zone_callable_does_not_break_the_indexing(memory):
    store, _conn = memory

    def broken(home, bbox, size):
        raise ValueError("no zones")

    indexer = SceneIndexer(_Detector([DetectedObject(label="keys", bbox=[1, 2, 3, 4])]),
                           store, zone_of=broken, masked=broken)
    result = indexer.index(HOME, JPEG)
    assert result.ok is True and result.sightings[0].zone == ""
    assert len(store.sightings(HOME)) == 1


def test_the_jpeg_size_is_read_from_the_header():
    assert jpeg_size(JPEG) == (100, 100)
    assert jpeg_size(_jpeg(640, 480)) == (640, 480)
    assert jpeg_size(b"") is None and jpeg_size(b"not a jpeg") is None


def test_the_owner_sees_the_zones_in_the_admin_list(memory, monkeypatch):
    from hub.admin_backend import AdminBackend

    _store, conn = memory
    cfg = Config(homes=[{"home_id": HOME, "name": "Living room", "tz": "America/Chicago",
                         "zones": [{"name": "дверь", "points": [[0, 0], [1, 0], [1, 1]]},
                                   {"name": "маска", "mask": True,
                                    "points": [[0, 0], [0.1, 0], [0.1, 0.1]]}]}])
    access = SimpleNamespace(is_hub_admin=lambda actor: True, is_owner=lambda actor: True)
    backend = AdminBackend(cfg, access, runtime=SimpleRuntime(),
                           get_room=lambda client_id=None: conn,
                           get_alerts=lambda: None,
                           get_home_owners=lambda: SimpleNamespace(
                               owners=lambda: {HOME: (7,)},
                               home_ids=lambda: (HOME,)))
    result = asyncio.run(backend.call("homes.list", {}, "owner"))
    assert result["ok"] is True
    item = result["items"][0]
    assert item["home_id"] == HOME
    assert [zone["name"] for zone in item["zones"]] == ["дверь", "маска"]
    assert item["zones"][1]["mask"] is True


class SimpleRuntime:
    """Минимальная заглушка runtime: панель спрашивает у него состояние хаба."""

    def __call__(self):
        return {}

    def snapshot(self):
        return {}
