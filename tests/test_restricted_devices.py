"""ТЗ F-606: `restricted: true` у устройств и сцен — и честный отказ гостю."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from common.client_config import DeviceConfig
from common.config import Config
from hub import app as hub_app
from hub import guest_access
from hub.devices import Device, DeviceStore
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.scenes import Scene, Step, scene_id_for
from hub.session import Session


def _hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    return conn


def _device(**overrides) -> Device:
    values = dict(id="server-rack", home_id="livingroom", name="Сервер",
                  kind="switch", capabilities=["press"], adapter="mqtt",
                  restricted=True)
    values.update(overrides)
    return Device(**values)


def _connection(tmp_path, *, role: str = guest_access.ROLE_GUEST, name: str = "Кай",
                devices: list[dict] | None = None):
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = "livingroom"
    conn.cfg = Config()
    conn.session = Session(client_id="room-pc", devices=devices or [], history_turns=2)
    conn._speaker_role = role
    conn._speaker_name = name
    conn._speaker_score = 0.9
    conn._reply_language = "ru"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    return conn


# --- the flag itself -------------------------------------------------------


def test_the_client_can_mark_a_device_restricted_in_its_config(tmp_path):
    device = DeviceConfig(name="Сервер", type="switchbot_bot", restricted=True)
    assert device.restricted is True
    # The flag is first class, not swallowed by the free-form ``params`` bag.
    assert "restricted" not in device.params
    plain = DeviceConfig(name="Лампа", type="magichome", host="192.168.1.50")
    assert plain.restricted is False
    assert plain.params["host"] == "192.168.1.50"


def test_the_flag_round_trips_through_the_hub_devices_table(tmp_path):
    store = DeviceStore(_hub_db(tmp_path))
    store.save(_device())
    assert store.get("server-rack").restricted is True
    assert store.devices("livingroom")[0].restricted is True
    plain = store.save(_device(id="lamp-living", name="Лампа", adapter="magichome",
                               restricted=False))
    assert plain.restricted is False
    assert store.get("lamp-living").restricted is False


# --- the hub resolves the name --------------------------------------------


def test_the_room_owns_list_marks_the_device(monkeypatch):
    conn = _connection(None, devices=[{"name": "Сервер", "type": "switchbot_bot",
                                        "restricted": True}])
    monkeypatch.setattr(hub_app, "_devices", False)
    assert conn._device_restricted("set_switch", {"device": "сервер"}) is True
    assert conn._device_restricted("set_switch", {"device": "лампа"}) is False
    # Only the three device tools can be restricted at all.
    assert conn._device_restricted("pc_control", {"device": "Сервер"}) is False


def test_the_hub_devices_table_marks_the_device(monkeypatch, tmp_path):
    store = DeviceStore(_hub_db(tmp_path))
    store.save(_device())
    conn = _connection(tmp_path)
    monkeypatch.setattr(hub_app, "_devices", store)
    assert conn._device_restricted("set_light", {"device": "Сервер"}) is True
    assert conn._device_restricted("set_light", {"device": "Телевизор"}) is False


# --- the honest refusal ----------------------------------------------------


def test_a_guest_is_refused_a_restricted_device_in_words_about_the_owner(
        monkeypatch, tmp_path):
    store = DeviceStore(_hub_db(tmp_path))
    store.save(_device())
    conn = _connection(tmp_path)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_devices", store)
    refusal = asyncio.run(conn._permission_check(
        "set_light", {"device": "Сервер", "state": "on"}))
    assert refusal is not None
    assert "хозяин" in refusal
    # The point of the task: not "I don't know that device".
    assert "не знаю" not in refusal.casefold()
    assert refusal == guest_access.denial(guest_access.RESTRICTED, "ru")


def test_an_unknown_device_stays_the_clients_own_answer(monkeypatch, tmp_path):
    store = DeviceStore(_hub_db(tmp_path))
    store.save(_device())
    conn = _connection(tmp_path)
    monkeypatch.setattr(hub_app, "_devices", store)
    # Nothing resolves the name, so the guest matrix does not invent a refusal.
    assert conn._device_restricted("set_light", {"device": "Чайник"}) is False


# --- scenes ---------------------------------------------------------------


def _scene_with(name: str, device: str) -> Scene:
    return Scene(scene_id=scene_id_for("livingroom", name), home_id="livingroom",
                 name=name, steps=[Step(kind="device", device=device,
                                        capability="press", value=True)])


def test_a_guest_cannot_run_a_scene_that_touches_a_restricted_device(
        monkeypatch, tmp_path):
    store = DeviceStore(_hub_db(tmp_path))
    store.save(_device())
    conn = _connection(tmp_path)
    monkeypatch.setattr(hub_app, "_devices", store)
    allowed_scene = _scene_with("вечер", "Лампа")
    restricted_scene = _scene_with("сервер", "Сервер")
    assert conn._scene_restricted(restricted_scene) is True
    assert conn._scene_restricted(allowed_scene) is False
    refusal = asyncio.run(conn._run_scene_turn(None, restricted_scene, "livingroom"))
    assert "хозяин" in refusal


def test_the_owner_is_not_stopped_by_the_guest_scene_rule(monkeypatch, tmp_path):
    store = DeviceStore(_hub_db(tmp_path))
    store.save(_device())
    conn = _connection(tmp_path, role="admin", name="Антон")
    monkeypatch.setattr(hub_app, "_devices", store)
    # A member of this home is not a guest, so the scene check does not fire.
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    assert conn._scene_restricted(_scene_with("сервер", "Сервер")) is True
    assert conn._home_guest() is False
    answer = asyncio.run(conn._run_scene_turn(None, _scene_with("сервер", "Сервер"),
                                              "livingroom"))
    assert "хозяин" not in answer, "the owner hears the real outcome, not the guest refusal"
