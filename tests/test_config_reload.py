"""Hot reload of room settings without restarting the hub (ТЗ 4.7)."""
from __future__ import annotations

import asyncio
import json

import pytest
import yaml

from common.config import Config
from common.protocol import MSG_CONFIG_UPDATE, SERVER_MESSAGE_TYPES, parse_message
from hub import app as hub_app
from hub import config_reload
from hub import migrations_runner as runner


def _write_config(path, homes, **server_overrides):
    data = json.loads(Config().model_dump_json())
    data["homes"] = homes
    for key, value in server_overrides.items():
        data["server"][key] = value
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _home(home_id, name, start="", end="", settings=None):
    return {
        "home_id": home_id,
        "name": name,
        "tz": "America/Chicago",
        "quiet_hours": {"start": start, "end": end},
        "settings": settings or {},
    }


def _conn(tmp_path):
    conn = runner.connect(str(tmp_path / "hub.db"))
    runner.migrate(conn)
    return conn


def test_reload_reports_changed_rooms_once(tmp_path):
    path = _write_config(tmp_path / "config.yaml", [_home("livingroom", "Living room", "23:00", "08:00")])
    conn = _conn(tmp_path)
    try:
        config, first = config_reload.reload_home_settings(conn, path)
        assert [change.home_id for change in first] == ["livingroom"]
        change = first[0]
        assert change.config_rev >= 1
        assert change.patch["quiet_hours"] == {"start": "23:00", "end": "08:00"}
        assert change.patch["name"] == "Living room"

        _, again = config_reload.reload_home_settings(conn, path)
        assert again == [], "an unchanged file must not re-announce the room"
    finally:
        conn.close()


def test_reload_bumps_the_revision_only_for_rooms_that_changed(tmp_path):
    path = _write_config(
        tmp_path / "config.yaml",
        [_home("livingroom", "Living room"), _home("dorm-max", "Max's room")],
    )
    conn = _conn(tmp_path)
    try:
        config_reload.reload_home_settings(conn, path)
        rev_before = {row[0]: row[1] for row in conn.execute("SELECT home_id, config_rev FROM homes")}

        _write_config(
            tmp_path / "config.yaml",
            [_home("livingroom", "Living room"), _home("dorm-max", "Max's room", "22:00", "07:00")],
        )
        _, changes = config_reload.reload_home_settings(conn, path)
        assert [change.home_id for change in changes] == ["dorm-max"]
        assert changes[0].config_rev == rev_before["dorm-max"] + 1
        rev_after = {row[0]: row[1] for row in conn.execute("SELECT home_id, config_rev FROM homes")}
        assert rev_after["livingroom"] == rev_before["livingroom"]
    finally:
        conn.close()


def test_invalid_config_leaves_the_room_table_untouched(tmp_path):
    path = _write_config(tmp_path / "config.yaml", [_home("livingroom", "Living room")])
    conn = _conn(tmp_path)
    try:
        config_reload.reload_home_settings(conn, path)
        path.write_text("homes:\n  - home_id: 'Bad Id'\n    name: x\n", encoding="utf-8")
        with pytest.raises(ValueError):
            config_reload.reload_home_settings(conn, path)
        row = conn.execute("SELECT name FROM homes WHERE home_id='livingroom'").fetchone()
        assert row is not None and row[0] == "Living room"
    finally:
        conn.close()


def test_missing_config_file_is_an_error(tmp_path):
    conn = _conn(tmp_path)
    try:
        with pytest.raises(ValueError, match="Config not found"):
            config_reload.reload_home_settings(conn, tmp_path / "nope.yaml")
    finally:
        conn.close()


def test_change_frame_is_a_valid_v2_server_message(tmp_path):
    path = _write_config(tmp_path / "config.yaml", [_home("livingroom", "Living room")])
    conn = _conn(tmp_path)
    try:
        _, changes = config_reload.reload_home_settings(conn, path)
        frame = changes[0].frame()
        assert MSG_CONFIG_UPDATE in SERVER_MESSAGE_TYPES
        parsed = parse_message(frame, direction="server")
        assert parsed is not None
        assert parsed.type == MSG_CONFIG_UPDATE
        assert parsed.config_rev == changes[0].config_rev
        assert parsed.home_id == "livingroom"
        assert parsed.patch["name"] == "Living room"
    finally:
        conn.close()


def test_current_room_frame_describes_the_stored_row(tmp_path):
    path = _write_config(tmp_path / "config.yaml", [_home("livingroom", "Living room", "23:00", "08:00")])
    conn = _conn(tmp_path)
    try:
        config_reload.reload_home_settings(conn, path)
        frame = config_reload.current_room_frame(conn, "livingroom")
        assert frame is not None
        assert frame["home_id"] == "livingroom"
        assert frame["patch"]["quiet_hours"]["start"] == "23:00"
        assert frame["config_rev"] >= 1
        assert config_reload.current_room_frame(conn, "missing") is None
    finally:
        conn.close()


def test_broadcast_reaches_only_the_room_that_changed(monkeypatch):
    from starlette.websockets import WebSocketState

    class _WS:
        client_state = WebSocketState.CONNECTED

    class _Live:
        def __init__(self, home_id):
            self.home_id = home_id
            self.peer = f"peer-{home_id}"
            self.received: list = []
            self.ws = _WS()

        async def queue_frame(self, payload, *, background=None):
            self.received.append(payload)
            return True

    living, other = _Live("livingroom"), _Live("dorm-max")
    monkeypatch.setattr(hub_app, "_connections", {living, other})
    frame = {"type": MSG_CONFIG_UPDATE, "proto": 2, "home_id": "livingroom", "config_rev": 3, "patch": {}}
    delivered = asyncio.run(hub_app.broadcast_config_update("livingroom", frame))
    assert delivered == 1
    assert living.received == [frame] and other.received == []


def test_broadcast_survives_a_dead_client(monkeypatch):
    from starlette.websockets import WebSocketState

    class _Live:
        peer = "broken"
        home_id = "livingroom"

        class ws:
            client_state = WebSocketState.CONNECTED

        async def queue_frame(self, payload, *, background=None):
            raise RuntimeError("socket is gone")

    monkeypatch.setattr(hub_app, "_connections", {_Live()})
    assert asyncio.run(hub_app.broadcast_config_update("livingroom", {"type": MSG_CONFIG_UPDATE})) == 0


def test_reload_room_configs_updates_live_config_and_returns_frames(tmp_path, monkeypatch):
    path = _write_config(tmp_path / "config.yaml", [_home("livingroom", "Living room", "23:00", "08:00")])
    db_path = tmp_path / "data" / "hub.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_connections", set())

    live = Config()
    monkeypatch.setattr(hub_app, "_config", live)

    frames = asyncio.run(hub_app.reload_room_configs(path))
    assert [frame["home_id"] for frame in frames] == ["livingroom"]
    # The reloaded config replaced the live object's fields - connections held
    # a reference to it, so open rooms see the new values without a reconnect.
    assert hub_app._config is live
    assert [home.home_id for home in live.homes] == ["livingroom"]

    again = asyncio.run(hub_app.reload_room_configs(path))
    assert again == [], "a second reload of an unchanged file must be a no-op"


def test_reload_room_configs_keeps_the_hub_on_an_invalid_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("homes:\n  - home_id: 'Bad Id'\n    name: x\n", encoding="utf-8")
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    live = Config()
    monkeypatch.setattr(hub_app, "_config", live)
    with pytest.raises(ValueError):
        asyncio.run(hub_app.reload_room_configs(path))
    assert hub_app._config is live


def test_client_stores_the_room_revision():
    from client.main import JarvisClient

    class _Overlay:
        def set_status(self, text):
            pass

    assistant = JarvisClient.__new__(JarvisClient)
    assistant.overlay = _Overlay()
    assistant.room_config_rev = 0
    assistant.room_config = {}
    assistant._apply_room_config(
        {"type": MSG_CONFIG_UPDATE, "config_rev": 7, "patch": {"quiet_hours": {"start": "23:00"}}}
    )
    assert assistant.room_config_rev == 7
    assert assistant.room_config == {"quiet_hours": {"start": "23:00"}}
    # Replaying the same revision and patch leaves the client state alone.
    assistant._apply_room_config(
        {"type": MSG_CONFIG_UPDATE, "config_rev": 7, "patch": {"quiet_hours": {"start": "23:00"}}}
    )
    assert assistant.room_config_rev == 7
    # A newer revision with a different patch is applied.
    assistant._apply_room_config({"type": MSG_CONFIG_UPDATE, "config_rev": 8, "patch": {"name": "Other"}})
    assert assistant.room_config_rev == 8 and assistant.room_config == {"name": "Other"}
