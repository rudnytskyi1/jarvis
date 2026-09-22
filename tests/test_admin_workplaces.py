"""``workplaces.list`` must not read the room's home off the wrong object.

The owner panel failed with ``AttributeError: 'Session' object has no attribute
'home_id'`` because ``_workplaces()`` asked the room state for the home. The
home belongs to the CONNECTION (the v2 hello takes it from the token), so the
listing has to read it there - and a v1 connection without a home still lists.
"""
from __future__ import annotations

from types import SimpleNamespace

from starlette.websockets import WebSocketState

from hub import app as hub_app
from hub.session import Session


def _connection(client_id: str, home_id: str, *, connected: bool = True):
    return SimpleNamespace(
        session=Session(client_id, [], 8),
        home_id=home_id,
        workplace_name=f"{client_id} workplace",
        camera_name=f"{client_id} camera",
        ws=SimpleNamespace(
            client_state=WebSocketState.CONNECTED if connected else WebSocketState.DISCONNECTED
        ),
    )


def test_the_listing_reports_the_home_of_the_connection(monkeypatch):
    monkeypatch.setattr(hub_app, "_telegram_access", None)
    monkeypatch.setattr(hub_app, "_connections",
                        [_connection("livingroom", "livingroom"),
                         _connection("dorm-max", "dorm-max")])

    rows = {row["id"]: row for row in hub_app._workplaces()}

    assert rows["livingroom"]["home_id"] == "livingroom"
    assert rows["dorm-max"]["home_id"] == "dorm-max"
    assert rows["dorm-max"]["name"] == "dorm-max workplace"


def test_a_v1_connection_without_a_home_is_listed_as_unknown(monkeypatch):
    monkeypatch.setattr(hub_app, "_telegram_access", None)
    monkeypatch.setattr(hub_app, "_connections", [_connection("legacy", "")])

    rows = hub_app._workplaces()

    assert [row["home_id"] for row in rows] == [""]


def test_a_closed_connection_is_not_listed(monkeypatch):
    monkeypatch.setattr(hub_app, "_telegram_access", None)
    monkeypatch.setattr(hub_app, "_connections",
                        [_connection("livingroom", "livingroom", connected=False)])

    assert hub_app._workplaces() == []


def test_the_home_lookup_uses_the_same_listing(monkeypatch):
    monkeypatch.setattr(hub_app, "_telegram_access", None)
    monkeypatch.setattr(hub_app, "_connections", [_connection("livingroom", "livingroom")])

    assert hub_app._workplace_home("livingroom") == "livingroom"
    assert hub_app._workplace_home("nobody") == ""
