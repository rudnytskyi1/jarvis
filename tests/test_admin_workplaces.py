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


KNOWN = {
    "room-9de3bed07b44": {"id": "room-9de3bed07b44", "name": "buro",
                          "camera_name": "Buro camera", "home_id": ""},
    "livingroom": {"id": "livingroom", "name": "AntonDorm",
                   "camera_name": "Основная камера", "home_id": ""},
}


def _two_rooms(monkeypatch, *, dorm_online=True):
    """Two computers the hub knows, one of them named 'buro' (ТЗ F-701)."""
    buro = _connection("room-9de3bed07b44", "")
    buro.workplace_name, buro.camera_name = "buro", "Buro camera"
    dorm = _connection("livingroom", "", connected=dorm_online)
    dorm.workplace_name, dorm.camera_name = "AntonDorm", "Основная камера"
    monkeypatch.setattr(hub_app, "_telegram_access",
                        SimpleNamespace(get_setting=lambda key, default=None:
                                        KNOWN if key == "workplaces" else default))
    monkeypatch.setattr(hub_app, "_connections", [buro, dorm])
    return buro, dorm


def _message(text, chat=-100, sender=8322835915):
    return {'text': text, 'chat': {'id': chat}, 'from': {'id': sender}}


def test_a_request_that_names_a_computer_routes_that_turn_to_it(monkeypatch):
    """No preselection: the name in the sentence is the choice (ТЗ F-701)."""
    buro, dorm = _two_rooms(monkeypatch)
    cases = {'сделай фото с камеры buro': buro,
             'take a photo from buro': buro,
             'screenshot from the Buro camera': buro,
             'AntonDorm, покажи экран': dorm}
    for text, expected in cases.items():
        assert hub_app._selected_telegram_room(_message(text)) is expected, text


def test_the_earliest_name_wins_and_case_and_spacing_are_ignored(monkeypatch):
    buro, dorm = _two_rooms(monkeypatch)
    assert hub_app._selected_telegram_room(_message('фото с buro, не antondorm')) is buro
    assert hub_app._selected_telegram_room(_message('foto s ANTON DORM')) is dorm


def test_naming_an_offline_computer_says_so_instead_of_picking_another(monkeypatch):
    _two_rooms(monkeypatch, dorm_online=False)
    answer = hub_app._selected_telegram_room(_message('покажи экран AntonDorm'))
    assert isinstance(answer, str) and 'AntonDorm' in answer and 'not connected' in answer


def test_a_request_that_names_nobody_keeps_working_as_before(monkeypatch):
    buro, dorm = _two_rooms(monkeypatch)
    # Two computers online and nobody named: the hub must not guess.
    assert hub_app._selected_telegram_room(_message('сделай фото')) is None
    # One computer online: it is the room, exactly as before.
    monkeypatch.setattr(hub_app, "_connections", [dorm])
    assert hub_app._selected_telegram_room(_message('сделай фото')) is dorm


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
