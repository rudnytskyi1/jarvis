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


# --- F-701: the owner's name for a computer survives a reconnect ------------


def test_a_client_that_repeats_its_own_id_does_not_rename_the_workplace():
    """The room PC used to send its client id as its name on every hello."""
    assert hub_app.workplace_display_name(
        "livingroom", "livingroom", stored="anton", home="anton") == "anton"


def test_the_room_name_is_the_fallback_when_nothing_was_set():
    assert hub_app.workplace_display_name("livingroom", "", home="anton") == "anton"


def test_a_stored_name_that_repeats_the_client_id_is_still_the_name():
    """buro: its own name is "buro", but the home it shares is called "anton".

    Владелец 2026-09-23: «почему buro pc называется anton». The stored name used
    to be skipped whenever it equalled the client id, so the friend's PC fell
    through to the home's name — "anton" — on every hello.
    """
    assert hub_app.workplace_display_name("buro", "buro", stored="buro", home="anton") == "buro"


def test_a_real_configured_name_from_the_client_still_wins():
    assert hub_app.workplace_display_name(
        "livingroom", "AntonDorm", stored="anton", home="anton") == "AntonDorm"


def test_an_unknown_workplace_falls_back_to_its_client_id():
    assert hub_app.workplace_display_name("room-9de3", "", stored="", home="") == "room-9de3"
    assert hub_app.workplace_display_name("", "") == "Room"


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


def test_a_stale_offline_namesake_does_not_steal_the_request(monkeypatch):
    """buro had a phantom twin: "room-9de3bed07b44", also called "buro".

    Владелец 2026-09-24: «Say to buro: buro hello this is anton» answered "the
    room is unreachable" while the real buro was online and answering - the
    named lookup took the stale entry, because it came first in the list.
    """
    live = _connection("buro", "livingroom")
    live.workplace_name, live.camera_name = "buro", "Buro camera"
    stored = {"room-9de3bed07b44": {"id": "room-9de3bed07b44", "name": "buro",
                                    "camera_name": "Buro camera", "home_id": ""}}
    monkeypatch.setattr(hub_app, "_telegram_access",
                        SimpleNamespace(get_setting=lambda key, default=None:
                                        stored if key == "workplaces" else default))
    monkeypatch.setattr(hub_app, "_connections", [live])

    assert hub_app._named_workplace('say to buro: buro hello this is anton') == "buro"


def test_an_offline_namesake_is_still_named_when_no_live_room_matches(monkeypatch):
    """A computer the owner really named offline must still hear about itself."""
    stored = {"room-9de3bed07b44": {"id": "room-9de3bed07b44", "name": "buro",
                                    "camera_name": "Buro camera", "home_id": ""}}
    monkeypatch.setattr(hub_app, "_telegram_access",
                        SimpleNamespace(get_setting=lambda key, default=None:
                                        stored if key == "workplaces" else default))
    monkeypatch.setattr(hub_app, "_connections", [])

    assert hub_app._named_workplace('photo from buro') == "room-9de3bed07b44"


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
