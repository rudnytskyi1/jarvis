"""Cloud decisions are not a switch any more (owner, 2026-09-23).

Владелец: «что за фигня? всм cloud decisions выключены? что это за мусор? убери
включение выключение этих cloud decisions, я хочу чтобы работало». Здесь
проверяется, что флаг дома больше никого не выключает и что комната без
``home_id`` (v1 ``hello`` без токена) всё равно получает чтение реплики: хаб с
одним домом точно знает, какой это дом.
"""
from types import SimpleNamespace

from common.config import HomeConfig
from hub import app as hub_app


def _config(*home_ids):
    return SimpleNamespace(homes=[HomeConfig(home_id=home, name=home) for home in home_ids])


def test_the_home_flag_no_longer_switches_the_cloud_off(monkeypatch):
    """The flag is inert: a home that says no is still read by Jev."""
    monkeypatch.setattr(hub_app, '_config', _config('livingroom'))
    hub_app._config.homes[0].cloud_decisions = False  # a config with the old taste
    assert hub_app._home_allows_cloud_decisions('livingroom') is True
    assert hub_app._home_allows_cloud_decisions('buro') is True
    assert hub_app._home_allows_cloud_decisions('') is True


def test_a_room_without_a_home_gets_the_only_home_of_the_hub(monkeypatch):
    monkeypatch.setattr(hub_app, '_config', _config('livingroom'))
    assert hub_app._home_for_decisions('') == 'livingroom'
    assert hub_app._home_for_decisions('livingroom') == 'livingroom'


def test_two_homes_are_never_guessed(monkeypatch):
    """With more than one room an unknown home stays unknown, honestly."""
    monkeypatch.setattr(hub_app, '_config', _config('livingroom', 'buro'))
    assert hub_app._home_for_decisions('') == ''
    assert hub_app._home_for_decisions('buro') == 'buro'


def test_a_room_without_a_home_still_follows_its_home_for_vision(monkeypatch):
    """Cloud vision keeps its own flag, but the flag now finds its home."""
    monkeypatch.setattr(hub_app, '_config', _config('livingroom'))
    hub_app._config.homes[0].cloud_vision = True
    connection = SimpleNamespace(home_id='', cfg=hub_app._config)
    assert hub_app.Connection._cloud_vision_allowed(connection) is True
    hub_app._config.homes[0].cloud_vision = False
    assert hub_app.Connection._cloud_vision_allowed(connection) is False
