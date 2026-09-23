"""The room client authenticates with a token when it has one (ТЗ 4.3).

Владелец 2026-09-23: комната подключалась без токена вообще, поэтому у хаба не
было её ``home_id`` — и всё привязанное к дому (облачное чтение реплики, запись
лиц, облачный взгляд на кадр) молча выключалось. Здесь проверяется и новое
поведение, и то, что клиент без токена остаётся прежним v1-клиентом.
"""
from types import SimpleNamespace

from client.main import build_hello

TOKEN_ENV = 'ROWAN_TEST_CLIENT_TOKEN'


def _config(**overrides):
    values = dict(client_id='livingroom', kind='room_pc', home_id='livingroom',
                  token_env=TOKEN_ENV, workplace_name='anton',
                  camera=SimpleNamespace(name='Main camera'), devices=[])
    values.update(overrides)
    return SimpleNamespace(**values)


def test_without_a_token_the_hello_stays_v1(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    hello = build_hello(_config())
    assert 'token' not in hello
    assert 'proto' not in hello
    assert hello['client_id'] == 'livingroom'
    assert hello['kind'] == 'room_pc'


def test_a_token_makes_the_hello_authenticate(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, 'a-real-token')
    hello = build_hello(_config())
    assert hello['token'] == 'a-real-token'
    assert hello['proto'] == 2
    assert hello['home_id'] == 'livingroom'


def test_an_empty_variable_is_not_a_token(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, '   ')
    assert 'token' not in build_hello(_config())


def test_a_config_without_a_token_name_never_sends_one(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, 'a-real-token')
    hello = build_hello(_config(token_env=''))
    assert 'token' not in hello


def test_a_client_without_a_home_does_not_declare_one(monkeypatch):
    """The hub takes ``home_id`` from the token, so an empty config omits it."""
    monkeypatch.setenv(TOKEN_ENV, 'a-real-token')
    hello = build_hello(_config(home_id=''))
    assert 'home_id' not in hello
    assert hello['token'] == 'a-real-token'


def test_the_token_is_not_copied_into_any_other_field(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, 'a-real-token')
    hello = build_hello(_config())
    everything_else = {key: value for key, value in hello.items() if key != 'token'}
    assert 'a-real-token' not in str(everything_else)
