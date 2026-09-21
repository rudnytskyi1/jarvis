"""Room-client configuration: hub identity fields are additive (ТЗ phase 0)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.client_config import ClientConfig, load_client_config


def test_legacy_client_config_keeps_working():
    cfg = ClientConfig(server_url="ws://192.168.1.100:8765/ws")
    assert cfg.home_id == "" and cfg.kind == "room_pc"
    assert cfg.token_env == "ROWAN_CLIENT_TOKEN"
    assert cfg.caps == []
    assert cfg.websocket_url == "ws://192.168.1.100:8765/ws", "hub_url falls back to the legacy URL"


def test_hub_url_takes_precedence_over_server_url():
    cfg = ClientConfig(server_url="ws://old/ws", hub_url="wss://hub.internal/ws")
    assert cfg.websocket_url == "wss://hub.internal/ws"


def test_client_template_carries_the_hub_identity():
    settings = load_client_config("config.client.example.yaml")
    client = settings.client
    assert client.home_id == "livingroom"
    assert client.client_id == "livingroom-pc"
    assert client.kind == "room_pc"
    assert client.token_env == "ROWAN_CLIENT_TOKEN"
    assert {"camera", "hud"} <= set(client.caps)
    assert client.websocket_url.endswith("/ws")


def test_unknown_kind_is_rejected():
    with pytest.raises(ValidationError):
        ClientConfig(server_url="ws://x/ws", kind="toaster")


def test_unknown_client_key_is_rejected():
    with pytest.raises(ValidationError):
        ClientConfig(server_url="ws://x/ws", homeid="typo")
