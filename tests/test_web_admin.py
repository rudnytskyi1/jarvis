"""The owner's web panel: overlay only, hash-only tokens (ТЗ F-705)."""
from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.config import Config, load_config
from hub import migrations_runner
from hub.web_admin import COOKIE, WebAdminAuth, WebAdminData, build_router, mount, overlay_only

OVERLAY, LAN = ("100.64.0.7", 50000), ("192.168.1.50", 50000)
PASSWORD = "dorm-owner-password"


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-alice', 'Alice')")
    conn.execute("INSERT INTO homes(home_id, name, tz, owner_person_id)"
                 " VALUES ('livingroom', 'Living room', 'Europe/Berlin', 'p-anton')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role, share_identity, share_presence)"
                 " VALUES ('p-anton', 'livingroom', 'admin', 1, 1)")
    conn.execute("INSERT INTO clients(client_id, home_id, kind, token_hash, version, hw, last_seen)"
                 " VALUES ('room-pc', 'livingroom', 'room_pc', 'hash-of-the-token', '1.6',"
                 " 'Ryzen', '2026-09-21T10:00:00')")
    conn.commit()
    return conn


def a_client(tmp_path, monkeypatch, *, client=OVERLAY, password=PASSWORD):
    monkeypatch.setenv("ROWAN_TEST_PASSWORD", password)
    cfg = Config()
    cfg.server.web_admin.password_env = "ROWAN_TEST_PASSWORD"
    auth = WebAdminAuth(password_env=cfg.server.web_admin.password_env, secret=b"test-key")
    database = migrated(tmp_path)
    database.close()
    app = FastAPI()
    app.include_router(build_router(cfg=cfg, data=WebAdminData(tmp_path / "hub.db"), auth=auth))
    return TestClient(app, client=client), auth


# --- the gate ---------------------------------------------------------------


@pytest.mark.parametrize("host,expected", [
    ("100.64.0.7", True), ("100.101.102.103", True), ("127.0.0.1", True),
    ("192.168.1.50", False), ("10.0.0.5", False), ("8.8.8.8", False),
    ("::ffff:100.64.0.9", True), ("::1", False), (None, False), ("not-an-ip", False),
])
def test_only_the_overlay_network_may_look(host, expected):
    assert overlay_only(["127.0.0.0/8", "100.64.0.0/10"])(host) is expected


def test_the_lan_never_sees_even_the_login_form(tmp_path, monkeypatch):
    client, _ = a_client(tmp_path, monkeypatch, client=LAN)
    for path in ("/admin", "/admin/homes", "/admin/clients", "/admin/people"):
        answer = client.get(path)
        assert answer.status_code == 404
        assert "overlay" in answer.text
    assert client.post("/admin/login", data={"password": PASSWORD}).status_code == 404


def test_config_declares_the_panel_in_both_files():
    for name in ("config.yaml", "config.example.yaml"):
        settings = load_config(name).server.web_admin
        assert settings.enabled is False
        assert settings.allowed_networks == ["127.0.0.0/8", "100.64.0.0/10"]
        assert settings.password_env == "ROWAN_ADMIN_PASSWORD"


def test_a_bad_network_is_a_typo_not_silence():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config.model_validate({"server": {"web_admin": {"allowed_networks": ["100.64.0.0/99"]}}})


# --- the password -----------------------------------------------------------


def test_the_password_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("ROWAN_MISSING_PASSWORD", raising=False)
    auth = WebAdminAuth(password_env="ROWAN_MISSING_PASSWORD")
    assert auth.verify("anything") is False, "an unset password lets nobody in"


def test_a_session_cookie_is_signed_and_expires(monkeypatch):
    monkeypatch.setenv("ROWAN_TEST_PASSWORD", PASSWORD)
    now = [1000.0]
    auth = WebAdminAuth(password_env="ROWAN_TEST_PASSWORD", session_minutes=1, secret=b"k",
                        clock=lambda: now[0])
    assert auth.verify(PASSWORD) is True and auth.verify("nope") is False
    cookie = auth.issue()
    assert auth.valid(cookie) is True
    assert auth.valid(cookie.replace(cookie[-1], "0" if cookie[-1] != "0" else "1")) is False
    assert auth.valid("garbage") is False and auth.valid(None) is False
    now[0] += 61
    assert auth.valid(cookie) is False, "the session does not outlive its minutes"


# --- the pages --------------------------------------------------------------


def test_the_owner_signs_in_and_sees_homes_clients_and_people(tmp_path, monkeypatch):
    client, _ = a_client(tmp_path, monkeypatch)
    assert "Sign in" in client.get("/admin").text
    assert client.post("/admin/login", data={"password": "wrong"}).status_code == 403
    assert COOKIE not in client.cookies

    signed = client.post("/admin/login", data={"password": PASSWORD}, follow_redirects=False)
    assert signed.status_code == 303 and signed.headers["location"] == "/admin/homes"
    assert COOKIE in client.cookies

    homes = client.get("/admin/homes")
    assert "Living room" in homes.text and "Europe/Berlin" in homes.text
    clients = client.get("/admin/clients")
    assert "room-pc" in clients.text and "1.6" in clients.text
    people = client.get("/admin/people")
    assert "Anton" in people.text and "admin" in people.text and "identity" in people.text

    client.get("/admin/logout")
    assert client.get("/admin/homes", follow_redirects=False).status_code == 303


def test_the_panel_never_shows_a_usable_token(tmp_path, monkeypatch):
    client, _ = a_client(tmp_path, monkeypatch)
    client.post("/admin/login", data={"password": PASSWORD}, follow_redirects=False)
    text = client.get("/admin/clients").text
    assert "hash-of-the-token" not in text
    hint = hashlib.sha256(b"hash-of-the-token").hexdigest()
    assert hint[:6] in text and hint[-4:] in text
    assert "the tokens themselves are never kept" in text


def test_a_request_without_a_session_goes_to_the_login(tmp_path, monkeypatch):
    client, _ = a_client(tmp_path, monkeypatch)
    answer = client.get("/admin/people", follow_redirects=False)
    assert answer.status_code == 303 and answer.headers["location"] == "/admin"


def test_mounting_is_off_until_the_config_says_otherwise(tmp_path, monkeypatch):
    app = FastAPI()
    cfg = Config()
    assert mount(app, cfg=cfg, data=None) is None
    assert not [route for route in app.routes if getattr(route, "path", "").startswith("/admin")]

    cfg.server.web_admin.enabled = True
    monkeypatch.setenv("ROWAN_ADMIN_PASSWORD", PASSWORD)
    assert mount(app, cfg=cfg, data=None) is not None
    assert [route for route in app.routes if getattr(route, "path", "") == "/admin/homes"]
