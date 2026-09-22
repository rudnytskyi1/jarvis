"""The owner's web panel: overlay only, hash-only tokens (ТЗ F-705)."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.config import Config, load_config
from hub import migrations_runner
from hub.labelling import day_of, day_window
from hub.web_admin import COOKIE, WebAdminAuth, WebAdminData, build_router, mount, overlay_only, panel_names

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
    # The panel also checks the name it was called by, so the browser in these
    # tests asks for it as the overlay address it really has.
    return TestClient(app, base_url="http://127.0.0.1", client=client), auth


# --- the gate ---------------------------------------------------------------


@pytest.mark.parametrize("host,expected", [
    ("100.64.0.7", True), ("100.101.102.103", True), ("127.0.0.1", True),
    ("192.168.1.50", False), ("10.0.0.5", False), ("8.8.8.8", False),
    ("::ffff:100.64.0.9", True), ("::1", False), (None, False), ("not-an-ip", False),
])
def test_only_the_overlay_network_may_look(host, expected):
    assert overlay_only(["127.0.0.0/8", "100.64.0.0/10"])(host) is expected


@pytest.mark.parametrize("name,expected", [
    ("127.0.0.1:8770", True), ("localhost:8770", True), ("100.64.0.7", True),
    ("[::ffff:100.64.0.9]:8770", True),
    ("dorm-smart-un-iversity-of-nebr-omaha.ngrok.app", False),
    ("rowan.example.com", False), ("8.8.8.8", False), ("", False), (None, False),
])
def test_the_panel_answers_under_its_own_names_only(name, expected):
    allowed = panel_names(["127.0.0.0/8", "100.64.0.0/10"])
    assert allowed(name) is expected, "публичное имя туннеля — не наш адрес"


def test_the_public_tunnel_never_shows_the_panel(tmp_path, monkeypatch):
    """The tunnel reaches the hub from loopback: only the name gives it away."""
    monkeypatch.setenv("ROWAN_TEST_PASSWORD", PASSWORD)
    cfg = Config()
    cfg.server.web_admin.password_env = "ROWAN_TEST_PASSWORD"
    auth = WebAdminAuth(password_env="ROWAN_TEST_PASSWORD", secret=b"test-key")
    migrated(tmp_path).close()
    app = FastAPI()
    app.include_router(build_router(cfg=cfg, data=WebAdminData(tmp_path / "hub.db"), auth=auth))
    tunnel = TestClient(app, base_url="http://dorm-smart-un-iversity-of-nebr-omaha.ngrok.app",
                        client=OVERLAY)
    assert tunnel.get("/admin").status_code == 404
    assert tunnel.post("/admin/login", data={"password": PASSWORD}).status_code == 404


def test_the_lan_never_sees_even_the_login_form(tmp_path, monkeypatch):
    client, _ = a_client(tmp_path, monkeypatch, client=LAN)
    for path in ("/admin", "/admin/homes", "/admin/clients", "/admin/people", "/admin/tracks",
                 "/admin/turns", "/admin/crop/c1.jpg"):
        answer = client.get(path)
        assert answer.status_code == 404
        assert "overlay" in answer.text
    assert client.post("/admin/login", data={"password": PASSWORD}).status_code == 404


def test_config_declares_the_panel_in_both_files():
    """This deployment switched the panel on; the example stays a safe default."""
    live = load_config("config.yaml").server.web_admin
    assert live.enabled is True, "владелец открывает /admin на своём хабе"
    example = load_config("config.example.yaml").server.web_admin
    assert example.enabled is False, "пример не включает панель сам по себе"
    for settings in (live, example):
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


# --- цепочка запроса (панель) -----------------------------------------------


def _seed_chain(tmp_path, turn_id="u-1", home="livingroom"):
    """Two steps of one request: a decision and a failed tool call."""
    conn = sqlite3.connect(str(tmp_path / "hub.db"))
    rows = [
        (turn_id, home, 1_700_000_000.0, "turn", "room", 1, 0,
         '{"transcript": "\u0442\u044b \u0442\u0443\u0442", "reply": "\u0442\u0443\u0442"}'),
        (turn_id, home, 1_700_000_000.4, "decision", "jev", 1, 475,
         '{"type": "route", "value": "chat", "confidence": 0.91}'),
        (turn_id, "", 1_700_000_001.2, "tool", "generate_image", 0, 2100,
         '{"args": {"prompt": "a cat"}, "result": {"ok": false, "error": "no key"}}'),
    ]
    conn.executemany("INSERT INTO turn_events(turn_id, home_id, ts, kind, name, ok,"
                     " latency_ms, payload_json) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_the_requests_page_lists_every_request_with_its_chain(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_chain(tmp_path)

    page = browser.get("/admin/turns")
    assert page.status_code == 200
    assert "u-1" in page.text and "/admin/turns/u-1" in page.text
    assert "1 decision(s)" in page.text and "1 tool call(s)" in page.text
    assert "failed" in page.text, "сломанный шаг виден в списке"
    assert "Requests" in browser.get("/admin/homes").text, "в навигации есть запросы"


def test_one_request_shows_its_steps_in_order(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_chain(tmp_path)

    page = browser.get("/admin/turns/u-1")
    assert page.status_code == 200
    # Every step is a card with a readable line, not only raw JSON.
    assert page.text.count('class="summary"') == 3, "у каждого шага своя строка"
    assert "ты тут" in page.text, "что услышали"
    assert "→ тут" in page.text, "и что ответили"
    assert "jev" in page.text and "0.91" in page.text, "кто решил и с какой уверенностью"
    assert "route = chat" in page.text, "решение читается словами"
    assert "generate_image" in page.text and "no key" in page.text, "что вернул инструмент"
    assert 'class="card step algorithm"' not in page.text, "класс шага — это его вид"
    assert page.text.index("decision") < page.text.index("generate_image"), "шаги по порядку"


def test_a_telegram_request_opens_with_its_colons_intact(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    turn_id = "telegram:-1003570242441:42"
    _seed_chain(tmp_path, turn_id=turn_id, home="livingroom")

    listing = browser.get("/admin/turns").text
    assert "telegram%3A-1003570242441%3A42" in listing, "ссылка кодирует двоеточия"
    detail = browser.get(f"/admin/turns/{turn_id}")
    assert detail.status_code == 200 and "u-1" not in detail.text
    assert "generate_image" in detail.text


def test_the_requests_page_asks_for_a_session(tmp_path, monkeypatch):
    browser, _auth = a_client(tmp_path, monkeypatch)
    assert browser.get("/admin/turns", follow_redirects=False).headers["location"] == "/admin"
    assert browser.get("/admin/turns/u-1", follow_redirects=False).headers["location"] == "/admin"


# --- ручная разметка (ТЗ F-216) ---------------------------------------------


def _seed_unknown_track(tmp_path, day: str = "") -> Path:
    """One track nobody named: crops of that day, samples, and a belief."""
    day = day or day_of()
    crop_dir = tmp_path / "homes" / "livingroom" / "body"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop = crop_dir / "t-unknown-1.jpg"
    crop.write_bytes(b"\xff\xd8\xff\xe0a-jpeg")
    start, _end = day_window(day)
    conn = sqlite3.connect(str(tmp_path / "hub.db"))
    conn.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen)"
                 " VALUES ('t-unknown-1','livingroom','room-pc',?,?)",
                 (f"{day}T09:00:00", f"{day}T09:30:00"))
    conn.execute("INSERT INTO body_crops(crop_id, home_id, client_id, track_id, ts, width,"
                 " height, path) VALUES ('c1','livingroom','room-pc','t-unknown-1',?,640,480,?)",
                 (start + 60, str(crop)))
    conn.execute("INSERT INTO face_embeddings(id, person_id, track_id, vector, dim, quality)"
                 " VALUES ('f1',NULL,'t-unknown-1',?,8,0.8)", (b"\x00" * 32,))
    conn.execute("INSERT INTO body_embeddings(id, person_id, track_id, session_day, vector,"
                 " dim, quality) VALUES ('b1',NULL,'t-unknown-1',?,?,8,0.6)",
                 (day, b"\x00" * 32))
    conn.execute("INSERT INTO voice_embeddings(id, person_id, track_id, vector, dim)"
                 " VALUES ('v1',NULL,'t-unknown-1',?,8)", (b"\x00" * 32,))
    conn.execute("INSERT INTO identity_belief(track_id, home_id, person_id, p, sources_json, at)"
                 " VALUES ('t-unknown-1','livingroom',NULL,0.41,'{\"voice\": 0.52}',123.0)")
    conn.commit()
    conn.close()
    return crop


def _signed_in(tmp_path, monkeypatch, *, client=OVERLAY):
    browser, _auth = a_client(tmp_path, monkeypatch, client=client)
    browser.post("/admin/login", data={"password": PASSWORD}, follow_redirects=False)
    return browser


def test_the_queue_shows_the_unknown_track_with_its_evidence(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_unknown_track(tmp_path)

    page = browser.get("/admin/tracks")
    assert page.status_code == 200
    assert "t-unknown-1" in page.text and "livingroom" in page.text
    assert "1 / 1 / 1" in page.text, "лицо, тело и голос видны владельцу"
    assert "p 0.41" in page.text and "voice 0.52" in page.text, "слияние — с числами"
    assert "/admin/crop/c1.jpg" in page.text
    assert "Anton" in page.text, "человека выбирают из настоящих людей"
    assert "Tracks" in browser.get("/admin/homes").text, "в навигации есть очередь"


def test_the_crop_route_serves_only_files_under_the_data_directory(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    crop = _seed_unknown_track(tmp_path)

    answer = browser.get("/admin/crop/c1.jpg")
    assert answer.status_code == 200
    assert answer.headers["content-type"] == "image/jpeg"
    assert answer.content == crop.read_bytes()
    assert browser.get("/admin/crop/nope.jpg").status_code == 404

    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"\xff\xd8\xff")
    conn = sqlite3.connect(str(tmp_path / "hub.db"))
    conn.execute("UPDATE body_crops SET path=? WHERE crop_id='c1'", (str(outside),))
    conn.commit()
    conn.close()
    assert browser.get("/admin/crop/c1.jpg").status_code == 404, "чужая папка — не наш файл"


def test_the_click_binds_the_track_links_the_samples_and_keeps_the_label(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_unknown_track(tmp_path)
    day = day_of()

    answer = browser.post("/admin/label", data={"track_id": "t-unknown-1",
                                                "person_id": "p-anton", "day": day,
                                                "crop_id": "c1"}, follow_redirects=False)
    assert answer.status_code == 303 and answer.headers["location"].startswith("/admin/tracks")

    conn = sqlite3.connect(str(tmp_path / "hub.db"))
    assert conn.execute("SELECT person_id FROM tracks WHERE track_id='t-unknown-1'"
                        ).fetchone()[0] == "p-anton"
    assert conn.execute("SELECT person_id FROM face_embeddings WHERE id='f1'"
                        ).fetchone()[0] == "p-anton"
    assert conn.execute("SELECT person_id FROM body_embeddings WHERE id='b1'"
                        ).fetchone()[0] == "p-anton"
    assert conn.execute("SELECT person_id FROM voice_embeddings WHERE id='v1'"
                        ).fetchone()[0] == "p-anton"
    assert conn.execute("SELECT track_id, person_id, day, crop_id, actor, source"
                        " FROM identity_labels").fetchone() == ("t-unknown-1", "p-anton",
                                                                day, "c1", "web-panel", "admin")
    assert conn.execute("SELECT action, actor_person_id, target, result FROM audit"
                        " WHERE action='identity.label'").fetchall() == [
                            ("identity.label", "web-panel", "t-unknown-1", "ok")]
    conn.close()
    assert "Track t-unknown-1" in browser.get(answer.headers["location"]).text
    assert "Nobody is waiting for a name" in browser.get("/admin/tracks").text


def test_a_click_about_nobody_known_changes_nothing(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_unknown_track(tmp_path)

    answer = browser.post("/admin/label", data={"track_id": "t-unknown-1",
                                                "person_id": "p-nope", "day": day_of()},
                          follow_redirects=False)
    assert answer.status_code == 303
    assert "Nothing changed" in browser.get(answer.headers["location"]).text
    conn = sqlite3.connect(str(tmp_path / "hub.db"))
    assert conn.execute("SELECT person_id FROM tracks WHERE track_id='t-unknown-1'"
                        ).fetchone()[0] is None
    assert conn.execute("SELECT COUNT(*) FROM identity_labels").fetchone()[0] == 0
    conn.close()


def test_an_unsigned_request_never_labels_anything(tmp_path, monkeypatch):
    browser, _auth = a_client(tmp_path, monkeypatch)
    _seed_unknown_track(tmp_path)

    assert browser.get("/admin/tracks", follow_redirects=False).headers["location"] == "/admin"
    assert browser.get("/admin/crop/c1.jpg").status_code == 404
    refused = browser.post("/admin/label", data={"track_id": "t-unknown-1",
                                                "person_id": "p-anton"}, follow_redirects=False)
    assert refused.status_code == 303 and refused.headers["location"] == "/admin"
    conn = sqlite3.connect(str(tmp_path / "hub.db"))
    assert conn.execute("SELECT COUNT(*) FROM identity_labels").fetchone()[0] == 0
    assert conn.execute("SELECT person_id FROM tracks WHERE track_id='t-unknown-1'"
                        ).fetchone()[0] is None
    conn.close()
