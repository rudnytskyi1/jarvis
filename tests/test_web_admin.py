"""The owner's web panel: overlay only, hash-only tokens (ТЗ F-705)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.config import Config, load_config
from hub import migrations_runner
from hub.appearance import AppearanceGallery
from hub.labelling import day_of, day_window
from hub.web_admin import (
    COOKIE,
    WebAdminAuth,
    WebAdminData,
    build_router,
    classifier_stages,
    mount,
    overlay_only,
    panel_names,
)

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
                 "/admin/turns", "/admin/crop/c1.jpg", "/admin/profiles",
                 "/admin/profiles/p-anton", "/admin/appearance/g/s/f.jpg"):
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


# --- классификатор профилей (ТЗ F-203, F-209, F-211) ------------------------


def _seed_archive(tmp_path, *, person_id="g-anton", name="Anton", samples=3):
    """An appearance archive written with the hub's own schema, not a copy of it."""
    root = tmp_path / "appearance"
    tag = person_id.split("-")[-1]
    gallery = AppearanceGallery(root)
    gallery._connect().close()  # the real DDL of ``AppearanceGallery``
    folder = root / person_id
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(gallery.database))
    try:
        conn.execute("INSERT OR IGNORE INTO people VALUES(?,?,?)",
                     (person_id, name, name.casefold()))
        made = []
        for index in range(samples):
            sample_id = f"{tag}-s{index + 1}"
            (folder / f"{sample_id}-face.jpg").write_bytes(b"\xff\xd8\xff\xe0face" + sample_id.encode())
            (folder / f"{sample_id}-body.jpg").write_bytes(b"\xff\xd8\xff\xe0body" + sample_id.encode())
            quality = {"identity_score": 0.90 + index / 100, "identity_margin": 0.40,
                       "detector_score": 0.85, "sharpness": 50.0 + index,
                       "face_pixels": [78, 94], "admission": "manual_anchor_confirmation",
                       "track_id": f"track-{sample_id}"}
            conn.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?)", (
                sample_id, person_id, name, 1_789_000_000.0 + index * 3600,
                f"{person_id}/{sample_id}-face.jpg", f"{person_id}/{sample_id}-body.jpg",
                json.dumps([0.0] * 8), json.dumps(quality), "a" * 64,
                "b" * 64 if index == 0 else "c" * 64))
            made.append(sample_id)
        conn.commit()
    finally:
        conn.close()
    return root, made


def test_the_pipeline_page_keeps_the_honest_state_of_every_signal():
    """The panel says what runs, at which number, and what is missing here."""
    stages = classifier_stages(Config())
    titles = [stage["title"] for stage in stages]
    assert "2 · Face signature" in titles and "5 · Fusion into one person" in titles
    face = next(stage for stage in stages if stage["key"] == "face")
    assert ("match threshold", "0.45") in face["numbers"], "число — из живого конфига"
    assert face["state"] == "on"

    missing = classifier_stages(Config(), body_reid=False)
    reid = next(stage for stage in missing if stage["key"] == "body")
    assert reid["state"] == "degraded", "тело без torchreid — не «on» и не «off»"
    assert "torchreid is not installed" in reid["note"]
    assert ("model", "osnet_x1_0") in reid["numbers"]
    assert ("same-day threshold", "0.5") in reid["numbers"]


def test_the_classifier_page_pairs_people_with_their_archive(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_archive(tmp_path)
    _seed_archive(tmp_path, person_id="g-vera", name="Vera", samples=1)

    page = browser.get("/admin/profiles")
    assert page.status_code == 200
    assert "Profile classifier" in page.text
    assert "0.45" in page.text, "пороги видны владельцу"
    assert 'href="/admin/profiles/p-anton"' in page.text, "человек и архив сходятся по имени"
    assert "/admin/appearance/g-anton/anton-s3/face.jpg" in page.text, "обложка — свежий кадр"
    assert 'href="/admin/profiles/archive:g-vera"' in page.text, "архив без человека виден тоже"
    assert "Profiles" in browser.get("/admin/homes").text, "в навигации есть профили"


def test_one_profile_spins_the_frames_that_were_really_archived(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_archive(tmp_path)

    page = browser.get("/admin/profiles/p-anton")
    assert page.status_code == 200
    assert 'id="turntable"' in page.text and 'id="filmstrip"' in page.text
    assert page.text.count('class="frame"') == 3, "на кольце — кадры архива"
    assert 'data-identity="0.920"' in page.text, "числа кадра — из quality архива"
    assert "/admin/appearance/g-anton/anton-s1/face.jpg" in page.text
    assert "translateZ" in page.text, "кольцо строится в браузере"
    assert "every outfit" in page.text and "0.920" in page.text
    assert browser.get("/admin/profiles/no-such-person").status_code == 404


def test_a_look_is_a_run_of_the_same_clothes():
    """A day is not a look: people change, and one odd frame is not a change."""

    def frame(stamp: str, index: int) -> dict:
        return {"day": stamp[:10], "captured_at": stamp, "sample_id": f"s{index}",
                "outfit": "a" * 12, "appearance_hash": "a" * 64, "face": f"f{index}.jpg",
                "body": f"b{index}.jpg", "identity": "0.900", "sharpness": "50.0",
                "pixels": "78×94", "admission": "manual_anchor_confirmation", "track": "t-1"}

    shirt, jacket = (1.0, 0.0), (0.0, 1.0)
    stamps = ["2026-09-23 09:10", "2026-09-23 09:40", "2026-09-23 14:00",
              "2026-09-23 14:30", "2026-09-23 14:45"]
    samples = [frame(stamp, index) for index, stamp in enumerate(stamps)]
    clothes = {"s0": shirt, "s1": shirt, "s2": jacket, "s3": jacket, "s4": jacket}

    looks = WebAdminData.outfits(samples, clothes)
    assert [look["count"] for look in looks] == [3, 2], "новая одежда — новый образ"
    assert looks[0]["index"] == 1 and looks[0]["first"] == "14:00"
    assert looks[0]["last"] == "14:45" and looks[0]["changed"] is True
    assert looks[0]["face"] == "f4.jpg", "обложка образа — самый свежий кадр"
    assert looks[1]["first"] == "09:10" and looks[1]["last"] == "09:40"

    # One frame that reads differently is not a change of clothes.
    one_off = [frame(stamp, index) for index, stamp in enumerate(stamps[:4])]
    clothes.update({"s2": jacket, "s3": shirt})
    looks = WebAdminData.outfits(one_off, clothes)
    assert len(looks) == 1 and looks[0]["count"] == 4
    assert looks[0]["changed"] is False

    # Frames without a body crop keep the look they arrived in.
    looks = WebAdminData.outfits(one_off, {})
    assert len(looks) == 1 and looks[0]["count"] == 4


def test_a_look_keeps_the_other_days_apart():
    def frame(stamp: str, index: int) -> dict:
        return {"day": stamp[:10], "captured_at": stamp, "sample_id": f"s{index}",
                "outfit": "a" * 12, "appearance_hash": "a" * 64, "face": f"f{index}.jpg",
                "body": "", "identity": "0.900", "sharpness": "50.0", "pixels": "78×94",
                "admission": "manual_anchor_confirmation", "track": "t-1"}

    samples = [frame("2026-09-23 10:00", 0), frame("2026-09-23 09:10", 1),
               frame("2026-09-22 21:00", 2)]
    looks = WebAdminData.outfits(samples, {})
    assert [look["day"] for look in looks] == ["2026-09-23", "2026-09-22"]
    assert [look["count"] for look in looks] == [2, 1]
    assert looks[0]["first"] == "09:10" and looks[0]["last"] == "10:00"


def test_the_classifier_holds_a_person_without_any_photograph(tmp_path, monkeypatch):
    """An empty archive is not an error: the manual profile is still listed."""
    browser = _signed_in(tmp_path, monkeypatch)

    page = browser.get("/admin/profiles")
    assert page.status_code == 200
    assert "Anton" in page.text and "no photo" in page.text
    detail = browser.get("/admin/profiles/p-anton")
    assert detail.status_code == 200
    assert "no confirmed sighting was kept" in detail.text


def test_an_archived_frame_is_served_only_for_the_person_it_belongs_to(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    root, _made = _seed_archive(tmp_path)

    face = browser.get("/admin/appearance/g-anton/anton-s1/face.jpg")
    assert face.status_code == 200 and face.headers["content-type"] == "image/jpeg"
    assert face.content == (root / "g-anton" / "anton-s1-face.jpg").read_bytes()
    assert browser.get("/admin/appearance/g-anton/anton-s1/body.jpg").status_code == 200
    assert browser.get("/admin/appearance/g-vera/vera-s1/face.jpg").status_code == 404, "чужой архив"
    assert browser.get("/admin/appearance/g-anton/anton-s9/face.jpg").status_code == 404
    assert browser.get("/admin/appearance/g-anton/anton-s1/nose.jpg").status_code == 404

    outside = tmp_path.parent / "outside-appearance.jpg"
    outside.write_bytes(b"\xff\xd8\xff")
    conn = sqlite3.connect(str(tmp_path / "appearance" / "gallery.sqlite3"))
    conn.execute("UPDATE samples SET face_path=? WHERE id='anton-s1'", (str(outside),))
    conn.commit()
    conn.close()
    assert browser.get("/admin/appearance/g-anton/anton-s1/face.jpg").status_code == 404, \
        "путь за пределы архива — не наш файл"


def test_the_archive_answers_nobody_without_a_session(tmp_path, monkeypatch):
    browser, _auth = a_client(tmp_path, monkeypatch)
    _seed_archive(tmp_path)

    assert browser.get("/admin/profiles", follow_redirects=False).headers["location"] == "/admin"
    assert browser.get("/admin/profiles/p-anton",
                       follow_redirects=False).headers["location"] == "/admin"
    assert browser.get("/admin/appearance/g-anton/anton-s1/face.jpg").status_code == 404


def _seed_model(tmp_path, gallery_id="g-anton", points=4, triangles=2):
    """The files ``scripts/build-3d-profile.py`` writes, without the build."""
    directory = tmp_path / "appearance" / gallery_id / "model"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(bytes(range(6 * points)))
    (directory / "model.json").write_text(json.dumps({
        "points": points, "triangles": triangles,
        "meta": {"person": "Anton", "source": "archive", "method": "hull",
                 "views": 3, "of_frames": 79, "yaw_range": [-7.3, 90.0],
                 "method_note": "visual hull from silhouettes"},
    }), encoding="utf-8")
    (directory / "model.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    return directory


def test_the_profile_shows_the_carved_3d_build(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_archive(tmp_path)
    _seed_model(tmp_path)

    page = browser.get("/admin/profiles/p-anton")
    assert page.status_code == 200
    assert 'id="model"' in page.text, "в панели есть просмотрщик модели"
    assert "/admin/model/g-anton/model.bin" in page.text
    assert "4 points" in page.text and "3 of 79" in page.text, "чем построено — видно"
    assert "silhouettes is the body" in page.text, "текст для силуэтного метода"
    assert browser.get("/admin/model/g-anton/model.bin").status_code == 200
    assert browser.get("/admin/model/g-anton/model.ply").status_code == 200
    assert json.loads(browser.get("/admin/model/g-anton/model.json").text)["points"] == 4


def test_a_profile_without_a_build_says_how_to_make_one(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_archive(tmp_path)

    page = browser.get("/admin/profiles/p-anton")
    assert page.status_code == 200
    assert 'id="model"' not in page.text
    assert "scripts/build-3d-profile.py" in page.text, "команда сборки — на странице"
    assert browser.get("/admin/model/g-anton/model.bin").status_code == 404


def test_a_model_is_served_only_from_that_persons_own_build(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_archive(tmp_path)
    _seed_model(tmp_path)

    assert browser.get("/admin/model/g-vera/model.bin").status_code == 404, "чужой архив"
    assert browser.get("/admin/model/g-anton/model.txt").status_code == 404
    assert browser.get("/admin/model/g-anton/../g-anton/model.bin").status_code in (404, 200)
    data = WebAdminData(tmp_path / "hub.db")
    assert data.model_file("g-anton", "../../../hub.db") is None, "только свои имена файлов"
    assert data.model_file("g-vera", "model.bin") is None


def test_the_3d_build_is_closed_without_a_session(tmp_path, monkeypatch):
    browser, _auth = a_client(tmp_path, monkeypatch)
    _seed_model(tmp_path)

    assert browser.get("/admin/model/g-anton/model.bin").status_code == 404
    assert browser.get("/admin/model/g-anton/model.json").status_code == 404


# --- меню сборки 3D ---------------------------------------------------------


def _seed_training_archive(tmp_path):
    """A training archive with one Anton event holding a body crop."""
    root = tmp_path / "training_archive"
    event = root / "2026-09-20" / "Anton--abc" / "events" / "001836-x"
    event.mkdir(parents=True)
    (event / "body.png").write_bytes(b"\x89PNG")
    (event / "original.jpg").write_bytes(b"\xff\xd8\xff")
    conn = sqlite3.connect(str(root / "index.sqlite3"))
    conn.executescript(
        "CREATE TABLE identities(id TEXT PRIMARY KEY, name TEXT NOT NULL);"
        "CREATE TABLE events(person_id TEXT, kind TEXT, captured_at REAL, record TEXT);")
    conn.execute("INSERT INTO identities VALUES('p-anton', 'Anton')")
    conn.execute("INSERT INTO events VALUES(?,?,?,?)", (
        "p-anton", "appearance", 1_789_881_516.0, json.dumps({
            "kind": "appearance", "event_path": "2026-09-20/Anton--abc/events/001836-x/event.json",
            "files": ["body.png", "original.jpg"]})))
    conn.commit()
    conn.close()
    return root


def test_the_menu_offers_every_source_with_its_real_numbers(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_archive(tmp_path)
    _seed_training_archive(tmp_path)

    page = browser.get("/admin/profiles/p-anton").text
    assert 'action="/admin/profiles/p-anton/build"' in page, "кнопка сборки"
    assert "Training archive" in page and "Appearance archive" in page, "варианты источников"
    assert "Everything (default)" in page and "Track body crops" in page
    assert "Real photographs (depth relief)" in page, "метод по умолчанию"
    assert "Silhouette hull" in page, "и быстрый метод рядом"
    assert "2 frame(s)" in page, "архив показывает, сколько у него кадров"
    assert 'name="device"' in page and "Where to compute" in page, \
        "владелец выбирает, считать на GPU или на CPU"
    assert "Automatic" in page, "автоматический выбор устройства"
    assert '<option value="cuda">GPU (CUDA' in page, "явный выбор карты"
    assert '<option value="cpu">CPU only' in page, "явный выбор процессора"
    assert 'id="point-size"' in page, "меню просмотра тоже справа"
    assert 'id="build-status"' in page, "статус сборки видно на странице"


def test_the_build_status_is_polled_from_the_panel(tmp_path, monkeypatch):
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_archive(tmp_path)
    _seed_model(tmp_path)
    status = tmp_path / "appearance" / "g-anton" / "model" / "status.json"
    status.write_text(json.dumps({"state": "running", "done": 30, "total": 600,
                                  "seconds": 12.5, "points": 100,
                                  "options": {"source": "archive", "method": "photos"}}),
                      encoding="utf-8")

    answer = browser.get("/admin/profiles/p-anton/build-status")
    assert answer.status_code == 200 and answer.json()["state"] == "running"
    page = browser.get("/admin/profiles/p-anton").text
    assert "running" in page and "30 / 600" in page
    assert "/admin/profiles/p-anton/build-status" in page, "страница сама спрашивает статус"


def test_the_build_button_starts_one_detached_build(tmp_path, monkeypatch):
    started: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            started.append(command)

    monkeypatch.setattr("hub.web_admin.subprocess.Popen", FakeProcess)
    browser = _signed_in(tmp_path, monkeypatch)
    _seed_archive(tmp_path)
    _seed_training_archive(tmp_path)

    answer = browser.post("/admin/profiles/p-anton/build",
                          data={"source": "archive", "method": "photos", "limit": "120",
                                "step": "2", "device": "cuda"}, follow_redirects=False)
    assert answer.status_code == 303 and answer.headers["location"].startswith(
        "/admin/profiles/p-anton?")
    assert len(started) == 1
    command = started[0]
    assert command[1].endswith("build-3d-profile.py") and "--method" in command
    assert "photos" in command and "--limit" in command and "120" in command
    assert "--source" in command and "archive" in command
    assert "--device" in command and "cuda" in command, "где считать — выбор владельца"
    status = json.loads((tmp_path / "appearance" / "g-anton" / "model" / "status.json")
                        .read_text(encoding="utf-8"))
    assert status["state"] == "running" and status["total"] == 120
    assert status["options"]["source"] == "archive"
    assert status["options"]["device"] == "cuda", "статус говорит, на чём считалось"

    # A second click does not start a second builder.
    again = browser.post("/admin/profiles/p-anton/build", data={"source": "all"},
                         follow_redirects=False)
    assert again.status_code == 303 and len(started) == 1
    assert "already running" in browser.get(again.headers["location"]).text


def test_a_build_cannot_be_started_without_a_session(tmp_path, monkeypatch):
    started: list[list[str]] = []
    monkeypatch.setattr("hub.web_admin.subprocess.Popen",
                        lambda command, **kwargs: started.append(command))
    browser, _auth = a_client(tmp_path, monkeypatch)
    _seed_archive(tmp_path)

    answer = browser.post("/admin/profiles/p-anton/build", data={"source": "all"},
                          follow_redirects=False)
    assert answer.status_code == 303 and answer.headers["location"] == "/admin"
    assert started == []
    assert browser.get("/admin/profiles/p-anton/build-status").status_code == 403


def test_the_panel_switches_the_live_camera_view(tmp_path, monkeypatch):
    """Владелец 2026-09-24: «открыть камеру на пк buro» — кнопкой из панели.

    Та же комната, что у голосового инструмента ``camera_preview``: панель только
    передаёт выбор, а окно открывает сам комнатный ПК.
    """
    calls: list[tuple[str, bool]] = []

    async def preview(room, on):
        calls.append((room, on))
        return {'ok': True, 'label': room or 'livingroom'}

    monkeypatch.setenv("ROWAN_TEST_PASSWORD", PASSWORD)
    cfg = Config()
    cfg.server.web_admin.password_env = "ROWAN_TEST_PASSWORD"
    auth = WebAdminAuth(password_env="ROWAN_TEST_PASSWORD", secret=b"test-key")
    migrated(tmp_path).close()
    app = FastAPI()
    app.include_router(build_router(cfg=cfg, data=WebAdminData(tmp_path / "hub.db"),
                                    auth=auth, preview=preview))
    browser = TestClient(app, base_url="http://127.0.0.1", client=OVERLAY)
    browser.post("/admin/login", data={"password": PASSWORD}, follow_redirects=False)

    answer = browser.get("/admin/preview?room=buro&on=1")
    assert answer.status_code == 200 and "Done" in answer.text
    assert calls == [("buro", True)]
    off = browser.get("/admin/preview?room=buro&on=0")
    assert off.status_code == 200
    assert calls[-1] == ("buro", False)


def test_the_live_camera_switch_needs_a_session_and_a_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("ROWAN_TEST_PASSWORD", PASSWORD)
    cfg = Config()
    cfg.server.web_admin.password_env = "ROWAN_TEST_PASSWORD"
    auth = WebAdminAuth(password_env="ROWAN_TEST_PASSWORD", secret=b"test-key")
    migrated(tmp_path).close()
    app = FastAPI()
    app.include_router(build_router(cfg=cfg, data=WebAdminData(tmp_path / "hub.db"), auth=auth))
    browser = TestClient(app, base_url="http://127.0.0.1", client=OVERLAY)

    # Без сессии — на вход, а не «включено».
    assert browser.get("/admin/preview", follow_redirects=False).status_code == 303
    browser.post("/admin/login", data={"password": PASSWORD}, follow_redirects=False)
    assert browser.get("/admin/preview").status_code == 503
