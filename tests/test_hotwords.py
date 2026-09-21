"""Hotwords follow the database, not one admin form (ТЗ F-104)."""
from __future__ import annotations

from types import SimpleNamespace

from common.config import Config
from hub import migrations_runner
from hub.admin_backend import AdminBackend
from hub.devices import Device, DeviceStore
from hub.hotwords import APP_WORDS, collect, engine_text, new_words, sync
from hub.scenes import Scene, SceneStore, Step

HOME = "livingroom"


def _db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES (?, ?)", (HOME, "Living room"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-1', 'Anton')")
    conn.commit()
    return conn


def _device(store):
    store.save(Device(id="lr-lamp", home_id=HOME, name="Ceiling lamp",
                      aliases=["люстра", "lamp"], kind="light", capabilities=["on_off"],
                      adapter="mqtt", adapter_config={"switch": 1}))


def test_names_of_people_devices_and_scenes_become_hotwords(tmp_path):
    conn = _db(tmp_path)
    try:
        devices = DeviceStore(conn)
        _device(devices)
        scenes = SceneStore(conn)
        scenes.save(Scene(scene_id="lr-evening", home_id=HOME, name="Evening",
                          aliases=["вечер"], steps=[Step(kind="say", text="Hi")]))
        words = collect(conn, homes=[HOME])
        assert "Anton" in words
        assert "Ceiling lamp" in words and "люстра" in words and "lamp" in words
        assert "Evening" in words and "вечер" in words
        assert set(APP_WORDS) <= set(words)
    finally:
        conn.close()


def test_the_configured_words_come_first_and_are_not_dropped(tmp_path):
    conn = _db(tmp_path)
    try:
        words = collect(conn, static=["Rowan", "Антон"], homes=[HOME])
        assert words[:2] == ["Rowan", "Антон"]
    finally:
        conn.close()


def test_a_word_is_never_listed_twice_whatever_its_case(tmp_path):
    conn = _db(tmp_path)
    try:
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-2', 'ANTON')")
        conn.commit()
        words = collect(conn, static=["anton"], homes=[HOME])
        assert [word.casefold() for word in words].count("anton") == 1
    finally:
        conn.close()


def test_the_list_respects_the_configured_ceiling(tmp_path):
    conn = _db(tmp_path)
    try:
        store = DeviceStore(conn)
        for index in range(60):
            store.save(Device(id=f"d{index}", home_id=HOME, name=f"Device {index}",
                              kind="light", capabilities=["on_off"], adapter="mqtt",
                              adapter_config={"switch": index}))
        words = collect(conn, static=["Rowan"], homes=[HOME], limit=32)
        assert len(words) == 32
        assert words[0] == "Rowan", "the owner's own word survives the cap"
    finally:
        conn.close()


def test_only_the_rooms_that_were_asked_for_contribute(tmp_path):
    conn = _db(tmp_path)
    try:
        conn.execute("INSERT INTO homes(home_id, name) VALUES ('dorm-max', 'Max')")
        conn.commit()
        stores = DeviceStore(conn)
        stores.save(Device(id="lr-lamp", home_id=HOME, name="Ceiling lamp", kind="light",
                           capabilities=["on_off"], adapter="mqtt", adapter_config={}))
        stores.save(Device(id="dm-tv", home_id="dorm-max", name="Max TV", kind="tv",
                           capabilities=["on_off"], adapter="mqtt", adapter_config={}))
        words = collect(conn, homes=[HOME])
        assert "Ceiling lamp" in words and "Max TV" not in words
        both = collect(conn, homes=[HOME, "dorm-max"])
        assert "Max TV" in both
    finally:
        conn.close()


def test_syncing_hands_the_engine_one_string_and_reports_new_words(tmp_path):
    conn = _db(tmp_path)
    try:
        _device(DeviceStore(conn))
        engine = SimpleNamespace(hotwords="Rowan")
        cfg = Config()
        cfg.server.stt.hotwords = ["Rowan"]
        added = sync(cfg, conn, engine, homes=[HOME])
        assert "Ceiling lamp" in added and "Anton" in added
        assert engine.hotwords.startswith("Rowan, ")
        assert "Ceiling lamp" in engine.hotwords and ", " in engine.hotwords
        # A second sync finds nothing new and does not rewrite the string.
        assert sync(cfg, conn, engine, homes=[HOME]) == []
        assert engine.hotwords == engine_text(collect(conn, static=["Rowan"], homes=[HOME]))
    finally:
        conn.close()


def test_syncing_without_an_engine_is_harmless(tmp_path):
    conn = _db(tmp_path)
    try:
        assert sync(Config(), conn, None, homes=[HOME]) == []
    finally:
        conn.close()


def test_new_words_ignores_case_and_order():
    assert new_words(["Rowan", "Anton"], ["rowan", "Ceiling lamp"]) == ["Ceiling lamp"]
    assert new_words(["a"], ["a", "A"]) == [], "case alone is not a new word"
    assert new_words([], ["a", "A"]) == ["a", "A"], "collect() is what deduplicates"


def test_the_panel_refreshes_the_recogniser_from_the_database(tmp_path):
    """The panel's own helper is what the device and scene actions call."""
    conn = _db(tmp_path)
    try:
        _device(DeviceStore(conn))
        cfg = Config()
        cfg.server.stt.hotwords = ["Rowan"]
        engine = SimpleNamespace(hotwords="Rowan")
        backend = AdminBackend(
            cfg, SimpleNamespace(),
            runtime=lambda: {"hub_conn": conn, "stt": engine},
            get_room=lambda: None, get_alerts=lambda: None,
        )
        added = backend._sync_hotwords()
        assert "Ceiling lamp" in added
        assert "люстра" in engine.hotwords
        # No database, no crash: the panel stays usable.
        backend.runtime = lambda: {"stt": engine}
        assert backend._sync_hotwords() == []
    finally:
        conn.close()
