"""Приветствия и прощания по событию входа (ТЗ F-302).

Three requirements have to hold together: the hello is PERSONAL (a name, no
guessing), it follows the TIME OF DAY and the room's QUIET HOURS, and it comes
from the TTS cache so it lands while the person is still in front of the
camera. The farewell is the same rule read on the `person_left` event of
F-301. The last requirement — "no more often than once per 20 minutes per
person" — is checked on the real gate the hub uses.
"""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from common.config import Config
from hub import app as hub_app
from hub import greetings as greet
from hub import migrations_runner
from hub.auth import ClientTokenStore
from hub.gateway import Gateway
from hub.greetings import (
    COOLDOWN_S,
    DAY,
    EVENING,
    MORNING,
    NIGHT,
    farewell,
    greeting,
    in_quiet_hours,
    part_of_day,
    stranger_greeting,
    window,
)
from hub.presence_state import KIND_LEFT, PresenceLog, PresenceState
from hub.room_state import RoomState
from hub.tts import TtsCache

# --- время суток и тихие часы ------------------------------------------------


def test_the_part_of_the_day_follows_the_local_clock():
    at = lambda hour: datetime(2026, 9, 21, hour, 30).timestamp()  # noqa: E731
    assert part_of_day(at(7)) == MORNING
    assert part_of_day(at(13)) == DAY
    assert part_of_day(at(19)) == EVENING
    assert part_of_day(at(2)) == NIGHT
    assert part_of_day(at(23)) == NIGHT
    assert part_of_day(at(5)) == MORNING and part_of_day(at(12)) == DAY
    assert part_of_day(at(18)) == EVENING


def test_the_greeting_is_personal_and_follows_the_time_of_day():
    at = lambda hour: datetime(2026, 9, 21, hour, 0).timestamp()  # noqa: E731
    assert greeting("Макс", "ru", moment=at(7)) == "Доброе утро, Макс!"
    assert greeting("Макс", "ru", moment=at(13), variant=1) == "Здравствуй, Макс!"
    assert greeting("Max", "en", moment=at(19)) == "Good evening, Max!"
    assert greeting("Max", "en", moment=at(2)) == "Good night, Max!"
    assert greeting("Ana", "es", moment=at(13)).startswith("¡Buenas tardes")
    assert greeting("", "ru", moment=at(7)) == "Доброе утро, ?!"
    assert greet.language_of("DE") == "ru" and greet.language_of("en-US") == "en"


def test_a_stranger_is_introduced_and_never_named():
    line = stranger_greeting("ru")
    assert "Rowan" in line and "Макс" not in line
    with_company = stranger_greeting("en", company=["Anton"])
    assert "Anton" in with_company and "Rowan" in with_company
    assert stranger_greeting("es").startswith("¡Hola!")
    assert stranger_greeting("ru", variant=1) != stranger_greeting("ru", variant=0)


def test_the_farewell_follows_the_time_of_day_too():
    at = lambda hour: datetime(2026, 9, 21, hour, 0).timestamp()  # noqa: E731
    assert farewell("Макс", "ru", moment=at(20)) == "Пока, Макс!"
    assert farewell("Макс", "ru", moment=at(20), variant=1) == "Хорошего вечера, Макс!"
    assert farewell("Max", "en", moment=at(2), variant=1) == "Good night, Max!"
    assert farewell("Ana", "es", moment=at(13), variant=1) == "¡Buen día, Ana!"


def test_quiet_hours_cover_the_night_and_can_cross_midnight():
    at = lambda hour: datetime(2026, 9, 21, hour, 15).timestamp()  # noqa: E731
    assert in_quiet_hours("23:00", "08:00", moment=at(2)) is True
    assert in_quiet_hours("23:00", "08:00", moment=at(23)) is True
    assert in_quiet_hours("23:00", "08:00", moment=at(12)) is False
    assert in_quiet_hours("13:00", "15:00", moment=at(14)) is True
    assert in_quiet_hours("13:00", "15:00", moment=at(16)) is False
    assert in_quiet_hours("", "", moment=at(2)) is False, "пустое окно = прежнее поведение"
    assert in_quiet_hours("25:00", "08:00", moment=at(2)) is False
    assert window("23:00", "08:00") == ("23:00", "08:00")
    assert window("", "") is None and window("23:00", "23:00") is None


def test_the_time_zone_of_the_home_moves_the_clock():
    moment = datetime(2026, 9, 21, 2, 0, tzinfo=UTC).timestamp()
    # 02:00 по Гринвичу — это 04:00 в Берлине и 19:00 в Лос-Анджелесе.
    assert part_of_day(moment, tz="UTC") == NIGHT
    assert part_of_day(moment, tz="Europe/Berlin") == NIGHT
    assert part_of_day(moment, tz="America/Los_Angeles") == EVENING
    assert part_of_day(moment, tz="Nowhere/Nothing") == part_of_day(moment), "чужой tz = местное"
    assert in_quiet_hours("03:00", "05:00", moment=moment, tz="Europe/Berlin") is True
    assert in_quiet_hours("03:00", "05:00", moment=moment, tz="UTC") is False


# --- кэш TTS -----------------------------------------------------------------


def test_the_tts_cache_remembers_short_lines_and_forgets_the_long_ones():
    cache = TtsCache(max_items=2, max_chars=10, max_bytes=4)
    assert cache.get("Пока, Макс!", 48000) is None and cache.misses == 1
    assert cache.put("Пока", 48000, b"ab") is True
    assert cache.get("Пока", 48000) == b"ab" and cache.hits == 1
    assert cache.get("Пока", 24000) is None, "другая частота — другой ключ"
    assert cache.put("Это слишком длинная строка", 48000, b"ab") is False
    assert cache.put("Кто дома?", 48000, b"") is False
    cache.put("Привет", 48000, b"cd")
    cache.put("Здравствуй", 48000, b"ef")
    assert cache.stats()["items"] == 2, "старое вытесняется по числу записей"
    cache.clear()
    assert cache.stats()["items"] == 0 and cache.stats()["bytes"] == 0


def test_the_cache_is_bounded_by_bytes_too():
    cache = TtsCache(max_items=10, max_chars=100, max_bytes=6)
    assert cache.put("один", 48000, b"1234") is True
    assert cache.put("два", 48000, b"5678") is True
    assert cache.stats()["bytes"] <= 6 and cache.get("один", 48000) is None


# --- порог «не чаще 20 минут на человека» -------------------------------------


def _conn(cfg: Config, *, known_gap_s: float = 900.0, greeting: Any = None) -> Any:
    """The part of a Connection the greeting gate reads (реальный cfg)."""
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.presence = hub_app.PresenceTracker(ttl_s=30.0)
    conn._last_seen_at = {}
    conn._due_greeting = set()
    conn._last_known_voice_at = 0.0
    conn._last_greeting_at = 0.0
    conn._last_audio_at = 0.0
    conn._quiet_until_wake = False
    conn.face_enabled = True
    conn.session = None
    conn.camera_state = {"persons": 1}
    cfg.server.face.greet_after_s = 10.0
    cfg.server.face.greeting_cooldown_s = 300.0
    cfg.server.face.greeting_cooldown_known_s = known_gap_s
    # Флаг ТЗ F-302: выключенный server.greeting = поведение фазы 2.
    cfg.server.greeting.enabled = bool(greeting)
    conn.cfg = cfg
    return conn


def test_the_twenty_minute_rule_of_the_spec_is_the_known_person_gap():
    cfg = Config()
    assert cfg.server.greeting.cooldown_s == COOLDOWN_S == 1200.0
    assert _conn(cfg)._greet_config() == (10.0, 300.0, 900.0), "флаг выключен = фаза 2"
    enabled = _conn(cfg, greeting=True)
    assert enabled._greet_config()[2] == 1200.0, "ТЗ F-302: 20 минут на человека"
    strict = _conn(cfg, known_gap_s=3600.0, greeting=True)
    assert strict._greet_config()[2] == 3600.0, "своё число владельца не уменьшается"


def test_a_room_in_its_quiet_hours_says_nothing(tmp_path, monkeypatch):
    cfg = Config()
    hour = datetime.now().hour
    start, end = (f"{hour:02d}:00", f"{(hour + 1) % 24:02d}:00")
    cfg.server.greeting.enabled = True
    cfg.server.greeting.quiet_start = start
    cfg.server.greeting.quiet_end = end
    monkeypatch.setattr(hub_app, "_face", SimpleNamespace(available=True))
    monkeypatch.setattr(hub_app, "_llm", object())
    monkeypatch.setattr(hub_app, "_tts", object())
    conn = _conn(cfg, greeting=True)
    conn.session = object()
    conn.receiving = False
    conn._task = None
    reasons: list[str] = []
    conn._greet_blocked = lambda key, reason: reasons.append(key) and None
    conn._greet_target = lambda *args: "Anton"
    assert conn._may_greet(10, 300, 900) is None
    assert reasons == ["quiet_hours"], "тихие часы дома важнее приветствия"
    assert conn._in_quiet_hours() is True


# --- приветствие, прощание и кэш в самом хабе --------------------------------


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name, tz) VALUES ('livingroom', 'Living', 'UTC')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role)"
                 " VALUES ('p-max', 'livingroom', 'admin')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _connection(hub_db, monkeypatch, *, cfg: Config | None = None, state=None,
                spoken: list[str] | None = None):
    config = cfg or Config()
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_gateway", Gateway(ClientTokenStore(hub_db)))
    monkeypatch.setattr(hub_app, "_presence", state if state is not None else PresenceState())
    monkeypatch.setattr(hub_app, "_presence_events", PresenceLog(hub_db))
    monkeypatch.setattr(hub_app, "_tts_cache", TtsCache())
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(
        sample_rate=48000, synth=lambda part: b"\0\1" * 4))
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = object()
    connection.cfg = config
    connection.receiving = False
    connection._task = None
    connection._presence_has_tracks = True
    connection._track_zones = {}
    connection._due_greeting = set()
    connection._last_seen_at = {}
    connection._last_known_voice_at = 0.0
    connection._last_greeting_at = 0.0
    connection._last_audio_at = 0.0
    connection._quiet_until_wake = False
    connection._farewell_at = {}
    connection._farewell_tasks = set()
    connection._greeting_variant = 0
    connection.face_enabled = True
    connection.presence = hub_app.PresenceTracker(ttl_s=30.0)
    connection.camera_state = {"persons": 1, "objects": {}, "ts": time.time()}
    room = RoomState()
    room.update([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}], now=time.monotonic())
    room.tracks["a:1"]["name"] = "Макс"
    connection.room = room
    connection._speaker_name = "Макс"
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection._speaker_language = "ru"
    connection._reply_language = "ru"
    connection._reply_lock = asyncio.Lock()
    connection._audio_lock = asyncio.Lock()
    connection._first_audio = None
    connection._recording_turn = None
    connection.send_json = _recorder()
    connection.send_bytes = _recorder()
    connection._stream_tts_calls = spoken if spoken is not None else []
    return connection


def _recorder():
    calls: list[Any] = []

    async def record(payload):
        calls.append(payload)

    record.calls = calls  # type: ignore[attr-defined]
    return record


def test_the_personal_greeting_follows_the_time_of_day_in_the_rooms_zone(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    text = connection._scripted_greeting("Макс")
    assert text == greet.greeting("Макс", "ru", moment=time.time(), tz="UTC")
    assert "Макс" in text
    connection.presence.note_faces(["Макс"])
    stranger = connection._scripted_greeting(hub_app.LABEL_UNKNOWN)
    assert "Rowan" in stranger and "Макс" in stranger, "незнакомцу называют тех, кто рядом"
    connection._speaker_language = "en"
    assert connection._scripted_greeting("Макс") == greet.greeting(
        "Макс", "en", moment=time.time(), tz="UTC", variant=2)
    assert connection._home_timezone() == "UTC"
    assert connection._greeting_language() == "en"


def test_two_greetings_of_the_same_face_come_from_the_cache(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    synthesised: list[str] = []

    async def scenario():
        voice = SimpleNamespace(sample_rate=48000,
                                synth=lambda part: synthesised.append(part) or b"\0\1" * 4)
        text = connection._scripted_greeting("Макс")
        await connection._stream_tts(voice, text, cache=True)
        await connection._stream_tts(voice, text, cache=True)

    asyncio.run(scenario())
    assert len(synthesised) == 1, "второе приветствие взято из кэша"
    assert synthesised[0] == greet.greeting("Макс", "ru", moment=time.time(), tz="UTC")
    assert hub_app._tts_cache.stats()["hits"] == 1


def test_a_reply_is_never_taken_from_the_cache(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    synthesised: list[str] = []

    async def scenario():
        voice = SimpleNamespace(sample_rate=48000,
                                synth=lambda part: synthesised.append(part) or b"\0\1" * 4)
        await connection._stream_tts(voice, "Готово.")
        await connection._stream_tts(voice, "Готово.")

    asyncio.run(scenario())
    assert synthesised == ["Готово.", "Готово."], "ответы модели не кэшируются"
    assert hub_app._tts_cache.stats()["items"] == 0


def test_a_person_who_left_is_seen_off_by_name(hub_db, monkeypatch):
    clock = [time.time()]
    state = PresenceState(absence_s=1.0, clock=lambda: clock[0])
    connection = _connection(hub_db, monkeypatch, state=state)
    spoken: list[str] = []

    async def scenario():
        waited = asyncio.Event()

        async def send_json(payload):  # pragma: no cover - only the TTS path is read
            waited.set()

        async def send_bytes(data):  # pragma: no cover - the goodbye is short
            waited.set()

        connection.send_json = send_json
        connection.send_bytes = send_bytes
        connection._stream_tts = _fake_tts(spoken)
        connection._observe_presence()
        clock[0] += 5.0
        connection.room.tracks.clear()
        connection._observe_presence()
        for _ in range(50):
            if spoken:
                break
            await asyncio.sleep(0)

    asyncio.run(scenario())
    assert spoken == [greet.farewell("Макс", "ru", moment=time.time(), tz="UTC")]
    rows = hub_db.execute("SELECT kind FROM presence_events ORDER BY rowid").fetchall()
    assert KIND_LEFT in [row[0] for row in rows], "прощание идёт по событию F-301"


def _fake_tts(spoken: list[str]):
    async def say(voice, text, purpose='reply', notice_id='', cache=False):
        spoken.append(text)

    return say


def test_the_same_person_is_not_seen_off_twice_in_twenty_minutes(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    assert connection._may_say_bye("p-max") is True
    assert connection._may_say_bye("p-max") is False
    assert connection._may_say_bye("p-max", cooldown_s=0.0) is True
    connection._farewell_at["p-max"] -= COOLDOWN_S + 1.0
    assert connection._may_say_bye("p-max") is True


def test_quiet_hours_stop_the_goodbye_too(hub_db, monkeypatch):
    cfg = Config()
    # Дом живёт в UTC (см. hub_db), поэтому окно тишины считаем по его часам.
    hour = datetime.now(UTC).hour
    cfg.server.greeting.quiet_start = f"{hour:02d}:00"
    cfg.server.greeting.quiet_end = f"{(hour + 1) % 24:02d}:00"
    connection = _connection(hub_db, monkeypatch, cfg=cfg)
    connection._schedule_farewell("p-max")
    assert not connection._farewell_tasks
    assert connection._in_quiet_hours() is True


def test_a_farewell_can_be_switched_off(hub_db, monkeypatch):
    cfg = Config()
    cfg.server.greeting.farewell = False
    connection = _connection(hub_db, monkeypatch, cfg=cfg)
    connection._schedule_farewell("p-max")
    assert not connection._farewell_tasks


def test_a_person_the_hub_cannot_name_gets_no_goodbye(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection._schedule_farewell("p-nobody")
    assert not connection._farewell_tasks
    assert connection._display_name("p-max") == "Макс"
    assert connection._display_name("p-nobody") == ""


def test_the_health_endpoint_publishes_the_cache(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(available=True, sample_rate=48000))
    hub_app._tts_cache.put("Пока, Макс!", 48000, b"\0\1" * 8)
    health = asyncio.run(hub_app.health())
    assert health["tts_cache"]["items"] == 1
    assert health["tts_cache"]["bytes"] == 16
    assert connection is not None
