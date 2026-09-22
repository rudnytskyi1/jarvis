"""P5-21 (F-608/F-407): «угадай, кто сказал» по голосам и таймер-соревнование."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config, GamesConfig
from hub import app as hub_app
from hub import guess_who as guess_mod
from hub import migrations_runner
from hub.homes import ensure_home
from hub.media import MediaStore
from hub.session import Session
from hub.skill_state import SkillStateStore
from hub.skills_registry import SkillRegistry
from hub.utterances import UtteranceMetrics
from hub.voice_clone import ConsentStore

REPO_ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
SPEAKER = "p-max"
SPEAKER_NAME = "Максим"


def _pcm(seconds: float = 1.0) -> bytes:
    return b"\x01\x02" * int(RATE * seconds)


def _env(tmp_path, *, window: float = 20.0, points: int = 5):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Гостиная", tz="UTC")
    ensure_home(conn, "office", name="Кабинет", tz="UTC")
    store = SkillStateStore(conn, skill="games")
    media = MediaStore(conn, tmp_path / "data", media_ttl_days=3, clip_ttl_days=7)
    consent = ConsentStore(tmp_path / "game_voice_consent.json", purpose="voice game")
    settings = GamesConfig(enabled=True, guess_window_s=window, guess_points=points)
    return conn, store, media, consent, settings


def _engine(store, consent, media, settings, *, clock=None, scheduler=None,
            home_name=lambda home: {"livingroom": "гостиная", "office": "кабинет"}.get(home, home)):
    return guess_mod.GuessEngine(store=store, consent=consent, settings=settings,
                                 scheduler=scheduler, media=media, clock=clock,
                                 home_name=home_name)


def _start(engine, *, home="livingroom", homes=("livingroom", "office"), seconds=1.0):
    return asyncio.run(engine.start(
        speaker_id=SPEAKER, speaker_name=SPEAKER_NAME, home_id=home, home_ids=list(homes),
        audio_pcm=_pcm(seconds), sample_rate=RATE,
        aliases=guess_mod.name_variants(f"{SPEAKER_NAME} Петров"), language="ru"))


# --- согласие ---------------------------------------------------------------


def test_a_round_needs_the_speaker_consent(tmp_path):
    conn, store, media, consent, settings = _env(tmp_path)
    try:
        engine = _engine(store, consent, media, settings)
        with pytest.raises(guess_mod.GuessUnavailable, match="has not allowed"):
            _start(engine)
        assert engine.active() is None, "без согласия партии нет"
        assert engine.snapshot()["refused"] == 1
        assert engine.grant_consent("livingroom", SPEAKER) is True
        round_, prompt = _start(engine)
        assert round_.speaker_name == SPEAKER_NAME
        assert "Чья это была фраза" in prompt and "5" in prompt
    finally:
        conn.close()


def test_consent_is_per_person_and_survives_a_restart(tmp_path):
    path = tmp_path / "game_voice_consent.json"
    consent = ConsentStore(path, purpose="voice game")
    assert consent.grant("livingroom", SPEAKER) is True
    again = ConsentStore(path, purpose="voice game")
    assert again.granted("livingroom", SPEAKER) is True
    assert again.granted("office", SPEAKER) is False, "согласие даёт человек в своей комнате"
    assert again.granted("livingroom", "p-other") is False
    assert again.revoke("livingroom", SPEAKER) is True
    assert ConsentStore(path, purpose="voice game").granted("livingroom", SPEAKER) is False


def test_the_clone_consent_does_not_open_the_voice_game(tmp_path):
    clone = ConsentStore(tmp_path / "voice_clone_consent.json", purpose="voice clone")
    game = ConsentStore(tmp_path / "game_voice_consent.json", purpose="voice game")
    clone.grant("livingroom", SPEAKER)
    assert clone.granted("livingroom", SPEAKER) is True
    assert game.granted("livingroom", SPEAKER) is False, "два разных согласия"


def test_consent_is_recognised_from_the_phrase(tmp_path):
    assert guess_mod.is_voice_consent("разрешаю использовать мой голос в игре") is True
    assert guess_mod.is_voice_consent("можно мой голос в игры?") is True
    assert guess_mod.is_voice_consent("you may use my voice in games") is True
    assert guess_mod.is_voice_consent("puedes usar mi voz en los juegos") is True
    assert guess_mod.is_voice_consent("включи игру") is False
    assert guess_mod.is_voice_consent("какой у меня голос?") is False


# --- запись -----------------------------------------------------------------


def test_the_mystery_is_a_real_recording_in_the_media_store(tmp_path):
    conn, store, media, consent, settings = _env(tmp_path)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings)
        round_, _ = _start(engine, seconds=2.0)
        row = conn.execute("SELECT path, kind FROM media WHERE media_ref=?",
                           (round_.audio_ref,)).fetchone()
        assert row is not None and row[1] == "audio", "запись лежит в медиах хаба"
        data = Path(row[0]).read_bytes()
        assert data.startswith(b"RIFF"), "это настоящий WAV, а не след в памяти"
        assert guess_mod.wav_seconds(data) == pytest.approx(2.0, abs=0.05)
        assert round_.audio_s == pytest.approx(2.0, abs=0.05)
    finally:
        conn.close()


def test_a_phrase_that_is_too_short_is_refused_and_a_huge_one_is_trimmed(tmp_path):
    conn, store, media, consent, settings = _env(tmp_path)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings)
        with pytest.raises(guess_mod.GuessUnavailable, match="too short"):
            _start(engine, seconds=0.1)
        round_, _ = _start(engine, seconds=guess_mod.MAX_PHRASE_S + 5)
        assert round_.truncated is True
        assert round_.audio_s == pytest.approx(guess_mod.MAX_PHRASE_S, abs=0.1)
    finally:
        conn.close()


def test_without_a_media_store_the_hub_refuses_instead_of_pretending(tmp_path):
    conn, store, _, consent, settings = _env(tmp_path)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, None, settings)
        with pytest.raises(guess_mod.GuessUnavailable, match="keep the recording"):
            _start(engine)
    finally:
        conn.close()


# --- таймер-соревнование ----------------------------------------------------


def test_the_fastest_room_scores_more_than_a_late_one(tmp_path):
    clock = {"now": 1000.0}
    conn, store, media, consent, settings = _env(tmp_path, window=20, points=5)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings, clock=lambda: clock["now"])
        _start(engine)
        clock["now"] = 1003.0
        fast = engine.guess(home_id="office", text="Максим", language="ru")
        assert fast is not None and fast.correct is True
        assert fast.points == 5, "самая быстрая комната забирает максимум"
        assert fast.scores == {"office": 5}
        assert "Максим" in fast.line
        engine.finish(language="ru")

        clock["now"] = 1000.0
        _start(engine)
        clock["now"] = 1015.0
        slow = engine.guess(home_id="office", text="Максим", language="ru")
        assert slow is not None and slow.correct is True
        assert slow.points == 2, "поздний ответ стоит меньше"
    finally:
        conn.close()


def test_speed_points_never_reach_zero(tmp_path):
    assert guess_mod.speed_points(20.0, 20.0, 5) == 5
    assert guess_mod.speed_points(10.0, 20.0, 5) == 3
    assert guess_mod.speed_points(0.5, 20.0, 5) == 1
    assert guess_mod.speed_points(-3.0, 20.0, 5) == 1
    assert guess_mod.speed_points(10.0, 0.0, 4) == 4, "без окна цена — полная"


def test_a_wrong_guess_does_not_score_and_the_round_goes_on(tmp_path):
    conn, store, media, consent, settings = _env(tmp_path)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings)
        _start(engine)
        wrong = engine.guess(home_id="office", text="это был не Максим", language="ru")
        assert wrong is not None and wrong.correct is False and wrong.scores == {}
        assert engine.active() is not None, "время ещё идёт"
        short = engine.guess(home_id="office", text="макс", language="ru")
        assert short.correct is False, "уменьшительное хаб не выдумывает"
        right = engine.guess(home_id="office", text="это Максим Петров", language="ru")
        assert right is not None and right.correct is True
    finally:
        conn.close()


def test_the_speaker_room_and_a_repeat_do_not_score_twice(tmp_path):
    conn, store, media, consent, settings = _env(tmp_path)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings)
        # Третья комната не отвечает: партия остаётся открытой после очка.
        _start(engine, homes=("livingroom", "office", "dorm-c"))
        spoiler = engine.guess(home_id="livingroom", text="Максим", language="ru")
        assert spoiler is not None and spoiler.spoiler is True and spoiler.scores == {}
        first = engine.guess(home_id="office", text="Максим", language="ru")
        assert first.correct is True and first.points >= 1
        again = engine.guess(home_id="office", text="Максим", language="ru")
        assert again is not None and again.already is True
        assert again.scores == first.scores, "второй раз очко не начисляют"
    finally:
        conn.close()


def test_a_late_guess_closes_the_round_and_names_the_answer(tmp_path):
    clock = {"now": 1000.0}
    conn, store, media, consent, settings = _env(tmp_path, window=20)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings, clock=lambda: clock["now"])
        _start(engine)
        clock["now"] = 1030.0
        verdict = engine.guess(home_id="office", text="Максим", language="ru")
        assert verdict is not None and verdict.late is True and verdict.finished is True
        assert SPEAKER_NAME in verdict.line and "вышло" in verdict.line.lower()
        assert engine.active() is None
        assert engine.snapshot()["timeouts"] == 1
    finally:
        conn.close()


def test_the_timer_closes_the_round_itself(tmp_path):
    async def scenario():
        conn, store, media, consent, settings = _env(tmp_path, window=20)
        timers: list[tuple[float, object, str]] = []

        class Scheduler:
            def in_(self, delay_s, callback, *, name=""):
                timers.append((delay_s, callback, name))
                return name

            def cancel(self, timer_id):
                return True

        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings, scheduler=Scheduler())
        await engine.start(speaker_id=SPEAKER, speaker_name=SPEAKER_NAME,
                           home_id="livingroom", home_ids=["livingroom", "office"],
                           audio_pcm=_pcm(), sample_rate=RATE, language="ru")
        assert timers and timers[0][0] == 20.0
        close = engine.close(reason="timeout")
        assert close is not None and close.round.speaker_name == SPEAKER_NAME
        assert SPEAKER_NAME in close.line
        assert engine.active() is None
        conn.close()

    asyncio.run(scenario())


def test_the_round_ends_when_every_room_has_guessed(tmp_path):
    conn, store, media, consent, settings = _env(tmp_path)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings)
        _start(engine)
        office = engine.guess(home_id="office", text="Максим", language="ru")
        # Единственная комната, которая могла угадать, угадала — партия закрыта,
        # а комната говорящего очка не получает вообще.
        assert office is not None and office.finished is True
        assert "Максим" in office.next_line
        assert engine.active() is None
    finally:
        conn.close()


def test_the_round_survives_a_restarted_engine(tmp_path):
    conn, store, media, consent, settings = _env(tmp_path)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings)
        round_, _ = _start(engine)
        other = _engine(SkillStateStore(conn, skill="games"), consent, media, settings)
        restored = other.active()
        assert restored is not None and restored.round_id == round_.round_id
        assert restored.audio_ref == round_.audio_ref
        guess = other.guess(home_id="office", text="Максим", language="ru")
        assert guess is not None and guess.correct is True
    finally:
        conn.close()


# --- разбор речи ------------------------------------------------------------


def test_only_a_request_to_play_opens_the_round():
    assert guess_mod.is_guess_start("давай сыграем в угадай, кто сказал") is True
    assert guess_mod.is_guess_start("угадай, кто это сказал") is True
    assert guess_mod.is_guess_start("let's play guess who said it") is True
    assert guess_mod.is_guess_start("adivina quién lo dijo") is True
    assert guess_mod.is_guess_start("кто это сказал?") is False, "просто вопрос"
    assert guess_mod.is_guess_start("включи свет") is False


def test_the_name_variants_are_the_full_name_and_the_first_word(tmp_path):
    assert guess_mod.name_variants("Максим Петров") == ["Максим Петров", "Максим"]
    assert guess_mod.name_variants("Максим") == ["Максим"]
    assert guess_mod.name_variants("   ") == []


def test_the_snapshot_names_the_round_and_the_consents(tmp_path):
    conn, store, media, consent, settings = _env(tmp_path)
    try:
        consent.grant("livingroom", SPEAKER)
        engine = _engine(store, consent, media, settings)
        empty = engine.snapshot()
        assert empty["active"] is False and empty["consents"] == 1
        _start(engine)
        snapshot = engine.snapshot()
        assert snapshot["active"] is True and snapshot["speaker"] == SPEAKER_NAME
        assert snapshot["rooms"] == ["livingroom", "office"]
    finally:
        conn.close()


# --- настоящий путь хода ----------------------------------------------------


def _registry():
    registry = SkillRegistry()
    registry.load_directory(REPO_ROOT / "skills")
    assert registry.errors == []
    return registry


def _hub(tmp_path, monkeypatch, *, enabled=True, speaker=SPEAKER_NAME):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Гостиная", tz="UTC")
    ensure_home(conn, "office", name="Кабинет", tz="UTC")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?, ?)",
                 (SPEAKER, SPEAKER_NAME))
    conn.commit()
    cfg = Config()
    cfg.server.games = GamesConfig(enabled=enabled, guess_window_s=20, guess_points=5)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_skills", _registry())
    monkeypatch.setattr(hub_app, "_game_consent",
                        ConsentStore(tmp_path / "game_voice_consent.json", purpose="voice game"))
    monkeypatch.setattr(hub_app, "_media",
                        MediaStore(conn, tmp_path / "data", media_ttl_days=3, clip_ttl_days=7))
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    connection = _connection(cfg, conn, home="livingroom", speaker=speaker)
    office = _connection(cfg, conn, home="office", speaker="Дрю")
    monkeypatch.setattr(hub_app, "_connections", [connection, office])
    return conn, connection, office


def _connection(cfg, conn, *, home, speaker):
    connection = hub_app.Connection(SimpleNamespace(client=None), cfg)
    connection.session = Session(client_id=f"pc-{home}", devices=[], history_turns=4)
    connection.home_id = home
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._speaker_name = speaker
    connection._speaker_role = "owner"
    connection.sample_rate = RATE
    connection.send_json = AsyncMock()
    connection.send_bytes = AsyncMock()
    connection._stream_tts = AsyncMock()
    connection._log_dialog = AsyncMock()
    return connection


def _said(connection) -> list[str]:
    return [str(call.args[0].get("text") or "") for call in connection.send_json.await_args_list]


def _played(connection) -> list[dict]:
    return [call.args[0] for call in connection.send_json.await_args_list
            if call.args[0].get("type") == "play_audio"]


def _consent(connection) -> None:
    asyncio.run(connection._guess_turn(
        "разрешаю использовать мой голос в игре", "ru", None, 100.0,
        connection.session, 40, b""))


def _arm(connection) -> None:
    asyncio.run(connection._guess_turn("давай сыграем в угадай, кто сказал", "ru", None,
                                       100.0, connection.session, 40, b""))


def _record(connection) -> bool:
    return asyncio.run(connection._guess_turn(
        "Сегодня отличная погода в общежитии", "ru", None, 100.0,
        connection.session, 40, _pcm()))


def test_the_turn_asks_for_a_phrase_then_records_it_and_the_other_room_hears_it(
        tmp_path, monkeypatch):
    conn, connection, office = _hub(tmp_path, monkeypatch)
    try:
        _consent(connection)
        assert any("разрешил" in text for text in _said(connection))
        connection.send_json.reset_mock()
        _arm(connection)
        said = " ".join(_said(connection))
        assert "Скажи короткую фразу" in said
        assert hub_app._guess_engine().active() is None, "фразы ещё нет"
        connection.send_json.reset_mock()
        assert _record(connection) is True
        assert any("Фраза записана" in text for text in _said(connection))
        round_ = hub_app._guess_engine().active()
        assert round_ is not None and round_.speaker_name == SPEAKER_NAME
        # Соседняя комната слышит и задание, и НАСТОЯЩУЮ запись голоса.
        headers = _played(office)
        assert headers and headers[0]["rate"] == RATE
        assert headers[0]["seconds"] == pytest.approx(1.0, abs=0.05)
        office.send_bytes.assert_awaited()
        sent = office.send_bytes.await_args.args[0]
        assert sent and len(sent) == len(_pcm())
        assert any("Чья это была фраза" in text for text in _said(office))
        assert _said(connection), "в комнате говорящего тоже отвечают"
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_without_consent_the_turn_refuses_and_starts_nothing(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        _arm(connection)
        connection.send_json.reset_mock()
        assert _record(connection) is True
        said = " ".join(_said(connection)).casefold()
        assert "разреш" in said
        assert hub_app._guess_engine().active() is None
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_an_unrecognised_speaker_cannot_be_the_mystery(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch, speaker="")
    try:
        _arm(connection)
        connection.send_json.reset_mock()
        assert _record(connection) is True
        assert any("узнать твой голос" in text for text in _said(connection))
        assert hub_app._guess_engine().active() is None
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_a_guess_from_another_room_scores_that_room_and_is_heard_everywhere(
        tmp_path, monkeypatch):
    conn, connection, office = _hub(tmp_path, monkeypatch)
    try:
        _consent(connection)
        _arm(connection)
        _record(connection)
        connection.send_json.reset_mock()
        office.send_json.reset_mock()
        handled = asyncio.run(office._guess_turn("Это Максим", "ru", None, 200.0,
                                                 office.session, 40, b""))
        assert handled is True
        assert any("Верно" in text and "Кабинет" in text for text in _said(office))
        round_ = hub_app._guess_engine().active()
        assert round_ is None, "единственная угадывающая комната закрыла партию"
        assert any("Верно" in text for text in _said(connection)), "счёт слышат все"
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_the_stop_word_and_the_score_question_answer_the_room(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        _consent(connection)
        _arm(connection)
        _record(connection)
        connection.send_json.reset_mock()
        assert asyncio.run(connection._guess_turn("какой счёт?", "ru", None, 200.0,
                                                  connection.session, 40, b"")) is True
        assert any("Счёт" in text or "угадал" in text for text in _said(connection))
        connection.send_json.reset_mock()
        assert asyncio.run(connection._guess_turn("хватит", "ru", None, 300.0,
                                                  connection.session, 40, b"")) is True
        said = " ".join(_said(connection)).casefold()
        assert "закончена" in said and "был" in said
        assert hub_app._guess_engine().active() is None
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_the_flag_off_and_ordinary_phrases_leave_the_turn_alone(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch, enabled=False)
    try:
        assert asyncio.run(connection._guess_turn(
            "давай сыграем в угадай, кто сказал", "ru", None, 100.0,
            connection.session, 40, b"")) is False
        connection.send_json.assert_not_awaited()
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()
    second = tmp_path / "second"
    second.mkdir(parents=True, exist_ok=True)
    conn2, other, _ = _hub(second, monkeypatch)
    try:
        assert asyncio.run(other._guess_turn(
            "включи свет в комнате", "ru", None, 100.0, other.session, 40, b"")) is False
        other.send_json.assert_not_awaited()
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn2.close()


def test_the_health_snapshot_names_the_voice_game(tmp_path, monkeypatch):
    conn, connection, _ = _hub(tmp_path, monkeypatch)
    try:
        _consent(connection)
        _arm(connection)
        _record(connection)
        snapshot = hub_app._games_snapshot()
        assert snapshot["enabled"] is True
        assert snapshot["guess"]["active"] is True
        assert snapshot["guess"]["speaker"] == SPEAKER_NAME
        assert snapshot["guess"]["consents"] == 1
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()
