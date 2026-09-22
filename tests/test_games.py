"""P5-20 (F-608/F-407): квиз между комнатами на каркасе скилла с состоянием."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config, GamesConfig
from hub import app as hub_app
from hub import games as games_mod
from hub import migrations_runner
from hub.homes import ensure_home
from hub.session import Session
from hub.skill_state import SkillStateStore
from hub.skills_registry import SkillRegistry
from hub.utterances import UtteranceMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- состояние скилла как хранилище партии ---------------------------------


def _store(tmp_path, *, skill="games"):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    return conn, SkillStateStore(conn, skill=skill)


class FakeQuiz:
    """Подставная модель квиза: отдаёт ровно те вопросы, что задал тест."""

    def __init__(self, questions=None):
        self.manual = questions or [
            games_mod.QuizQuestion(prompt="Кто написал «Евгения Онегина»?",
                                   answer="Пушкин", aliases=["Александр Пушкин"]),
            games_mod.QuizQuestion(prompt="В каком году человек полетел в космос?",
                                   answer="1961"),
        ]
        self.calls: list[tuple] = []

    async def questions(self, topic, *, count=5, language="ru", avoid=()):
        self.calls.append((topic, count, language, tuple(avoid)))
        return list(self.manual)


def _engine(tmp_path, *, generator=None, settings=None, scheduler=None, clock=None):
    conn, store = _store(tmp_path)
    engine = games_mod.QuizEngine(
        store=store, generator=generator or FakeQuiz(),
        settings=settings or GamesConfig(enabled=True, max_questions=2, answer_window_s=30),
        scheduler=scheduler, clock=clock or (lambda: 1000.0),
        home_name=lambda home: {"livingroom": "гостиная"}.get(home, home))
    return conn, engine


def test_a_new_engine_sees_the_round_the_old_one_started(tmp_path):
    conn, engine = _engine(tmp_path)
    try:
        round_, line = asyncio.run(engine.start(topic="история", language="ru",
                                                home_id="livingroom",
                                                homes=["livingroom", "office"]))
        assert "история" in line and "Кто написал" in line
        assert round_.home_ids == ["livingroom", "office"]
        # Партия в состоянии скилла, а не в памяти движка.
        other = games_mod.QuizEngine(store=SkillStateStore(conn, skill="games"),
                                     generator=FakeQuiz(), settings=GamesConfig(enabled=True))
        restored = other.active()
        assert restored is not None and restored.topic == "история"
        assert restored.current().prompt == round_.current().prompt
    finally:
        conn.close()


def test_a_correct_answer_scores_that_room_and_asks_the_next_question(tmp_path):
    conn, engine = _engine(tmp_path)
    try:
        asyncio.run(engine.start(topic="история", language="ru", home_id="livingroom"))
        verdict = engine.answer(home_id="livingroom", text="Пушкин", language="ru")
        assert verdict is not None and verdict.correct is True and verdict.late is False
        assert verdict.scores == {"livingroom": 1}
        assert "гостиная" in verdict.line and "1" in verdict.line
        assert "космос" in verdict.next_line
    finally:
        conn.close()


def test_a_wrong_answer_does_not_score_and_does_not_move_on(tmp_path):
    conn, engine = _engine(tmp_path)
    try:
        asyncio.run(engine.start(topic="история", language="ru", home_id="livingroom"))
        verdict = engine.answer(home_id="livingroom", text="Толстой", language="ru")
        assert verdict is not None and verdict.correct is False
        assert verdict.scores == {} and verdict.next_line == ""
        assert engine.active().index == 0, "вопрос остаётся тем же"
    finally:
        conn.close()


def test_a_late_answer_is_not_scored_and_names_the_right_answer(tmp_path):
    clock = {"now": 1000.0}
    conn, engine = _engine(tmp_path, clock=lambda: clock["now"])
    try:
        asyncio.run(engine.start(topic="история", language="ru", home_id="livingroom"))
        clock["now"] = 1000.0 + 31.0
        verdict = engine.answer(home_id="livingroom", text="Пушкин", language="ru")
        assert verdict is not None and verdict.late is True and verdict.correct is False
        assert verdict.scores == {} and "Пушкин" in verdict.line
        assert "космос" in verdict.next_line
    finally:
        conn.close()


def test_the_last_question_ends_the_game_with_the_winner(tmp_path):
    conn, engine = _engine(tmp_path)
    try:
        asyncio.run(engine.start(topic="история", language="ru", home_id="livingroom"))
        engine.answer(home_id="livingroom", text="Пушкин", language="ru")
        engine.answer(home_id="office", text="1961", language="ru")
        assert engine.active() is None, "партия закрыта"
        last = SkillStateStore(conn, skill="games").get(games_mod.STATE_LAST)
        assert last["scores"] == {"livingroom": 1, "office": 1}
    finally:
        conn.close()


def test_finishing_by_voice_says_the_score_and_clears_the_round(tmp_path):
    conn, engine = _engine(tmp_path)
    try:
        asyncio.run(engine.start(topic="история", language="ru", home_id="livingroom"))
        engine.answer(home_id="livingroom", text="Пушкин", language="ru")
        line = engine.finish(language="ru", reason="stop")
        assert line is not None and "гостиная" in line and "закончена" in line.lower()
        assert engine.active() is None
        assert engine.finish(language="ru") is None
    finally:
        conn.close()


def test_the_next_game_avoids_the_questions_of_the_previous_one(tmp_path):
    generator = FakeQuiz()
    conn, engine = _engine(tmp_path, generator=generator)
    try:
        asyncio.run(engine.start(topic="история", language="ru", home_id="livingroom"))
        engine.finish(language="ru")
        asyncio.run(engine.start(topic="история", language="ru", home_id="livingroom"))
        assert generator.calls[-1][3] == ("Кто написал «Евгения Онегина»?",
                                          "В каком году человек полетел в космос?")
    finally:
        conn.close()


def test_a_second_round_does_not_replace_the_running_one(tmp_path):
    conn, engine = _engine(tmp_path)
    try:
        asyncio.run(engine.start(topic="история", language="ru", home_id="livingroom"))
        with pytest.raises(games_mod.QuizError, match="already running"):
            asyncio.run(engine.start(topic="кино", language="ru", home_id="office"))
    finally:
        conn.close()


def test_a_timer_closes_the_question_and_rearms_the_next_one(tmp_path):
    async def scenario():
        conn = migrations_runner.connect(str(tmp_path / "hub.db"))
        migrations_runner.migrate(conn)
        timers: list[tuple[float, object, str]] = []

        class Scheduler:
            def in_(self, delay_s, callback, *, name=""):
                timers.append((delay_s, callback, name))
                return name

            def cancel(self, timer_id):
                return True

        engine = games_mod.QuizEngine(
            store=SkillStateStore(conn, skill="games"), generator=FakeQuiz(),
            settings=GamesConfig(enabled=True, max_questions=2, answer_window_s=30.0),
            scheduler=Scheduler(), clock=lambda: 1000.0)
        await engine.start(topic="история", language="ru", home_id="livingroom")
        assert timers[0][0] == 30.0
        close = engine.close_question(reason="timeout")
        assert close is not None and close.expected == "Пушкин"
        assert "Поздно" in close.line and "космос" in close.next_line
        assert len(timers) == 2, "таймер поднят для следующего вопроса"
        conn.close()

    asyncio.run(scenario())


# --- разбор речи ------------------------------------------------------------


def test_only_a_request_to_play_starts_a_quiz():
    assert games_mod.is_quiz_start("давайте сыграем в квиз по истории") is True
    assert games_mod.is_quiz_start("сыграем в викторину") is True
    assert games_mod.is_quiz_start("let's play a quiz about movies") is True
    # Разговор о квизе — не просьба начать.
    assert games_mod.is_quiz_start("вчера был хороший квиз") is False
    assert games_mod.is_quiz_start("какая сегодня погода") is False


def test_the_topic_comes_from_the_phrase_or_the_config():
    assert games_mod.quiz_topic("сыграем в квиз по теме история") == "история"
    assert games_mod.quiz_topic("давай квиз про фильмы 90-х") == "фильмы 90-х"
    assert games_mod.quiz_topic("сыграем в квиз", ["история", "наука"]) == ""
    assert games_mod.quiz_topic("сыграем в квиз, тема наука", ["наука"]) == "наука"
    assert "тему" in games_mod.topic_missing_line(language="ru").lower()


def test_the_stop_and_score_phrases_are_recognised():
    assert games_mod.is_stop("хватит, закончим") is True
    assert games_mod.is_stop("stop the game") is True
    assert games_mod.is_stop("какой хороший день") is False
    assert games_mod.is_score_request("какой счёт?") is True
    assert games_mod.is_score_request("покажи очки") is True
    assert games_mod.is_score_request("включи свет") is False


def test_answers_match_by_word_boundaries_not_by_substring():
    assert games_mod.answers_match("пушкин", "Пушкин") is True
    assert games_mod.answers_match("Александр Пушкин", "Пушкин") is True
    assert games_mod.answers_match("пушкин", "Александр Пушкин") is True, "часть ответа"
    assert games_mod.answers_match("one thousand nine hundred sixty one",
                                   "1961", ["1961 год"]) is False, "числа словами не угадываем"
    assert games_mod.answers_match("это был не толстой", "Толстой") is False
    assert games_mod.answers_match("это толстой", "Толстой") is True
    assert games_mod.answers_match("strip", "trip") is False, "подстрока — не ответ"
    assert games_mod.answers_match("", "Пушкин") is False


def test_the_model_answer_is_parsed_strictly():
    payload = {"questions": [
        {"question": "Столица Франции?", "answer": "Париж", "aliases": ["Paris"]},
        {"question": "", "answer": "мусор", "aliases": []},
        {"question": "2+2?", "answer": "4"},
        "не объект",
    ]}
    questions = games_mod.questions_from_payload(payload, topic="наука", count=5)
    assert [item.answer for item in questions] == ["Париж", "4"]
    assert questions[0].aliases == ["Paris"] and questions[0].topic == "наука"
    assert games_mod.questions_from_payload({"questions": []}, count=5) == []
    assert games_mod.questions_from_payload(None) == []


def test_the_generator_names_the_reason_when_the_model_is_missing():
    async def scenario():
        generator = games_mod.LlmQuizGenerator(None)
        with pytest.raises(games_mod.QuizUnavailable, match="no language model"):
            await generator.questions("история")
        assert generator.snapshot()["failures"] == 0

    asyncio.run(scenario())


def test_the_generator_uses_the_schema_and_falls_back_to_json_text():
    class Model:
        def __init__(self, *, structured=None, text=""):
            self.structured = structured
            self.text = text
            self.schemas: list[dict] = []

        async def structured_json(self, messages, schema, *, name="result"):
            self.schemas.append(schema)
            if self.structured is None:
                raise RuntimeError("the endpoint refused guided JSON")
            return self.structured

        async def reply_text(self, messages):
            return self.text

    async def scenario():
        model = Model(structured={"questions": [
            {"question": "Столица Франции?", "answer": "Париж", "aliases": []}]})
        questions = await games_mod.LlmQuizGenerator(model).questions("география")
        assert [item.answer for item in questions] == ["Париж"]
        assert model.schemas and "properties" in model.schemas[0]

        fallback = Model(text='вот они: {"questions": [{"question": "2+2?", '
                              '"answer": "4", "aliases": ["четыре"]}]}')
        questions = await games_mod.LlmQuizGenerator(fallback).questions("математика")
        assert [item.answer for item in questions] == ["4"]

        broken = Model(text="никакого json тут нет")
        generator = games_mod.LlmQuizGenerator(broken)
        with pytest.raises(games_mod.QuizUnavailable, match="no usable questions"):
            await generator.questions("история")
        assert generator.snapshot()["failures"] == 1

    asyncio.run(scenario())


# --- настоящий путь хода ----------------------------------------------------


def _registry():
    registry = SkillRegistry()
    registry.load_directory(REPO_ROOT / "skills")
    assert registry.errors == []
    return registry


def _hub(tmp_path, monkeypatch, *, enabled=True, generator=None, config=None):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Гостиная", tz="UTC")
    ensure_home(conn, "office", name="Кабинет", tz="UTC")
    cfg = config or Config()
    cfg.server.games = GamesConfig(enabled=enabled, max_questions=2, answer_window_s=30,
                                   topics=["история"])
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_skills", _registry())
    monkeypatch.setattr(hub_app, "_quiz_generator", generator or FakeQuiz())
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    connection = hub_app.Connection(SimpleNamespace(client=None), cfg)
    connection.session = Session(client_id="room-pc", devices=[], history_turns=4)
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._speaker_name = "Anton"
    connection.send_json = AsyncMock()
    connection._stream_tts = AsyncMock()
    connection._log_dialog = AsyncMock()
    return conn, connection


def _said(connection) -> list[str]:
    return [str(call.args[0].get("text") or "") for call in connection.send_json.await_args_list]


def test_the_turn_starts_a_quiz_and_stores_the_round(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    spoken: list[tuple[str, str]] = []

    async def say(home, line, *, name=""):
        spoken.append((home, line))
        return True

    monkeypatch.setattr(hub_app, "_say_in_home", say)
    try:
        handled = asyncio.run(connection._game_turn(
            "давайте сыграем в квиз по теме история", "ru", None, 100.0,
            connection.session, 40))
        assert handled is True
        assert any("Кто написал" in text for text in _said(connection))
        round_ = games_mod.QuizEngine(
            store=SkillStateStore(conn, skill="games"), generator=FakeQuiz(),
            settings=connection.cfg.server.games).active()
        assert round_ is not None and round_.topic == "история"
        assert round_.home_ids == ["livingroom", "office"]
        # Вопрос услышали и в соседней комнате: партия между комнатами.
        assert [home for home, _ in spoken] == ["office"]
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_an_answer_from_another_room_scores_that_room(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    spoken: list[tuple[str, str]] = []

    async def say(home, line, *, name=""):
        spoken.append((home, line))
        return True

    monkeypatch.setattr(hub_app, "_say_in_home", say)
    try:
        asyncio.run(connection._game_turn("сыграем в квиз по теме история", "ru",
                                          None, 100.0, connection.session, 40))
        connection.send_json.reset_mock()
        connection.home_id = "office"
        handled = asyncio.run(connection._game_turn("Пушкин", "ru", None, 200.0,
                                                    connection.session, 40))
        assert handled is True
        assert any("Верно" in text and "Кабинет" in text for text in _said(connection))
        round_ = games_mod.QuizEngine(
            store=SkillStateStore(conn, skill="games"), generator=FakeQuiz(),
            settings=connection.cfg.server.games).active()
        assert round_.scores == {"office": 1}
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_the_score_and_the_stop_word_answer_the_room(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)

    async def say(home, line, *, name=""):
        return True

    monkeypatch.setattr(hub_app, "_say_in_home", say)
    try:
        asyncio.run(connection._game_turn("сыграем в квиз по теме история", "ru",
                                          None, 100.0, connection.session, 40))
        connection.send_json.reset_mock()
        assert asyncio.run(connection._game_turn("какой счёт?", "ru", None, 200.0,
                                                 connection.session, 40)) is True
        assert any("Счёт" in text or "очк" in text for text in _said(connection))
        connection.send_json.reset_mock()
        assert asyncio.run(connection._game_turn("хватит", "ru", None, 300.0,
                                                 connection.session, 40)) is True
        assert any("закончена" in text.lower() for text in _said(connection))
        assert games_mod.QuizEngine(
            store=SkillStateStore(conn, skill="games"), generator=FakeQuiz(),
            settings=connection.cfg.server.games).active() is None
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_a_quiz_without_a_topic_asks_instead_of_inventing_one(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch, generator=FakeQuiz())
    try:
        handled = asyncio.run(connection._game_turn("сыграем в квиз", "ru", None, 100.0,
                                                   connection.session, 40))
        assert handled is True
        assert any("тему" in text.lower() for text in _said(connection))
        assert SkillStateStore(conn, skill="games").get(games_mod.STATE_ROUND) is None
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_no_model_means_no_game_and_the_reason_is_spoken(tmp_path, monkeypatch):
    class Blind:
        async def questions(self, topic, *, count=5, language="ru", avoid=()):
            raise games_mod.QuizUnavailable("no language model is loaded")

    conn, connection = _hub(tmp_path, monkeypatch, generator=Blind())
    try:
        handled = asyncio.run(connection._game_turn(
            "сыграем в квиз по теме история", "ru", None, 100.0, connection.session, 40))
        assert handled is True
        said = " ".join(_said(connection)).casefold()
        assert "no language model" in said
        assert SkillStateStore(conn, skill="games").get(games_mod.STATE_ROUND) is None
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_a_second_quiz_request_does_not_start_a_second_round(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        asyncio.run(connection._game_turn("сыграем в квиз по теме история", "ru",
                                          None, 100.0, connection.session, 40))
        connection.send_json.reset_mock()
        assert asyncio.run(connection._game_turn("сыграем в квиз по теме фильмы", "ru",
                                                 None, 200.0, connection.session, 40)) is True
        said = " ".join(_said(connection)).casefold()
        assert "уже идёт" in said and "хватит" in said
        round_ = games_mod.QuizEngine(
            store=SkillStateStore(conn, skill="games"), generator=FakeQuiz(),
            settings=connection.cfg.server.games).active()
        assert round_.topic == "история"
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_the_flag_off_leaves_the_turn_alone(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch, enabled=False)
    try:
        assert asyncio.run(connection._game_turn(
            "сыграем в квиз по теме история", "ru", None, 100.0,
            connection.session, 40)) is False
        connection.send_json.assert_not_awaited()
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_an_ordinary_phrase_is_not_swallowed_by_the_game(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        assert asyncio.run(connection._game_turn(
            "включи свет в комнате", "ru", None, 100.0, connection.session, 40)) is False
        connection.send_json.assert_not_awaited()
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_the_skill_context_carries_state_and_a_scheduler(tmp_path, monkeypatch):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Гостиная", tz="UTC")
    cfg = Config()
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_config", cfg)
    try:
        ctx = hub_app._skill_context("livingroom", "p-anton", "ru", skill="games")
        assert isinstance(ctx.state, SkillStateStore)
        assert ctx.state.skill_id == "livingroom:games"
        assert ctx.scheduler is hub_app._skill_scheduler()
        assert ctx.home_name("livingroom") == "Гостиная"
        assert ctx.game_settings is cfg.server.games
        plain = hub_app._skill_context("livingroom", "", "ru")
        assert plain.state is None, "скиллу без имени состояние не подсовывают"
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_speaking_into_a_room_without_a_client_is_a_clear_false(monkeypatch):
    monkeypatch.setattr(hub_app, "_connections", [])
    assert asyncio.run(hub_app._say_in_home("nowhere", "привет")) is False


def test_the_health_snapshot_names_the_flag_and_the_round(tmp_path, monkeypatch):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Гостиная", tz="UTC")
    cfg = Config()
    cfg.server.games = GamesConfig(enabled=True, max_questions=2, homes=["livingroom"])
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_quiz_generator", FakeQuiz())
    try:
        asyncio.run(games_mod.QuizEngine(
            store=SkillStateStore(conn, skill="games"), generator=FakeQuiz(),
            settings=cfg.server.games).start(topic="история", home_id="livingroom"))
        snapshot = hub_app._games_snapshot()
        assert snapshot["enabled"] is True
        assert snapshot["homes"] == ["livingroom"]
        assert snapshot["engine"]["active"] is True
        assert snapshot["engine"]["topic"] == "история"
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()
