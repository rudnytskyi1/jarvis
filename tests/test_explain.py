"""Объяснимость идентичности (ТЗ F-215).

The point of the feature is honesty: every number in the answer has to come
from the belief F-206 wrote, and everything the hub did NOT see has to be said
out loud. So the tests check both halves - the numbers that are there and the
signals that are not - and the case where there is no belief at all.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from hub import app as hub_app
from hub import migrations_runner
from hub.explain import (
    MISSING,
    age_phrase,
    explain,
    language_of,
    number,
    seen_and_missing,
    why_question,
)
from hub.identity_fusion import Belief, BeliefStore
from hub.room_state import RoomState


def _belief(**overrides) -> Belief:
    values = {"track_id": "a:1", "person_id": "p-max", "p": 0.94,
              "sources": {"voice": 0.71, "face": 0.9, "p": 0.94, "lead": 0.4},
              "at": time.time()}
    values.update(overrides)
    return Belief(**values)  # type: ignore[arg-type]


# --- вопрос ----------------------------------------------------------------


def test_the_question_is_understood_in_three_languages():
    assert why_question("Rowan, почему ты решил, что это Макс?").who == "Макс"
    assert why_question("Rowan, почему ты так решил, что это Макс").who == "Макс"
    assert why_question("Rowan, с чего ты взял, что это Макс?").who == "Макс"
    assert why_question("Rowan, why do you think this is Max?").who == "Max"
    assert why_question("Rowan, how do you know it's Max").who == "Max"
    assert why_question("Rowan, por qué crees que es Max").who == "Max"
    assert why_question("Rowan, почему ты решил, что это Макс?").language == "ru"
    assert why_question("why do you think that's Max").language == "en"


def test_the_question_about_the_hub_itself_is_understood():
    asked = why_question("Rowan, почему ты думаешь, что это я?")
    assert asked is not None and asked.about_me and asked.who == ""
    assert why_question("Rowan, how do you know it's me?").about_me
    assert why_question("Rowan, cómo sabes que soy yo").about_me


def test_ordinary_speech_is_not_this_question():
    assert why_question("Rowan, включи свет") is None
    assert why_question("Rowan, почему небо синее?") is None
    assert why_question("") is None
    assert why_question(None) is None


# --- числа и время ---------------------------------------------------------


def test_numbers_are_written_the_way_each_language_writes_them():
    assert number(0.71, "ru") == "0,71"
    assert number(0.71, "es") == "0,71"
    assert number(0.71, "en") == "0.71"
    assert number(1.0, "ru") == "1"
    assert number("не число", "ru") == "не число"
    assert language_of("RU") == "ru" and language_of("de") == "ru"


def test_the_age_of_the_decision_is_spoken():
    assert age_phrase(0.4, "ru") == "только что"
    assert age_phrase(12, "ru") == "12 секунд назад"
    assert age_phrase(41, "ru") == "41 секунду назад"
    assert age_phrase(125, "ru") == "2 минуты назад"
    assert age_phrase(7200, "ru") == "2 часа назад"
    assert age_phrase(12, "en") == "12 seconds ago"
    assert age_phrase(12, "es") == "hace 12 segundos"


# --- ответ -----------------------------------------------------------------


def test_the_answer_names_the_signals_and_their_numbers():
    spoken = explain(_belief(), language="ru", name="Макс")
    assert spoken.startswith("Макс?")
    assert "голос 0,71" in spoken and "лицо 0,9" in spoken
    assert "уверенность 0,94" in spoken
    assert "тела того же дня нет" in spoken
    english = explain(_belief(), language="en", name="Max")
    assert english.startswith("Max?") and "voice 0.71" in english


def test_what_the_hub_did_not_see_is_said_out_loud():
    only_voice = _belief(sources={"voice": 0.71, "p": 0.71})
    spoken = explain(only_voice, language="ru")
    assert "лица не видно" in spoken and "тела того же дня нет" in spoken
    assert "голоса не слышно" not in spoken
    seen, missing = seen_and_missing(only_voice, "ru")
    assert seen == ["голос 0,71"] and len(missing) == 2


def test_an_ambiguous_decision_says_what_decided_it():
    ambiguous = _belief(sources={"voice": 0.66, "p": 0.52,
                                 "ambiguous": [{"person_id": "p-other", "p": 0.5}]})
    assert "плохо различали" in explain(ambiguous, language="ru")
    with_context = _belief(sources={"voice": 0.66, "p": 0.52, "ambiguous": [],
                                    "context": "expected here / already in the room"})
    spoken = explain(with_context, language="ru")
    assert "контекст дома" in spoken


def test_a_decision_that_named_nobody_says_so_and_why():
    none = _belief(person_id=None, p=0.4,
                   sources={"reason": "no signal past its threshold"})
    spoken = explain(none, language="ru")
    assert "никого не назвал" in spoken
    assert "ни один сигнал не дотянул" in spoken
    assert "уверенность" not in spoken, "уверенность без имени ни о чём не говорит"


def test_no_belief_at_all_is_an_honest_answer():
    assert "не решал" in explain(None, language="ru")
    assert explain(None, language="ru", name="Макс").startswith("Макс?")
    assert "no belief" in explain(None, language="en")
    assert json.dumps({"ru": explain(None, language="ru")})


def test_a_belief_without_numbers_still_answers():
    bare = _belief(sources={})
    spoken = explain(bare, language="ru")
    assert "уверенность 0,94" in spoken, "уверенность берётся из поля p"
    assert "голоса не слышно" in spoken and "лица не видно" in spoken
    assert MISSING["voice"]["ru"] in spoken


# --- комната ---------------------------------------------------------------


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role)"
                 " VALUES ('p-max', 'livingroom', 'admin')")
    conn.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen,"
                 " person_id) VALUES ('a:1','livingroom','pc-1','now','now','p-max')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _room(tracks) -> RoomState:
    room = RoomState()
    room.update(tracks, now=time.monotonic())
    return room


def _connection(hub_db, monkeypatch, *, store=None, speaker="Макс"):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_identity_beliefs", store)
    monkeypatch.setattr(hub_app, "_voices", None)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}])
    connection._speaker_name = speaker
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection._reply_language = "ru"
    connection._spoofed_tracks = set()
    return connection


def test_the_room_answers_with_the_numbers_of_its_own_belief(hub_db, monkeypatch):
    store = BeliefStore(hub_db)
    store.save(_belief(track_id="a:1", person_id="p-max"), home_id="livingroom")
    connection = _connection(hub_db, monkeypatch, store=store)

    spoken = asyncio.run(connection._explain_turn("Rowan, почему ты решил, что это Макс?"))

    assert spoken and "голос 0,71" in spoken and "лицо 0,9" in spoken
    assert not asyncio.run(connection._explain_turn("Rowan, включи свет"))


def test_the_room_says_it_did_not_decide_when_there_is_no_belief(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, store=BeliefStore(hub_db))
    spoken = asyncio.run(connection._explain_turn("Rowan, why do you think this is Max?"))
    assert spoken and ("не решал" in spoken or "did not decide" in spoken)


def test_the_room_does_not_make_up_a_person_it_never_saw(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, store=BeliefStore(hub_db))
    spoken = asyncio.run(connection._explain_turn("Rowan, почему ты решил, что это Гость?"))
    assert spoken and spoken.startswith("Гость?") and "не решал" in spoken


def test_a_hub_without_the_belief_store_still_answers_honestly(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, store=None)
    spoken = asyncio.run(connection._explain_turn("Rowan, почему ты так решил?"))
    assert spoken and "не решал" in spoken
