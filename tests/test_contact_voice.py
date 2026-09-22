"""Голосовое согласие между людьми: разбор фразы и ход хаба (ТЗ F-602)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import contacts as contacts_mod
from hub.contacts import ContactStore, contact_command, spoken_name_variants
from hub.homes import ensure_home
from hub.interhome import InterhomeLimiter
from hub.migrations_runner import connect, migrate
from hub.session import Session
from hub.utterances import UtteranceMetrics

AMY = "p-amy"
MAX = "p-max"
CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    ensure_home(conn, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)",
                 (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)",
                 (MAX, "Макс"))
    conn.commit()
    yield conn
    conn.close()


# --- разбор фразы -----------------------------------------------------------


@pytest.mark.parametrize("text,name", [
    ("добавь Макса в контакты", "Макса"),
    ("Добавь Макса в контакты!", "Макса"),
    ("добавь в контакты Макса", "Макса"),
    ("add Max to my contacts", "Max"),
    ("Add Max as a contact", "Max"),
    ("agrega a Max a mis contactos", "Max"),
    ("añade a Макс a los contactos", "Макс"),
])
def test_an_invitation_phrase_names_the_other_person(text, name):
    command = contact_command(text)
    assert command is not None
    assert command.intent is contacts_mod.ContactIntent.INVITE
    assert command.name == name


@pytest.mark.parametrize("text,name", [
    ("подтверди контакт с Антоном", "Антоном"),
    ("Подтверждаю контакт с Антоном", "Антоном"),
    ("confirm the contact with Anton", "Anton"),
    ("confirmar el contacto con Anton", "Anton"),
])
def test_a_confirmation_phrase_names_the_other_person(text, name):
    command = contact_command(text)
    assert command is not None
    assert command.intent is contacts_mod.ContactIntent.CONFIRM
    assert command.name == name


@pytest.mark.parametrize("text", ["", "включи свет", "позвони Максу",
                                  "what time is it?", "спасибо", "добавь молока"])
def test_other_phrases_are_left_to_the_model(text):
    assert contact_command(text) is None


@pytest.mark.parametrize("text,intent,name", [
    ("отмени контакт с Максом", "revoke", "Максом"),
    ("убери Макса из контактов", "revoke", "Макса"),
    ("remove Max from my contacts", "revoke", "Max"),
    ("quita a Max de mis contactos", "revoke", "Max"),
    ("заблокируй Макса", "block", "Макса"),
    ("block Max", "block", "Max"),
    ("bloquea a Max", "block", "Max"),
    ("разблокируй Макса", "unblock", "Макса"),
    ("сними блокировку с Макса", "unblock", "Макса"),
    ("unblock Max", "unblock", "Max"),
    ("desbloquea a Max", "unblock", "Max"),
    ("разреши Максу видеть, что я дома", "presence_on", "Максу"),
    ("let Max know when I am home", "presence_on", "Max"),
    ("share my presence with Max", "presence_on", "Max"),
    ("permite que Max sepa cuando estoy en casa", "presence_on", "Max"),
    ("запрети Максу видеть, что я дома", "presence_off", "Максу"),
    ("stop sharing my presence with Max", "presence_off", "Max"),
    ("no permitas que Max sepa cuando estoy en casa", "presence_off", "Max"),
])
def test_lifecycle_phrases_are_understood(text, intent, name):
    command = contact_command(text)
    assert command is not None, text
    assert command.intent.value == intent
    assert command.name == name


@pytest.mark.parametrize("text", ["кто дома?", "где Макс?", "Макс дома?", "включи музыку",
                                  "покажи камеру"])
def test_lifecycle_phrases_do_not_swallow_other_turns(text):
    assert contact_command(text) is None


def test_spoken_names_are_looked_up_without_their_case_ending():
    assert "макс" in spoken_name_variants("Макса")
    assert "антон" in spoken_name_variants("Антоном")
    assert spoken_name_variants("Max") == ("max",)
    assert spoken_name_variants("") == ()


# --- ход хаба ---------------------------------------------------------------


def _connection(monkeypatch, hub_db, *, speaker: str = "Антон", home: str = "livingroom"):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_contacts", None)
    monkeypatch.setattr(hub_app, "_audit", None)
    # Частота сообщений — своя задача (P4-06); здесь проверяется согласие.
    monkeypatch.setattr(hub_app, "_interhome_limits", InterhomeLimiter(enabled=False))
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = home
    conn.cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room",
                              "tz": CHICAGO}])
    conn._reply_language = "ru"
    conn._speaker_name = speaker
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    return conn


def test_an_invitation_from_a_recognised_person_waits_for_the_other_side(
        monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru"))
    assert "Макс" in answer and "подтвердит" in answer
    contact = ContactStore(hub_db).get(AMY, MAX)
    assert contact is not None and contact.status == "pending"
    assert contact.requested_by == AMY and not contact.confirmed
    # Оба конца пары «видят» приглашение, но доступ ещё закрыт.
    assert ContactStore(hub_db).status(MAX, AMY) == "pending"
    assert ContactStore(hub_db).allowed(AMY, MAX) is False


def test_the_other_person_confirms_it_in_their_own_room(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru"))
    max_room = _connection(monkeypatch, hub_db, speaker="Макс", home="kyiv")
    answer = asyncio.run(max_room._contact_turn("подтверди контакт с Антоном", "ru"))
    assert "контакт" in answer.lower()
    contact = ContactStore(hub_db).get(AMY, MAX)
    assert contact is not None and contact.confirmed
    assert ContactStore(hub_db).allowed(MAX, AMY) is True


def test_the_inviter_cannot_confirm_their_own_invitation(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru"))
    answer = asyncio.run(conn._contact_turn("подтверди контакт с Максом", "ru"))
    assert "сам" in answer
    assert ContactStore(hub_db).get(AMY, MAX).status == "pending"


def test_confirming_without_an_invitation_is_refused(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._contact_turn("подтверди контакт с Максом", "ru"))
    assert "подтверждать нечего" in answer
    assert ContactStore(hub_db).get(AMY, MAX) is None


def test_a_stranger_voice_cannot_change_contacts(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db, speaker="")
    answer = asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru"))
    assert "голос" in answer
    assert ContactStore(hub_db).get(AMY, MAX) is None


def test_an_unknown_name_is_said_back_instead_of_guessed(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._contact_turn("добавь Геннадия в контакты", "ru"))
    assert "Геннадия" in answer
    assert not list(hub_db.execute("SELECT 1 FROM contacts"))


def test_a_person_cannot_add_themselves(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._contact_turn("добавь Антона в контакты", "ru"))
    assert "Себя" in answer
    assert not list(hub_db.execute("SELECT 1 FROM contacts"))


def test_a_blocked_pair_cannot_be_invited_again(monkeypatch, hub_db):
    ContactStore(hub_db).block(MAX, AMY)
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru"))
    assert "заблокирована" in answer
    assert ContactStore(hub_db).status(AMY, MAX) == "blocked"


def test_contact_answers_follow_the_speakers_language(monkeypatch, hub_db):
    max_room = _connection(monkeypatch, hub_db, speaker="Макс")
    english = asyncio.run(max_room._contact_turn("add Антон to my contacts", "en"))
    assert "invitation" in english
    amy_room = _connection(monkeypatch, hub_db, speaker="Антон")
    spanish = asyncio.run(amy_room._contact_turn("confirma el contacto con Макс", "es"))
    assert "contactos" in spanish
    assert ContactStore(hub_db).allowed(AMY, MAX) is True


def test_a_phrase_that_is_not_about_contacts_stays_with_the_model(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    assert asyncio.run(conn._contact_turn("включи свет в комнате", "ru")) is None


def test_a_hub_without_a_database_keeps_the_old_behaviour(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_contacts", False)
    assert asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru")) is None


def test_every_contact_change_is_audited(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru"))
    max_room = _connection(monkeypatch, hub_db, speaker="Макс")
    asyncio.run(max_room._contact_turn("подтверди контакт с Антоном", "ru"))
    rows = list(hub_db.execute(
        "SELECT action, actor_person_id, target, result, home_id FROM audit"
        " ORDER BY rowid"))
    assert [(row[0], row[1], row[2], row[3]) for row in rows] == [
        ("contact.invite", AMY, MAX, "ok"),
        ("contact.confirm", MAX, AMY, "ok"),
    ]
    assert rows[0][4] == "livingroom"


def test_a_refused_contact_change_is_audited_too(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru"))
    asyncio.run(conn._contact_turn("подтверди контакт с Максом", "ru"))
    rows = list(hub_db.execute(
        "SELECT action, result, detail_json FROM audit ORDER BY rowid"))
    assert rows[-1][0] == "contact.confirm" and rows[-1][1] == "denied"
    assert "own invitation" in rows[-1][2]


def test_a_shortened_name_still_finds_the_person(monkeypatch, hub_db):
    hub_db.execute("UPDATE persons SET display_name = 'Максим' WHERE person_id = ?", (MAX,))
    hub_db.commit()
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._contact_turn("добавь Макса в контакты", "ru"))
    assert ContactStore(hub_db).status(AMY, MAX) == "pending"
    assert "Максим" in answer, "the hub says the name it knows, not the spoken case"


# --- отзыв, блокировка и флаг присутствия (F-602) ---------------------------


def _linked(monkeypatch, hub_db) -> ContactStore:
    """Подтверждённая пара Антон ↔ Макс, как её оставили бы P4-01..02."""
    store = ContactStore(hub_db)
    store.invite(AMY, MAX)
    store.confirm(MAX, AMY)
    return store


def test_blocking_by_voice_closes_the_connection(monkeypatch, hub_db):
    store = _linked(monkeypatch, hub_db)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    answer = asyncio.run(conn._contact_turn("заблокируй Макса", "ru"))
    assert "закрыла" in answer
    assert store.status(AMY, MAX) == "blocked"
    assert store.allowed(AMY, MAX) is False
    rows = list(hub_db.execute("SELECT action, result FROM audit"))
    assert rows[-1] == ("contact.block", "ok")


def test_a_blocked_person_cannot_invite_back(monkeypatch, hub_db):
    _linked(monkeypatch, hub_db)
    amy = _connection(monkeypatch, hub_db, speaker="Антон")
    asyncio.run(amy._contact_turn("заблокируй Макса", "ru"))
    max_room = _connection(monkeypatch, hub_db, speaker="Макс")
    answer = asyncio.run(max_room._contact_turn("добавь Антона в контакты", "ru"))
    assert "заблокирована" in answer
    assert ContactStore(hub_db).status(AMY, MAX) == "blocked"


def test_only_the_person_who_blocked_can_unblock(monkeypatch, hub_db):
    store = _linked(monkeypatch, hub_db)
    store.block(AMY, MAX)
    max_room = _connection(monkeypatch, hub_db, speaker="Макс")
    refused = asyncio.run(max_room._contact_turn("разблокируй Антона", "ru"))
    assert "поставил" in refused
    assert store.status(AMY, MAX) == "blocked"
    amy = _connection(monkeypatch, hub_db, speaker="Антон")
    done = asyncio.run(amy._contact_turn("разблокируй Макса", "ru"))
    assert "снята" in done and ContactStore(hub_db).get(AMY, MAX) is None
    rows = [row[0] for row in hub_db.execute("SELECT action FROM audit")]
    assert rows == ["contact.unblock", "contact.unblock"]
    assert list(hub_db.execute("SELECT result FROM audit"))[0][0] == "denied"


def test_revoking_by_voice_removes_the_contact(monkeypatch, hub_db):
    store = _linked(monkeypatch, hub_db)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    answer = asyncio.run(conn._contact_turn("убери Макса из контактов", "ru"))
    assert "убрала" in answer
    assert store.get(AMY, MAX) is None and store.allowed(AMY, MAX) is False


def test_revoking_a_stranger_is_refused_honestly(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._contact_turn("убери Макса из контактов", "ru"))
    assert "не в контактах" in answer
    rows = list(hub_db.execute("SELECT action, result FROM audit"))
    assert rows[-1] == ("contact.revoke", "denied")


def test_each_person_shares_their_presence_with_their_own_words(monkeypatch, hub_db):
    store = _linked(monkeypatch, hub_db)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    answer = asyncio.run(conn._contact_turn("разреши Максу видеть, что я дома", "ru"))
    assert "будет видеть" in answer
    assert store.presence_shared(AMY, MAX) is True
    assert store.presence_shared(MAX, AMY) is False, "Amy's yes is not Max's"
    rows = list(hub_db.execute("SELECT action, result FROM audit"))
    assert rows[-1] == ("contact.presence_on", "ok")


def test_presence_can_be_taken_back(monkeypatch, hub_db):
    store = _linked(monkeypatch, hub_db)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    asyncio.run(conn._contact_turn("разреши Максу видеть, что я дома", "ru"))
    answer = asyncio.run(conn._contact_turn("запрети Максу видеть, что я дома", "ru"))
    assert "больше не видит" in answer
    assert store.presence_shared(AMY, MAX) is False


def test_presence_cannot_be_shared_without_a_confirmed_contact(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    answer = asyncio.run(conn._contact_turn("разреши Максу видеть, что я дома", "ru"))
    assert "Сначала нужен подтверждённый контакт" in answer
    assert list(hub_db.execute("SELECT result FROM audit"))[-1][0] == "denied"


def test_a_question_about_being_home_stays_with_presence(monkeypatch, hub_db):
    _linked(monkeypatch, hub_db)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    assert asyncio.run(conn._contact_turn("Макс дома?", "ru")) is None


# --- одна точка межкомнатного (F-602, критерий приёмки фазы 4) ---------------


class _Delivery:
    """Подставная доставка: считает, сколько раз её вообще позвали."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self) -> str:
        self.calls.append("delivered")
        return "передано"


def test_without_mutual_consent_nothing_leaves_the_room(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    delivery = _Delivery()
    answer = asyncio.run(conn._interhome_send(MAX, "ru", delivery))
    assert "не в контактах" in answer
    assert delivery.calls == [], "доставка не должна была произойти вовсе"


def test_a_one_sided_invitation_does_not_open_the_door(monkeypatch, hub_db):
    ContactStore(hub_db).invite(AMY, MAX)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    delivery = _Delivery()
    answer = asyncio.run(conn._interhome_send(MAX, "ru", delivery))
    assert "не подтвердили" in answer
    assert delivery.calls == []


def test_a_confirmed_contact_lets_the_delivery_through(monkeypatch, hub_db):
    _linked(monkeypatch, hub_db)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    delivery = _Delivery()
    assert asyncio.run(conn._interhome_send(MAX, "ru", delivery)) == "передано"
    assert delivery.calls == ["delivered"]


def test_a_blocked_contact_closes_the_door_too(monkeypatch, hub_db):
    store = _linked(monkeypatch, hub_db)
    store.block(MAX, AMY)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    delivery = _Delivery()
    answer = asyncio.run(conn._interhome_send(MAX, "ru", delivery))
    assert "заблокирована" in answer and delivery.calls == []


def test_the_gate_speaks_the_language_of_the_room(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    english = asyncio.run(conn._interhome_send(MAX, "en", _Delivery()))
    assert "not contacts" in english
    spanish = asyncio.run(conn._interhome_send(MAX, "es", _Delivery()))
    assert "no son contactos" in spanish


def test_an_unrecognised_voice_and_a_stranger_name_are_refused(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db, speaker="")
    delivery = _Delivery()
    assert "голос" in asyncio.run(conn._interhome_send(MAX, "ru", delivery))
    assert delivery.calls == []
    stranger = _connection(monkeypatch, hub_db, speaker="Антон")
    assert "не в контактах" in asyncio.run(stranger._interhome_send("p-nobody", "ru", delivery))
    assert delivery.calls == []


def test_two_different_interhome_callers_share_the_same_gate(monkeypatch, hub_db):
    """Гейт один: что интерком, что опрос — обе доставки отбиваются одинаково."""
    _linked(monkeypatch, hub_db)
    conn = _connection(monkeypatch, hub_db, speaker="Антон")
    first, second = _Delivery(), _Delivery()
    assert asyncio.run(conn._interhome_send(MAX, "ru", first)) == "передано"
    assert asyncio.run(conn._interhome_send(MAX, "ru", second)) == "передано"
    assert first.calls == second.calls == ["delivered"]
    ContactStore(hub_db).block(AMY, MAX)
    third = _Delivery()
    assert "заблокирована" in asyncio.run(conn._interhome_send(MAX, "ru", third))
    assert third.calls == []
