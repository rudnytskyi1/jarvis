"""P3-20 (F-418): «запомни, что …», «забудь, что …», «что ты обо мне знаешь?».

The three explicit memory requests are recognised by the hub, not by the
model; the two that change stored data wait for the spoken "yes" of F-113, and
the read-only question is answered from the rows themselves.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from common.config import Config
from hub import app as hub_app
from hub import memories, memory_admin
from hub.memories import Kind, Scope
from hub.migrations_runner import connect, migrate
from hub.room_state import RoomState
from hub.storage import Memory


@pytest.fixture
def hub_db(tmp_path):
    """A real hub database: the schema of section 14, applied."""
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    yield conn
    conn.close()


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **row: Any) -> None:
        self.rows.append(row)


def _connection(hub_db, monkeypatch, tmp_path, *, speaker: str = "Anton",
                role: str = "admin", memory: Memory | None = None):
    audit = _Audit()
    store = memory if memory is not None else Memory(data_dir=tmp_path)
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    # The hub builds its own database lazily on some paths and would put ITS
    # connection into the global; this harness hands it a real, migrated one
    # and keeps the lazy builder from replacing it (the same guard P3-15's
    # tests use).
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    monkeypatch.setattr(hub_app, "_memory", store)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = RoomState()
    connection._speaker_name = speaker
    connection._speaker_role = role
    connection._speaker_score = 0.9
    connection._pending_confirmation = None
    connection._approved_call = ""
    connection._confirmation_opened = None
    connection._untrusted_reads = []
    connection._utterance_actions = []
    connection._memory_seq = 1
    connection._personal_facts = {}
    connection.cfg = Config()
    connection._reply_lock = asyncio.Lock()
    connection.utterance_id = "utt-1"
    connection.spoken: list[str] = []
    connection.sent: list[dict[str, Any]] = []

    async def _say(voice, say_text, **kwargs):
        connection.spoken.append(str(say_text))

    async def _send_json(payload):
        connection.sent.append(payload)

    connection._stream_tts = _say
    connection.send_json = _send_json
    connection._finish_utterance = lambda **kwargs: None

    async def _log_dialog(*args, **kwargs):
        return None

    connection._log_dialog = _log_dialog
    return connection, audit, store


def _answer(connection: Any, text: str, language: str = "ru") -> bool:
    return asyncio.run(connection._resolve_confirmation(
        text, voice=None, session=None, started_at=None, language=language,
        stt_ms=0, t_start=0.0))


# --- the parsing ------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Rowan, запомни, что я пью кофе по утрам", "я пью кофе по утрам"),
    ("запомни что моя пара в девять", "моя пара в девять"),
    ("Remember that I drink coffee in the morning", "I drink coffee in the morning"),
    ("please note down that the spare key is in the drawer", "the spare key is in the drawer"),
    ("Recuerda que tomo café por la mañana", "tomo café por la mañana"),
])
def test_a_remember_request_yields_the_fact(text, expected):
    request = memory_admin.parse(text)
    assert request is not None and request.action == "remember"
    assert request.text == expected


@pytest.mark.parametrize("text,expected", [
    ("забудь, что я пью кофе", "я пью кофе"),
    ("Забудь про мой старый адрес", "мой старый адрес"),
    ("forget that I like jazz", "I like jazz"),
    ("forget about my old address", "my old address"),
    ("olvida que tomo café", "tomo café"),
])
def test_a_forget_request_yields_the_query(text, expected):
    request = memory_admin.parse(text)
    assert request is not None and request.action == "forget"
    assert request.text == expected


@pytest.mark.parametrize("text", [
    "что ты обо мне знаешь?",
    "Что ты знаешь обо мне",
    "что ты помнишь обо мне",
    "What do you know about me?",
    "what do you remember about me",
    "¿Qué sabes de mí?",
])
def test_a_knowledge_question_is_recognised(text):
    request = memory_admin.parse(text)
    assert request is not None and request.action == "knowledge"


@pytest.mark.parametrize("text", [
    "забудь меня",
    "Rowan, forget me",
    "не забудь купить хлеб",
    "don't forget the milk",
    "запомни меня",
    "remember my voice",
    "forget it",
    "забудь",
    "открой браузер",
    "",
])
def test_other_flows_are_not_memory_requests(text):
    assert memory_admin.parse(text) is None


# --- choosing the one fact --------------------------------------------------


def _candidate(text: str) -> memory_admin.Candidate:
    return memory_admin.Candidate(text=text)


def test_the_best_matching_fact_wins():
    selection = memory_admin.select(
        [_candidate("Любит кофе с молоком"), _candidate("Пара начинается в девять")],
        "люблю кофе с молоком")
    assert selection.candidate is not None
    assert selection.candidate.text == "Любит кофе с молоком"
    assert not selection.ambiguous


def test_a_near_tie_is_refused_instead_of_guessed():
    selection = memory_admin.select(
        [_candidate("Люблю кофе с молоком"), _candidate("Люблю кофе без сахара")],
        "люблю кофе")
    assert selection.candidate is None and selection.ambiguous
    assert len(selection.alternatives) == 2


def test_nothing_matching_is_no_candidate():
    selection = memory_admin.select([_candidate("Любит кофе")], "умею летать")
    assert selection.candidate is None and not selection.ambiguous


def test_a_shared_stem_still_counts():
    assert memory_admin.score("Ключи лежат на столе", "где мои ключи") > 0.0


# --- wording ----------------------------------------------------------------


@pytest.mark.parametrize("language,needle", [("ru", "запомню"), ("en", "remember"), ("es", "recordaré")])
def test_the_remember_question_names_the_fact(language, needle):
    text = memory_admin.remember_question("I drink coffee", language, window_s=8)
    assert needle in text and "I drink coffee" in text


@pytest.mark.parametrize("language,needle", [("ru", "забуду"), ("en", "forget"), ("es", "olvidaré")])
def test_the_forget_question_warns_that_it_is_final(language, needle):
    text = memory_admin.forget_question("I drink coffee", language, window_s=8)
    assert needle in text and "coffee" in text


@pytest.mark.parametrize("language,needle", [
    ("ru", "я о тебе знаю"), ("en", "I know about you"), ("es", "sé de ti")])
def test_the_knowledge_answer_lists_the_facts(language, needle):
    text = memory_admin.knowledge_answer(["Пьёт кофе", "Пара в девять"], language)
    assert needle in text and "Пьёт кофе" in text and "Пара в девять" in text


def test_an_empty_memory_is_admitted():
    assert "ничего" in memory_admin.knowledge_answer([], "ru")


# --- the live room ----------------------------------------------------------


def test_a_remember_request_asks_first_and_writes_only_after_yes(hub_db, monkeypatch, tmp_path):
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path)
    turn = asyncio.run(connection._memory_turn("Rowan, запомни, что я пью кофе по утрам", "ru"))
    assert turn is not None and "запомню" in turn
    assert connection._pending_confirmation is not None
    assert store.facts("Anton") == [], "nothing is saved before the yes"
    assert memories.MemoryIndex(hub_db).active() == []

    assert _answer(connection, "да") is True
    assert store.facts("Anton") == ["я пью кофе по утрам"]
    rows = memories.MemoryIndex(hub_db).active()
    assert [(fact.scope, fact.owner_id, fact.text) for fact in rows] == [
        (Scope.PERSON, "Anton", "я пью кофе по утрам")]
    assert connection.spoken and "Запомнил" in connection.spoken[-1]


def test_a_no_leaves_memory_untouched(hub_db, monkeypatch, tmp_path):
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path)
    asyncio.run(connection._memory_turn("запомни, что я пью кофе", "ru"))
    assert _answer(connection, "нет") is True
    assert store.facts("Anton") == []
    assert memories.MemoryIndex(hub_db).active() == []
    assert connection.spoken and "ничего" in connection.spoken[-1]


def test_a_forget_request_deletes_exactly_one_fact_after_yes(hub_db, monkeypatch, tmp_path):
    connection, audit, store = _connection(hub_db, monkeypatch, tmp_path)
    store.add("я пью кофе по утрам", "Anton")
    store.add("моя пара в девять", "Anton")
    memories.MemoryIndex(hub_db).write(memories.fact_from(
        scope=Scope.PERSON, owner_id="Anton", kind=Kind.PERSON_FACT,
        text="я пью кофе по утрам"))

    turn = asyncio.run(connection._memory_turn("забудь, что я пью кофе", "ru"))
    assert turn is not None and "забуду" in turn
    assert connection._pending_confirmation is not None
    assert store.facts("Anton") == ["я пью кофе по утрам", "моя пара в девять"]

    assert _answer(connection, "да") is True
    assert store.facts("Anton") == ["моя пара в девять"]
    assert [fact.text for fact in memories.MemoryIndex(hub_db).active()] == []
    actions = [row["action"] for row in audit.rows]
    assert "memory.forget" in actions and "confirm.dangerous" in actions
    forgotten = next(row for row in audit.rows if row["action"] == "memory.forget")
    assert forgotten["result"] == "ok"
    assert connection.spoken and "Забыл" in connection.spoken[-1]


def test_forgetting_an_unknown_fact_changes_nothing(hub_db, monkeypatch, tmp_path):
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path)
    store.add("я пью кофе", "Anton")
    turn = asyncio.run(connection._memory_turn("забудь, что я умею летать", "ru"))
    assert turn is not None and "не нашёл" in turn
    assert connection._pending_confirmation is None
    assert store.facts("Anton") == ["я пью кофе"]


def test_an_ambiguous_forget_asks_instead_of_deleting(hub_db, monkeypatch, tmp_path):
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path)
    store.add("Люблю кофе с молоком", "Anton")
    store.add("Люблю кофе без сахара", "Anton")
    turn = asyncio.run(connection._memory_turn("забудь, что я люблю кофе", "ru"))
    assert turn is not None and "больше одного" in turn
    assert connection._pending_confirmation is None
    assert len(store.facts("Anton")) == 2


def test_the_knowledge_question_is_answered_without_a_confirmation(hub_db, monkeypatch, tmp_path):
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path)
    store.add("я пью кофе по утрам", "Anton")
    memories.MemoryIndex(hub_db).write(memories.fact_from(
        scope=Scope.PERSON, owner_id="Anton", kind=Kind.PREFERENCE,
        text="Anton wants Chrome"))
    turn = asyncio.run(connection._memory_turn("что ты обо мне знаешь?", "ru"))
    assert turn is not None
    assert "я пью кофе по утрам" in turn and "Anton wants Chrome" in turn
    assert connection._pending_confirmation is None


def test_an_unknown_speaker_cannot_file_a_personal_fact(hub_db, monkeypatch, tmp_path):
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path, speaker="")
    turn = asyncio.run(connection._memory_turn("запомни, что я пью кофе", "ru"))
    assert turn is not None and "recognize" in turn
    assert connection._pending_confirmation is None
    assert store.facts() == []


def test_a_guest_does_not_get_the_rooms_facts(hub_db, monkeypatch, tmp_path):
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path, speaker="")
    store.add("в комнате холодно")           # a fact of the room itself
    store.add("гость пьёт чай", "Anton")     # somebody else's fact
    memories.MemoryIndex(hub_db).write(memories.fact_from(
        scope=Scope.HOME, owner_id="livingroom", kind=Kind.HOME_FACT,
        text="The light switch is by the door"))
    turn = asyncio.run(connection._memory_turn("что ты обо мне знаешь?", "ru"))
    assert turn is not None
    assert "в комнате холодно" not in turn and "гость пьёт чай" not in turn
    assert "light switch" not in turn


def test_memory_management_can_be_switched_off(hub_db, monkeypatch, tmp_path):
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path)
    connection.cfg.server.memory.management_enabled = False
    assert asyncio.run(connection._memory_turn("запомни, что я пью кофе", "ru")) is None
    assert connection._pending_confirmation is None


def test_the_forget_tool_also_waits_for_its_yes(hub_db, monkeypatch, tmp_path):
    """A model-issued ``forget_fact`` is a deletion and asks, by config."""
    connection, _audit, store = _connection(hub_db, monkeypatch, tmp_path)
    store.add("я пью кофе", "Anton")
    result = asyncio.run(connection._execute_tool(
        "forget_fact", {"query": "я пью кофе", "about": "me"}))
    assert result.get("needs_confirmation") is True
    assert store.facts("Anton") == ["я пью кофе"], "nothing deleted before the yes"
