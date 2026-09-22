"""P4-23 (F-606/F-415/F-212): гость не получает память дома и общий профиль."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common.config import Config
from hub import app, memories
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.session import Session
from hub.storage import Memory

OWNER = "p-owner"
GUEST = "p-guest"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (OWNER, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (GUEST, "Кай"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (OWNER, "livingroom", "admin"))
    # F-210 writes exactly this: a guest of the room.
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (GUEST, "livingroom", "guest"))
    conn.commit()
    yield conn
    conn.close()


def _voices():
    roles = {"Антон": "admin", "Кай": "guest"}
    registry = Mock()
    registry.enabled = True
    registry.role_of = lambda name: roles.get(str(name or ""), "unknown")
    registry.identify = lambda *a: ("Антон", "admin", 0.9)
    registry.identify_ex = lambda *a: ("Антон", "admin", 0.9, None)
    registry.people = lambda: dict(roles)
    registry.voice_profiles = lambda: {}
    registry.face_profiles = lambda: {}
    return registry


def _memory(tmp_path):
    memory = Memory(data_dir=tmp_path)
    memory.add("Кот живёт на кухне.", "")                      # a ROOM fact
    memory.add("У Антона ключи в верхнем ящике.", "Антон")      # somebody else's fact
    memory.add("Кай любит зелёный чай.", "Кай")                 # the guest's own fact
    return memory


def _filled_memory_table(hub_db):
    # ``owner_id`` is the display name - the same convention ``_run_remember``
    # uses (``_memory_profile`` returns the recognised name, not the row id).
    index = memories.MemoryIndex(hub_db)
    index.write(memories.fact_from(scope=memories.Scope.HOME, owner_id="livingroom",
                                   kind=memories.Kind.HOME_FACT,
                                   text="Дом: чайник стоит на столе."))
    index.write(memories.fact_from(scope=memories.Scope.PERSON, owner_id="Кай",
                                   kind=memories.Kind.PERSON_FACT,
                                   text="Кай пьёт чай без сахара."))
    return index


def _brain():
    return SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(
            text="Хорошо.", history=[], tool_calls=[])),
        verify=AsyncMock())


def _run_turn(monkeypatch, tmp_path, hub_db, *, speaker: str, role: str,
              text: str = "что здесь стоит на столе?"):
    cfg = Config()
    cfg.server.diarization.enabled = False
    cfg.server.llm.verify_actions = False
    brain = _brain()
    monkeypatch.setattr(app, "_llm", brain)
    monkeypatch.setattr(app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: (text, "ru")))
    voices = _voices()
    voices.identify = lambda *a: (speaker, role, 0.9)
    # The hub identifies through ``identify_ex`` (F-205: the vector comes back
    # with the name), so the stub has to answer there too.
    voices.identify_ex = lambda *a: (speaker, role, 0.9, None)
    monkeypatch.setattr(app, "_voices", voices)
    monkeypatch.setattr(app, "_tts", object())
    monkeypatch.setattr(app, "_conversations", None)
    monkeypatch.setattr(app, "_memory", _memory(tmp_path))
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_audit", None)
    monkeypatch.setattr(app, "_guest_grants", False)
    # Every lazily-built global store is reset for this test only: the hub
    # keeps them per process, and a store from another test points at a
    # database this one has already closed.
    for name in ("_polls", "_intercom", "_contacts", "_scenes", "_objects",
                 "_device_states", "_switches"):
        monkeypatch.setattr(app, name, None)
    for name in ("_devices", "_tools", "_interhome_limits", "_media"):
        monkeypatch.setattr(app, name, False)
    conn = app.Connection(SimpleNamespace(client=None), cfg)
    conn.home_id = "livingroom"
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.send_json = AsyncMock()
    conn._announce_speaker = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn.presence = SimpleNamespace(present=lambda: {"Антон"}, unknown_count=0)
    asyncio.run(conn._handle_utterance(b"\0" * 3200 * 2))
    return brain.generate.call_args.args[0]


# --- the rule itself -------------------------------------------------------


def test_a_registered_guest_counts_as_a_guest_of_the_home(hub_db, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_voices", _voices())
    conn = app.Connection.__new__(app.Connection)
    conn.home_id = "livingroom"
    conn.session = Session(client_id="room-pc", devices=[], history_turns=2)
    conn._speaker_name = "Кай"
    assert conn._home_guest() is True, "a guest of F-210 is a guest, not a member"
    conn._speaker_name = "Антон"
    assert conn._home_guest() is False, "the owner is not a guest"
    conn._speaker_name = "unknown"
    assert conn._home_guest() is True


def test_a_guest_gets_no_room_memory_in_the_guest_s_own_turn(
        hub_db, monkeypatch, tmp_path):
    _filled_memory_table(hub_db)
    messages = _run_turn(monkeypatch, tmp_path, hub_db, speaker="Кай", role="guest",
                         text="что стоит на столе?")
    system, user = messages[0]["content"], messages[-1]["content"]
    assert "Кот живёт на кухне." not in system, "no room facts for a guest"
    assert "ключи в верхнем ящике" not in system, "nor another person's facts"
    assert "Кай любит зелёный чай." in system, "their own facts stay theirs (F-415)"
    assert "чайник стоит на столе" not in user, "nor the home's rows in the retrieval"
    # The guest's OWN rows still follow them: the same question, asked about
    # their own fact, retrieves it (and the home's row is not even a candidate).
    own = _run_turn(monkeypatch, tmp_path, hub_db, speaker="Кай", role="guest",
                    text="какой чай я пью?")
    assert "Кай пьёт чай без сахара." in own[-1]["content"]


def test_a_guest_is_not_given_the_household_s_profile(hub_db, monkeypatch, tmp_path):
    _filled_memory_table(hub_db)
    messages = _run_turn(monkeypatch, tmp_path, hub_db, speaker="Кай", role="guest")
    user = messages[-1]["content"]
    assert "in the room: Антон" not in user, "who lives here is the house's profile"
    assert "speaker: Кай" in user and "role: guest" in user


def test_the_owner_still_gets_the_home_memory_and_the_people(hub_db, monkeypatch, tmp_path):
    _filled_memory_table(hub_db)
    messages = _run_turn(monkeypatch, tmp_path, hub_db, speaker="Антон", role="admin",
                         text="что стоит на столе?")
    system, user = messages[0]["content"], messages[-1]["content"]
    assert "Кот живёт на кухне." in system
    assert "ключи в верхнем ящике" in system
    assert "чайник стоит на столе" in user
    assert "in the room: Антон" in user
