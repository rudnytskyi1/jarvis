"""P3-18 (F-415): a person's facts follow them, a home's facts stay home.

ТЗ 9.4: «Факты о человеке доступны в любом его доме; факты дома — только в
доме; гость не получает память дома». The isolation is checked through the
retrieval that feeds the prompt AND through the system-prompt channel of the
file store, with the flags of the phase still OFF: this is the test that has to
pass BEFORE memory v2 is switched on (ТЗ section 1, `PLAN.md` risks of phase 3).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from common.config import Config
from hub import app, memories, memory_search
from hub.migrations_runner import connect, migrate
from hub.session import Session
from hub.storage import Memory


def hub_db(tmp_path):
    """A real hub database with the schema of section 14, applied."""
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    return conn


def fact(text, *, owner="Anton", scope=memories.Scope.PERSON, kind=memories.Kind.PERSON_FACT):
    return memories.fact_from(scope=scope, owner_id=owner, kind=kind, text=text)


def home_fact(text, home="livingroom"):
    return fact(text, owner=home, scope=memories.Scope.HOME, kind=memories.Kind.HOME_FACT)


HUB_FACT = fact("The wifi password is on the fridge.", owner="",
                scope=memories.Scope.HUB, kind=memories.Kind.HOME_FACT)


# --- the rule itself --------------------------------------------------------


def test_a_persons_own_facts_follow_them_anywhere():
    row = fact("Anton keeps his keys in the top drawer.")
    assert memory_search.visible(row, person="Anton", home_id="livingroom")
    assert memory_search.visible(row, person="Anton", home_id="dorm-max")
    assert memory_search.visible(row, person="anton", home_id="dorm-max")
    assert not memory_search.visible(row, person="Max", home_id="livingroom")
    assert not memory_search.visible(row, person="", home_id="livingroom")


def test_a_home_fact_stays_in_its_home_and_reaches_no_guest():
    row = home_fact("The kettle lives on the desk.", home="livingroom")
    assert memory_search.visible(row, person="Anton", home_id="livingroom")
    assert not memory_search.visible(row, person="Anton", home_id="dorm-max")
    assert not memory_search.visible(row, person="Anton", home_id="")
    assert not memory_search.visible(row, person="Anton", home_id="livingroom",
                                     member=False), "a guest reads no home memory"
    assert not memory_search.visible(row, person="Max", home_id="dorm-max",
                                     member=False)


def test_a_hub_fact_is_for_everybody_including_a_guest():
    assert memory_search.visible(HUB_FACT, person="", home_id="livingroom")
    assert memory_search.visible(HUB_FACT, person="Anton", home_id="livingroom",
                                 member=False)
    assert memory_search.visible(HUB_FACT, person="Max", home_id="dorm-max", member=False)


# --- through the hub's own turn ---------------------------------------------


def _connection(monkeypatch, tmp_path, conn, *, name="Anton", role="admin",
                home="livingroom", memory=None):
    cfg = Config()
    cfg.server.permissions_enabled = False
    cfg.server.identity.enabled = False
    monkeypatch.setattr(app, "get_config", lambda: cfg)
    monkeypatch.setattr(app, "_memory", memory if memory is not None else Memory(tmp_path))
    monkeypatch.setattr(app, "_voices", None)
    monkeypatch.setattr(app, "_hub_db_path", lambda: tmp_path / "hub.db")
    connection = app.Connection(SimpleNamespace(client=None), cfg)
    connection.home_id = home
    connection.session = Session("room-pc", [], 8)
    connection._speaker_name, connection._speaker_role = name, role
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    app._hub_gateway()
    monkeypatch.setattr(app, "_hub_conn", conn)
    return connection, cfg


def _fill(conn, rows):
    index = memories.MemoryIndex(conn)
    for row in rows:
        index.write(row)
    return index


def _rows(conn, *, person="p-anton", home="livingroom"):
    """Register the person as a member of ``home`` (or of another house)."""
    conn.execute("INSERT OR IGNORE INTO homes(home_id, name) VALUES (?, ?)", (home, home))
    conn.execute("INSERT OR IGNORE INTO persons(person_id, display_name) VALUES (?, ?)",
                 (person, person.replace("p-", "").title()))
    conn.execute("INSERT OR IGNORE INTO memberships(person_id, home_id, role) VALUES (?, ?, 'admin')",
                 (person, home))
    conn.commit()


def test_a_member_reads_the_home_facts_and_a_guest_does_not(monkeypatch, tmp_path):
    conn = hub_db(tmp_path)
    try:
        _fill(conn, [home_fact("The kettle lives on the desk."),
                     fact("Anton keeps his keys in the top drawer."),
                     HUB_FACT])
        query = "Where is the kettle and my keys?"

        _rows(conn, person="p-anton", home="livingroom")
        member, _ = _connection(monkeypatch, tmp_path, conn, name="Anton")
        block = asyncio.run(member._memory_block(query))
        assert "kettle" in block and "keys in the top drawer" in block

        guest, _ = _connection(monkeypatch, tmp_path, conn, name="unknown", role="guest")
        block = asyncio.run(guest._memory_block(query))
        assert "kettle" not in block, "a guest never reads the home's memory"
        assert "keys in the top drawer" not in block, "nor somebody else's"
        assert "wifi password" in block, "the hub's own fact is for everybody"
    finally:
        conn.close()


def test_a_person_from_another_house_reads_their_own_facts_only(monkeypatch, tmp_path):
    conn = hub_db(tmp_path)
    try:
        _fill(conn, [home_fact("The kettle lives on the desk."),
                     fact("Max hates mint tea.", owner="Max"),
                     HUB_FACT])
        _rows(conn, person="p-max", home="dorm-max")  # a member of ANOTHER house
        visitor, _ = _connection(monkeypatch, tmp_path, conn, name="Max")
        block = asyncio.run(visitor._memory_block("Is there mint tea and a kettle?"))
        assert "mint tea" in block, "their own facts follow them"
        assert "kettle" not in block, "another home's facts do not"
        assert visitor._home_guest() is True
    finally:
        conn.close()


def test_a_shared_profile_reads_the_home_of_the_room_it_visits(monkeypatch, tmp_path):
    conn = hub_db(tmp_path)
    try:
        _fill(conn, [home_fact("The kettle lives on the desk.")])
        _rows(conn, person="p-max", home="dorm-max")
        # F-212: a person who is a member of TWO houses is seen in the second
        # one only when at least one of their memberships allows sharing.
        _rows(conn, person="p-max", home="livingroom")
        conn.execute("UPDATE memberships SET share_identity=1 WHERE person_id='p-max'"
                     " AND home_id='livingroom'")
        conn.commit()
        visitor, _ = _connection(monkeypatch, tmp_path, conn, name="Max")
        assert visitor._home_guest() is False
        assert "kettle" in asyncio.run(visitor._memory_block("Where is the kettle?"))
    finally:
        conn.close()


def test_a_hub_without_the_identity_database_keeps_the_old_behaviour(monkeypatch, tmp_path):
    """A named profile from before the identity database is not a guest."""
    conn = hub_db(tmp_path)
    try:
        _fill(conn, [home_fact("The kettle lives on the desk.")])
        connection, _ = _connection(monkeypatch, tmp_path, conn, name="Anton")
        assert connection._home_guest() is False
        assert "kettle" in asyncio.run(connection._memory_block("Where is the kettle?"))
    finally:
        conn.close()


def test_the_system_prompt_gives_a_guest_no_room_facts(monkeypatch, tmp_path):
    """The file store is the other channel into the prompt (P3-15 note)."""
    conn = hub_db(tmp_path)
    try:
        memory = Memory(data_dir=tmp_path)
        memory.add("The light switch is by the door.")          # a room fact
        memory.add("Anton wants short answers.", "", key="speech.verbosity", value="brief")
        memory.add("Anton keeps his keys in the top drawer.", "Anton")

        member, _ = _connection(monkeypatch, tmp_path, conn, name="Anton", memory=memory)
        assert member._prompt_memory() == memory.effective("Anton")
        assert any("light switch" in row for row in member._prompt_memory())

        guest, _ = _connection(monkeypatch, tmp_path, conn, name="unknown",
                               role="guest", memory=memory)
        rows = guest._prompt_memory()
        assert not any("light switch" in row for row in rows), "no room memory for a guest"
        assert any("short answers" in row for row in rows), "the room's settings still apply"
        assert not any("keys in the top drawer" in row for row in rows)
    finally:
        conn.close()


def test_the_isolation_holds_with_every_flag_of_the_phase_off(monkeypatch, tmp_path):
    """The guarantee must not depend on the switches P3-16/P3-17 introduced."""
    conn = hub_db(tmp_path)
    try:
        cfg = Config()
        assert cfg.server.memory.dialogs_from_db is False
        assert cfg.server.memory.retrieval_enabled is True
        cfg.server.permissions_enabled = False
        cfg.server.identity.enabled = False
        monkeypatch.setattr(app, "get_config", lambda: cfg)
        monkeypatch.setattr(app, "_memory", Memory(data_dir=tmp_path))
        monkeypatch.setattr(app, "_hub_db_path", lambda: tmp_path / "hub.db")
        connection = app.Connection(SimpleNamespace(client=None), cfg)
        connection.home_id = "livingroom"
        connection.session = Session("room-pc", [], 8)
        connection._speaker_name, connection._speaker_role = "unknown", "guest"
        app._hub_gateway()
        monkeypatch.setattr(app, "_hub_conn", conn)
        _fill(conn, [home_fact("The kettle lives on the desk."),
                     fact("Anton keeps his keys in the top drawer.")])
        block = asyncio.run(connection._memory_block("kettle keys"))
        assert block == "", "a guest reads neither the home's facts nor anybody's"
        prefix = connection._turn_prefix(datetime.now(UTC), "kettle keys", memory=block)
        assert "[memory:" not in prefix
    finally:
        conn.close()
