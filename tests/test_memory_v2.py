"""P3-15 (F-414): a fact is a typed row of ``memories``, not just a sentence.

TZ 9.4 asks for a kind, a scope, a TTL and an embedding in the table of section
14. The file store keeps feeding the prompt; this table is what the hybrid
search (P3-16) and the nightly consolidation (P3-19) will read.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from common import protocol as proto
from common.config import Config
from hub import app, memories, vectors
from hub.migrations_runner import connect, migrate
from hub.session import Session
from hub.storage import Memory


def hub_db(tmp_path):
    """A real hub database: the schema of section 14, applied."""
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    return conn


# --- the model --------------------------------------------------------------


def test_a_fact_carries_its_kind_scope_owner_and_time():
    fact = memories.fact_from(scope=memories.Scope.PERSON, owner_id=" Anton ",
                              kind=memories.Kind.PERSON_FACT, text=" Likes tea. ")
    assert fact.owner_id == "Anton" and fact.text == "Likes tea."
    assert fact.kind is memories.Kind.PERSON_FACT and fact.weight == 1.0
    assert fact.expires_at is None and not fact.expired()
    assert fact.dim is None and fact.vector is None
    assert len(fact.memory_id) >= 20, "ids are ULIDs, like the rest of the hub"


@pytest.mark.parametrize("kind", ["person_fact", "home_fact", "preference", "todo", "event"])
def test_every_kind_of_the_tz_is_accepted(kind):
    fact = memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                              kind=memories.Kind(kind), text="A fact.")
    assert fact.kind.value == kind


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValidationError):
        memories.MemoryFact(scope=memories.Scope.PERSON, owner_id="Anton",
                            kind="rumour", text="Something.")


def test_a_scoped_fact_needs_an_owner_and_a_hub_fact_needs_none():
    with pytest.raises(ValidationError):
        memories.MemoryFact(scope=memories.Scope.PERSON, kind="person_fact", text="x")
    with pytest.raises(ValidationError):
        memories.MemoryFact(scope=memories.Scope.HOME, kind="home_fact", text="x")
    with pytest.raises(ValidationError):
        memories.MemoryFact(scope=memories.Scope.HUB, owner_id="livingroom",
                            kind="home_fact", text="x")
    fact = memories.MemoryFact(scope=memories.Scope.HUB, kind="home_fact", text="The hub speaks Russian.")
    assert fact.owner_id == ""


def test_an_empty_or_overlong_fact_is_refused_or_cut():
    with pytest.raises(ValidationError):
        memories.fact_from(scope=memories.Scope.HUB, owner_id="", kind=memories.Kind.HOME_FACT,
                           text="   ")
    long_text = memories.fact_from(scope=memories.Scope.HUB, owner_id="",
                                   kind=memories.Kind.HOME_FACT, text="x" * 900)
    assert len(long_text.text) == memories.MAX_TEXT_CHARS


def test_a_symbolic_weight_is_refused():
    with pytest.raises(ValidationError):
        memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                           kind=memories.Kind.PERSON_FACT, text="x", weight=1.5)
    with pytest.raises(ValidationError):
        memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                           kind=memories.Kind.PERSON_FACT, text="x", weight=-0.1)


def test_the_ttl_becomes_the_moment_the_fact_expires():
    now = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    fact = memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                              kind=memories.Kind.TODO, text="Hand in the lab report.",
                              ttl_s=3600, now=now)
    assert fact.expires_at == now + timedelta(hours=1)
    assert not fact.expired(now=now)
    assert fact.expired(now=now + timedelta(hours=1, seconds=1))
    assert not memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                                  kind=memories.Kind.PERSON_FACT, text="x").expired(
        now=now + timedelta(days=3650)), "a fact without a TTL never expires"


def test_a_fact_that_is_already_over_is_refused():
    now = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    with pytest.raises(ValidationError):
        memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                           kind=memories.Kind.EVENT, text="Yesterday's party.",
                           ttl_s=0, now=now)


def test_an_embedding_must_match_its_dimension():
    fact = memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                              kind=memories.Kind.PERSON_FACT, text="x",
                              vector=[0.1, 0.2, 0.3])
    assert fact.dim == 3 and len(fact.vector or b"") == 12
    with pytest.raises(ValidationError):
        memories.MemoryFact(scope=memories.Scope.PERSON, owner_id="Anton",
                            kind="person_fact", text="x", vector=b"\0" * 12, dim=4)
    with pytest.raises(ValidationError):
        memories.MemoryFact(scope=memories.Scope.PERSON, owner_id="Anton",
                            kind="person_fact", text="x", vector=b"\0" * 12)


def test_the_models_are_strict():
    with pytest.raises(ValidationError):
        memories.MemoryFact(scope=memories.Scope.HUB, kind="home_fact", text="x", bogus=1)


def test_the_kind_follows_what_was_asked_for():
    assert memories.kind_for(key="apps.browser") is memories.Kind.PREFERENCE
    assert memories.kind_for(shared=True) is memories.Kind.HOME_FACT
    assert memories.kind_for() is memories.Kind.PERSON_FACT


# --- the table --------------------------------------------------------------


def test_a_fact_round_trips_through_the_memories_table(tmp_path):
    conn = hub_db(tmp_path)
    try:
        index = memories.MemoryIndex(conn)
        fact = memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                                  kind=memories.Kind.PREFERENCE, text="Anton drinks tea.",
                                  ttl_s=7200, vector=[0.5, 0.25, 0.0])
        index.write(fact)

        read = index.read(fact.memory_id)
        assert read is not None
        assert read.scope is memories.Scope.PERSON and read.owner_id == "Anton"
        assert read.kind is memories.Kind.PREFERENCE and read.text == "Anton drinks tea."
        assert read.vector == fact.vector and read.dim == 3
        assert read.expires_at == fact.expires_at
        row = conn.execute("SELECT scope, owner_id, kind, text, dim, weight FROM memories"
                           " WHERE memory_id=?", (fact.memory_id,)).fetchone()
        assert row == ("person", "Anton", "preference", "Anton drinks tea.", 3, 1.0)
    finally:
        conn.close()


def test_active_hides_what_is_expired_and_purge_removes_it(tmp_path):
    conn = hub_db(tmp_path)
    try:
        index = memories.MemoryIndex(conn)
        now = datetime.now(UTC)
        live = memories.fact_from(scope=memories.Scope.HOME, owner_id="livingroom",
                                  kind=memories.Kind.HOME_FACT, text="The kettle is on the desk.",
                                  now=now)
        spent = memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                                   kind=memories.Kind.TODO, text="Buy milk.",
                                   ttl_s=60, now=now - timedelta(minutes=5))
        index.write(live)
        index.write(spent)

        assert index.count() == 1 and index.count(include_expired=True) == 2
        assert [fact.memory_id for fact in index.active()] == [live.memory_id]
        assert [fact.memory_id for fact in index.active(scope=memories.Scope.HOME,
                                                        owner_id="livingroom")] == [live.memory_id]
        assert index.active(scope=memories.Scope.PERSON, owner_id="Anton") == []
        assert index.purge_expired() == 1
        assert index.count(include_expired=True) == 1
        assert index.read(spent.memory_id) is None
    finally:
        conn.close()


def test_a_persons_facts_are_found_by_their_owner(tmp_path):
    conn = hub_db(tmp_path)
    try:
        index = memories.MemoryIndex(conn)
        for owner, text in (("Anton", "Anton likes tea."), ("Max", "Max likes coffee."),
                            ("Anton", "Anton studies at nine.")):
            index.write(memories.fact_from(scope=memories.Scope.PERSON, owner_id=owner,
                                           kind=memories.Kind.PERSON_FACT, text=text))
        mine = index.active(scope=memories.Scope.PERSON, owner_id="anton")
        assert {fact.text for fact in mine} == {"Anton likes tea.", "Anton studies at nine."}
        assert index.active(kind=memories.Kind.PREFERENCE) == []
    finally:
        conn.close()


def test_a_deleted_fact_is_gone_from_the_table(tmp_path):
    conn = hub_db(tmp_path)
    try:
        index = memories.MemoryIndex(conn)
        fact = memories.fact_from(scope=memories.Scope.HUB, owner_id="",
                                  kind=memories.Kind.HOME_FACT, text="The wifi password is on the fridge.")
        index.write(fact)
        assert index.delete(fact.memory_id) is True
        assert index.delete(fact.memory_id) is False
        assert index.count(include_expired=True) == 0
    finally:
        conn.close()


def test_an_embedding_is_mirrored_into_the_vector_index(tmp_path):
    if vectors.locate_extension() is None:
        pytest.skip("sqlite-vec is not available in this environment")
    conn = hub_db(tmp_path)
    try:
        vectors.ensure_index(conn, "memory", dimension=3)
        index = memories.MemoryIndex(conn)
        fact = memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                                  kind=memories.Kind.PERSON_FACT, text="Anton likes tea.",
                                  vector=[1.0, 0.0, 0.0])
        index.write(fact)
        assert vectors.count(conn, "memory") == 1
        hits = vectors.search(conn, "memory", [1.0, 0.0, 0.0], limit=1)
        assert [hit.row_id for hit in hits] == [fact.memory_id]
        index.delete(fact.memory_id)
        assert vectors.count(conn, "memory") == 0
    finally:
        conn.close()


def test_a_hub_without_the_extension_still_stores_the_fact(tmp_path, monkeypatch):
    conn = hub_db(tmp_path)

    def refuse(*args, **kwargs):
        raise vectors.VectorExtensionUnavailable("no index here")

    monkeypatch.setattr(vectors, "store", refuse)
    try:
        index = memories.MemoryIndex(conn)
        fact = memories.fact_from(scope=memories.Scope.PERSON, owner_id="Anton",
                                  kind=memories.Kind.PERSON_FACT, text="Anton likes tea.",
                                  vector=[1.0, 0.0, 0.0])
        index.write(fact)
        assert index.read(fact.memory_id) is not None, "the row is authoritative"
    finally:
        conn.close()


def test_a_row_written_by_the_schema_date_default_is_readable(tmp_path):
    """``created_at`` may come from ``datetime('now')``, not from this model."""
    conn = hub_db(tmp_path)
    try:
        conn.execute(
            "INSERT INTO memories(memory_id, scope, owner_id, kind, text)"
            " VALUES ('m-1', 'hub', '', 'home_fact', 'An older row.')")
        conn.commit()
        fact = memories.MemoryIndex(conn).read("m-1")
        assert fact is not None and fact.created_at.tzinfo is not None
        assert fact.kind is memories.Kind.HOME_FACT
    finally:
        conn.close()


# --- the remember tool writes the row --------------------------------------


def _turn(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(app, "_memory", Memory(data_dir=tmp_path))
    monkeypatch.setattr(app, "_voices", None)
    cfg = Config()
    cfg.server.permissions_enabled = False
    # A fact saved for the whole room is a privileged call (F-208), and that
    # gate wants a face or a phone. This file tests the write-through, so the
    # gate is off here the way the other tool tests turn it off.
    cfg.server.identity.enabled = False
    connection = app.Connection(SimpleNamespace(client=None), cfg)
    connection.home_id = "livingroom"
    connection.session = Session("room-pc", [], 8)
    connection._speaker_name, connection._speaker_role = "Anton", "admin"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    # The hub prepares its own database the first time a turn needs it, and
    # that puts ITS connection into the global. Do it once here, then hand the
    # global the test's database (the hub keeps what it is given).
    app._hub_gateway()
    monkeypatch.setattr(app, "_hub_conn", conn)
    return connection


def test_remember_writes_a_person_fact_to_the_table(monkeypatch, tmp_path):
    conn = hub_db(tmp_path)
    try:
        connection = _turn(monkeypatch, tmp_path, conn)
        result = asyncio.run(connection._execute_tool(
            "remember", {"fact": "Anton studies at nine.", "about": "me"}))
        assert result["ok"] is True
        rows = memories.MemoryIndex(conn).active()
        assert [(fact.scope, fact.owner_id, fact.kind, fact.text) for fact in rows] == [
            (memories.Scope.PERSON, "Anton", memories.Kind.PERSON_FACT, "Anton studies at nine.")]
    finally:
        conn.close()


def test_remember_writes_a_preference_and_a_home_fact(monkeypatch, tmp_path):
    conn = hub_db(tmp_path)
    try:
        connection = _turn(monkeypatch, tmp_path, conn)
        asyncio.run(connection._execute_tool("remember", {
            "fact": "Anton wants Chrome.", "about": "me",
            "key": "apps.browser", "value": "Google Chrome"}))
        asyncio.run(connection._execute_tool("remember", {
            "fact": "The light switch is by the door.", "scope": "global"}))

        rows = memories.MemoryIndex(conn).active()
        by_kind = {fact.kind: fact for fact in rows}
        assert by_kind[memories.Kind.PREFERENCE].owner_id == "Anton"
        assert by_kind[memories.Kind.HOME_FACT].scope is memories.Scope.HOME
        assert by_kind[memories.Kind.HOME_FACT].owner_id == "livingroom"
    finally:
        conn.close()


def test_a_hub_without_a_database_still_remembers(monkeypatch, tmp_path, caplog):
    connection = _turn(monkeypatch, tmp_path, None)
    result = asyncio.run(connection._execute_tool("remember", {"fact": "Anton studies at nine.",
                                                              "about": "me"}))
    assert result["ok"] is True
    assert connection.session is not None


def test_a_broken_table_does_not_lose_the_remembered_fact(monkeypatch, tmp_path):
    class Broken:
        def execute(self, *args, **kwargs):
            raise RuntimeError("the table is gone")

        def commit(self):
            raise RuntimeError("the table is gone")

    connection = _turn(monkeypatch, tmp_path, Broken())
    result = asyncio.run(connection._execute_tool("remember", {"fact": "Anton studies at nine.",
                                                              "about": "me"}))
    assert result["ok"] is True, "the file store already saved it"


def test_the_protocol_model_of_a_device_state_is_still_the_one_for_lights():
    """Sanity: this file's schema is the protocol's, not a private copy."""
    assert proto.DeviceState(device_id="lamp", capability="on_off", value=True).value is True
