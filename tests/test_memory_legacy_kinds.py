"""A legacy row must not blind every turn of the hub to its memory (F-414).

Owner's report (2026-09-22): every turn logged
``Could not read the memories table ('fact' is not a valid Kind)``. The rows
below are exactly what ``hub/legacy_migrate.py`` used to write before the typed
kinds existed: ``kind='fact'`` (sometimes ``'setting'``). ``MemoryFact`` refused
one of them, ``MemoryIndex.active()`` raised, and the whole memory block - all
63 facts - was dropped while the warning was printed on every turn.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hub import memories
from hub.migrations_runner import connect, migrate


def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    return conn


def write_legacy(conn, memory_id, *, scope, owner, kind, text, created="2026-09-17T03:11:13"):
    conn.execute(
        "INSERT INTO memories(memory_id, scope, owner_id, kind, text, weight, created_at) "
        "VALUES (?, ?, ?, ?, ?, 1.0, ?)",
        (memory_id, scope, owner, kind, text, created),
    )
    conn.commit()


def test_a_legacy_fact_row_is_translated_not_refused(tmp_path):
    conn = hub_db(tmp_path)
    write_legacy(conn, "legacy-mem-1", scope="person", owner="Anton", kind="fact",
                 text="Anton prefers the lamp on at night.")
    write_legacy(conn, "legacy-mem-2", scope="home", owner="livingroom", kind="fact",
                 text="The room's browser is Chrome.")
    write_legacy(conn, "legacy-mem-3", scope="person", owner="Anton", kind="setting",
                 text="Preferred browser: Google Chrome")

    rows = {fact.memory_id: fact for fact in memories.MemoryIndex(conn).active()}

    assert rows["legacy-mem-1"].kind is memories.Kind.PERSON_FACT
    assert rows["legacy-mem-2"].kind is memories.Kind.HOME_FACT
    assert rows["legacy-mem-3"].kind is memories.Kind.PREFERENCE


def test_one_unreadable_row_does_not_hide_the_rest_of_the_table(tmp_path):
    conn = hub_db(tmp_path)
    write_legacy(conn, "legacy-mem-ok", scope="person", owner="Anton", kind="person_fact",
                 text="Anton likes tea.")
    # "kind" itself is free text in the schema, so anything can be in there;
    # one such row must not blind the rest of the table.
    write_legacy(conn, "broken", scope="person", owner="Anton", kind="nonsense",
                 text="A row no model can read.")

    facts = memories.MemoryIndex(conn).active()

    assert [fact.memory_id for fact in facts] == ["legacy-mem-ok"]


def test_the_kind_filter_finds_both_spellings(tmp_path):
    conn = hub_db(tmp_path)
    write_legacy(conn, "typed", scope="person", owner="Anton", kind="person_fact",
                 text="A typed fact.")
    write_legacy(conn, "legacy", scope="person", owner="Anton", kind="fact",
                 text="A legacy fact.")

    found = memories.MemoryIndex(conn).active(kind=memories.Kind.PERSON_FACT)

    assert {fact.memory_id for fact in found} == {"typed", "legacy"}


@pytest.mark.parametrize("raw,scope,expected", [
    ("fact", "home", memories.Kind.HOME_FACT),
    ("fact", "person", memories.Kind.PERSON_FACT),
    ("setting", "person", memories.Kind.PREFERENCE),
    ("todo", "person", memories.Kind.TODO),
])
def test_kind_of_translates_the_legacy_names(raw, scope, expected):
    assert memories.kind_of(raw, scope=scope) is expected


def test_kind_of_still_refuses_a_value_nobody_wrote():
    with pytest.raises(ValueError):
        memories.kind_of("nonsense", scope="person")


def test_the_read_is_still_a_read_of_the_live_table(tmp_path):
    """Sanity: the timestamps of the schema are what the model expects."""
    conn = hub_db(tmp_path)
    conn.execute(
        "INSERT INTO memories(memory_id, scope, owner_id, kind, text, weight, created_at) "
        "VALUES ('now-row', 'person', 'Anton', 'person_fact', 'Written now.', 1.0, ?)",
        (datetime.now(UTC).isoformat(timespec="seconds"),),
    )
    conn.commit()
    fact = memories.MemoryIndex(conn).read("now-row")
    assert fact is not None and fact.text == "Written now."
