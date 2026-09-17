"""server/storage.py + server/app.py: memory scoped to a person (SPEC v1.7).

A fact is either about ONE person — their preference, their habit, how they
want Rowan to behave with them — or about the room and everybody in it.
Personal facts are read back only while that person is speaking; room facts
live in the system prompt, where they must stay byte-identical between turns
so the model's prompt cache survives.
"""
from server.app import _mentions_the_speaker
from server.storage import Memory


def _memory(tmp_path) -> Memory:
    return Memory(data_dir=tmp_path)


def test_a_fact_with_no_owner_belongs_to_the_room(tmp_path):
    memory = _memory(tmp_path)
    memory.add("The light switch is by the door.")
    assert memory.facts() == ["The light switch is by the door."]


def test_a_personal_fact_is_not_a_room_fact(tmp_path):
    memory = _memory(tmp_path)
    memory.add("Prefers the lights dim in the evening.", "Anton")
    # The system prompt must not learn it: it is Anton's, not the room's.
    assert memory.facts() == []
    assert memory.facts("Anton") == ["Prefers the lights dim in the evening."]


def test_one_person_never_sees_another_persons_facts(tmp_path):
    memory = _memory(tmp_path)
    memory.add("Prefers tea.", "Anton")
    memory.add("Prefers coffee.", "Bob")
    assert memory.facts("Anton") == ["Prefers tea."]
    assert memory.facts("Bob") == ["Prefers coffee."]


def test_facts_come_back_oldest_first(tmp_path):
    memory = _memory(tmp_path)
    memory.add("First.", "Anton")
    memory.add("Second.", "Anton")
    assert memory.facts("Anton") == ["First.", "Second."]


def test_people_lists_only_those_with_facts_of_their_own(tmp_path):
    memory = _memory(tmp_path)
    memory.add("A room fact.")
    memory.add("Prefers tea.", "Anton")
    memory.add("Also likes tea.", "Anton")
    memory.add("Prefers coffee.", "Bob")
    assert memory.people() == ["Anton", "Bob"]


def test_an_owner_name_is_whitespace_normalised(tmp_path):
    memory = _memory(tmp_path)
    memory.add("Prefers tea.", "  Anton  ")
    assert memory.facts("Anton") == ["Prefers tea."]


def test_old_records_without_a_person_still_load_as_room_facts(tmp_path):
    # memory.jsonl predates the person field; those lines must keep working.
    path = tmp_path / "memory.jsonl"
    path.write_text(
        '{"ts": "2026-01-01T00:00:00", "fact": "An old fact."}\n', encoding="utf-8"
    )
    memory = _memory(tmp_path)
    assert memory.facts() == ["An old fact."]
    assert memory.facts("Anton") == []


def test_an_unreadable_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "memory.jsonl"
    path.write_text(
        'not json at all\n{"fact": "A good fact.", "person": "Anton"}\n',
        encoding="utf-8",
    )
    assert _memory(tmp_path).facts("Anton") == ["A good fact."]


# -- attributing a fact the model forgot to label ----------------------------

def test_first_person_wording_is_recognised_as_the_speakers_own():
    for fact in [
        "I prefer the lights dim.",
        "My lectures start at nine.",
        "Call me Tony.",
        "I've stopped drinking coffee.",
        "Wake me at seven, not me at eight",
    ]:
        assert _mentions_the_speaker(fact), fact


def test_room_wording_is_not_mistaken_for_a_personal_fact():
    for fact in [
        "The light switch is by the door.",
        "The couch is against the blue wall.",
        "Lectures in this building start at nine.",
        "",
    ]:
        assert not _mentions_the_speaker(fact), fact
