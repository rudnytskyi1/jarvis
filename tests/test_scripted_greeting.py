"""server/app.py v1.7: greetings are spoken from a script, not generated.

Going through the model put 3-5 seconds (once 61) between recognising a face
and saying anything, by which time a stranger has stopped looking at the camera
and the greeting reads as random noise instead. The script says exactly what the
model was being asked to say, instantly.
"""
from types import SimpleNamespace

import pytest

from hub.app import (
    LABEL_UNKNOWN,
    SCRIPTED_GREETING_KNOWN,
    SCRIPTED_GREETING_UNKNOWN,
    SCRIPTED_GREETING_UNKNOWN_WITH_COMPANY,
    Connection,
    PresenceTracker,
)


def _conn(greeting_llm: bool = False) -> Connection:
    conn = Connection.__new__(Connection)
    conn.presence = PresenceTracker(ttl_s=30.0)
    conn._greeting_variant = 0
    conn.cfg = SimpleNamespace(
        server=SimpleNamespace(face=SimpleNamespace(greeting_llm=greeting_llm))
    )
    return conn


def test_scripted_is_the_default_and_the_flag_flips_it():
    assert _conn()._greeting_llm is False
    assert _conn(greeting_llm=True)._greeting_llm is True


def test_a_stranger_alone_is_introduced_to_and_asked_their_name():
    text = _conn()._scripted_greeting(LABEL_UNKNOWN)
    assert text in SCRIPTED_GREETING_UNKNOWN
    assert "Rowan" in text
    assert "name" in text.lower()


def test_every_stranger_line_introduces_rowan_and_asks_the_name():
    for line in SCRIPTED_GREETING_UNKNOWN:
        assert "Rowan" in line
        assert "name" in line.lower() or "call you" in line.lower()


def test_a_stranger_beside_somebody_known_is_told_who_they_are_with():
    conn = _conn()
    conn.presence.note_faces(["Anton", LABEL_UNKNOWN])
    text = conn._scripted_greeting(LABEL_UNKNOWN)
    assert "Anton" in text
    assert "Rowan" in text


def test_several_known_people_are_all_named():
    conn = _conn()
    conn.presence.note_faces(["Anton", "Drew", LABEL_UNKNOWN])
    text = conn._scripted_greeting(LABEL_UNKNOWN)
    assert "Anton" in text and "Drew" in text


def test_a_known_face_is_greeted_by_name():
    text = _conn()._scripted_greeting("Anton")
    assert "Anton" in text
    assert "Rowan" not in text  # they already know who he is


def test_the_wording_rotates_so_the_room_does_not_hear_one_sentence():
    conn = _conn()
    seen = {conn._scripted_greeting("Anton") for _ in range(len(SCRIPTED_GREETING_KNOWN))}
    assert len(seen) == len(SCRIPTED_GREETING_KNOWN)


def test_rotation_wraps_around_without_failing():
    conn = _conn()
    for _ in range(len(SCRIPTED_GREETING_UNKNOWN) * 3 + 1):
        assert conn._scripted_greeting(LABEL_UNKNOWN)


@pytest.mark.parametrize("line", SCRIPTED_GREETING_UNKNOWN_WITH_COMPANY)
def test_company_lines_have_exactly_the_names_placeholder(line):
    assert "{names}" in line
    assert "{name}" not in line.replace("{names}", "")


@pytest.mark.parametrize("line", SCRIPTED_GREETING_KNOWN)
def test_known_lines_have_the_name_placeholder(line):
    assert "{name}" in line


def test_no_line_is_long_enough_to_be_annoying_out_loud():
    for line in (
        SCRIPTED_GREETING_UNKNOWN
        + SCRIPTED_GREETING_UNKNOWN_WITH_COMPANY
        + SCRIPTED_GREETING_KNOWN
    ):
        assert len(line) <= 120, line
