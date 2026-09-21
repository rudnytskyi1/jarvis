"""ULID identifiers for utterances (ТЗ 4.5)."""
from __future__ import annotations

import re

import pytest

from common import ids
from common.ids import (
    ULID_ALPHABET,
    ULID_LENGTH,
    is_ulid,
    new_ulid,
    ulid_timestamp_ms,
)

ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
#: Far enough in the future that the module's clock guard cannot interfere.
FUTURE_MS = 4_000_000_000_000


@pytest.fixture(autouse=True)
def reset_generator(monkeypatch):
    """Each test starts from a virgin generator, whatever ran before it."""
    monkeypatch.setattr(ids, "_last_ms", 0)
    monkeypatch.setattr(ids, "_last_random", 0)


def test_a_new_ulid_is_26_crockford_characters():
    value = new_ulid()
    assert len(value) == ULID_LENGTH
    assert ULID_PATTERN.match(value), value
    assert is_ulid(value)
    # I, L, O and U are not part of the alphabet: a hand-copied id stays readable.
    assert not set("ILOU") & set(value)
    assert set(value) <= set(ULID_ALPHABET)


def test_ulids_are_unique():
    values = {new_ulid() for _ in range(5000)}
    assert len(values) == 5000


def test_ulids_created_in_the_same_millisecond_still_sort_in_order():
    stamp = FUTURE_MS
    values = [new_ulid(stamp) for _ in range(200)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert all(ulid_timestamp_ms(value) == stamp for value in values)


def test_ulids_sort_by_creation_time_across_milliseconds():
    first = new_ulid(FUTURE_MS)
    second = new_ulid(FUTURE_MS + 1)
    assert first < second
    assert ulid_timestamp_ms(second) - ulid_timestamp_ms(first) == 1


def test_a_clock_that_steps_backwards_does_not_rewind_the_sequence(monkeypatch):
    ticks = iter([FUTURE_MS, FUTURE_MS - 1_000, FUTURE_MS - 2_000])
    monkeypatch.setattr(ids, "_now_ms", lambda: next(ticks))

    values = [new_ulid() for _ in range(3)]
    assert values == sorted(values)
    assert len(set(values)) == 3
    assert {ulid_timestamp_ms(value) for value in values} == {FUTURE_MS}


@pytest.mark.parametrize(
    "value",
    ["", "not-a-ulid", "0" * 25, "0" * 27, "I" * 26, "U" * 26, "01ARZ3NDEKTSV4RRFFQ69G5FA9!"],
)
def test_malformed_values_are_not_ulids(value):
    assert is_ulid(value) is False
    with pytest.raises(ValueError):
        ulid_timestamp_ms(value)


def test_is_ulid_accepts_only_strings():
    assert is_ulid(None) is False
    assert is_ulid(26) is False
    assert is_ulid(b"01ARZ3NDEKTSV4RRFFQ69G5FAV") is False
    # The canonical form is upper case, but a lower-case copy still decodes.
    assert is_ulid("01arz3ndektsv4rrffq69g5fav")


def test_a_timestamp_outside_the_48_bit_range_is_rejected():
    with pytest.raises(ValueError):
        new_ulid(1 << 49)
    with pytest.raises(ValueError):
        new_ulid(-1)
