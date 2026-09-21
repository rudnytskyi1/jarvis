"""Hub configuration: the multi-room section is additive (ТЗ section 4.7)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.config import Config, load_config


def test_single_room_config_still_loads_without_homes():
    cfg = load_config("config.example.yaml")
    assert cfg.homes == [], "an existing single-room config must keep working unchanged"
    assert cfg.server.port > 0


def test_multi_room_template_parses():
    cfg = load_config("config.hub.example.yaml")
    assert [home.home_id for home in cfg.homes] == ["livingroom", "dorm-max"]
    assert cfg.homes[0].name == "Living room"
    assert cfg.homes[0].quiet_hours.start == "23:00"
    assert cfg.homes[0].tz == "America/Chicago"
    assert cfg.homes[1].quiet_hours.start == ""


def test_duplicate_home_ids_are_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"homes": [{"home_id": "a", "name": "A"}, {"home_id": "a", "name": "B"}]})


@pytest.mark.parametrize("home_id", ["", "LivingRoom", "with space", "-leading"])
def test_home_id_pattern_is_enforced(home_id):
    with pytest.raises(ValidationError):
        Config.model_validate({"homes": [{"home_id": home_id, "name": "x"}]})


def test_invalid_timezone_is_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"homes": [{"home_id": "a", "name": "A", "tz": "Mars/Olympus"}]})


@pytest.mark.parametrize("quiet", [{"start": "23:00"}, {"start": "23:00", "end": "23:00"},
                                   {"start": "25:00", "end": "08:00"}])
def test_quiet_hours_are_validated(quiet):
    with pytest.raises(ValidationError):
        Config.model_validate({"homes": [{"home_id": "a", "name": "A", "quiet_hours": quiet}]})


def test_unknown_home_key_is_a_typo_not_silence():
    with pytest.raises(ValidationError):
        Config.model_validate({"homes": [{"home_id": "a", "name": "A", "tjmezone": "UTC"}]})
