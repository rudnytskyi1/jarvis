"""The cinema scene by voice inside the 2-second budget (ТЗ сценарий 1, 15.1).

``tests/test_scene_e2e.py`` proves the scene runs end to end; this file proves
the sentence of the criterion is understood as that scene and that the turn is
timed against the budget - on the real hub path, with the model forbidden and
the room's engines stubbed, so the number is the hub's own work.
"""
from __future__ import annotations

import json

import pytest

from hub.scenes import cinema_request, plain_scene_text
from scripts import measure_scene_latency as measure_mod


@pytest.mark.parametrize("language", sorted(measure_mod.PHRASES))
def test_the_sentence_of_the_criterion_asks_for_the_cinema_scene(language):
    assert cinema_request(measure_mod.PHRASES[language]) is True


@pytest.mark.parametrize("text", [
    "включи свет",                       # the light, but nobody asked to put it out
    "расскажи про кино в городе",        # a film, but no light in the sentence
    "выключи компьютер и включи фильм",  # nothing about the light
    "выключи свет",                      # just the light
    "",
])
def test_a_request_that_is_not_the_cinema_sentence_is_left_alone(text):
    assert cinema_request(text) is False


def test_the_wake_word_and_politeness_do_not_hide_the_scene():
    for text in ("Rowan, выключи свет и включи фильм",
                 "эй Роуан, пожалуйста выключи свет и включи фильм",
                 "Rowan AI, turn off the light and put on a film",
                 "Hey Rowan, please apaga la luz y pon una película"):
        assert cinema_request(text) is True, text
        assert plain_scene_text(text) != ""


def test_the_measurement_runs_the_scene_without_the_model():
    report = measure_mod.measure(repeats=2)
    assert report["answer"] == "Cinema mode."
    assert report["applied_steps"] == [["lr-lamp", "on_off", False],
                                       ["lr-strip", "color_rgb", [0x22, 0x11, 0x00]],
                                       ["lr-tv", "on_off", True]]
    assert report["answered_turns"] == 2
    assert report["within_budget"] is True


def test_the_hub_turn_is_measured_against_the_2_second_budget():
    report = measure_mod.measure(repeats=3)
    assert report["budget_s"] == 2.0
    assert report["engines_included"] is False
    assert report["hub_turn_s"]["max_s"] < report["budget_s"], (
        "the hub's own work must leave room for the room's engines")
    assert report["seconds_left_s"] > 0


def test_the_declared_engine_costs_are_counted_in_the_total():
    report = measure_mod.measure(repeats=1, stt_ms=300, tts_ms=200)
    assert report["engines_included"] is True
    assert report["total_s"]["median_s"] >= 0.5
    assert report["within_budget"] is True


def test_an_engine_cost_that_overruns_a_stage_budget_is_reported_not_hidden():
    """900 ms of STT is past the hub's own 700 ms stage budget (ТЗ 15.1).

    The room still gets its answer - a late transcript beats silence - but the
    measurement has to say the stage was degraded instead of printing a
    comfortable total.
    """
    report = measure_mod.measure(repeats=1, stt_ms=900)
    assert report["answered_turns"] == 1, "a slow engine must not leave the room in silence"
    assert report["degraded_stages"] == ["stt"]
    assert report["stage_budget_met"] is False
    assert "note" in report


def test_the_command_prints_the_report_and_fails_over_budget(capsys):
    assert measure_mod.main(["--repeats", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["phrase"] == measure_mod.PHRASES["ru"]
    assert report["within_budget"] is True

    assert measure_mod.main(["--repeats", "1", "--stt-ms", "5000"]) == 1
    over = json.loads(capsys.readouterr().out)
    assert over["within_budget"] is False


def test_every_language_of_the_criterion_runs_the_scene():
    for language in sorted(measure_mod.PHRASES):
        report = measure_mod.measure(language=language, repeats=1)
        assert report["answer"] == "Cinema mode.", language
        assert report["within_budget"] is True, language
