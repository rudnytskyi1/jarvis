"""P3-12 (F-412): the prompt carries the speaker's profile and the home state.

The TZ asks for the speaker (name, language, role, preferences, the last facts
about them) and the home (who is in the room, the light, the time, the quiet
hours). It rides in the per-turn prefix of the user message, not in the system
prompt, because the system prompt has to stay byte-identical for the model's
prompt cache (see ``hub/session.py``). Everything here is measured against what
the hub really knows: a light whose state nobody reported is not called "off".
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from common.config import Config, HomeConfig
from hub import app
from hub.session import Session
from hub.speaker_context import (
    KIND_LIGHT,
    KIND_OTHER,
    KIND_SWITCH,
    MAX_PREFERENCES,
    HomeDevice,
    HomeState,
    SpeakerProfile,
    device_kind,
    home_state_from,
    local_time,
    profile_from,
    render_home,
    render_prefix,
    render_profile,
)
from hub.storage import Memory


def _memory(tmp_path) -> Memory:
    memory = Memory(data_dir=tmp_path)
    memory.add("Prefers oolong tea in the evening.", "Anton")
    memory.add("Anton likes his browser in English.", "Anton",
               key="apps.browser", value="Google Chrome")
    memory.add("Anton wants short answers.", "", key="speech.verbosity", value="brief")
    memory.add("The light switch is by the door.")
    return memory


# --- the speaker's profile -------------------------------------------------


def test_the_profile_carries_the_name_the_role_and_the_language():
    profile = profile_from(name="Anton", role="admin", language="ru", memory=None)
    assert profile.name == "Anton" and profile.role == "admin" and profile.language == "ru"
    line = render_profile(profile)
    assert "speaker: Anton" in line and "role: admin" in line and "language: ru" in line


def test_preferences_are_the_keyed_settings_with_the_personal_one_winning(tmp_path):
    memory = _memory(tmp_path)
    memory.add("Anton wants brief answers.", "Anton", key="speech.verbosity", value="brief")
    memory.add("The room wants long answers.", "", key="speech.verbosity", value="long")
    profile = profile_from(name="Anton", role="user", language="en", memory=memory)
    assert profile.preferences["apps.browser"] == "Google Chrome"
    assert profile.preferences["speech.verbosity"] == "brief", "the person's own setting wins"
    assert 'apps.browser="Google Chrome"' in render_profile(profile)


def test_only_the_newest_facts_ride_in_the_profile(tmp_path):
    memory = Memory(data_dir=tmp_path)
    for number in range(5):
        memory.add(f"Fact number {number}.", "Anton")
    profile = profile_from(name="Anton", role="user", language="en", memory=memory)
    assert profile.facts == ["Fact number 2.", "Fact number 3.", "Fact number 4."]


def test_a_person_with_nothing_remembered_gets_no_invented_facts(tmp_path):
    memory = Memory(data_dir=tmp_path)
    profile = profile_from(name="Anton", role="user", language="en", memory=memory)
    assert profile.preferences == {} and profile.facts == []
    assert "recent facts" not in render_profile(profile)


def test_an_unrecognised_voice_gets_no_personal_facts(tmp_path):
    memory = _memory(tmp_path)
    profile = profile_from(name="unknown", role="unknown", language="en", memory=memory)
    assert profile.facts == []
    # A global setting belongs to the whole room and reaches everybody; a
    # personal one must not.
    assert profile.preferences == {"speech.verbosity": "brief"}
    line = render_profile(profile)
    assert "speaker: unknown" in line and "role: unknown" in line
    assert "oolong" not in line, "a stranger never inherits somebody's facts"


def test_a_fact_is_quoted_and_kept_on_one_line():
    profile = SpeakerProfile(name="Anton", role="user", language="en",
                             facts=['He said "the light" and\nleft'])
    line = render_profile(profile)
    assert "\n" not in line
    assert r'recent facts: "He said \"the light\" and left"' in line
    assert "]" not in line, "a fact cannot close the block early"


def test_an_empty_profile_says_nothing():
    assert render_profile(SpeakerProfile()) == ""


# --- the state of the home -------------------------------------------------


def _home(**values) -> HomeState:
    base = dict(home_id="livingroom", name="Living room",
                devices=[HomeDevice(name="lamp", kind=KIND_LIGHT, area="desk")])
    return HomeState(**{**base, **values})


def test_the_home_state_names_the_room_and_who_is_in_it():
    line = render_home(_home(people=["Anton", "Max"]))
    assert line.startswith("Living room")
    assert "in the room: Anton, Max" in line


def test_people_the_camera_did_not_recognise_are_counted_not_named():
    assert "1 unknown person" in render_home(_home(unknown_people=1))
    assert "2 unknown people" in render_home(_home(unknown_people=2))
    assert "unknown" not in render_home(_home(people=["Anton"]))


def test_the_lights_are_listed_without_inventing_their_state():
    assert "lights: lamp (on/off not reported by the room)" in render_home(_home())
    assert "not reported" not in render_home(_home(lights_known=True))


def test_quiet_hours_are_shown_with_whether_they_are_on():
    assert "quiet hours: 23:00-08:00 (off)" in render_home(
        _home(quiet_hours="23:00-08:00", quiet_now=False))
    assert "keep it quiet" in render_home(_home(quiet_hours="23:00-08:00", quiet_now=True))
    assert "quiet hours" not in render_home(_home())


def test_a_room_with_nothing_to_say_renders_nothing():
    assert render_home(HomeState()) == ""


def test_the_devices_are_split_by_what_they_are():
    assert device_kind("magichome") == KIND_LIGHT
    assert device_kind("yeelight") == KIND_LIGHT
    assert device_kind("switchbot_bot") == KIND_SWITCH
    assert device_kind("unknown-thing") == KIND_OTHER
    line = render_home(_home(devices=[HomeDevice(name="lamp", kind=KIND_LIGHT),
                                      HomeDevice(name="wall switch", kind=KIND_SWITCH),
                                      HomeDevice(name="speaker", kind=KIND_OTHER)]))
    assert "lights: lamp" in line and "switches: wall switch" in line
    assert "speaker" not in line, "a plain device is not called a light"


def test_the_rooms_own_clock_is_used():
    # 21:04 in Chicago is 04:04 the next day in Berlin.
    moment = datetime(2026, 9, 21, 21, 4).timestamp()
    assert local_time(moment, "Europe/Berlin").startswith("2026-09-22T04:0")
    assert local_time(moment, "not-a-zone").startswith("2026-09-21T21:0")


def test_the_home_state_comes_from_the_room_config_and_the_client():
    config_home = HomeConfig(home_id="livingroom", name="Living room", tz="Europe/Berlin",
                             quiet_hours={"start": "23:00", "end": "08:00"})
    state = home_state_from(
        home_id="livingroom", config_home=config_home, people=["Anton", "Anton", ""],
        unknown_people="2", devices=[{"name": "lamp", "type": "magichome", "area": "desk"},
                                     {"name": ""}, "not a device"],
        moment=datetime(2026, 9, 21, 21, 4).timestamp())
    assert state.name == "Living room" and state.people == ["Anton"]
    assert state.unknown_people == 2 and state.quiet_hours == "23:00-08:00"
    assert state.quiet_now is True
    assert [device.name for device in state.devices] == ["lamp"]
    assert state.devices[0].kind == KIND_LIGHT and state.devices[0].area == "desk"


def test_a_home_without_quiet_hours_is_not_given_any():
    state = home_state_from(home_id="livingroom",
                            config_home=HomeConfig(home_id="livingroom", name="Room"))
    assert state.quiet_hours == "" and state.quiet_now is False
    assert home_state_from(home_id="", config_home=None).home_id == ""


# --- the prefix the model is shown -----------------------------------------


def test_the_prefix_keeps_its_shape_and_ends_with_the_persons_words():
    profile = SpeakerProfile(name="Anton", role="admin", language="en",
                             preferences={"apps.browser": "Google Chrome"},
                             facts=["Prefers tea."])
    prefix = render_prefix(at=datetime(2026, 9, 21, 21, 4, 11), profile=profile,
                           home=_home(people=["Anton"], quiet_hours="23:00-08:00"),
                           live_view="Room now: Anton at the desk",
                           language="Answer in Russian.", text="turn the light on")
    assert prefix.startswith("[at 2026-09-21T21:04:11 | speaker: Anton | role: admin")
    assert "[home: Living room | in the room: Anton | lights: lamp" in prefix
    assert "[room: Room now: Anton at the desk]" in prefix
    assert "[Answer in Russian.] turn the light on" in prefix
    assert prefix.endswith("turn the light on")


def test_the_models_are_strict():
    with pytest.raises(ValidationError):
        SpeakerProfile(name="Anton", bogus="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        HomeDevice(name="lamp", bogus="on")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        HomeState(unknown_people=-1)
    with pytest.raises(ValidationError):
        HomeDevice(name="")


def test_the_preference_list_is_capped(tmp_path):
    memory = Memory(data_dir=tmp_path)
    for number in range(MAX_PREFERENCES + 3):
        memory.add(f"Setting {number}.", "Anton", key=f"custom.key{number}", value=f"v{number}")
    profile = profile_from(name="Anton", role="user", language="en", memory=memory)
    assert len(profile.preferences) == MAX_PREFERENCES


# --- a real turn -----------------------------------------------------------


def test_a_real_turn_shows_the_model_the_speaker_and_the_home(monkeypatch, tmp_path):
    memory = _memory(tmp_path)

    async def run():
        brain = SimpleNamespace(
            generate=AsyncMock(return_value=SimpleNamespace(
                text="It is nine in the evening.", history=[], tool_calls=[])),
            verify=AsyncMock())
        monkeypatch.setattr(app, "_llm", brain)
        monkeypatch.setattr(app, "_stt", SimpleNamespace(
            transcribe_pcm=lambda *a: ("what time is it?", "en")))
        monkeypatch.setattr(app, "_voices", SimpleNamespace(
            enabled=True, identify=lambda *a: ("Anton", "admin", 0.9)))
        monkeypatch.setattr(app, "_tts", object())
        monkeypatch.setattr(app, "_conversations", None)
        monkeypatch.setattr(app, "_memory", memory)
        cfg = Config()
        cfg.homes = [HomeConfig(home_id="livingroom", name="Living room",
                                quiet_hours={"start": "23:00", "end": "08:00"})]
        conn = app.Connection(SimpleNamespace(client=None), cfg)
        conn.home_id = "livingroom"
        conn.cfg.server.llm.verify_actions = False
        conn.cfg.server.diarization.enabled = False
        conn.session = Session("room-pc", [{"name": "lamp", "type": "magichome",
                                            "area": "desk"}], 25)
        conn.send_json = AsyncMock()
        conn._announce_speaker = AsyncMock()
        conn._log_dialog = AsyncMock()
        conn._stream_tts = AsyncMock()
        conn._execute_tool = AsyncMock()
        conn.presence = SimpleNamespace(present=lambda: {"Anton", "unknown"}, unknown_count=1)
        await conn._handle_utterance(b"\0" * 16000)
        return brain.generate.call_args.args[0]

    messages = asyncio.run(run())
    system, user = messages[0]["content"], messages[-1]["content"]

    assert messages[0]["role"] == "system"
    assert "Prefers oolong tea in the evening." in system, "the system prompt keeps the facts"
    assert "speaker: Anton" in user and "role: admin" in user
    assert "language: en" in user
    assert 'apps.browser="Google Chrome"' in user
    assert '"Prefers oolong tea in the evening."' in user
    assert "in the room: Anton" in user and "1 unknown person" in user
    assert "lights: lamp (on/off not reported by the room)" in user
    assert "quiet hours: 23:00-08:00" in user
    assert user.endswith("what time is it?")


def test_the_early_start_draft_gets_the_same_prefix(monkeypatch):
    """A held round must see what the real one sees, or the two disagree."""
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn.home_id = "livingroom"
    conn.session = Session("room-pc", [], 4)
    conn._speaker_name, conn._speaker_role, conn._reply_language = "Anton", "admin", "en"
    conn.presence = SimpleNamespace(present=lambda: {"Anton"}, unknown_count=0)
    monkeypatch.setattr(app, "_memory", None)

    real = conn._turn_prefix(datetime(2026, 9, 21, 21, 4, 11), "turn it on", live_view="view")
    draft = conn._turn_prefix(datetime(2026, 9, 21, 21, 4, 11), "turn it", live_view="view")

    assert real.replace("turn it on", "") == draft.replace("turn it", "")
