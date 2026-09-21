"""The client's OTA: hourly check, restart, rollback (ТЗ 4.9)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from client.ota import ClientUpdater
from common.config import Config, load_config
from hub.ota import desired_tag, release_frame


class Git:
    """A repository in a list: records commands and answers like git would."""

    def __init__(self, *, tag="v1.0.0", fail_on=()):
        self.tag = tag
        self.fail_on = tuple(fail_on)
        self.commands: list[tuple[str, ...]] = []
        self.checked_out: list[str] = []

    def __call__(self, *args):
        self.commands.append(tuple(args))
        if any(args and args[0] == name for name in self.fail_on):
            raise RuntimeError(f"git {args[0]} failed: boom")
        if args[:1] == ("describe",):
            if not self.tag:
                raise RuntimeError("no tag here")
            return self.tag + "\n"
        if args[:1] == ("checkout",):
            self.checked_out.append(args[1])
            self.tag = args[1]
            return ""
        return ""


def updater(tmp_path, git=None, *, now=None, healthy_after_s=60.0, migrate=None, **overrides):
    clock = lambda: now[0] if now else 1000.0  # noqa: E731
    return ClientUpdater(repo=tmp_path, state_path=tmp_path / "ota_state.json",
                         git=git or Git(), clock=clock, healthy_after_s=healthy_after_s,
                         migrate=migrate, **overrides)


# --- the hub side -----------------------------------------------------------


def test_the_hub_tells_the_room_which_tag_to_run():
    cfg = Config()
    assert desired_tag(cfg) == "" and release_frame(cfg) is None
    cfg.server.client_release = "v1.6.0"
    assert release_frame(cfg) == {"type": "release", "tag": "v1.6.0", "utf8": True}


def test_the_client_config_declares_ota_off_by_default():
    for name in ("config.yaml", "config.client.example.yaml"):
        settings = load_config(name).client.ota
        assert settings.enabled is False
        assert settings.interval_s == 3600
        assert settings.healthy_after_s == 60


def test_a_nonsense_interval_is_a_typo_not_silence():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config.model_validate({"client": {"ota": {"interval_s": 5}}})


# --- the hourly check -------------------------------------------------------


def test_the_hub_is_checked_once_an_hour(tmp_path):
    now = [1000.0]
    git = Git(tag="v1.0.0")
    updater_ = updater(tmp_path, git, now=now)
    assert updater_.due() is True, "nothing checked yet"
    asyncio.run(updater_.check("v1.0.0"))
    assert updater_.due() is False
    now[0] += 3599
    assert updater_.due() is False
    now[0] += 2
    assert updater_.due() is True


def test_a_matching_tag_changes_nothing(tmp_path):
    git = Git(tag="v1.0.0")
    updater_ = updater(tmp_path, git)
    assert asyncio.run(updater_.check("v1.0.0")) is None
    assert git.commands == [("describe", "--tags", "--exact-match")]


def test_a_new_tag_is_fetched_checked_out_migrated_and_restarted(tmp_path):
    git = Git(tag="v1.0.0")
    migrated: list[str] = []
    updater_ = updater(tmp_path, git, migrate=lambda: migrated.append("migrated"))
    report = asyncio.run(updater_.check("v1.1.0"))
    assert report == {"action": "restart", "tag": "v1.1.0", "previous": "v1.0.0"}
    assert git.commands[1:3] == [("fetch", "origin", "--tags", "--prune"), ("checkout", "v1.1.0")]
    assert migrated == ["migrated"]
    state = json.loads((tmp_path / "ota_state.json").read_text(encoding="utf-8"))
    assert state["tag"] == "v1.1.0" and state["previous"] == "v1.0.0"
    assert state["pending"]["attempts"] == 0


def test_a_failed_fetch_leaves_the_room_on_the_tag_it_had(tmp_path):
    git = Git(tag="v1.0.0", fail_on=("fetch",))
    report = asyncio.run(updater(tmp_path, git).check("v1.1.0"))
    assert report["action"] == "failed" and "boom" in report["error"]
    assert git.checked_out == []


def test_a_config_migration_that_fails_rolls_straight_back(tmp_path):
    git = Git(tag="v1.0.0")

    def broken():
        raise RuntimeError("old key, new schema")

    report = asyncio.run(updater(tmp_path, git, migrate=broken).check("v1.1.0"))
    assert report["action"] == "rollback" and report["tag"] == "v1.0.0"
    assert git.checked_out == ["v1.1.0", "v1.0.0"]


def test_an_untagged_checkout_is_updated_too(tmp_path):
    git = Git(tag="")
    report = asyncio.run(updater(tmp_path, git).check("v1.1.0"))
    assert report["previous"] == "" and report["tag"] == "v1.1.0"


# --- the guard window -------------------------------------------------------


def test_a_release_that_survives_the_guard_window_is_kept(tmp_path):
    now = [1000.0]
    git = Git(tag="v1.0.0")
    updater_ = updater(tmp_path, git, now=now)
    asyncio.run(updater_.check("v1.1.0"))
    now[0] += 61
    assert updater_.startup_guard() == {"action": "accepted", "tag": "v1.1.0"}
    assert git.checked_out == ["v1.1.0"], "no rollback for a healthy release"
    assert updater_.state()["pending"] is None


def test_a_release_that_dies_inside_the_window_is_rolled_back(tmp_path):
    now = [1000.0]
    git = Git(tag="v1.0.0")
    updater_ = updater(tmp_path, git, now=now)
    asyncio.run(updater_.check("v1.1.0"))
    first = updater_.startup_guard()
    assert first["action"] == "starting" and first["attempt"] == 1
    second = updater_.startup_guard()
    assert second["action"] == "rollback" and second["tag"] == "v1.0.0"
    assert git.checked_out == ["v1.1.0", "v1.0.0"]
    assert updater_.state()["pending"] is None


def test_a_healthy_release_is_marked_and_a_later_crash_does_not_roll_back(tmp_path):
    now = [1000.0]
    git = Git(tag="v1.0.0")
    updater_ = updater(tmp_path, git, now=now)
    asyncio.run(updater_.check("v1.1.0"))
    now[0] += 61
    updater_.startup_guard()
    updater_.mark_healthy()
    now[0] += 5
    assert updater_.startup_guard() is None
    assert git.checked_out == ["v1.1.0"]


def test_a_rollback_without_a_previous_tag_is_reported_not_guessed(tmp_path):
    git = Git(tag="")
    updater_ = updater(tmp_path, git)
    updater_.save_state({"tag": "", "pending": {"tag": "v1.1.0", "previous": "",
                                                "started_at": 999.0, "attempts": 1}})
    report = updater_.startup_guard()
    assert report["action"] == "failed" and "no tag to go back to" in report["error"]


def test_the_state_file_survives_a_corrupt_write(tmp_path):
    git = Git(tag="v1.0.0")
    updater_ = updater(tmp_path, git)
    (tmp_path / "ota_state.json").write_text("not json", encoding="utf-8")
    assert updater_.state() == {}
    assert updater_.due() is True


# --- the wiring in the room client -----------------------------------------


def test_a_release_message_moves_the_room_and_restarts_it(tmp_path, monkeypatch):
    from client import main as client_main

    git = Git(tag="v1.0.0")
    restarted: list[str] = []
    client = client_main.JarvisClient.__new__(client_main.JarvisClient)
    client._updater = ClientUpdater(repo=tmp_path, git=git, state_path=tmp_path / "s.json",
                                    clock=lambda: 1000.0)
    client._wanted_release = ""
    client.ccfg = SimpleNamespace(ota=SimpleNamespace(enabled=True))
    client._restart_into_release = lambda: restarted.append(git.tag)

    asyncio.run(client._ota_move(client._ota(), "v1.1.0"))
    assert git.checked_out == ["v1.1.0"]
    assert restarted == ["v1.1.0"], "the new tag only runs after a real restart"


def test_a_release_message_understands_the_hub_and_skips_manual_rooms(tmp_path):
    from client import main as client_main

    client = client_main.JarvisClient.__new__(client_main.JarvisClient)
    client.ccfg = SimpleNamespace(ota=SimpleNamespace(enabled=False))
    client._updater = None
    client._wanted_release = ""
    client._on_release({"tag": "v1.1.0"})
    assert client._wanted_release == "v1.1.0"
    assert client._ota() is None, "a room with OTA switched off never updates"


def test_the_updater_state_lives_next_to_the_checkout():
    from client.main import _ota_enabled, _ota_state_path

    assert _ota_enabled(SimpleNamespace(ota=SimpleNamespace(enabled=True))) is True
    assert _ota_enabled(SimpleNamespace()) is False
    path = _ota_state_path(SimpleNamespace(state_path="data/ota_state.json"))
    assert path.name == "ota_state.json" and path.is_absolute()


def test_restarting_replaces_the_process(monkeypatch):
    from client import main as client_main

    called: list[tuple] = []
    monkeypatch.setattr(client_main.os, "execv", lambda exe, args: called.append((exe, args)))
    monkeypatch.setattr(client_main.sys, "argv", ["client", "--config", "config.yaml"])
    client_main.JarvisClient._restart_into_release(None)
    assert called and called[0][1][-2:] == ["--config", "config.yaml"]
