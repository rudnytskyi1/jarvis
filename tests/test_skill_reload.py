"""Hot reload of skills by file change, dev mode only (ТЗ F-405)."""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from common.config import Config, load_config
from hub import app as hub_app
from hub.skills_registry import SkillRegistry, SkillWatcher
from hub.skills_runtime import SkillResult


def write_skill(root, name, *, scope="hub", spoken="first", body=None):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.yaml").write_text(
        f"name: {name}\ndescription: test skill\nscope: {scope}\nrole: user\n"
        f"caps: []\nversion: 0.1.0\nenabled: true\n", encoding="utf-8")
    (directory / "skill.py").write_text(
        body or body_returning(spoken), encoding="utf-8")
    return directory


def body_returning(spoken):
    return ("from hub.skills_runtime import SkillResult\n"
            "async def run(ctx, args):\n"
            f"    return SkillResult(ok=True, spoken={spoken!r})\n")


def touch(path, *, seconds=5.0):
    """Make sure the file looks newer than the copy that is loaded."""
    moment = time.time() + seconds
    os.utime(path, (moment, moment))


def spoken_of(registry, name, *, home_id=None):
    return asyncio.run(registry.run(name, None, home_id=home_id)).spoken


@pytest.fixture()
def roots(tmp_path):
    hub_dir, home_dir = tmp_path / "hub_skills", tmp_path / "homes"
    hub_dir.mkdir()
    (home_dir / "livingroom" / "skills").mkdir(parents=True)
    return hub_dir, home_dir


def test_dev_mode_reloads_an_edited_skill_file(roots):
    hub_dir, _ = roots
    directory = write_skill(hub_dir, "coffee", spoken="first")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    watcher = SkillWatcher(registry, enabled=True)
    assert spoken_of(registry, "coffee") == "first"

    (directory / "skill.py").write_text(body_returning("second"), encoding="utf-8")
    touch(directory / "skill.py")
    assert watcher.tick() == ["coffee"]
    assert spoken_of(registry, "coffee") == "second"
    assert watcher.reloaded == ["coffee"]


def test_a_skill_that_nobody_edited_is_left_alone(roots):
    hub_dir, _ = roots
    write_skill(hub_dir, "coffee", spoken="first")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    watcher = SkillWatcher(registry, enabled=True)
    assert watcher.tick() == [] and watcher.reloaded == []


def test_the_watcher_does_nothing_unless_dev_mode_is_on(roots):
    hub_dir, _ = roots
    directory = write_skill(hub_dir, "coffee", spoken="first")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    watcher = SkillWatcher(registry, enabled=False)
    (directory / "skill.py").write_text(body_returning("second"), encoding="utf-8")
    touch(directory / "skill.py")
    assert watcher.tick() == []
    assert spoken_of(registry, "coffee") == "first"
    assert asyncio.run(watcher.start()) is None, "a switched-off watcher has no loop"


def test_a_broken_edit_keeps_the_copy_that_works(roots):
    hub_dir, _ = roots
    directory = write_skill(hub_dir, "coffee", spoken="first")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    watcher = SkillWatcher(registry, enabled=True)

    (directory / "skill.py").write_text("this is not python\n", encoding="utf-8")
    touch(directory / "skill.py")
    assert watcher.tick() == []
    assert spoken_of(registry, "coffee") == "first", "the room keeps its skill"
    assert any("reload failed" in error for error in registry.errors)
    # ...and the same broken file is not retried on every tick.
    assert watcher.tick() == []


def test_a_fixed_file_is_picked_up_after_a_broken_edit(roots):
    hub_dir, _ = roots
    directory = write_skill(hub_dir, "coffee", spoken="first")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    watcher = SkillWatcher(registry, enabled=True)
    (directory / "skill.py").write_text("broken\n", encoding="utf-8")
    touch(directory / "skill.py")
    watcher.tick()

    (directory / "skill.py").write_text(body_returning("fixed"), encoding="utf-8")
    touch(directory / "skill.py", seconds=10.0)
    assert watcher.tick() == ["coffee"]
    assert spoken_of(registry, "coffee") == "fixed"


def test_the_manifest_is_watched_too(roots):
    hub_dir, _ = roots
    directory = write_skill(hub_dir, "coffee", spoken="first")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    watcher = SkillWatcher(registry, enabled=True)
    (directory / "manifest.yaml").write_text(
        "name: coffee\ndescription: test skill\nscope: hub\nrole: admin\n"
        "caps: []\nversion: 0.2.0\nenabled: true\n", encoding="utf-8")
    touch(directory / "manifest.yaml")
    assert watcher.tick() == ["coffee"]
    assert registry.get("coffee", home_id=None).manifest.role == "admin"


def test_a_home_skill_stays_in_its_home_after_reloading(roots):
    _, homes = roots
    directory = write_skill(homes / "livingroom" / "skills", "coffee", scope="home",
                            spoken="first")
    registry = SkillRegistry()
    registry.load_directory(homes / "livingroom" / "skills", home_id="livingroom")
    watcher = SkillWatcher(registry, enabled=True)
    (directory / "skill.py").write_text(body_returning("second"), encoding="utf-8")
    touch(directory / "skill.py")
    assert watcher.tick() == ["coffee"]
    assert spoken_of(registry, "coffee", home_id="livingroom") == "second"
    assert registry.get("coffee", home_id="kitchen") is None


def test_the_background_loop_reloads_without_being_asked(roots):
    hub_dir, _ = roots
    directory = write_skill(hub_dir, "coffee", spoken="first")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    watcher = SkillWatcher(registry, enabled=True, interval_s=0.25)

    async def run():
        await watcher.start()
        (directory / "skill.py").write_text(body_returning("second"), encoding="utf-8")
        touch(directory / "skill.py")
        for _ in range(20):
            await asyncio.sleep(0.05)
            if watcher.reloaded:
                break
        await watcher.stop()

    asyncio.run(run())
    assert watcher.reloaded == ["coffee"]
    assert spoken_of(registry, "coffee") == "second"


# --- configuration --------------------------------------------------------


def test_hot_reload_is_off_by_default():
    settings = Config().server.skills
    assert settings.dev_reload is False
    assert settings.interval_s == 2.0


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_configs_declare_the_skills_section(name):
    cfg = load_config(name)
    assert cfg.server.skills.dev_reload is False
    assert cfg.server.skills.interval_s == 2.0


def test_the_interval_cannot_be_absurd():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config.model_validate({"server": {"skills": {"interval_s": 0.01}}})


# --- how the hub arms it --------------------------------------------------


def test_the_hub_loads_both_scopes_and_arms_the_watcher_in_dev_mode(roots):
    hub_dir, homes = roots
    write_skill(hub_dir, "clock", spoken="hub")
    write_skill(homes / "livingroom" / "skills", "coffee", scope="home", spoken="home")
    for flag in (False, True):
        cfg = Config()
        cfg.server.skills.dev_reload = flag
        registry, watcher = hub_app._skill_hot_reload(cfg, hub_root=hub_dir, homes_root=homes)
        assert {skill.manifest.name for skill in registry.all()} == {"clock", "coffee"}
        assert watcher.enabled is flag


def test_a_hub_without_skill_directories_still_starts(tmp_path):
    cfg = Config()
    registry, watcher = hub_app._skill_hot_reload(cfg, hub_root=tmp_path / "none",
                                                 homes_root=tmp_path / "nowhere")
    assert registry.all() == [] and watcher.enabled is False
    assert asyncio.run(registry.run("coffee", None, home_id=None)) == SkillResult(
        ok=False, error="unknown skill 'coffee'")
