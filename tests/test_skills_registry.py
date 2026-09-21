"""Skill discovery, isolation, role gating and failure containment (F-405/F-406)."""
from __future__ import annotations

import asyncio

import pytest

from hub.skill_scaffold import scaffold
from hub.skills_registry import SkillRegistry
from hub.skills_runtime import SkillResult


def write_skill(root, name, *, scope, role="user", enabled=True, body):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "manifest.yaml").write_text(
        f"name: {name}\ndescription: test skill\nscope: {scope}\nrole: {role}\n"
        f"caps: []\nversion: 0.1.0\nenabled: {str(enabled).lower()}\n", encoding="utf-8")
    (directory / "skill.py").write_text(body, encoding="utf-8")
    return directory


OK_BODY = (
    "from hub.skills_runtime import SkillResult\n"
    "async def run(ctx, args):\n"
    "    return SkillResult(ok=True, spoken='done')\n"
)


@pytest.fixture()
def roots(tmp_path):
    hub_dir = tmp_path / "hub_skills"
    home_dir = tmp_path / "home_skills"
    hub_dir.mkdir()
    home_dir.mkdir()
    return hub_dir, home_dir


def test_home_skills_are_invisible_to_other_homes(roots):
    hub_dir, home_dir = roots
    write_skill(hub_dir, "clock", scope="hub", body=OK_BODY)
    write_skill(home_dir, "coffee", scope="home", body=OK_BODY)
    registry = SkillRegistry()
    assert registry.load_directory(hub_dir, home_id=None) == ["clock"]
    assert registry.load_directory(home_dir, home_id="dorm-max") == ["coffee"]
    assert not registry.errors
    assert [m.name for m in registry.available(home_id="dorm-max", role="user")] == ["clock", "coffee"]
    assert [m.name for m in registry.available(home_id="livingroom", role="user")] == ["clock"]
    assert registry.get("coffee", home_id="livingroom") is None


def test_disabled_and_privileged_skills_are_gated(roots):
    hub_dir, _ = roots
    write_skill(hub_dir, "hidden", scope="hub", enabled=False, body=OK_BODY)
    write_skill(hub_dir, "unlock", scope="hub", role="admin", body=OK_BODY)
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    assert registry.available(home_id="a", role="user") == []
    assert [m.name for m in registry.available(home_id="a", role="admin")] == ["unlock"]


def test_a_broken_skill_is_reported_and_does_not_stop_the_rest(roots):
    hub_dir, _ = roots
    write_skill(hub_dir, "broken", scope="hub", body="this is not python\n")
    write_skill(hub_dir, "good", scope="hub", body=OK_BODY)
    registry = SkillRegistry()
    assert registry.load_directory(hub_dir) == ["good"]
    assert registry.errors and "broken" in registry.errors[0]


def test_scope_mismatch_is_rejected(roots):
    hub_dir, home_dir = roots
    write_skill(hub_dir, "coffee", scope="home", body=OK_BODY)
    registry = SkillRegistry()
    assert registry.load_directory(hub_dir, home_id=None) == []
    assert registry.errors and "scope" in registry.errors[0]


def test_running_a_skill_returns_a_result_and_never_raises(roots):
    hub_dir, _ = roots
    write_skill(hub_dir, "good", scope="hub", body=OK_BODY)
    write_skill(hub_dir, "boom", scope="hub",
                body="async def run(ctx, args):\n    raise RuntimeError('nope')\n")
    write_skill(hub_dir, "slow", scope="hub",
                body="import asyncio\nasync def run(ctx, args):\n    await asyncio.sleep(5)\n")
    registry = SkillRegistry(timeout_s=0.05)
    registry.load_directory(hub_dir)
    assert asyncio.run(registry.run("good", None, home_id=None)) == SkillResult(ok=True, spoken="done")
    failed = asyncio.run(registry.run("boom", None, home_id=None))
    assert failed.ok is False and "RuntimeError" in (failed.error or "")
    timed_out = asyncio.run(registry.run("slow", None, home_id=None))
    assert timed_out.ok is False and "timed out" in (timed_out.error or "")
    unknown = asyncio.run(registry.run("missing", None, home_id=None))
    assert unknown.ok is False and "unknown skill" in (unknown.error or "")


def test_scaffolded_skill_loads_and_reports_not_implemented(tmp_path):
    scaffold(tmp_path, "coffee")
    skills_root = tmp_path / "skills"
    # The scaffold writes a home-scoped manifest, so it loads into a home.
    registry = SkillRegistry()
    assert registry.load_directory(skills_root, home_id="livingroom") == ["coffee"]
    result = asyncio.run(registry.run("coffee", None, home_id="livingroom"))
    assert result.ok is False and "not implemented" in (result.error or "")
