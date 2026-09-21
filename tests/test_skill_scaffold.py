"""The skill scaffold creates a valid manifest and never fakes a result."""
from __future__ import annotations

import asyncio

import pytest

from hub import skill_scaffold
from hub.skills_runtime import SkillManifest, SkillResult, load_manifest


def test_scaffold_writes_a_valid_manifest_and_body(tmp_path):
    directory = skill_scaffold.scaffold(tmp_path, "coffee")
    assert (directory / "manifest.yaml").is_file()
    manifest = load_manifest(directory / "manifest.yaml")
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "coffee" and manifest.scope == "home" and manifest.enabled is False


@pytest.mark.parametrize("name", ["Coffee", "cof fee", "1coffee", "", "x"])
def test_scaffold_rejects_bad_names(tmp_path, name):
    with pytest.raises(ValueError):
        skill_scaffold.scaffold(tmp_path, name)


def test_scaffold_refuses_to_overwrite(tmp_path):
    skill_scaffold.scaffold(tmp_path, "coffee")
    with pytest.raises(FileExistsError):
        skill_scaffold.scaffold(tmp_path, "coffee")


def test_generated_body_raises_instead_of_pretending(tmp_path):
    directory = skill_scaffold.scaffold(tmp_path, "coffee")
    namespace: dict = {}
    exec(compile((directory / "skill.py").read_text(encoding="utf-8"),
                 str(directory / "skill.py"), "exec"), namespace)
    args = namespace["Args"]()
    with pytest.raises(NotImplementedError):
        asyncio.run(namespace["run"](None, args))


def test_skill_result_rejects_unknown_fields():
    with pytest.raises(Exception):
        SkillResult(ok=True, unexpected=1)
