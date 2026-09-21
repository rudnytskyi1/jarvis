"""Skill registry with per-home isolation (ТЗ F-405, F-406).

A skill is a directory with ``manifest.yaml`` and ``skill.py``. Hub-scoped
skills are visible to every home; home-scoped skills only inside their own
home. A broken or slow skill is reported as a failed ``SkillResult`` - it never
raises into the hub, and never takes the whole assistant down.
"""
from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from hub.skills_runtime import SkillManifest, SkillResult, load_manifest

SKILL_TIMEOUT_S = 20.0
ROLE_RANK = {"guest": 0, "user": 1, "trusted": 2, "admin": 3}


@dataclass(frozen=True)
class Skill:
    manifest: SkillManifest
    module: ModuleType
    home_id: str | None

    @property
    def key(self) -> str:
        return f"{self.home_id or 'hub'}:{self.manifest.name}"


class SkillRegistry:
    def __init__(self, *, timeout_s: float = SKILL_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s
        self._skills: dict[str, Skill] = {}
        self.errors: list[str] = []

    # --- loading ------------------------------------------------------------

    def load_directory(self, root: Path | str, *, home_id: str | None = None) -> list[str]:
        """Load every skill under ``root``; returns the names that loaded."""
        loaded: list[str] = []
        for directory in sorted(Path(root).iterdir()):
            manifest_path = directory / "manifest.yaml"
            if not directory.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = load_manifest(manifest_path)
                if (manifest.scope == "home") != (home_id is not None):
                    raise ValueError("manifest scope does not match its directory")
                module = self._load_module(directory / "skill.py", manifest.name)
                self.register(manifest, module, home_id=home_id)
                loaded.append(manifest.name)
            except Exception as exc:  # noqa: BLE001 - one bad skill must not stop the rest
                self.errors.append(f"{directory.name}: {type(exc).__name__}: {exc}")
        return loaded

    @staticmethod
    def _load_module(path: Path, name: str) -> ModuleType:
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} is missing")
        spec = importlib.util.spec_from_file_location(f"rowan_skill_{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "run", None)):
            raise ImportError(f"{path.name} has no run(ctx, args)")
        return module

    def register(self, manifest: SkillManifest, module: ModuleType, *, home_id: str | None = None) -> Skill:
        skill = Skill(manifest=manifest, module=module, home_id=home_id)
        self._skills[skill.key] = skill
        return skill

    # --- lookup -------------------------------------------------------------

    def available(self, *, home_id: str | None, role: str) -> list[SkillManifest]:
        """Enabled skills this home may use, given the speaker's role."""
        rank = ROLE_RANK.get(role, 0)
        return [skill.manifest for skill in self._skills.values()
                if skill.manifest.enabled
                and (skill.home_id is None or skill.home_id == home_id)
                and ROLE_RANK.get(skill.manifest.role, 0) <= rank]

    def get(self, name: str, *, home_id: str | None) -> Skill | None:
        return self._skills.get(f"{home_id or 'hub'}:{name}") or self._skills.get(f"hub:{name}")

    # --- execution ----------------------------------------------------------

    async def run(self, name: str, args: Any, *, home_id: str | None, ctx: Any = None) -> SkillResult:
        skill = self.get(name, home_id=home_id)
        if skill is None:
            return SkillResult(ok=False, error=f"unknown skill {name!r}")
        try:
            result = await asyncio.wait_for(skill.module.run(ctx, args), self.timeout_s)
        except TimeoutError:
            return SkillResult(ok=False, error=f"{name} timed out after {self.timeout_s:g}s")
        except NotImplementedError as exc:
            return SkillResult(ok=False, error=str(exc) or f"{name} is not implemented")
        except Exception as exc:  # noqa: BLE001 - skill failures stay inside the skill
            self.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return SkillResult(ok=False, error=f"{name} failed: {type(exc).__name__}")
        if not isinstance(result, SkillResult):
            return SkillResult(ok=False, error=f"{name} returned {type(result).__name__}, not SkillResult")
        return result
