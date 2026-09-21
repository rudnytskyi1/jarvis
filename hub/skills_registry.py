"""Skill registry with per-home isolation (ТЗ F-405, F-406).

A skill is a directory with ``manifest.yaml`` and ``skill.py``. Hub-scoped
skills are visible to every home; home-scoped skills only inside their own
home. A broken or slow skill is reported as a failed ``SkillResult`` - it never
raises into the hub, and never takes the whole assistant down.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from hub.skills_runtime import SkillManifest, SkillResult, load_manifest

SKILL_TIMEOUT_S = 20.0
ROLE_RANK = {"guest": 0, "user": 1, "trusted": 2, "admin": 3}
SKILL_FILES = ("manifest.yaml", "skill.py")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    manifest: SkillManifest
    module: ModuleType
    home_id: str | None
    #: Where the skill was loaded from; the reload watcher needs it back.
    directory: Path | None = None

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
                module = self.load_module(directory / "skill.py", manifest.name)
                self.register(manifest, module, home_id=home_id, directory=directory)
                loaded.append(manifest.name)
            except Exception as exc:  # noqa: BLE001 - one bad skill must not stop the rest
                self.errors.append(f"{directory.name}: {type(exc).__name__}: {exc}")
        return loaded

    @staticmethod
    def load_module(path: Path, name: str) -> ModuleType:
        """Import one ``skill.py``; a module without ``run`` is not a skill."""
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

    def register(self, manifest: SkillManifest, module: ModuleType, *, home_id: str | None = None,
                 directory: Path | None = None) -> Skill:
        skill = Skill(manifest=manifest, module=module, home_id=home_id, directory=directory)
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

    def all(self) -> list[Skill]:
        """Every loaded skill, whatever its home (the reload watcher's view)."""
        return list(self._skills.values())

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


class SkillWatcher:
    """Reload skills whose files changed — only in the dev mode of ТЗ F-405.

    A hub that reloads skill code the moment a file is saved behaves differently
    from the one that ships, so the watcher is off unless
    ``server.skills.dev_reload`` is on. Reloading is also contained: a skill
    whose new code does not import keeps the version that works, and the reason
    lands in :attr:`SkillRegistry.errors` instead of taking the feature away.
    """

    def __init__(self, registry: SkillRegistry, *, enabled: bool = False,
                 interval_s: float = 2.0, clock: Any = time.monotonic) -> None:
        self.registry = registry
        self.enabled = bool(enabled)
        self.interval_s = max(0.25, float(interval_s))
        self.clock = clock
        #: Keys reloaded since the watcher started, newest last.
        self.reloaded: list[str] = []
        self._stamps = self.stamps()
        self._task: asyncio.Task[None] | None = None

    # --- what changed -------------------------------------------------------

    def stamps(self) -> dict[str, float]:
        """Newest modification time of each skill's files."""
        return {skill.key: self._stamp(skill) for skill in self.registry.all()}

    @staticmethod
    def _stamp(skill: Skill) -> float:
        if skill.directory is None:
            return 0.0
        newest = 0.0
        for name in SKILL_FILES:
            try:
                newest = max(newest, (skill.directory / name).stat().st_mtime)
            except OSError:
                continue
        return newest

    def changed(self) -> list[Skill]:
        """Skills whose files are newer than the copy that is loaded."""
        return [skill for skill in self.registry.all()
                if self._stamp(skill) > self._stamps.get(skill.key, 0.0)]

    # --- reloading ----------------------------------------------------------

    def tick(self) -> list[str]:
        """One pass: reload every changed skill. Never raises."""
        if not self.enabled:
            return []
        done: list[str] = []
        for skill in self.changed():
            try:
                self._reload(skill)
            except Exception as exc:  # noqa: BLE001 - the old copy keeps working
                self.registry.errors.append(f"{skill.manifest.name}: reload failed: {exc}")
                log.warning("Skill %s was not reloaded (%s); keeping the loaded copy",
                            skill.manifest.name, exc)
            else:
                done.append(skill.manifest.name)
            # The stamp moves either way: a half-written file is not retried on
            # every tick, only after the next save.
            self._stamps[skill.key] = self._stamp(skill)
        if done:
            self.reloaded.extend(done)
            log.info("Reloaded skill(s): %s", ", ".join(done))
        return done

    def _reload(self, skill: Skill) -> Skill:
        directory = skill.directory
        if directory is None:
            raise ValueError("the skill was not loaded from a directory")
        manifest = load_manifest(directory / "manifest.yaml")
        if (manifest.scope == "home") != (skill.home_id is not None):
            raise ValueError("manifest scope does not match its directory")
        module = self.registry.load_module(directory / "skill.py", manifest.name)
        return self.registry.register(manifest, module, home_id=skill.home_id, directory=directory)

    # --- the background loop ------------------------------------------------

    async def start(self) -> asyncio.Task[None] | None:
        """Begin watching; a watcher that is off returns ``None`` and does nothing."""
        if not self.enabled or self._task is not None:
            return self._task
        self._task = asyncio.create_task(self._loop())
        return self._task

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - stopping is best effort
            pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            await asyncio.to_thread(self.tick)


__all__ = ["ROLE_RANK", "SKILL_FILES", "SKILL_TIMEOUT_S", "Skill", "SkillRegistry", "SkillWatcher"]
