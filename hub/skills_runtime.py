"""Skill runtime primitives (ТЗ F-405, F-406).

A skill is a Python module in its own directory with a ``manifest.yaml`` plus a
``skill.py`` exposing ``run(ctx, args) -> SkillResult``. Phase 0 delivers the
manifest model and result type; the loader/registry and per-home isolation are
phase 1 work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

SCOPE = Literal["hub", "home"]
ROLE = Literal["admin", "trusted", "user", "guest"]


class SkillResult(BaseModel):
    """What a skill returns to the LLM loop; nothing else is accepted."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    spoken: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SkillManifest(BaseModel):
    """``manifest.yaml`` of one skill (ТЗ F-405)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=41, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=280)
    scope: SCOPE = "home"
    role: ROLE = "user"
    caps: list[str] = Field(default_factory=list)
    version: str = "0.1.0"
    enabled: bool = False


def load_manifest(path: Path) -> SkillManifest:
    """Parse and validate a manifest; a bad file raises pydantic's error."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return SkillManifest.model_validate(data)
