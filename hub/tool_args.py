"""Pydantic validation of tool arguments BEFORE anything runs (ТЗ F-409).

F-409: «все вызовы инструментов валидируются Pydantic до выполнения;
невалидные возвращаются модели с текстом ошибки валидации (максимум 2
повтора)». The schemas the model was given (`hub.tools.TOOLS`) are the single
source of truth: the models here are BUILT FROM them, so a tool's Pydantic
contract cannot drift away from the JSON schema that was advertised.

An unknown construct in a schema is an error at import time, never a silently
unvalidated tool: a door that opens by accident is worse than a loud failure.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from difflib import get_close_matches
from typing import Any, Literal, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from hub.tools import TOOLS

log = logging.getLogger("jarvis.server.tool_args")

#: ТЗ F-409: how many times one tool may be sent back to the model with bad
#: arguments before this turn stops trying that call ("максимум 2 повтора").
MAX_ARGUMENT_RETRIES = 2

#: JSON-schema scalar types we know how to enforce.
_SCALARS: dict[str, Any] = {"string": str, "integer": int, "number": float,
                            "boolean": bool}


def _annotation(name: str, spec: Mapping[str, Any]) -> tuple[Any, bool]:
    """The Python type one property validates against, and whether null is one."""
    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        return Literal[tuple(enum)], False  # type: ignore[valid-type]
    raw = spec.get("type")
    allows_null = False
    if isinstance(raw, list):
        kinds = [item for item in raw if item != "null"]
        allows_null = "null" in raw
        if not kinds:
            raise ValueError(f"{name}: {raw!r} names no type")
        raw = kinds
    if isinstance(raw, list):
        # e.g. `["string", "integer", "null"]`: a small union of scalars.
        if any(kind not in _SCALARS for kind in raw):
            raise ValueError(f"{name}: unsupported union {raw!r}")
        types = [_SCALARS[kind] for kind in raw]
        annotation: Any = types[0]
        for extra_type in types[1:]:
            annotation = annotation | extra_type
        return annotation, allows_null
    if raw == "array":
        items = spec.get("items") or {}
        if items.get("type") != "string":
            raise ValueError(f"{name}: only arrays of strings are supported, got {items!r}")
        return list[str], allows_null
    if raw not in _SCALARS:
        raise ValueError(f"{name}: unsupported type {raw!r}")
    return _SCALARS[raw], allows_null


def _field(name: str, spec: Mapping[str, Any], *, required: bool) -> tuple[Any, Any]:
    annotation, allows_null = _annotation(name, spec)
    limits: dict[str, Any] = {}
    if "minimum" in spec:
        limits["ge"] = spec["minimum"]
    if "maximum" in spec:
        limits["le"] = spec["maximum"]
    if get_origin(annotation) is list:
        if "maxItems" in spec:
            limits["max_length"] = spec["maxItems"]
        if "minItems" in spec:
            limits["min_length"] = spec["minItems"]
    for unsupported in ("minLength", "maxLength", "pattern"):
        if unsupported in spec:
            # No tool uses these today. Silently ignoring them would mean
            # validating against something other than the published contract.
            raise ValueError(f"{name}: {unsupported} is not supported yet")
    if required and not allows_null:
        return annotation, Field(**limits)
    if required:
        return (annotation | None), Field(**limits)
    return (annotation | None), Field(default=None, **limits)


def _model_for(name: str, parameters: Mapping[str, Any]) -> type[BaseModel]:
    """A Pydantic model for one tool, built from the schema the model was given."""
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or ())
    if required - set(properties):
        raise ValueError(f"{name}: required fields {sorted(required - set(properties))} "
                         f"are not declared")
    fields: dict[str, Any] = {
        field_name: _field(field_name, spec, required=field_name in required)
        for field_name, spec in properties.items()
    }
    # A published tool contract is a closed set: an unexpected key is a mistake
    # the model has to fix, not something to forward into the room.
    return create_model(f"ToolArgs_{name}", __config__=ConfigDict(extra="forbid"), **fields)


def drop_undeclared(name: str, model: type[BaseModel],
                    args: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the declared arguments, drop a stray extra one with a warning.

    The model does send an argument that belongs to a different tool — the live
    audit of 2026-09-23 caught ``pc_control`` arriving with the ``url`` of the
    browser call it had just made. Refusing the whole call for that cost a
    retry and, often, the request itself: the room heard "it didn't work" about
    a command that was one key away from running. The declared fields are still
    validated strictly, so a wrong command, a bad enum value or a missing
    required field is still handed back to the model.
    """
    declared = set(model.model_fields)
    extra = sorted(key for key in args if key not in declared)
    if not extra:
        return dict(args)
    log.warning("Tool %s: dropping undeclared argument(s) %s", name, ", ".join(extra))
    return {key: value for key, value in args.items() if key in declared}


def _misspelled_field(key: str, declared: set[str]) -> str:
    """The declared field ``key`` was probably meant to be, or ``""``.

    A stray field that belongs to another tool is noise and goes away
    (AUDIT-02). A field that is one letter away from a real one is a typo, and
    dropping it would run the call with a default the person never asked for:
    ``{"brightnesss": 50}`` on a light would turn it on at full brightness. That
    one still goes back to the model to fix.
    """
    matches = get_close_matches(key, sorted(declared), n=1, cutoff=0.8)
    return matches[0] if matches else ""


#: Built once, at import: an unsupported schema fails loudly here, not per turn.
TOOL_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    tool["function"]["name"]: _model_for(tool["function"]["name"],
                                        tool["function"].get("parameters") or {})
    for tool in TOOLS
}


def _first_errors(error: ValidationError) -> str:
    """One line per bad field: which field, and what was wrong with it."""
    parts: list[str] = []
    for item in error.errors():
        field = ".".join(str(step) for step in item.get("loc") or ("?",))
        parts.append(f"{field}: {item.get('msg') or 'invalid'}")
    return "; ".join(parts)


def validate_args(name: str, args: Mapping[str, Any] | None) -> tuple[dict[str, Any], str]:
    """Validate one tool call's arguments (ТЗ F-409).

    Returns ``(arguments, "")`` when the call may run — the dictionary is what
    the tool gets, with the fields the model did not send left out, so a tool's
    own default still applies — or ``({}, reason)`` when it must not, where
    ``reason`` is the text handed back to the model.
    """
    model = TOOL_ARGUMENT_MODELS.get(name)
    if model is None:
        return {}, f"unknown tool: {name}"
    declared = set(model.model_fields)
    misspelled = {key: _misspelled_field(key, declared)
                  for key in (args or {}) if key not in declared}
    misspelled = {key: fixed for key, fixed in misspelled.items() if fixed}
    if misspelled:
        return {}, "; ".join(f"{key} is not a field of {name} (did you mean {fixed}?)"
                             for key, fixed in sorted(misspelled.items()))
    try:
        validated = model.model_validate(drop_undeclared(name, model, dict(args or {})))
    except ValidationError as error:
        return {}, _first_errors(error)
    except Exception as exc:  # noqa: BLE001 - a broken call must not raise here
        log.warning("Could not validate the arguments of %s (%s)", name, exc)
        return {}, f"{type(exc).__name__}: {exc}"
    return validated.model_dump(exclude_unset=True, exclude_none=True), ""


def tool_argument_schemas() -> dict[str, dict[str, Any]]:
    """The JSON schema of each tool's Pydantic model (for tests and the panel)."""
    return {name: model.model_json_schema() for name, model in TOOL_ARGUMENT_MODELS.items()}


__all__ = [
    "MAX_ARGUMENT_RETRIES",
    "TOOL_ARGUMENT_MODELS",
    "drop_undeclared",
    "tool_argument_schemas",
    "validate_args",
]
