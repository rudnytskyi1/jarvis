"""Shared code for both Jarvis processes (brain server and room client).

Contains only machine-agnostic pieces:

* :mod:`common.config`   -- YAML config loading/validation (SPEC section 6).
* :mod:`common.protocol` -- WebSocket message-type constants (SPEC section 4).

This package intentionally imports nothing at package level so that
``import common.protocol`` works even in an environment where pydantic/pyyaml
are not installed.
"""

__all__ = ["config", "protocol"]
