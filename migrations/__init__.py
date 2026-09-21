"""Numbered hub database migrations (ТЗ section 4.6).

Each module exposes ``VERSION`` (int), ``NAME`` (str) and ``apply(conn)``.
``hub.migrations_runner`` discovers, orders and applies them exactly once and
records the result in the ``schema_version`` table.
"""
