"""Apply the numbered SQLite migrations to the hub database (ТЗ section 4.6).

Usage:
    python -m hub.migrations_runner --db data/hub.db

Every migration runs inside its own transaction and is recorded in
``schema_version``; a failure rolls the whole migration back and stops the
runner, so a half-applied schema is never left behind.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = REPO_ROOT / "migrations"
_PATTERN = re.compile(r"^(\d{4})_([a-z0-9_]+)\.py$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path

    def load(self) -> ModuleType:
        spec = importlib.util.spec_from_file_location(f"rowan_migration_{self.version:04d}", self.path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load migration {self.path.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if getattr(module, "VERSION", None) != self.version:
            raise RuntimeError(f"{self.path.name} declares VERSION {getattr(module, 'VERSION', None)!r}")
        if not callable(getattr(module, "apply", None)):
            raise RuntimeError(f"{self.path.name} has no apply(conn)")
        return module


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Every ``NNNN_name.py`` migration, ordered by version."""
    found: list[Migration] = []
    for path in sorted(Path(directory).glob("*.py")):
        match = _PATTERN.match(path.name)
        if match:
            found.append(Migration(int(match.group(1)), match.group(2), path))
    versions = [item.version for item in found]
    if len(set(versions)) != len(versions):
        raise RuntimeError("Duplicate migration version in migrations/")
    return sorted(found, key=lambda item: item.version)


def connect(db_path: str) -> sqlite3.Connection:
    """Open the hub database in WAL mode with foreign keys enforced."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def ensure_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    ensure_version_table(conn)
    return {int(row[0]) for row in conn.execute("SELECT version FROM schema_version")}


def migrate(conn: sqlite3.Connection, directory: Path = MIGRATIONS_DIR) -> list[int]:
    """Apply pending migrations in order; return the versions actually applied."""
    done = applied_versions(conn)
    applied: list[int] = []
    for migration in discover(directory):
        if migration.version in done:
            continue
        module = migration.load()
        conn.execute("BEGIN IMMEDIATE")
        try:
            module.apply(conn)
            conn.execute(
                "INSERT INTO schema_version(version, name, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.name, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        applied.append(migration.version)
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hub.migrations_runner", description="Apply hub SQLite migrations.")
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "hub.db"),
                        help="path to the hub database (default: data/hub.db)")
    args = parser.parse_args(argv)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(str(db_path))
    try:
        applied = migrate(conn)
    finally:
        conn.close()
    print(f"{db_path}: applied {len(applied)} migration(s) {applied}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
