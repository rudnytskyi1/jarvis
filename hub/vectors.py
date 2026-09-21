"""sqlite-vec vector indexes for hub embeddings (ТЗ 4.6 and section 14).

The ordinary tables (``voice_embeddings``, ``face_embeddings``,
``body_embeddings``, ``memories``, ``objects_index``) stay authoritative: their
float32 little-endian BLOB ``vector`` column is the stored embedding. When the
``sqlite-vec`` loadable extension is available, a matching ``vec0`` virtual
table mirrors those rows for KNN search. The virtual table row id is the
metadata row's implicit integer rowid, which is the "external key on the
metadata row" of section 14; nothing else is duplicated.

The extension is discovered in this order:

1. ``ROWAN_SQLITE_VEC_PATH`` environment variable (explicit override),
2. the ``sqlite_vec`` Python package, if it is installed,
3. the copy vendored in ``hub/vendor/`` (see ``hub/vendor/README.md``).

No silent fallback: when the extension cannot be loaded the module raises
:class:`VectorExtensionUnavailable` and the caller decides (the hub logs the
reason and runs without vector search, the metadata columns stay intact).
"""
from __future__ import annotations

import importlib.util
import logging
import math
import os
import re
import sqlite3
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.server.vectors")

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
EXTENSION_ENV_VAR = "ROWAN_SQLITE_VEC_PATH"
MAX_DIMENSION = 8192

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_FLOAT_DIM_RE = re.compile(r"float\[(\d+)\]")


class VectorExtensionUnavailable(RuntimeError):
    """The sqlite-vec extension could not be found or loaded."""


@dataclass(frozen=True)
class VectorIndexSpec:
    """One embedding kind and the metadata row it belongs to."""

    kind: str
    metadata_table: str
    index_table: str
    id_column: str
    default_dimension: int


#: Every embedding store the hub mirrors into sqlite-vec, with the dimension of
#: the model that produces it (ТЗ 4.6, 14). Defaults are used only until the
#: first row with a ``dim`` arrives, so a different embedding model is picked up
#: from the data instead of silently mismatching.
INDEX_SPECS: dict[str, VectorIndexSpec] = {
    "voice": VectorIndexSpec("voice", "voice_embeddings", "vec_voice_embeddings", "id", 192),
    "face": VectorIndexSpec("face", "face_embeddings", "vec_face_embeddings", "id", 512),
    "body": VectorIndexSpec("body", "body_embeddings", "vec_body_embeddings", "id", 512),
    "memory": VectorIndexSpec("memory", "memories", "vec_memories", "memory_id", 384),
    "objects": VectorIndexSpec("objects", "objects_index", "vec_objects_index", "id", 512),
}

DEFAULT_DIMENSIONS: dict[str, int] = {kind: spec.default_dimension for kind, spec in INDEX_SPECS.items()}


@dataclass(frozen=True)
class VectorHit:
    """One KNN result: the metadata primary key and its distance."""

    row_id: str
    distance: float


def spec_for(kind: str) -> VectorIndexSpec:
    try:
        return INDEX_SPECS[kind]
    except KeyError:
        known = ", ".join(sorted(INDEX_SPECS))
        raise ValueError(f"unknown vector kind {kind!r}; expected one of {known}") from None


# ---------------------------------------------------------------------------
# extension discovery and loading
# ---------------------------------------------------------------------------


def extension_filename() -> str:
    """File name of the loadable extension on this platform."""
    if sys.platform.startswith("win"):
        return "vec0.dll"
    if sys.platform == "darwin":
        return "vec0.dylib"
    return "vec0.so"


def locate_extension() -> Path | None:
    """Path of the extension to load, or ``None`` when none is available."""
    override = os.environ.get(EXTENSION_ENV_VAR, "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    try:
        found = importlib.util.find_spec("sqlite_vec")
    except (ImportError, ValueError):
        found = None
    if found is not None and found.origin:
        candidate = Path(found.origin).resolve().parent / extension_filename()
        if candidate.is_file():
            return candidate

    vendored = VENDOR_DIR / extension_filename()
    return vendored if vendored.is_file() else None


def is_loaded(conn: sqlite3.Connection) -> bool:
    """True when this connection already has the extension loaded."""
    try:
        conn.execute("SELECT vec_version()").fetchone()
    except sqlite3.Error:
        return False
    return True


def load_extension(conn: sqlite3.Connection, *, path: Path | str | None = None) -> str:
    """Load the extension into ``conn`` and return the sqlite-vec version.

    :raises VectorExtensionUnavailable: no extension file was found or SQLite
        refused to load it.
    """
    if is_loaded(conn):
        row = conn.execute("SELECT vec_version()").fetchone()
        return str(row[0]) if row else "unknown"

    target = Path(path) if path is not None else locate_extension()
    if target is None:
        raise VectorExtensionUnavailable(
            "sqlite-vec is not available: set "
            f"{EXTENSION_ENV_VAR}, install the sqlite-vec package, or vendor "
            f"{extension_filename()} into {VENDOR_DIR}"
        )
    conn.enable_load_extension(True)
    try:
        conn.load_extension(str(target))
    except sqlite3.Error as exc:
        raise VectorExtensionUnavailable(f"cannot load sqlite-vec from {target}: {exc}") from exc
    finally:
        conn.enable_load_extension(False)
    row = conn.execute("SELECT vec_version()").fetchone()
    version = str(row[0]) if row else "unknown"
    log.debug("sqlite-vec %s loaded from %s", version, target)
    return version


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------


def pack_vector(vector: Sequence[float]) -> bytes:
    """Pack a vector as float32 little-endian, the schema's storage format."""
    values = [float(value) for value in vector]
    if not values:
        raise ValueError("vector is empty")
    if len(values) > MAX_DIMENSION:
        raise ValueError(f"vector has {len(values)} dimensions (max {MAX_DIMENSION})")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vector contains a non-finite value")
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of :func:`pack_vector`; raises ``ValueError`` on bad input."""
    if not blob or len(blob) % 4:
        raise ValueError("vector blob must be a non-empty multiple of 4 bytes")
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


# ---------------------------------------------------------------------------
# virtual tables
# ---------------------------------------------------------------------------


def index_dimension(conn: sqlite3.Connection, kind: str) -> int | None:
    """Dimension declared by the existing virtual table, or ``None``."""
    spec = spec_for(kind)
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (spec.index_table,)
    ).fetchone()
    if row is None or not row[0]:
        return None
    match = _FLOAT_DIM_RE.search(str(row[0]))
    return int(match.group(1)) if match else None


def metadata_dimension(conn: sqlite3.Connection, kind: str) -> int | None:
    """Dimension of the stored vectors in the metadata table.

    :raises ValueError: rows disagree about the dimension, which would make a
        single index wrong for some of them.
    """
    spec = spec_for(kind)
    rows = conn.execute(
        f"SELECT DISTINCT dim FROM {spec.metadata_table} WHERE vector IS NOT NULL AND dim IS NOT NULL"
    ).fetchall()
    dimensions = {int(row[0]) for row in rows}
    if len(dimensions) > 1:
        raise ValueError(
            f"{spec.metadata_table} holds mixed vector dimensions {sorted(dimensions)}; "
            "re-embed the rows before building the index"
        )
    return next(iter(dimensions)) if dimensions else None


def ensure_index(conn: sqlite3.Connection, kind: str, *, dimension: int | None = None) -> int:
    """Create the ``vec0`` virtual table for ``kind`` and return its dimension.

    :raises VectorExtensionUnavailable: sqlite-vec is missing.
    :raises ValueError: the requested dimension contradicts the index or the
        already stored rows.
    """
    spec = spec_for(kind)
    if dimension is not None and not 0 < dimension <= MAX_DIMENSION:
        raise ValueError(f"dimension must be between 1 and {MAX_DIMENSION}, got {dimension}")
    load_extension(conn)
    existing = index_dimension(conn, kind)
    detected = metadata_dimension(conn, kind)
    if dimension is not None:
        for label, value in (("existing index", existing), ("stored rows", detected)):
            if value is not None and value != dimension:
                raise ValueError(
                    f"{spec.index_table}: {label} use {value} dimensions, configuration says {dimension}"
                )
        target = dimension
    elif existing is not None:
        if detected is not None and detected != existing:
            raise ValueError(
                f"{spec.index_table} uses {existing} dimensions but {spec.metadata_table} "
                f"rows use {detected}; rebuild the index after re-embedding"
            )
        return existing
    else:
        target = detected or spec.default_dimension
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {spec.index_table} USING vec0(embedding float[{target}])"
    )
    conn.commit()
    log.info("sqlite-vec index %s ready (%d dimensions)", spec.index_table, target)
    return target


def ensure_indexes(
    conn: sqlite3.Connection,
    *,
    dimensions: Mapping[str, int] | None = None,
    extension_path: Path | str | None = None,
) -> dict[str, int]:
    """Create every index; return the effective ``kind -> dimension`` map.

    Per-kind ``dimensions`` overrides come from ``server.vectors.dimensions``;
    unknown kinds are rejected instead of being ignored.
    """
    load_extension(conn, path=extension_path)
    overrides = dict(dimensions or {})
    for kind in overrides:
        spec_for(kind)
    return {kind: ensure_index(conn, kind, dimension=overrides.get(kind)) for kind in INDEX_SPECS}


# ---------------------------------------------------------------------------
# index maintenance
# ---------------------------------------------------------------------------


def metadata_rowid(conn: sqlite3.Connection, kind: str, row_id: str) -> int:
    """Implicit integer rowid of a metadata row.

    :raises ValueError: the metadata row does not exist (index first, delete
        the metadata row after :func:`remove`).
    """
    spec = spec_for(kind)
    row = conn.execute(f"SELECT rowid FROM {spec.metadata_table} WHERE {spec.id_column}=?", (row_id,)).fetchone()
    if row is None:
        raise ValueError(f"{spec.metadata_table} has no row {row_id!r}")
    return int(row[0])


def store(conn: sqlite3.Connection, kind: str, row_id: str, vector: Sequence[float] | bytes) -> None:
    """Upsert one embedding into the index.

    The metadata row must already exist; the vector must match the index
    dimension.
    """
    spec = spec_for(kind)
    blob = bytes(vector) if isinstance(vector, bytes) else pack_vector(vector)
    dimension = index_dimension(conn, kind)
    if dimension is None:
        dimension = ensure_index(conn, kind, dimension=len(blob) // 4)
    if dimension != len(blob) // 4:
        raise ValueError(f"{spec.index_table} expects {dimension} dimensions, got {len(blob) // 4}")
    rowid = metadata_rowid(conn, kind, row_id)
    conn.execute(f"DELETE FROM {spec.index_table} WHERE rowid=?", (rowid,))
    conn.execute(f"INSERT INTO {spec.index_table}(rowid, embedding) VALUES (?, ?)", (rowid, blob))
    conn.commit()


def remove(conn: sqlite3.Connection, kind: str, row_id: str) -> bool:
    """Drop one embedding from the index; ``False`` when it was not indexed."""
    spec = spec_for(kind)
    row = conn.execute(f"SELECT rowid FROM {spec.metadata_table} WHERE {spec.id_column}=?", (row_id,)).fetchone()
    if row is None:
        return False
    cursor = conn.execute(f"DELETE FROM {spec.index_table} WHERE rowid=?", (int(row[0]),))
    conn.commit()
    return bool(cursor.rowcount)


def count(conn: sqlite3.Connection, kind: str) -> int:
    """Number of rows in the index (``0`` when it does not exist yet)."""
    spec = spec_for(kind)
    if index_dimension(conn, kind) is None:
        return 0
    row = conn.execute(f"SELECT count(*) FROM {spec.index_table}").fetchone()
    return int(row[0]) if row else 0


def prune(conn: sqlite3.Connection, kind: str) -> int:
    """Delete index rows whose metadata row is gone; return how many."""
    spec = spec_for(kind)
    if index_dimension(conn, kind) is None:
        return 0
    indexed = {int(row[0]) for row in conn.execute(f"SELECT rowid FROM {spec.index_table}")}
    alive = {int(row[0]) for row in conn.execute(f"SELECT rowid FROM {spec.metadata_table}")}
    stale = sorted(indexed - alive)
    for rowid in stale:
        conn.execute(f"DELETE FROM {spec.index_table} WHERE rowid=?", (rowid,))
    conn.commit()
    return len(stale)


def rebuild(conn: sqlite3.Connection, kind: str, *, dimension: int | None = None) -> int:
    """Recreate the index from the metadata table; return the number of rows.

    Nightly consolidation and re-indexing (GPU queue class 3) rewrite the index
    from the authoritative BLOB column, so it can never drift silently.
    """
    spec = spec_for(kind)
    load_extension(conn)
    if index_dimension(conn, kind) is not None:
        conn.execute(f"DROP TABLE IF EXISTS {spec.index_table}")
        conn.commit()
    target = ensure_index(conn, kind, dimension=dimension)
    rows = conn.execute(
        f"SELECT rowid, {spec.id_column}, vector FROM {spec.metadata_table} WHERE vector IS NOT NULL"
    ).fetchall()
    for rowid, row_id, blob in rows:
        if len(blob) // 4 != target:
            raise ValueError(
                f"{spec.metadata_table} row {row_id!r} has {len(blob) // 4} dimensions, index expects {target}"
            )
        conn.execute(f"INSERT INTO {spec.index_table}(rowid, embedding) VALUES (?, ?)", (int(rowid), blob))
    conn.commit()
    return len(rows)


def search(
    conn: sqlite3.Connection,
    kind: str,
    query: Sequence[float] | bytes,
    *,
    limit: int = 10,
    filter_column: str | None = None,
    filter_value: Any = None,
) -> list[VectorHit]:
    """Nearest neighbours of ``query`` in the index, closest first.

    ``filter_column`` is an optional metadata column (for example
    ``person_id`` or ``home_id``); both are bound as parameters.
    """
    spec = spec_for(kind)
    if limit < 1:
        raise ValueError("limit must be positive")
    blob = bytes(query) if isinstance(query, bytes) else pack_vector(query)
    dimension = index_dimension(conn, kind)
    if dimension is None:
        raise ValueError(f"{spec.index_table} does not exist; call ensure_index() first")
    if dimension != len(blob) // 4:
        raise ValueError(f"{spec.index_table} expects {dimension} dimensions, got {len(blob) // 4}")
    if filter_column is not None and not _IDENTIFIER_RE.fullmatch(filter_column):
        raise ValueError(f"invalid filter column {filter_column!r}")

    sql = (
        f"SELECT m.{spec.id_column}, v.distance FROM {spec.index_table} v "
        f"JOIN {spec.metadata_table} m ON m.rowid = v.rowid "
        "WHERE v.embedding MATCH ? AND k = ?"
    )
    params: list[Any] = [blob, int(limit)]
    if filter_column is not None:
        sql += f" AND m.{filter_column} = ?"
        params.append(filter_value)
    sql += " ORDER BY v.distance"
    rows = conn.execute(sql, params).fetchall()
    return [VectorHit(row_id=str(row[0]), distance=float(row[1])) for row in rows]


__all__ = [
    "DEFAULT_DIMENSIONS",
    "EXTENSION_ENV_VAR",
    "INDEX_SPECS",
    "MAX_DIMENSION",
    "VENDOR_DIR",
    "VectorExtensionUnavailable",
    "VectorHit",
    "VectorIndexSpec",
    "count",
    "ensure_index",
    "ensure_indexes",
    "extension_filename",
    "index_dimension",
    "is_loaded",
    "load_extension",
    "locate_extension",
    "metadata_dimension",
    "metadata_rowid",
    "pack_vector",
    "prune",
    "rebuild",
    "remove",
    "search",
    "spec_for",
    "store",
    "unpack_vector",
]
