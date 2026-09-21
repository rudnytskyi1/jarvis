"""sqlite-vec indexes over the hub embedding tables (ТЗ 4.6, section 14)."""
from __future__ import annotations

import json
import sqlite3

import pytest

from common.config import Config
from hub import migrations_runner as runner
from hub import vectors


def _conn(tmp_path, *, index: bool = True) -> sqlite3.Connection:
    conn = runner.connect(str(tmp_path / "hub.db"))
    runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('anton', 'Anton')")
    conn.commit()
    if index:
        vectors.ensure_indexes(conn)
    return conn


def _add_person(conn: sqlite3.Connection, person_id: str, name: str) -> None:
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?, ?)", (person_id, name))
    conn.commit()


def _add_voice(conn: sqlite3.Connection, embedding_id: str, person_id: str, vector: list[float]) -> None:
    conn.execute(
        "INSERT INTO voice_embeddings(id, person_id, vector, dim) VALUES (?, ?, ?, ?)",
        (embedding_id, person_id, vectors.pack_vector(vector), len(vector)),
    )
    conn.commit()


def test_extension_is_available_and_reports_a_version(tmp_path):
    conn = _conn(tmp_path)
    try:
        assert vectors.locate_extension() is not None
        version = vectors.load_extension(conn)
        assert version.startswith("v")
        assert vectors.is_loaded(conn)
    finally:
        conn.close()


def test_ensure_indexes_creates_every_kind_with_its_default_dimension(tmp_path):
    conn = _conn(tmp_path)
    try:
        for kind, spec in vectors.INDEX_SPECS.items():
            assert vectors.index_dimension(conn, kind) == spec.default_dimension
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {spec.index_table for spec in vectors.INDEX_SPECS.values()} <= names
    finally:
        conn.close()


def test_ensure_indexes_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    try:
        first = vectors.ensure_indexes(conn)
        assert vectors.ensure_indexes(conn) == first
    finally:
        conn.close()


def test_dimension_comes_from_stored_rows_not_the_default(tmp_path):
    conn = _conn(tmp_path)
    try:
        conn.execute("DROP TABLE vec_voice_embeddings")
        _add_voice(conn, "v1", "anton", [0.1] * 320)
        assert vectors.ensure_index(conn, "voice") == 320
        assert vectors.index_dimension(conn, "voice") == 320
    finally:
        conn.close()


def test_mixed_stored_dimensions_are_refused(tmp_path):
    conn = _conn(tmp_path)
    try:
        conn.execute("DROP TABLE vec_voice_embeddings")
        _add_voice(conn, "v1", "anton", [0.1] * 192)
        _add_voice(conn, "v2", "anton", [0.1] * 256)
        with pytest.raises(ValueError, match="mixed vector dimensions"):
            vectors.ensure_index(conn, "voice")
    finally:
        conn.close()


def test_store_search_returns_closest_row_first(tmp_path):
    conn = _conn(tmp_path)
    try:
        _add_voice(conn, "near", "anton", [1.0, 0.0, 0.0])
        _add_voice(conn, "far", "anton", [0.0, 1.0, 0.0])
        conn.execute("DROP TABLE vec_voice_embeddings")
        vectors.ensure_index(conn, "voice", dimension=3)
        vectors.store(conn, "voice", "near", [1.0, 0.0, 0.0])
        vectors.store(conn, "voice", "far", [0.0, 1.0, 0.0])

        hits = vectors.search(conn, "voice", [0.9, 0.1, 0.0], limit=2)
        assert [hit.row_id for hit in hits] == ["near", "far"]
        assert hits[0].distance < hits[1].distance
    finally:
        conn.close()


def test_search_can_filter_by_metadata_column(tmp_path):
    conn = _conn(tmp_path)
    try:
        _add_person(conn, "max", "Max")
        _add_voice(conn, "a1", "anton", [1.0, 0.0, 0.0])
        _add_voice(conn, "a2", "max", [0.99, 0.01, 0.0])
        conn.execute("DROP TABLE vec_voice_embeddings")
        vectors.ensure_index(conn, "voice", dimension=3)
        vectors.store(conn, "voice", "a1", [1.0, 0.0, 0.0])
        vectors.store(conn, "voice", "a2", [0.99, 0.01, 0.0])

        hits = vectors.search(conn, "voice", [1.0, 0.0, 0.0], limit=5, filter_column="person_id",
                              filter_value="anton")
        assert [hit.row_id for hit in hits] == ["a1"]
    finally:
        conn.close()


def test_store_upserts_instead_of_duplicating(tmp_path):
    conn = _conn(tmp_path)
    try:
        _add_voice(conn, "v1", "anton", [1.0, 0.0, 0.0])
        conn.execute("DROP TABLE vec_voice_embeddings")
        vectors.ensure_index(conn, "voice", dimension=3)
        vectors.store(conn, "voice", "v1", [1.0, 0.0, 0.0])
        vectors.store(conn, "voice", "v1", [0.0, 1.0, 0.0])

        assert vectors.count(conn, "voice") == 1
        hits = vectors.search(conn, "voice", [0.0, 1.0, 0.0], limit=1)
        assert hits[0].row_id == "v1"
        assert hits[0].distance == pytest.approx(0.0)
    finally:
        conn.close()


def test_wrong_dimension_is_refused(tmp_path):
    conn = _conn(tmp_path)
    try:
        _add_voice(conn, "v1", "anton", [1.0, 0.0, 0.0])
        conn.execute("DROP TABLE vec_voice_embeddings")
        vectors.ensure_index(conn, "voice", dimension=3)
        with pytest.raises(ValueError, match="expects 3 dimensions"):
            vectors.store(conn, "voice", "v1", [1.0, 0.0])
        with pytest.raises(ValueError, match="expects 3 dimensions"):
            vectors.search(conn, "voice", [1.0, 0.0])
    finally:
        conn.close()


def test_store_requires_the_metadata_row(tmp_path):
    conn = _conn(tmp_path)
    try:
        with pytest.raises(ValueError, match="has no row"):
            vectors.store(conn, "voice", "missing", [0.0] * 192)
    finally:
        conn.close()


def test_remove_and_prune_drop_stale_index_rows(tmp_path):
    conn = _conn(tmp_path)
    try:
        _add_voice(conn, "v1", "anton", [0.1] * 192)
        _add_voice(conn, "v2", "anton", [0.2] * 192)
        vectors.store(conn, "voice", "v1", [0.1] * 192)
        vectors.store(conn, "voice", "v2", [0.2] * 192)

        assert vectors.remove(conn, "voice", "v1") is True
        assert vectors.remove(conn, "voice", "v1") is False
        assert vectors.count(conn, "voice") == 1

        conn.execute("DELETE FROM voice_embeddings WHERE id='v2'")
        conn.commit()
        assert vectors.prune(conn, "voice") == 1
        assert vectors.count(conn, "voice") == 0
    finally:
        conn.close()


def test_rebuild_restores_the_index_from_the_metadata_rows(tmp_path):
    conn = _conn(tmp_path)
    try:
        _add_voice(conn, "v1", "anton", [0.1] * 192)
        _add_voice(conn, "v2", "anton", [0.2] * 192)
        assert vectors.rebuild(conn, "voice") == 2
        assert vectors.count(conn, "voice") == 2
        assert [hit.row_id for hit in vectors.search(conn, "voice", [0.1] * 192, limit=2)][0] == "v1"

        vectors.store(conn, "voice", "v1", [0.0] * 191 + [1.0])
        assert vectors.count(conn, "voice") == 2
        assert vectors.rebuild(conn, "voice") == 2
        assert [hit.row_id for hit in vectors.search(conn, "voice", [0.1] * 191 + [0.0], limit=1)] == ["v1"]
    finally:
        conn.close()


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown vector kind"):
        vectors.spec_for("smell")


def test_pack_vector_rejects_bad_input():
    with pytest.raises(ValueError, match="empty"):
        vectors.pack_vector([])
    with pytest.raises(ValueError, match="non-finite"):
        vectors.pack_vector([float("nan")])
    packed = vectors.pack_vector([1.5, -2.0])
    assert vectors.unpack_vector(packed) == [1.5, -2.0]
    with pytest.raises(ValueError, match="multiple of 4"):
        vectors.unpack_vector(b"\x00\x01\x02")


def test_missing_extension_raises_instead_of_faking_results(tmp_path, monkeypatch):
    monkeypatch.setenv(vectors.EXTENSION_ENV_VAR, str(tmp_path / "nope.dll"))
    conn = _conn(tmp_path, index=False)
    try:
        with pytest.raises(vectors.VectorExtensionUnavailable, match="not available"):
            vectors.load_extension(conn)
    finally:
        conn.close()


def test_vendored_extension_matches_the_platform_name():
    assert (vectors.VENDOR_DIR / vectors.extension_filename()).is_file()


def test_vector_kinds_in_the_config_match_the_index_specs():
    cfg = Config.model_validate({"server": {"vectors": {"dimensions": {"voice": 256}}}})
    assert cfg.server.vectors.dimensions == {"voice": 256}
    assert set(vectors.INDEX_SPECS) == {"voice", "face", "body", "memory", "objects"}
    for kind in vectors.INDEX_SPECS:
        Config.model_validate({"server": {"vectors": {"dimensions": {kind: 8}}}})


def test_unknown_vector_kind_in_config_is_a_typo_not_silence():
    with pytest.raises(Exception, match="unknown vector kind"):
        Config.model_validate({"server": {"vectors": {"dimensions": {"smell": 8}}}})


def test_vector_config_defaults_are_on_and_empty():
    vectors_cfg = Config().server.vectors
    assert vectors_cfg.enabled is True
    assert vectors_cfg.extension_path == ""
    assert vectors_cfg.dimensions == {}


def test_hub_startup_prepares_the_indexes(tmp_path, monkeypatch):
    from hub import main as hub_main

    conn = _conn(tmp_path)
    try:
        cfg = Config()
        assert hub_main._prepare_vector_indexes(conn, cfg) == vectors.DEFAULT_DIMENSIONS

        cfg.server.vectors.enabled = False
        assert hub_main._prepare_vector_indexes(conn, cfg) == {}
    finally:
        conn.close()


def test_search_on_missing_index_is_an_error_not_an_empty_result(tmp_path):
    conn = runner.connect(str(tmp_path / "hub.db"))
    runner.migrate(conn)
    try:
        with pytest.raises(ValueError, match="does not exist"):
            vectors.search(conn, "voice", [0.0] * 192)
    finally:
        conn.close()


def test_memory_and_objects_indexes_use_their_own_primary_keys(tmp_path):
    conn = _conn(tmp_path)
    try:
        conn.execute(
            "INSERT INTO memories(memory_id, scope, owner_id, kind, text, vector, dim) "
            "VALUES ('m1', 'home', 'livingroom', 'fact', 'the kettle is blue', ?, 4)",
            (vectors.pack_vector([0.0, 1.0, 0.0, 0.0]),),
        )
        conn.execute(
            "INSERT INTO objects_index(id, home_id, ts, label, vector, dim) "
            "VALUES ('o1', 'livingroom', 1.0, 'kettle', ?, 4)",
            (vectors.pack_vector([0.0, 0.0, 1.0, 0.0]),),
        )
        conn.commit()
        assert vectors.rebuild(conn, "memory", dimension=4) == 1
        assert vectors.rebuild(conn, "objects", dimension=4) == 1
        assert vectors.search(conn, "memory", [0.0, 1.0, 0.0, 0.0], limit=1)[0].row_id == "m1"
        assert vectors.search(conn, "objects", [0.0, 0.0, 1.0, 0.0], limit=1)[0].row_id == "o1"
    finally:
        conn.close()


def test_search_filter_column_is_validated(tmp_path):
    conn = _conn(tmp_path)
    try:
        with pytest.raises(ValueError, match="invalid filter column"):
            vectors.search(conn, "voice", [0.0] * 192, filter_column="person_id; DROP TABLE homes")
    finally:
        conn.close()


def test_vectors_are_stored_as_float32_little_endian():
    packed = vectors.pack_vector([1.0, 0.5])
    assert len(packed) == 8
    assert packed == b"\x00\x00\x80?\x00\x00\x00?"
    assert json.dumps(vectors.unpack_vector(packed)) == "[1.0, 0.5]"
