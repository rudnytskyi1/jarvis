"""Entry point for the brain server: ``python -m server.main`` from the repo root."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn  # noqa: E402  (import after sys.path setup)

from common.config import load_config  # noqa: E402
from hub import app as server_app  # noqa: E402

log = logging.getLogger("jarvis.server.main")

DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
#: The server runs in its own console window opened by start-jarvis-server.bat,
#: so its scrollback is gone the moment that window is closed - and it is the
#: only record of why a greeting did or did not fire, why a reply was slow, and
#: what the model actually called. Mirror everything to a file (same crude
#: rotation as the client's data/client.log) so it can be read afterwards.
LOG_FILE_PATH = REPO_ROOT / "data" / "server.log"
LOG_FILE_MAX_BYTES = 5 * 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hub.main",
        description="Jarvis brain server: Whisper STT + LLM with tools + Silero TTS",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"path to config.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def _force_utf8_console() -> None:
    """Log messages must not blow up on a cp866/cp1252 Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _add_file_logging() -> None:
    """Mirror the console log to :data:`LOG_FILE_PATH` (best-effort, never fatal)."""
    try:
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_FILE_PATH.exists() and LOG_FILE_PATH.stat().st_size > LOG_FILE_MAX_BYTES:
            LOG_FILE_PATH.unlink()  # crude rotation: start over past 5 MB
        handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(handler)
    except Exception as exc:  # noqa: BLE001 - file logging is a convenience, not a dependency
        log.warning("File logging unavailable (%s) - console only", exc)


def _prepare_hub_database(cfg) -> None:
    """Create/upgrade data/hub.db and seed the configured rooms (ТЗ 4.6, 4.7)."""
    from hub import migrations_runner
    from hub.homes import sync_homes_from_config

    db_path = REPO_ROOT / "data" / "hub.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = migrations_runner.connect(str(db_path))
    try:
        applied = migrations_runner.migrate(conn)
        changed = sync_homes_from_config(conn, getattr(cfg, "homes", []) or [])
        legacy_counts = _import_legacy_data(conn)
        media_counts = _cleanup_media(conn, cfg)
        vector_dims = _prepare_vector_indexes(conn, cfg)
    finally:
        conn.close()
    if applied:
        log.info("Hub database migrated: %s", applied)
    if changed:
        log.info("Rooms refreshed from config: %s", changed)
    if legacy_counts:
        log.info("Legacy data imported into the hub database: %s", legacy_counts)
    if media_counts and media_counts.get("expired_rows"):
        log.info("Expired media cleaned up: %s", media_counts)
    if vector_dims:
        log.info("Vector indexes ready: %s", vector_dims)


def _import_legacy_data(conn) -> dict:
    """Copy the single-room stores into the hub database once (ТЗ 4.6)."""
    try:
        from hub.legacy_migrate import migrate_legacy

        return migrate_legacy(conn)
    except Exception as exc:  # noqa: BLE001 - legacy import must not stop the hub
        log.warning("Legacy data import skipped (%s); old files stay as backup", exc)
        return {}


def _cleanup_media(conn, cfg) -> dict:
    """Remove expired room media rows and files at startup (ТЗ 4.6/F-304)."""
    try:
        from hub.media import MediaStore

        media = getattr(cfg.server, "media", None)
        store = MediaStore(
            conn,
            REPO_ROOT / "data",
            media_ttl_days=getattr(media, "media_ttl_days", 3),
            clip_ttl_days=getattr(media, "clip_ttl_days", 7),
        )
        return store.cleanup_expired()
    except Exception as exc:  # noqa: BLE001 - retention cleanup must not stop the hub
        log.warning("Media cleanup skipped (%s); expired rows stay until the next start", exc)
        return {}


def _prepare_vector_indexes(conn, cfg) -> dict:
    """Create/refresh the sqlite-vec search indexes at startup (ТЗ 4.6).

    Vector search is an index over data that stays in the ordinary tables, so a
    machine without the extension only loses KNN search: the reason is logged
    and startup continues.
    """
    vectors = getattr(getattr(cfg, "server", None), "vectors", None)
    if not getattr(vectors, "enabled", True):
        log.info("Vector indexes disabled by config (server.vectors.enabled: false)")
        return {}
    from hub.vectors import VectorExtensionUnavailable, ensure_indexes

    try:
        return ensure_indexes(
            conn,
            dimensions=getattr(vectors, "dimensions", None),
            extension_path=getattr(vectors, "extension_path", "") or None,
        )
    except (VectorExtensionUnavailable, ValueError) as exc:
        log.warning("Vector search unavailable (%s); embeddings stay in the metadata tables", exc)
        return {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _force_utf8_console()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
    _add_file_logging()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        log.error(
            "Config %s not found. Copy config.example.yaml to config.yaml and adjust it.",
            config_path,
        )
        return 2

    try:
        cfg = load_config(str(config_path))
    except Exception as exc:
        log.error("Could not load the config %s: %s", config_path, exc)
        return 2

    try:
        _prepare_hub_database(cfg)
    except Exception as exc:  # noqa: BLE001 - the hub must still start single-room
        log.warning("Hub database is unavailable (%s); multi-room features stay off", exc)

    server_app.configure(cfg)
    log.info("GPU queue: %s", server_app.gpu_queue_status())
    uvicorn_config = uvicorn.Config(
        app=server_app.app,
        host=str(cfg.server.host),
        port=int(cfg.server.port),
        log_level=args.log_level.lower(),
        access_log=False,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
        ws_max_size=24_000_000,
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(uvicorn_config)
    log.info("Listening on ws://%s:%s/ws", cfg.server.host, cfg.server.port)
    try:
        server.run()
    except KeyboardInterrupt:
        log.info("Stopped by the user (Ctrl+C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
