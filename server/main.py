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
from server import app as server_app  # noqa: E402

log = logging.getLogger("jarvis.server.main")

DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="server.main",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _force_utf8_console()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

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

    server_app.configure(cfg)
    uvicorn_config = uvicorn.Config(
        app=server_app.app,
        host=str(cfg.server.host),
        port=int(cfg.server.port),
        log_level=args.log_level.lower(),
        access_log=False,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
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
