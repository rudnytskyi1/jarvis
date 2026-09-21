"""Make the room client's cloud-mode profile from its own device settings.

Run on the room PC. No OpenAI key is needed or copied to this machine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from common.config import Config, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "config.openai.yaml")
    parser.add_argument("--server-url", required=True)
    args = parser.parse_args()
    url = urlsplit(args.server_url)
    if url.scheme not in {"ws", "wss"} or not url.hostname or url.username or url.password:
        parser.error("Use a ws:// or wss:// server URL without credentials")
    data = load_config(args.source).model_dump()
    data["client"].update(server_url=args.server_url, attention_mode="wake_word", followup_window_s=0)
    data["client"]["vad"]["pre_roll_ms"] = max(1500, data["client"]["vad"]["pre_roll_ms"])
    Config.model_validate(data)
    with args.output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    print("Room profile created; microphone, speakers, camera and device settings preserved.")


if __name__ == "__main__":
    main()
