"""Download/check local Community-1, then optionally enable a selected profile.

Uses HF_TOKEN or the existing Hugging Face login; never prints credentials.
The user must already have accepted the model's access conditions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config
from hub.diarization import DiarizationEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.openai.yaml")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    target = Path(cfg.server.diarization.model_path)
    if not target.is_absolute():
        target = ROOT / target
    if args.download:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
        try:
            snapshot_download("pyannote/speaker-diarization-community-1", local_dir=target,
                              allow_patterns=["*.yaml", "*.bin", "*.safetensors", "*.pt", "*.npz", "*.md"])
        except HfHubHTTPError as exc:
            print(f"Model download failed (HTTP {exc.response.status_code}). "
                  "Check Hugging Face access/login; no profile was changed.")
            return 1
    engine = DiarizationEngine(cfg.server.diarization)
    engine.load()
    engine.diarize(b"\0" * (16000 * 2 * 2), 16000)
    print(f"Local diarization loaded and inference completed on {cfg.server.diarization.device}.")
    if args.enable:
        import yaml
        data = cfg.model_dump()
        data["server"]["diarization"]["enabled"] = True
        # Keep enough pre-trigger audio to attribute the wake word itself.
        data["client"]["vad"]["pre_roll_ms"] = max(1500, data["client"]["vad"]["pre_roll_ms"])
        args.config.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"Enabled in {args.config.name}. Restart both peers with this profile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
