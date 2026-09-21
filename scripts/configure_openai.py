"""Create a separate budgeted profile, preserving this machine's local devices.

Run with the jarvis Python: python scripts/configure_openai.py
No credentials or network I/O. Never modifies config.yaml or overwrites output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml

from common.config import Config, load_config


def create_profile(source: Path, output: Path) -> None:
    data = load_config(source).model_dump()
    llm = data["server"]["llm"]
    local_vision = llm.get("vision_base_url") or llm["base_url"]
    if llm["provider"] != "ollama_native" and not llm.get("vision_base_url"):
        local_vision = "http://127.0.0.1:11434/v1"
    llm.update(provider="openai_responses", model="gpt-5.4-mini",
               base_url="https://api.openai.com/v1", api_key="", api_key_env="OPENAI_API_KEY",
               monthly_budget_usd=18.0, max_input_bytes=64000,
               max_tokens=600, max_tool_rounds=4, history_turns=25,
               vision_base_url=local_vision, prompt_file="prompts/cloud.md", verify_actions=False)
    data["server"]["face"]["greeting_llm"] = False
    data["client"].update(attention_mode="wake_word", followup_window_s=0)
    Config.model_validate(data)
    with output.open("x", encoding="utf-8") as handle:
        handle.write("# Local OpenAI profile. API key is read from OPENAI_API_KEY, never from this file.\n")
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "config.openai.yaml")
    args = parser.parse_args()
    create_profile(args.source, args.output)
    print(f"Created {args.output}. Start both sides with --config {args.output.name}.")


if __name__ == "__main__":
    main()
