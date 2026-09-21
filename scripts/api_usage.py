"""Print this month's recorded Jarvis API usage without any network requests."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config
from hub.api_budget import ApiBudget

if __name__ == "__main__":
    path = ROOT / "data" / "api_usage.sqlite3"
    if not path.exists():
        print("No API usage ledger yet; Jarvis has not reserved any API requests here.")
    else:
        profile = ROOT / "config.openai.yaml"
        limit = load_config(profile).server.llm.monthly_budget_usd if profile.exists() else 18.0
        print(json.dumps(ApiBudget(path, limit).status(), indent=2))
