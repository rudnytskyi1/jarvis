"""Remove the facts the audit bench wrote into the live memory (AU-08).

The bench of the mass audit used the live hub's own stores until AU-08 fixed
it: `scripts/live-eval.py` called a bare ``Memory()`` (that is
``data/memory.jsonl``) and let ``hub.app`` open the live ``data/hub.db``, so
every scenario that asked "remember that I like tea" wrote a sentence into the
OWNER's memory, and the room then told the owner "you already told me that"
(DECISIONS.md TEST-DB-01, AU-08).

This script is the one-off cleanup of what those runs left behind. It is
deliberately conservative:

* the file store keeps every row written before the first bench run
  (``--file-cutoff``, local time, default the start of 2026-09-23);
* the ``memories`` table keeps the owner's own facts, named explicitly in
  ``OWNER_TEXTS`` and printed before anything is written;
* every removed row is copied to ``--backup`` first.

    python scripts/clean-bench-memory.py                    # dry run
    python scripts/clean-bench-memory.py --write
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_FILE = REPO_ROOT / "data" / "memory.jsonl"
HUB_DB = REPO_ROOT / "data" / "hub.db"

#: The owner's own facts in the ``memories`` table. Everything else in that
#: table on 2026-09-23 is a sentence the audit corpus asked for ("Anton likes
#: tea.", "... allergic to peanuts.") or an artifact of its runs
#: ("placeholder", "This person should be saved as Max."), so it goes.
OWNER_TEXTS = {
    "i keep my keys in the top drawer.",
    "я пью кофе по утрам",
    "anton studies at nine.",
    "preferred browser: google chrome",
    "likes jazz",
}
#: Long owner sentences, matched by prefix because they end with a count.
OWNER_PREFIXES = ("when the owner asks to close an app",)


def is_owner_fact(text: str) -> bool:
    lowered = str(text or "").strip().casefold()
    return lowered in OWNER_TEXTS or lowered.startswith(OWNER_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-cutoff", default="2026-09-23T00:16:00",
                        help="локальное время первого прогона стенда")
    parser.add_argument("--backup", default="data/audit/memory-bench-cleanup.jsonl")
    parser.add_argument("--write", action="store_true", help="иначе только показать")
    args = parser.parse_args()
    backup = Path(args.backup)
    if not backup.is_absolute():
        backup = REPO_ROOT / backup

    rows = [json.loads(line) for line in MEMORY_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    keep = [row for row in rows if str(row.get("ts", "")) < args.file_cutoff]
    dropped = [row for row in rows if str(row.get("ts", "")) >= args.file_cutoff]
    print(f"data/memory.jsonl: {len(rows)} строк, оставить {len(keep)}, убрать {len(dropped)}")
    for row in keep:
        print(f"  keep  {row.get('ts')}  {str(row.get('fact'))[:60]}")

    conn = sqlite3.connect(f"file:{HUB_DB}?mode=ro", uri=True)
    table = conn.execute(
        "SELECT memory_id, scope, owner_id, kind, text, created_at FROM memories").fetchall()
    conn.close()
    keep_sql = [row for row in table if is_owner_fact(row[4])]
    drop_sql = [row for row in table if not is_owner_fact(row[4])]
    print(f"memories: {len(table)} строк, оставить {len(keep_sql)}, убрать {len(drop_sql)}")
    for row in keep_sql:
        print(f"  keep  {row[2]:10s} {str(row[4])[:60]}")

    if not args.write:
        print("dry run: ничего не записано (--write чтобы применить)")
        return 0

    backup.parent.mkdir(parents=True, exist_ok=True)
    with backup.open("w", encoding="utf-8") as handle:
        for row in dropped:
            handle.write(json.dumps({"store": "file", **row}, ensure_ascii=False) + "\n")
        for row in drop_sql:
            handle.write(json.dumps({"store": "memories", "memory_id": row[0], "scope": row[1],
                                     "owner_id": row[2], "kind": row[3], "text": row[4],
                                     "created_at": row[5]}, ensure_ascii=False) + "\n")
    MEMORY_FILE.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keep),
                           encoding="utf-8")
    conn = sqlite3.connect(str(HUB_DB), timeout=15)
    with conn:
        conn.executemany("DELETE FROM memories WHERE memory_id=?",
                         [(row[0],) for row in drop_sql])
    left = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    print(f"записано: backup {backup} ({len(dropped) + len(drop_sql)} строк), "
          f"memory.jsonl {len(keep)} строк, memories {left} строк")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
