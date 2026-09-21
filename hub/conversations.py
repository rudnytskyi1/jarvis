"""Durable, person-scoped conversation archive; never pool unknown voices."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


class Conversations:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "conversations.sqlite3"
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, person TEXT NOT NULL, ts TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, source TEXT UNIQUE)")
            db.execute("CREATE INDEX IF NOT EXISTS person_turns ON turns(person, id)")
            db.execute("CREATE TABLE IF NOT EXISTS imports (path TEXT PRIMARY KEY)")
        # Import existing known-person dialogs once, without exposing mixed speech.
        for path in sorted((root / "dialogs").glob("*.jsonl")):
            with self._db() as db:
                if db.execute("SELECT 1 FROM imports WHERE path=?", (str(path),)).fetchone():
                    continue
                for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    person = self.key(row.get("speaker"))
                    if person and not row.get("note") and row.get("transcript") and row.get("reply"):
                        # A log written after midnight may already have been
                        # archived live before this process was restarted.
                        if db.execute("SELECT 1 FROM turns WHERE person=? AND ts=? AND question=? AND answer=? LIMIT 1",
                                      (person, row.get('ts', ''), row['transcript'], row['reply'])).fetchone():
                            continue
                        db.execute("INSERT OR IGNORE INTO turns(person,ts,question,answer,source) VALUES(?,?,?,?,?)",
                                   (person, row.get("ts", ""), row["transcript"], row["reply"], f"{path}:{index}"))
                db.execute("INSERT INTO imports VALUES(?)", (str(path),))

    def _db(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def key(person):
        name = str(person or "").strip().casefold()
        return "" if name in {"", "unknown", "guest", "user", "friend", "none"} else name

    def append(self, person, ts, question, answer):
        name = self.key(person)
        if name and question and answer:
            with self._db() as db:
                db.execute("INSERT INTO turns(person,ts,question,answer) VALUES(?,?,?,?)", (name, ts, question, answer))

    def begin(self, person, ts, question):
        name = self.key(person)
        if name and question:
            with self._db() as db:
                return db.execute('INSERT INTO turns(person,ts,question,answer) VALUES(?,?,?,?)',
                                  (name, ts, question, '[Request interrupted or answer not completed.]')).lastrowid
        return None

    def finish(self, turn_id, answer):
        if turn_id:
            with self._db() as db:
                db.execute('UPDATE turns SET answer=? WHERE id=?', (answer, turn_id))

    def recent(self, person, limit=30):
        name = self.key(person)
        if not name:
            return []
        with self._db() as db:
            rows = db.execute("SELECT id,ts,question,answer FROM turns WHERE person=? ORDER BY id DESC LIMIT ?", (name, min(100, max(1, limit)))).fetchall()
        return [dict(id=r[0], person=name, ts=r[1], question=r[2], answer=r[3]) for r in reversed(rows)]

    def recall(self, person, query, limit=12, since='', until=''):
        # Parameterized and person-filtered before ranking. The full archive stays
        # on disk; only a bounded relevant subset enters the model's context.
        name = self.key(person)
        terms = list(dict.fromkeys(re.findall(r"[\w]{3,}", query.casefold())))[:12]
        if not name:
            return []
        filters, params = ['person=?'], [name]
        if since:
            filters.append('ts >= ?')
            params.append(since)
        if until:
            filters.append('ts <= ?')
            params.append(until)
        with self._db() as db:
            rows = db.execute('SELECT id,ts,question,answer FROM turns WHERE ' + ' AND '.join(filters) + ' ORDER BY id DESC', params).fetchall()
        if terms:
            rows = [r for r in rows if any(t in (r[2] + ' ' + r[3]).casefold() for t in terms)]
        rows.sort(key=lambda r: sum(t in (r[2] + " " + r[3]).casefold() for t in terms), reverse=True)
        return [dict(id=r[0], person=name, ts=r[1], question=r[2], answer=r[3]) for r in rows[:min(25, max(1, int(limit)))]]

    def rename(self, old, new):
        if self.key(old) and self.key(new):
            with self._db() as db:
                db.execute("UPDATE turns SET person=? WHERE person=?", (self.key(new), self.key(old)))
