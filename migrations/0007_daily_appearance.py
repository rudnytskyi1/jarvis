"""«Внешность дня»: усреднённый вектор тела за сутки (ТЗ F-209).

The body vectors of a day (F-203) are raw observations; the nightly job of
F-209 turns the ones that belong to a known person into ONE "appearance of the
day" row and keeps it for a week of statistics, while the unattached clusters
are deleted (nobody was identified there, so there is no person to keep them
for). ``expires_at`` is what the same job reads to drop the row a week later.
"""
VERSION = 7
NAME = "daily_appearance"


def apply(conn):
    conn.execute(
        """
        CREATE TABLE daily_appearance (
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            day TEXT NOT NULL,
            home_id TEXT REFERENCES homes(home_id) ON DELETE CASCADE,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            samples INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            expires_at REAL NOT NULL,
            PRIMARY KEY (person_id, day)
        )"""
    )
    conn.execute("CREATE INDEX idx_daily_appearance_expires ON daily_appearance(expires_at)")
    conn.execute("CREATE INDEX idx_daily_appearance_person ON daily_appearance(person_id, day)")


__all__ = ["NAME", "VERSION", "apply"]
