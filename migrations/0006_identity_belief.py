"""Итог слияния сигналов: belief по каждому треку (ТЗ F-206).

One row per live track: the person the fusion of voice, face and body believes
this body is, with the probability and the numbers behind it (``sources_json``)
that F-215 quotes when somebody asks "почему ты решил, что это Макс". The row is
replaced by every second-long pass, and disappears with its track.
"""
VERSION = 6
NAME = "identity_belief"


def apply(conn):
    conn.execute(
        """
        CREATE TABLE identity_belief (
            track_id TEXT PRIMARY KEY REFERENCES tracks(track_id) ON DELETE CASCADE,
            home_id TEXT REFERENCES homes(home_id) ON DELETE CASCADE,
            person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            p REAL NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '{}',
            at REAL NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX idx_identity_belief_home ON identity_belief(home_id, p)")
    conn.execute("CREATE INDEX idx_identity_belief_person ON identity_belief(person_id)")


__all__ = ["NAME", "VERSION", "apply"]
