"""Индекс «векторы одного трека за день» (ТЗ F-203/F-206).

F-203 writes one body vector per crop, and F-206 asks "what does this track
look like today" for every live track once a second. Without an index that
question scans the whole ``body_embeddings`` table, which grows by a few rows
per person per minute.
"""
VERSION = 5
NAME = "body_embedding_track"


def apply(conn):
    conn.execute(
        "CREATE INDEX idx_body_embeddings_track ON body_embeddings(track_id, session_day)"
    )


__all__ = ["NAME", "VERSION", "apply"]
