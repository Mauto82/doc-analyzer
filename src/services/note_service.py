"""
NoteService: persist and retrieve Q&A notes in a local SQLite database.

The DB file lives in the same data directory as Qdrant (configurable via
NOTE_DB_PATH env var) so it is covered by the same Docker volume mount.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

from src.models.note import Note

_DEFAULT_DB_PATH = os.path.join(
    os.getenv("QDRANT_DB_PATH", "./data/qdrant").rsplit("/", 1)[0],
    "notes.db",
)
NOTE_DB_PATH = os.getenv("NOTE_DB_PATH", _DEFAULT_DB_PATH)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    sources     TEXT    NOT NULL DEFAULT '[]',
    collection  TEXT    NOT NULL DEFAULT 'default',
    role        TEXT    NOT NULL DEFAULT 'default',
    created_at  TEXT    NOT NULL
);
"""


class NoteService:
    def __init__(self, db_path: str = NOTE_DB_PATH) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    # ── Internal ──────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> Note:
        return Note(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            sources=json.loads(row["sources"]),
            collection=row["collection"],
            role=row["role"],
            created_at=row["created_at"],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, note: Note) -> Note:
        """Insert a new note and return it with the assigned id."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO notes (question, answer, sources, collection, role, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (note.question, note.answer, json.dumps(note.sources),
                 note.collection, note.role, note.created_at),
            )
            note.id = cur.lastrowid
        return note

    def list(self, limit: int = 100, offset: int = 0) -> List[Note]:
        """Return notes ordered by most recent first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM notes ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def get(self, note_id: int) -> Optional[Note]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        return self._row_to_note(row) if row else None

    def delete(self, note_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cur.rowcount > 0
