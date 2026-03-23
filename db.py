"This module acts as the metadata layer in the RAG pipeline, mapping FAISS vector indices to "
"actual text chunks and tracking ingestion to avoid redundant processing."
"""
db.py — SQLite operations for chunk storage and ingestion tracking.

Why SQLite alongside FAISS?
  FAISS stores float32 vectors and returns integer positions (0, 1, 2...).
  It has no concept of text or source filenames.
  SQLite maps each FAISS position → chunk text + source filename,
  so the retriever can return human-readable context to the user.

Tables:
  chunks        — (id, source, text) one row per text chunk
  ingestion_log — (filename) tracks which files have been embedded
                  so we skip them on restart instead of re-embedding
"""

import sqlite3

from config import settings


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row   # row["text"] instead of row[0]
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT    NOT NULL,
                text   TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                filename    TEXT PRIMARY KEY,
                ingested_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


def insert_chunks(source: str, chunks: list[str]) -> list[int]:
    """
    Insert chunk rows and return their auto-assigned row IDs.
    The IDs are stored in faiss_id_map so FAISS positions can be
    translated back to readable text at query time.
    """
    ids: list[int] = []
    with get_connection() as conn:
        for text in chunks:
            cur = conn.execute(
                "INSERT INTO chunks (source, text) VALUES (?, ?)",
                (source, text),
            )
            ids.append(cur.lastrowid)
        conn.commit()
    return ids


def get_chunks_by_ids(ids: list[int]) -> list[sqlite3.Row]:
    """Fetch chunk rows by primary-key IDs, preserving the input order."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT id, source, text FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    row_map = {row["id"]: row for row in rows}
    return [row_map[i] for i in ids if i in row_map]


def mark_ingested(filename: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ingestion_log (filename) VALUES (?)",
            (filename,),
        )
        conn.commit()


def is_ingested(filename: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM ingestion_log WHERE filename = ?", (filename,)
        ).fetchone()
    return row is not None


def get_total_chunks() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]