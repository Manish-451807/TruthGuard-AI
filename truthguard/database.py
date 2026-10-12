"""
truthguard/database.py - SQLite history using Python's built-in sqlite3.

All queries use ? placeholders (never string formatting) so user text cannot
inject SQL. Only a short excerpt of the input is stored, never API keys.
"""
import json
import sqlite3
from contextlib import contextmanager

import config

VALID_TYPES = {"credibility", "sentiment", "clustering", "summary"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_type TEXT NOT NULL,
    excerpt       TEXT NOT NULL,
    input_length  INTEGER NOT NULL,
    result_label  TEXT NOT NULL,
    confidence    REAL,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def _connect(db_path=None):
    conn = sqlite3.connect(str(db_path or config.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=None):
    """Create the table if it does not exist (safe to call every start-up)."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def _row(row):
    item = dict(row)
    try:
        item["metadata"] = json.loads(item["metadata"])
    except (TypeError, ValueError):
        item["metadata"] = {}
    return item


def add_analysis(analysis_type, text, result_label, confidence=None,
                 metadata=None, db_path=None):
    """Store one analysis; returns the new row id."""
    if analysis_type not in VALID_TYPES:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    limit = config.SAVED_EXCERPT_LENGTH
    excerpt = text if len(text) <= limit else text[:limit].rstrip() + "..."
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO analyses (analysis_type, excerpt, input_length, "
            "result_label, confidence, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_type, excerpt, len(text), str(result_label), confidence,
             json.dumps(metadata or {})))
        return cur.lastrowid


def get_recent(limit=50, analysis_type=None, db_path=None):
    sql, params = "SELECT * FROM analyses", []
    if analysis_type:
        sql += " WHERE analysis_type = ?"
        params.append(analysis_type)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _connect(db_path) as conn:
        return [_row(r) for r in conn.execute(sql, params).fetchall()]


def delete_analysis(entry_id, db_path=None):
    """Delete one entry. Returns True if a row was removed."""
    with _connect(db_path) as conn:
        return conn.execute("DELETE FROM analyses WHERE id = ?", (entry_id,)).rowcount > 0


def clear_history(db_path=None):
    """Delete everything. Returns the number of rows removed."""
    with _connect(db_path) as conn:
        return conn.execute("DELETE FROM analyses").rowcount


def count_by_type(db_path=None):
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT analysis_type, COUNT(*) AS n FROM analyses GROUP BY analysis_type")
        return {r["analysis_type"]: r["n"] for r in rows}


def label_distribution(analysis_type, db_path=None):
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT result_label, COUNT(*) AS n FROM analyses "
            "WHERE analysis_type = ? GROUP BY result_label ORDER BY n DESC",
            (analysis_type,))
        return {r["result_label"]: r["n"] for r in rows}


def latest_metadata(analysis_type, db_path=None):
    rows = get_recent(limit=1, analysis_type=analysis_type, db_path=db_path)
    return rows[0]["metadata"] if rows else None
