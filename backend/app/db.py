"""Tiny SQLite store for inspections (swap for PostgreSQL later; the API stays the same)."""
import json
import sqlite3
from datetime import datetime, timezone

from app.config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY, created_at TEXT, farmer TEXT, lot_ref TEXT, centre TEXT,
            image TEXT, annotated TEXT, image_sha256 TEXT, result_sha256 TEXT, result_json TEXT)""")


def save(rec):
    with _conn() as c:
        c.execute("INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?)", (
            rec["id"], rec["created_at"], rec["farmer"], rec["lot_ref"], rec["centre"], rec["image"],
            rec["annotated"], rec["image_sha256"], rec["result_sha256"], json.dumps(rec["result"])))


def _row(r):
    d = dict(r)
    d["result"] = json.loads(d.pop("result_json"))
    return d


def get(inspection_id):
    with _conn() as c:
        r = c.execute("SELECT * FROM inspections WHERE id=?", (inspection_id,)).fetchone()
    return _row(r) if r else None


def list_recent(limit=50):
    with _conn() as c:
        rows = c.execute("SELECT * FROM inspections ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
