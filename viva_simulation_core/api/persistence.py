"""SQLite persistence for twin definitions (source + metadata + owner + hash).

Live stepped state is deliberately NOT persisted: after a restart a twin is
re-warmed lazily from its source and starts again at t=0.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field

__all__ = ["TwinMeta", "TwinStore"]


@dataclass
class TwinMeta:
    twin_id: str
    source: str
    model_format: str
    metadata: dict = field(default_factory=dict)
    owner_key: str = ""
    model_hash: str = ""
    created: float = 0.0


_SCHEMA = """CREATE TABLE IF NOT EXISTS twins (
    twin_id TEXT PRIMARY KEY, source TEXT NOT NULL, model_format TEXT NOT NULL,
    metadata TEXT NOT NULL, owner_key TEXT NOT NULL, model_hash TEXT NOT NULL,
    created REAL NOT NULL)"""


class TwinStore:
    def __init__(self, db_path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(_SCHEMA)
            self._db.commit()

    @staticmethod
    def _row(r) -> TwinMeta:
        return TwinMeta(r[0], r[1], r[2], json.loads(r[3]), r[4], r[5], r[6])

    def save(self, meta: TwinMeta) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO twins VALUES (?,?,?,?,?,?,?)",
                (meta.twin_id, meta.source, meta.model_format, json.dumps(meta.metadata),
                 meta.owner_key, meta.model_hash, meta.created))
            self._db.commit()

    def get(self, twin_id: str) -> TwinMeta | None:
        with self._lock:
            r = self._db.execute(
                "SELECT twin_id,source,model_format,metadata,owner_key,model_hash,created "
                "FROM twins WHERE twin_id=?", (twin_id,)).fetchone()
        return self._row(r) if r else None

    def delete(self, twin_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM twins WHERE twin_id=?", (twin_id,))
            self._db.commit()

    def load_all(self) -> list[TwinMeta]:
        with self._lock:
            rows = self._db.execute(
                "SELECT twin_id,source,model_format,metadata,owner_key,model_hash,created "
                "FROM twins ORDER BY created, twin_id").fetchall()
        return [self._row(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
