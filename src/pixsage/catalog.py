from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_PHOTOS = """
CREATE TABLE IF NOT EXISTS photos (
  sha256 TEXT PRIMARY KEY,
  current_path TEXT,
  filename TEXT,
  filesize INTEGER,
  mtime REAL,
  last_tagged_at TEXT,
  model_versions TEXT,
  added_at TEXT,
  last_seen_at TEXT,
  error_reason TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA_PHOTOS)

    def close(self) -> None:
        self._conn.close()

    def upsert_photo(self, sha256: str, path: Path, filesize: int, mtime: float) -> None:
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO photos (sha256, current_path, filename, filesize, mtime, added_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    current_path = excluded.current_path,
                    filename = excluded.filename,
                    filesize = excluded.filesize,
                    mtime = excluded.mtime,
                    last_seen_at = excluded.last_seen_at
                """,
                (sha256, str(path), path.name, filesize, mtime, now, now),
            )

    def get_photo(self, sha256: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM photos WHERE sha256 = ?", (sha256,))
        row = cur.fetchone()
        return dict(row) if row else None

    def mark_tagged(self, sha256: str, model_versions: dict[str, str]) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE photos SET last_tagged_at = ?, model_versions = ?, error_reason = NULL WHERE sha256 = ?",
                (_now(), json.dumps(model_versions, sort_keys=True), sha256),
            )

    def mark_error(self, sha256: str, reason: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE photos SET error_reason = ? WHERE sha256 = ?",
                (reason, sha256),
            )

    def needs_tagging(self, sha256: str, model_versions: dict[str, str]) -> bool:
        row = self.get_photo(sha256)
        if row is None:
            return True
        if row["last_tagged_at"] is None:
            return True
        if row["model_versions"] is None:
            return True
        existing = json.loads(row["model_versions"])
        return existing != model_versions
