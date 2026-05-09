from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pixsage.taggers.base import Tag

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

SCHEMA_TAGS = """
CREATE TABLE IF NOT EXISTS tags (
  sha256 TEXT NOT NULL,
  tag TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL,
  hierarchy TEXT,
  user_rejected INTEGER NOT NULL DEFAULT 0,
  applied_at TEXT,
  PRIMARY KEY (sha256, tag, source),
  FOREIGN KEY (sha256) REFERENCES photos(sha256) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tags_sha256 ON tags(sha256);
CREATE INDEX IF NOT EXISTS idx_tags_source ON tags(source);
"""

SCHEMA_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT,
  finished_at TEXT,
  photos_processed INTEGER,
  photos_skipped INTEGER,
  photos_errored INTEGER,
  config_hash TEXT,
  model_versions TEXT
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
            self._conn.executescript(SCHEMA_TAGS)
            self._conn.executescript(SCHEMA_RUNS)

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

    def record_tags(self, sha256: str, tags: list[Tag]) -> None:
        now = _now()
        with self._conn:
            for t in tags:
                self._conn.execute(
                    """
                    INSERT INTO tags (sha256, tag, source, confidence, hierarchy, applied_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sha256, tag, source) DO UPDATE SET
                        confidence = excluded.confidence,
                        hierarchy = excluded.hierarchy,
                        applied_at = excluded.applied_at
                    """,
                    (sha256, t.name, t.source, t.confidence, t.hierarchy, now),
                )

    def get_tags(self, sha256: str) -> list[Tag]:
        from pixsage.taggers.base import Tag as _Tag  # noqa: PLC0415 — runtime construction
        cur = self._conn.execute(
            "SELECT tag, confidence, hierarchy, source FROM tags WHERE sha256 = ?",
            (sha256,),
        )
        return [_Tag(name=r["tag"], confidence=r["confidence"] or 0.0, hierarchy=r["hierarchy"], source=r["source"]) for r in cur]

    def flag_user_rejections(self, sha256: str, surviving_xmp_tags: set[str]) -> None:
        """Any tag we previously applied that's NOT in surviving_xmp_tags becomes user_rejected."""
        with self._conn:
            cur = self._conn.execute(
                "SELECT tag, source FROM tags WHERE sha256 = ?",
                (sha256,),
            )
            for r in cur.fetchall():
                if r["tag"] not in surviving_xmp_tags:
                    self._conn.execute(
                        "UPDATE tags SET user_rejected = 1 WHERE sha256 = ? AND tag = ? AND source = ?",
                        (sha256, r["tag"], r["source"]),
                    )

    def is_user_rejected(self, sha256: str, tag: str, source: str) -> bool:
        cur = self._conn.execute(
            "SELECT user_rejected FROM tags WHERE sha256 = ? AND tag = ? AND source = ?",
            (sha256, tag, source),
        )
        row = cur.fetchone()
        return bool(row and row["user_rejected"])

    def get_user_rejected(self, sha256: str) -> set[tuple[str, str]]:
        cur = self._conn.execute(
            "SELECT tag, source FROM tags WHERE sha256 = ? AND user_rejected = 1",
            (sha256,),
        )
        return {(r["tag"], r["source"]) for r in cur}

    def delete_tags(self, sha256: str) -> None:
        """Wipe every tag row (and their user_rejected flags) for this photo.

        Used by `pixsage tag --rewrite` so the next run starts as if we'd
        never tagged this photo before. The photo row itself is preserved
        (we still want to track that we've seen it).
        """
        with self._conn:
            self._conn.execute("DELETE FROM tags WHERE sha256 = ?", (sha256,))

    def rekey_photo(self, old_sha256: str, new_sha256: str) -> None:
        """Update the primary key of a photo + its tags. No-op if old==new.

        Defers FK checking to commit time so the intermediate state
        (photos.sha256 updated but tags.sha256 not yet, or vice versa)
        doesn't trip the tags→photos foreign key.
        """
        if old_sha256 == new_sha256:
            return
        with self._conn:
            self._conn.execute("PRAGMA defer_foreign_keys = ON")
            self._conn.execute(
                "UPDATE photos SET sha256 = ? WHERE sha256 = ?",
                (new_sha256, old_sha256),
            )
            self._conn.execute(
                "UPDATE tags SET sha256 = ? WHERE sha256 = ?",
                (new_sha256, old_sha256),
            )

    def cleanup_orphans(self) -> int:
        """Remove stale photo rows.

        For each (current_path) that has multiple sha256 rows (which happens
        when a previous run errored mid-write and left an old-sha row behind),
        keep only the row with the most recent last_seen_at. CASCADE deletes
        the orphan tags rows automatically.

        Returns the number of photo rows deleted.
        """
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            cur = self._conn.execute(
                """
                DELETE FROM photos
                WHERE rowid NOT IN (
                    SELECT rowid FROM photos p1
                    WHERE last_seen_at = (
                        SELECT MAX(last_seen_at) FROM photos p2
                        WHERE p2.current_path = p1.current_path
                    )
                )
                """
            )
            return int(cur.rowcount or 0)

    def start_run(self, config_hash: str, model_versions: dict[str, str]) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs (started_at, config_hash, model_versions) VALUES (?, ?, ?)",
                (_now(), config_hash, json.dumps(model_versions, sort_keys=True)),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, processed: int, skipped: int, errored: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, photos_processed = ?, photos_skipped = ?, photos_errored = ? WHERE run_id = ?",
                (_now(), processed, skipped, errored, run_id),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        cur = self._conn.execute("SELECT * FROM runs ORDER BY run_id")
        return [dict(r) for r in cur]
