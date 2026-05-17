from __future__ import annotations

import itertools
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorStore:
    """Parquet vector storage, one logical store per vector_kind, dedup on sha256.

    Two physical layouts coexist and are merged on read:

      * Legacy single file ``root/<kind>.parquet`` — written by ``append`` and
        by older pixsage versions.
      * Append-only part-files ``root/<kind>/<ts>_<seq>.parquet`` — written by
        ``extend``. Each flush is its own file, so a flush costs only its own
        rows regardless of how many were written before (O(1) amortised per
        row instead of O(n) rewrite-the-world).

    Merge order is legacy-first then part-files in name order (which is write
    order). Last write wins on sha256 conflict, so a forced re-embed correctly
    supersedes an earlier vector.

    Schema per file:
        sha256: string
        vector: list<float32>     (fixed length per kind, enforced by validation)
        created_at: string        (ISO timestamp)
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._seq = itertools.count()

    # ---- paths -----------------------------------------------------------

    def _path(self, kind: str) -> Path:
        """Legacy single-file path for the kind."""
        return self.root / f"{kind}.parquet"

    def _part_dir(self, kind: str) -> Path:
        return self.root / kind

    def _part_files(self, kind: str) -> list[Path]:
        d = self._part_dir(kind)
        if not d.is_dir():
            return []
        return sorted(d.glob("*.parquet"))

    def _ordered_files(self, kind: str) -> list[Path]:
        """Legacy file (oldest) first, then part-files in write order."""
        files: list[Path] = []
        legacy = self._path(kind)
        if legacy.exists():
            files.append(legacy)
        files.extend(self._part_files(kind))
        return files

    # ---- reads -----------------------------------------------------------

    def _read_all(self, kind: str) -> dict[str, dict]:
        """Return {sha256 -> {sha256, vector, created_at}} for the kind.

        Later files override earlier ones on sha256 (last write wins)."""
        out: dict[str, dict] = {}
        for path in self._ordered_files(kind):
            table = pq.read_table(path)
            shas = table.column("sha256").to_pylist()
            vecs = table.column("vector").to_pylist()
            ts = table.column("created_at").to_pylist()
            for s, v, t in zip(shas, vecs, ts):
                out[s] = {"sha256": s, "vector": v, "created_at": t}
        return out

    def index(self, kind: str) -> dict[str, str]:
        """Return {sha256 -> created_at} without materialising vectors.

        Cheap one-shot view for skip/staleness checks — reads only the
        sha256 and created_at columns."""
        out: dict[str, str] = {}
        for path in self._ordered_files(kind):
            table = pq.read_table(path, columns=["sha256", "created_at"])
            shas = table.column("sha256").to_pylist()
            ts = table.column("created_at").to_pylist()
            for s, t in zip(shas, ts):
                out[s] = t
        return out

    def load(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (sha_array, matrix). matrix is (N, D) float32; sha_array is (N,) object."""
        rows = self._read_all(kind)
        if not rows:
            return np.array([], dtype=object), np.zeros((0, 0), dtype=np.float32)
        shas = np.array([r["sha256"] for r in rows.values()], dtype=object)
        matrix = np.array([r["vector"] for r in rows.values()], dtype=np.float32)
        return shas, matrix

    def missing_for(self, kind: str, all_shas: set[str]) -> set[str]:
        return all_shas - self.index(kind).keys()

    def get_one(self, kind: str, sha256: str) -> np.ndarray | None:
        row = self._read_all(kind).get(sha256)
        if row is None:
            return None
        return np.array(row["vector"], dtype=np.float32)

    def created_at(self, kind: str, sha256: str) -> str | None:
        return self.index(kind).get(sha256)

    # ---- writes ----------------------------------------------------------

    @staticmethod
    def _validate(rows: list[tuple[str, np.ndarray]]) -> None:
        for sha, vec in rows:
            if vec.dtype != np.float32:
                raise ValueError(f"vector for {sha!r} must be float32, got {vec.dtype}")
            if vec.ndim != 1:
                raise ValueError(f"vector for {sha!r} must be 1-D, got shape {vec.shape}")

    def _table(self, rows: list[dict]) -> pa.Table:
        return pa.table({
            "sha256": [r["sha256"] for r in rows],
            "vector": pa.array([r["vector"] for r in rows], type=pa.list_(pa.float32())),
            "created_at": [r["created_at"] for r in rows],
        })

    def extend(self, kind: str, rows: list[tuple[str, np.ndarray]]) -> None:
        """Append rows as a new part-file. O(len(rows)) — never rewrites
        previously written vectors. Caller is responsible for not re-adding
        a sha it has already embedded this run (cheap via ``index``)."""
        if not rows:
            return
        self._validate(rows)
        now = _now()
        records = [
            {"sha256": sha, "vector": vec.tolist(), "created_at": now}
            for sha, vec in rows
        ]
        part_dir = self._part_dir(kind)
        part_dir.mkdir(parents=True, exist_ok=True)
        # Compact UTC timestamp + per-instance sequence keeps part names in
        # write order under a lexical sort, even within the same second.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        name = f"{stamp}_{next(self._seq):06d}.parquet"
        pq.write_table(self._table(records), part_dir / name)

    def append(self, kind: str, rows: list[tuple[str, np.ndarray]]) -> None:
        """Add or replace rows, consolidating into the single legacy file.

        Reads everything (legacy + any part-files), merges, writes one file,
        and drops the part directory so the legacy file is the sole source.
        Heavier than ``extend``; kept for callers that want read-modify-write
        semantics. Not used on the embed hot path."""
        if not rows:
            return
        self._validate(rows)
        existing = self._read_all(kind)
        now = _now()
        for sha, vec in rows:
            existing[sha] = {"sha256": sha, "vector": vec.tolist(), "created_at": now}
        self._write_legacy(kind, list(existing.values()))
        part_dir = self._part_dir(kind)
        if part_dir.is_dir():
            shutil.rmtree(part_dir)

    def _write_legacy(self, kind: str, rows: list[dict]) -> None:
        if not rows:
            self._path(kind).unlink(missing_ok=True)
            return
        pq.write_table(self._table(rows), self._path(kind))
