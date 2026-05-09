from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorStore:
    """Parquet-per-vector_kind storage. One file per kind, dedup on sha256.

    Schema per file:
        sha256: string
        vector: list<float32>     (fixed length per kind, enforced by validation)
        created_at: string        (ISO timestamp)
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str) -> Path:
        return self.root / f"{kind}.parquet"

    def _read_all(self, kind: str) -> dict[str, dict]:
        """Return {sha256 -> {sha256, vector, created_at}} for the kind."""
        path = self._path(kind)
        if not path.exists():
            return {}
        table = pq.read_table(path)
        out: dict[str, dict] = {}
        shas = table.column("sha256").to_pylist()
        vecs = table.column("vector").to_pylist()
        ts = table.column("created_at").to_pylist()
        for s, v, t in zip(shas, vecs, ts):
            out[s] = {"sha256": s, "vector": v, "created_at": t}
        return out

    def append(self, kind: str, rows: list[tuple[str, np.ndarray]]) -> None:
        """Add or replace rows. Each vector must be 1-D float32."""
        if not rows:
            return
        existing = self._read_all(kind)
        now = _now()
        for sha, vec in rows:
            if vec.dtype != np.float32:
                raise ValueError(f"vector for {sha!r} must be float32, got {vec.dtype}")
            if vec.ndim != 1:
                raise ValueError(f"vector for {sha!r} must be 1-D, got shape {vec.shape}")
            existing[sha] = {"sha256": sha, "vector": vec.tolist(), "created_at": now}
        self._write(kind, list(existing.values()))

    def _write(self, kind: str, rows: list[dict]) -> None:
        if not rows:
            self._path(kind).unlink(missing_ok=True)
            return
        table = pa.table({
            "sha256": [r["sha256"] for r in rows],
            "vector": pa.array([r["vector"] for r in rows], type=pa.list_(pa.float32())),
            "created_at": [r["created_at"] for r in rows],
        })
        pq.write_table(table, self._path(kind))

    def load(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (sha_array, matrix). matrix is (N, D) float32; sha_array is (N,) object."""
        path = self._path(kind)
        if not path.exists():
            return np.array([], dtype=object), np.zeros((0, 0), dtype=np.float32)
        table = pq.read_table(path)
        shas = np.array(table.column("sha256").to_pylist(), dtype=object)
        vecs = table.column("vector").to_pylist()
        if not vecs:
            return shas, np.zeros((0, 0), dtype=np.float32)
        matrix = np.array(vecs, dtype=np.float32)
        return shas, matrix

    def missing_for(self, kind: str, all_shas: set[str]) -> set[str]:
        existing = self._read_all(kind)
        return all_shas - existing.keys()

    def get_one(self, kind: str, sha256: str) -> np.ndarray | None:
        existing = self._read_all(kind)
        row = existing.get(sha256)
        if row is None:
            return None
        return np.array(row["vector"], dtype=np.float32)

    def created_at(self, kind: str, sha256: str) -> str | None:
        existing = self._read_all(kind)
        row = existing.get(sha256)
        return row["created_at"] if row else None
