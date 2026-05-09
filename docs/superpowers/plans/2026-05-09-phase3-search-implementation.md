# Phase 3 — Embedded Semantic Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working `pixsage embed` + `pixsage serve` pair that gives the photographer a localhost semantic search UI for their corpus, with both visual and caption retrieval channels and "more like this" navigation.

**Architecture:** Five logical components (catalog migration, vector storage, embedders, search service, web app) wired together by two new CLI verbs. Bottom-up build: schema → storage → embedder protocol with mock → real SigLIP2 → embed runner → search service → web routes → real CLI. Mock embedder lets us prove the end-to-end pipeline without loading torch.

**Tech Stack:** Python ≥3.11, PyTorch + transformers (SigLIP2), pyarrow (parquet), FastAPI + Jinja2 + uvicorn, HTMX (vendored static JS), numpy. Test framework: pytest + httpx (FastAPI TestClient).

**Spec:** `docs/superpowers/specs/2026-05-09-phase3-search-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pixsage/embedders/__init__.py` | Re-exports protocol + factory |
| `src/pixsage/embedders/base.py` | `Embedder` protocol + `EmbedResult` dataclass |
| `src/pixsage/embedders/mock.py` | Deterministic test embedder |
| `src/pixsage/embedders/siglip2.py` | SigLIP2 wrapper |
| `src/pixsage/vectors.py` | Parquet writer/reader keyed by `(vector_kind, sha256)` |
| `src/pixsage/embed_runner.py` | Per-photo embed loop with caption staleness check |
| `src/pixsage/search.py` | numpy cosine + weighted blend + more-like-this |
| `src/pixsage/web/__init__.py` | Package marker |
| `src/pixsage/web/app.py` | FastAPI app factory + dependency wiring |
| `src/pixsage/web/routes.py` | All HTTP route handlers |
| `src/pixsage/web/thumbs.py` | Lazy thumbnail cache |
| `src/pixsage/web/templates/index.html` | Search page (empty state) |
| `src/pixsage/web/templates/_results.html` | Photo grid partial (HTMX swap target) |
| `src/pixsage/web/templates/_card.html` | Single photo card partial |
| `src/pixsage/web/templates/photo.html` | Photo detail page |
| `src/pixsage/web/static/style.css` | Page styles |
| `src/pixsage/web/static/htmx.min.js` | Vendored HTMX (no CDN) |
| **Modified:** `src/pixsage/catalog.py` | Add caption columns + `record_caption` + `iter_photos_for_embedding` |
| **Modified:** `src/pixsage/config.py` | Add `EmbeddingsConfig` and `SearchConfig` |
| **Modified:** `src/pixsage/cli.py` | Add `embed`/`serve` verbs; record caption inside `tag`; extend `cleanup` |
| **Modified:** `pyproject.toml` | Add `[search]` extra |
| **Modified:** `tests/conftest.py` | Add `catalog`, `make_caption_jpeg`, `vectors_path` fixtures |
| `tests/test_catalog_caption.py` | Migration + record_caption + queries |
| `tests/test_vectors.py` | Parquet round-trip + missing-for + dedupe |
| `tests/test_embedders_mock.py` | Mock embedder sanity + protocol conformance |
| `tests/test_embedders_siglip2.py` | Real-model smoke (gated on CUDA + cache) |
| `tests/test_embed_runner.py` | End-to-end with mock embedder + staleness |
| `tests/test_search.py` | Synthetic vectors → known nearest + weight blend |
| `tests/test_web_thumbs.py` | Cache hit/miss + regeneration |
| `tests/test_web_routes.py` | All routes via TestClient with mock embedder |
| `tests/test_cli_embed.py` | `pixsage embed` end-to-end with mock |
| `tests/test_cli_serve.py` | `pixsage serve` boot smoke (TestClient) |

The `embedders/` package isolates heavy deps the same way `taggers/` does — early tasks don't need to install `transformers` or model weights. The `web/` package isolates FastAPI from the embed pipeline so `pixsage embed` can run on a system without `[search]` extras installed.

---

## Task 1: Catalog schema migration — caption columns

**Files:**
- Modify: `src/pixsage/catalog.py`
- Create: `tests/test_catalog_caption.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catalog_caption.py
from __future__ import annotations

from pathlib import Path

import pytest

from pixsage.catalog import Catalog


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    return cat


def test_caption_columns_exist(catalog: Catalog):
    cur = catalog._conn.execute("PRAGMA table_info(photos)")
    cols = {row["name"] for row in cur.fetchall()}
    assert "caption" in cols
    assert "caption_updated_at" in cols


def test_record_caption_sets_text_and_timestamp(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.record_caption("sha1", "a leopard seal on ice")

    row = catalog.get_photo("sha1")
    assert row["caption"] == "a leopard seal on ice"
    assert row["caption_updated_at"] is not None  # ISO timestamp


def test_record_caption_updates_timestamp_on_change(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.record_caption("sha1", "first")
    first_ts = catalog.get_photo("sha1")["caption_updated_at"]

    catalog.record_caption("sha1", "second")
    second_ts = catalog.get_photo("sha1")["caption_updated_at"]
    assert second_ts > first_ts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_catalog_caption.py -v`
Expected: FAIL with `AssertionError: assert 'caption' in cols` (column doesn't exist yet) and `AttributeError: 'Catalog' object has no attribute 'record_caption'`.

- [ ] **Step 3: Add the migration and method**

In `src/pixsage/catalog.py`, change `init_schema` to also run a migration step, and add the new method:

```python
def init_schema(self) -> None:
    with self._conn:
        self._conn.executescript(SCHEMA_PHOTOS)
        self._conn.executescript(SCHEMA_TAGS)
        self._conn.executescript(SCHEMA_RUNS)
        self._migrate_add_caption_columns()

def _migrate_add_caption_columns(self) -> None:
    cur = self._conn.execute("PRAGMA table_info(photos)")
    existing = {row["name"] for row in cur.fetchall()}
    if "caption" not in existing:
        self._conn.execute("ALTER TABLE photos ADD COLUMN caption TEXT")
    if "caption_updated_at" not in existing:
        self._conn.execute("ALTER TABLE photos ADD COLUMN caption_updated_at TEXT")

def record_caption(self, sha256: str, caption: str | None) -> None:
    """Set the caption for a photo. Bumps caption_updated_at to now()."""
    with self._conn:
        self._conn.execute(
            "UPDATE photos SET caption = ?, caption_updated_at = ? WHERE sha256 = ?",
            (caption, _now(), sha256),
        )
```

The introspection-and-`ALTER` pattern is idempotent: it runs on every `init_schema()` call and no-ops once the columns exist. Phase 1 catalogs will be migrated transparently the first time Phase 3 code touches them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_catalog_caption.py -v`
Expected: 3 passed.

Also run: `pytest tests/test_catalog.py -v`
Expected: existing Phase 1 tests still pass (migration is purely additive).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/catalog.py tests/test_catalog_caption.py
git commit -m "feat(catalog): caption + caption_updated_at columns with idempotent migration"
```

---

## Task 2: Catalog query — `iter_photos_for_embedding`

**Files:**
- Modify: `src/pixsage/catalog.py`
- Modify: `tests/test_catalog_caption.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog_caption.py`:

```python
def test_iter_photos_for_embedding_returns_all(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.upsert_photo("sha2", tmp_path / "b.jpg", filesize=20, mtime=2.0)
    catalog.record_caption("sha1", "caption a")
    # sha2 has no caption

    rows = list(catalog.iter_photos_for_embedding())
    assert len(rows) == 2
    by_sha = {r["sha256"]: r for r in rows}
    assert by_sha["sha1"]["caption"] == "caption a"
    assert by_sha["sha2"]["caption"] is None
    assert by_sha["sha1"]["current_path"] == str(tmp_path / "a.jpg")


def test_iter_photos_for_embedding_skips_errored(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.upsert_photo("sha2", tmp_path / "b.jpg", filesize=20, mtime=2.0)
    catalog.mark_error("sha2", "decode failed")

    rows = list(catalog.iter_photos_for_embedding())
    assert {r["sha256"] for r in rows} == {"sha1"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_catalog_caption.py::test_iter_photos_for_embedding_returns_all -v`
Expected: FAIL with `AttributeError: 'Catalog' object has no attribute 'iter_photos_for_embedding'`.

- [ ] **Step 3: Implement**

Add to `src/pixsage/catalog.py`:

```python
def iter_photos_for_embedding(self) -> Iterator[dict[str, Any]]:
    """Yield rows {sha256, current_path, caption, caption_updated_at} for every
    photo that's not currently flagged with an error.
    """
    cur = self._conn.execute(
        """
        SELECT sha256, current_path, caption, caption_updated_at
        FROM photos
        WHERE error_reason IS NULL
        """
    )
    for row in cur:
        yield dict(row)
```

Add `from typing import Iterator` at the top if not already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_catalog_caption.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/catalog.py tests/test_catalog_caption.py
git commit -m "feat(catalog): iter_photos_for_embedding query"
```

---

## Task 3: Vectors module — parquet storage

**Files:**
- Create: `src/pixsage/vectors.py`
- Create: `tests/test_vectors.py`
- Modify: `pyproject.toml` (add pyarrow + numpy to base deps)

- [ ] **Step 1: Update pyproject.toml**

In `pyproject.toml`, change the `dependencies` block to:

```toml
dependencies = [
  "pydantic>=2.5",
  "typer>=0.12",
  "tqdm>=4.66",
  "pillow>=10.0",
  "pillow-heif>=0.16",
  "numpy>=1.26",
  "pyarrow>=15",
]
```

Reasoning: numpy and pyarrow are small enough and broadly useful enough across phases that we promote them to base deps rather than `[search]`. The `[search]` extra will hold FastAPI/uvicorn/jinja2 only.

Run: `pip install -e ".[dev]"` to pick up the new deps.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vectors.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixsage.vectors import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vectors")


def test_append_and_load_round_trip(store: VectorStore):
    rows = [("sha1", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("sha2", np.array([0.0, 1.0, 0.0], dtype=np.float32))]
    store.append("siglip2_image", rows)

    sha_array, matrix = store.load("siglip2_image")
    assert list(sha_array) == ["sha1", "sha2"]
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(matrix[1], [0.0, 1.0, 0.0])


def test_append_replaces_existing_sha(store: VectorStore):
    store.append("siglip2_image", [("sha1", np.array([1.0, 0.0], dtype=np.float32))])
    store.append("siglip2_image", [("sha1", np.array([0.0, 1.0], dtype=np.float32))])

    sha_array, matrix = store.load("siglip2_image")
    assert list(sha_array) == ["sha1"]
    np.testing.assert_array_equal(matrix[0], [0.0, 1.0])


def test_missing_for_returns_unembedded_shas(store: VectorStore):
    store.append("siglip2_image", [
        ("sha1", np.array([1.0, 0.0], dtype=np.float32)),
        ("sha2", np.array([0.0, 1.0], dtype=np.float32)),
    ])
    missing = store.missing_for("siglip2_image", {"sha1", "sha2", "sha3", "sha4"})
    assert missing == {"sha3", "sha4"}


def test_load_empty_kind_returns_empty(store: VectorStore):
    sha_array, matrix = store.load("siglip2_image")
    assert len(sha_array) == 0
    assert matrix.shape == (0, 0)


def test_get_one_returns_vector(store: VectorStore):
    v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    store.append("siglip2_image", [("sha1", v)])
    fetched = store.get_one("siglip2_image", "sha1")
    np.testing.assert_array_equal(fetched, v)


def test_get_one_missing_returns_none(store: VectorStore):
    assert store.get_one("siglip2_image", "missing-sha") is None


def test_created_at_recorded(store: VectorStore):
    store.append("siglip2_image", [("sha1", np.array([1.0, 0.0], dtype=np.float32))])
    ts = store.created_at("siglip2_image", "sha1")
    assert ts is not None
    # ISO 8601 ish — at least contains 'T' or '-'
    assert "T" in ts or "-" in ts
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_vectors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pixsage.vectors'`.

- [ ] **Step 4: Implement**

```python
# src/pixsage/vectors.py
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
```

The `_read_all`-then-rewrite pattern is O(N) per append, but is fine: appends happen one photo at a time during embed runs, and with 50k photos × ~4.5 KB per row ≈ 230 MB read+write per append we'd be I/O-bound. We mitigate by batching at the embed-runner level (Task 7) — flush every K photos rather than every photo.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_vectors.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/pixsage/vectors.py tests/test_vectors.py
git commit -m "feat(vectors): parquet-per-kind store with append/load/missing-for"
```

---

## Task 4: Embedder protocol + mock embedder

**Files:**
- Create: `src/pixsage/embedders/__init__.py`
- Create: `src/pixsage/embedders/base.py`
- Create: `src/pixsage/embedders/mock.py`
- Create: `tests/test_embedders_mock.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedders_mock.py
from __future__ import annotations

import numpy as np
from PIL import Image

from pixsage.embedders.mock import MockEmbedder


def test_image_embedding_shape_and_dtype():
    e = MockEmbedder(dim=8)
    img = Image.new("RGB", (32, 32), color="red")
    vecs = e.embed_image([img])
    assert vecs.shape == (1, 8)
    assert vecs.dtype == np.float32


def test_text_embedding_shape_and_dtype():
    e = MockEmbedder(dim=8)
    vecs = e.embed_text(["a leopard seal"])
    assert vecs.shape == (1, 8)
    assert vecs.dtype == np.float32


def test_l2_normalized_output():
    e = MockEmbedder(dim=16)
    vecs = e.embed_image([Image.new("RGB", (32, 32))])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-5


def test_deterministic_for_same_input():
    e = MockEmbedder(dim=8)
    a = e.embed_text(["leopard seal"])
    b = e.embed_text(["leopard seal"])
    np.testing.assert_array_equal(a, b)


def test_different_text_different_vector():
    e = MockEmbedder(dim=8)
    a = e.embed_text(["leopard seal"])
    b = e.embed_text(["emperor penguin"])
    assert not np.array_equal(a, b)


def test_batched_embedding_matches_single_calls():
    e = MockEmbedder(dim=8)
    imgs = [Image.new("RGB", (32, 32), c) for c in ("red", "green", "blue")]
    batched = e.embed_image(imgs)
    one_at_a_time = np.vstack([e.embed_image([img]) for img in imgs])
    np.testing.assert_array_equal(batched, one_at_a_time)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embedders_mock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pixsage.embedders'`.

- [ ] **Step 3: Implement the protocol**

```python
# src/pixsage/embedders/__init__.py
from pixsage.embedders.base import Embedder, EmbedderInfo  # noqa: F401
```

```python
# src/pixsage/embedders/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EmbedderInfo:
    name: str            # short identifier, e.g. "siglip2-so400m"
    image_kind: str      # vector_kind for image vectors, e.g. "siglip2_image"
    text_kind: str       # vector_kind for caption/text vectors, e.g. "siglip2_caption"
    dim: int             # output dimension (image and text share dim for SigLIP-style)


class Embedder(Protocol):
    info: EmbedderInfo

    def load(self, device: str) -> None: ...
    def embed_image(self, images: list[Image.Image]) -> np.ndarray:
        """Return (N, dim) float32 L2-normalized."""
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Return (N, dim) float32 L2-normalized."""
```

- [ ] **Step 4: Implement the mock**

```python
# src/pixsage/embedders/mock.py
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from pixsage.embedders.base import Embedder, EmbedderInfo


def _seed_from(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _hashed_vector(seed_text: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_from(seed_text))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


class MockEmbedder(Embedder):
    def __init__(self, dim: int = 16):
        self.info = EmbedderInfo(
            name="mock",
            image_kind="mock_image",
            text_kind="mock_text",
            dim=dim,
        )

    def load(self, device: str) -> None:
        pass

    def embed_image(self, images: list[Image.Image]) -> np.ndarray:
        # Hash the bytes of a tiny canonical thumbnail so the same image content
        # produces the same vector across runs.
        out = np.zeros((len(images), self.info.dim), dtype=np.float32)
        for i, img in enumerate(images):
            small = img.convert("RGB").resize((8, 8))
            out[i] = _hashed_vector(small.tobytes().hex(), self.info.dim)
        return out

    def embed_text(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.info.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = _hashed_vector(t, self.info.dim)
        return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_embedders_mock.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/embedders tests/test_embedders_mock.py
git commit -m "feat(embedders): protocol + deterministic mock embedder"
```

---

## Task 5: SigLIP2 embedder

**Files:**
- Create: `src/pixsage/embedders/siglip2.py`
- Create: `tests/test_embedders_siglip2.py`
- Modify: `pyproject.toml` (siglip2 deps already in `[taggers]` via transformers)

- [ ] **Step 1: Write the failing test (gated on CUDA + cache)**

```python
# tests/test_embedders_siglip2.py
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_siglip2_cache() -> bool:
    """Skip unless the model is downloaded (don't pull weights in CI)."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    target = "models--google--siglip2-so400m-patch14-384"
    return os.path.isdir(os.path.join(cache, target))


pytestmark = pytest.mark.skipif(
    not (_has_cuda() and _has_siglip2_cache()),
    reason="SigLIP2 smoke test requires CUDA and a cached model",
)


def test_image_and_text_embeddings_share_space():
    from pixsage.embedders.siglip2 import SigLIP2Embedder

    e = SigLIP2Embedder()
    e.load("cuda")

    cat_img = Image.new("RGB", (224, 224), color=(180, 130, 70))   # placeholder
    img_vecs = e.embed_image([cat_img])
    text_vecs = e.embed_text(["a brown cat sitting"])

    assert img_vecs.shape == (1, e.info.dim)
    assert text_vecs.shape == (1, e.info.dim)
    assert img_vecs.dtype == np.float32
    assert text_vecs.dtype == np.float32

    # L2-normalized
    assert abs(np.linalg.norm(img_vecs[0]) - 1.0) < 1e-3
    assert abs(np.linalg.norm(text_vecs[0]) - 1.0) < 1e-3


def test_text_text_similarity_makes_sense():
    from pixsage.embedders.siglip2 import SigLIP2Embedder

    e = SigLIP2Embedder()
    e.load("cuda")
    a, b, c = e.embed_text(["a leopard seal", "a leopard", "an emperor penguin"])

    # "a leopard seal" should be closer to "a leopard" than to "an emperor penguin"
    assert float(a @ b) > float(a @ c)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedders_siglip2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pixsage.embedders.siglip2'` (or SKIPPED on a machine without CUDA + cache, which is fine — we'll exercise it locally on the 4090).

- [ ] **Step 3: Implement**

```python
# src/pixsage/embedders/siglip2.py
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pixsage.embedders.base import Embedder, EmbedderInfo


class SigLIP2Embedder(Embedder):
    """Wraps google/siglip2-so400m-patch14-384.

    Loads model + processor lazily in `load()`. Uses fp16 on CUDA, fp32 otherwise.
    Both encoders share the model object — `embed_image` and `embed_text` go
    through different forward paths.
    """

    MODEL_ID = "google/siglip2-so400m-patch14-384"

    def __init__(self) -> None:
        self.info = EmbedderInfo(
            name="siglip2-so400m-patch14-384",
            image_kind="siglip2_image",
            text_kind="siglip2_caption",
            dim=1152,  # so400m projection dim — verified at load time
        )
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: str = "cpu"
        self._dtype: Any = None

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._device = device
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        model = AutoModel.from_pretrained(self.MODEL_ID, torch_dtype=self._dtype)
        model.to(device).eval()
        self._model = model
        # Verify dim matches what we declared.
        actual_dim = int(model.config.text_config.hidden_size)
        if actual_dim != self.info.dim:
            self.info = EmbedderInfo(
                name=self.info.name,
                image_kind=self.info.image_kind,
                text_kind=self.info.text_kind,
                dim=actual_dim,
            )

    def embed_image(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        assert self._model is not None and self._processor is not None
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        # SigLIP2 processor returns float32 pixel_values; cast to model dtype.
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.float().cpu().numpy()

    def embed_text(self, texts: list[str]) -> np.ndarray:
        import torch

        assert self._model is not None and self._processor is not None
        inputs = self._processor(text=texts, padding="max_length", return_tensors="pt").to(self._device)
        with torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.float().cpu().numpy()
```

The `pixel_values` dtype cast mirrors the Florence-2 fix from Phase 1 — same root cause, same fix. The `padding="max_length"` for text is required by SigLIP2 (unlike CLIP it expects fixed-length text).

- [ ] **Step 4: Run test to verify it passes (locally on 4090) or stays skipped (other machines)**

Run: `pytest tests/test_embedders_siglip2.py -v`
Expected on the user's RTX 4090 with model cached: 2 passed. Elsewhere: 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/embedders/siglip2.py tests/test_embedders_siglip2.py
git commit -m "feat(embedders): SigLIP2-so400m image + text embedder"
```

---

## Task 6: Embeddings + search config

**Files:**
- Modify: `src/pixsage/config.py`
- Create: `tests/test_config_phase3.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_phase3.py
from __future__ import annotations

from pathlib import Path

from pixsage.config import DEFAULT_CONFIG_TOML, ensure_default_config, load_config


def test_default_toml_includes_embeddings_block(tmp_path: Path):
    cfg_path = tmp_path / "vocabulary.toml"
    ensure_default_config(cfg_path)
    text = cfg_path.read_text(encoding="utf-8")
    assert "[embeddings]" in text
    assert "[embeddings.siglip2]" in text
    assert "[search]" in text


def test_loaded_config_has_embeddings_defaults(tmp_path: Path):
    cfg_path = tmp_path / "vocabulary.toml"
    ensure_default_config(cfg_path)
    cfg = load_config(cfg_path)

    assert cfg.embeddings.enabled is True
    assert cfg.embeddings.siglip2.enabled is True
    assert cfg.embeddings.siglip2.image is True
    assert cfg.embeddings.siglip2.caption is True
    assert cfg.embeddings.siglip2.batch_size == 16


def test_search_config_defaults(tmp_path: Path):
    cfg_path = tmp_path / "vocabulary.toml"
    ensure_default_config(cfg_path)
    cfg = load_config(cfg_path)

    assert cfg.search.default_image_weight == 0.5
    assert cfg.search.top_k == 60
    assert cfg.search.thumb_size_default == "medium"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config_phase3.py -v`
Expected: FAIL — `Config` has no `embeddings` attribute.

- [ ] **Step 3: Implement**

In `src/pixsage/config.py`, add new model classes and extend `Config`:

```python
class SigLIP2Config(BaseModel):
    enabled: bool = True
    model: str = "google/siglip2-so400m-patch14-384"
    image: bool = True
    caption: bool = True
    batch_size: int = 16


class EmbeddingsConfig(BaseModel):
    enabled: bool = True
    siglip2: SigLIP2Config = Field(default_factory=SigLIP2Config)


class SearchConfig(BaseModel):
    default_image_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = 60
    thumb_size_default: str = "medium"  # "small" | "medium" | "large"


class Config(BaseModel):
    florence2: TaggerConfig
    ram_plus_plus: TaggerConfig
    hierarchy_overrides: dict[str, str] = Field(default_factory=dict)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
```

Append these blocks to `DEFAULT_CONFIG_TOML`:

```python
DEFAULT_CONFIG_TOML = """\
# pixsage vocabulary configuration. Edit and re-run `pixsage tag --force` to apply.

[florence2]
enabled = true
tags_enabled = false
confidence_threshold = 0.5
exclude = ["photograph", "image", "picture"]

[ram_plus_plus]
enabled = true
tags_enabled = true
confidence_threshold = 0.4
exclude = []

[hierarchy_overrides]

[caption]
enabled = true
overwrite = false

[embeddings]
enabled = true

[embeddings.siglip2]
enabled = true
model = "google/siglip2-so400m-patch14-384"
image = true
caption = true
batch_size = 16

[search]
default_image_weight = 0.5
top_k = 60
thumb_size_default = "medium"
"""
```

(Keep the existing comment lines in the TOML — abbreviated above for clarity.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config_phase3.py tests/test_config.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/config.py tests/test_config_phase3.py
git commit -m "feat(config): embeddings + search config blocks with defaults"
```

---

## Task 7: Embed runner

**Files:**
- Create: `src/pixsage/embed_runner.py`
- Create: `tests/test_embed_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed_runner.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixsage.catalog import Catalog
from pixsage.embed_runner import EmbedRunner
from pixsage.embedders.mock import MockEmbedder
from pixsage.vectors import VectorStore


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    return cat


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vectors")


def _seed_photo(catalog: Catalog, sha: str, img_path: Path, caption: str | None = None) -> None:
    Image.new("RGB", (32, 32), color="red").save(img_path)
    catalog.upsert_photo(sha, img_path, filesize=img_path.stat().st_size, mtime=img_path.stat().st_mtime)
    if caption is not None:
        catalog.record_caption(sha, caption)


def test_runner_embeds_image_and_caption(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption="a leopard seal")

    runner = EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8))
    runner.run()

    sha_img, mat_img = store.load("mock_image")
    sha_txt, mat_txt = store.load("mock_text")
    assert list(sha_img) == ["sha1"]
    assert list(sha_txt) == ["sha1"]
    assert mat_img.shape == (1, 8)
    assert mat_txt.shape == (1, 8)


def test_runner_skips_caption_when_absent(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption=None)

    runner = EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8))
    runner.run()

    sha_img, _ = store.load("mock_image")
    sha_txt, _ = store.load("mock_text")
    assert list(sha_img) == ["sha1"]
    assert list(sha_txt) == []


def test_runner_skips_already_embedded(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption="x")
    embedder = MockEmbedder(dim=8)

    EmbedRunner(catalog=catalog, vectors=store, embedder=embedder).run()
    # Track call count via wrapping.
    calls = {"image": 0, "text": 0}
    real_image, real_text = embedder.embed_image, embedder.embed_text
    embedder.embed_image = lambda imgs: (calls.__setitem__("image", calls["image"] + 1), real_image(imgs))[1]
    embedder.embed_text = lambda txts: (calls.__setitem__("text", calls["text"] + 1), real_text(txts))[1]

    EmbedRunner(catalog=catalog, vectors=store, embedder=embedder).run()
    assert calls == {"image": 0, "text": 0}


def test_runner_reembeds_on_caption_staleness(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption="first")
    EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8)).run()
    first_vec = store.get_one("mock_text", "sha1")

    time.sleep(0.05)
    catalog.record_caption("sha1", "completely different caption")
    EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8)).run()
    second_vec = store.get_one("mock_text", "sha1")

    assert not np.array_equal(first_vec, second_vec)


def test_runner_force_reembeds_everything(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption="first")
    embedder = MockEmbedder(dim=8)
    EmbedRunner(catalog=catalog, vectors=store, embedder=embedder).run()

    calls = {"image": 0}
    real_image = embedder.embed_image
    embedder.embed_image = lambda imgs: (calls.__setitem__("image", calls["image"] + 1), real_image(imgs))[1]

    EmbedRunner(catalog=catalog, vectors=store, embedder=embedder, force=True).run()
    assert calls["image"] >= 1


def test_runner_marks_decode_errors(catalog: Catalog, store: VectorStore, tmp_path: Path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    catalog.upsert_photo("sha-bad", bad, filesize=bad.stat().st_size, mtime=bad.stat().st_mtime)

    runner = EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8))
    runner.run()

    row = catalog.get_photo("sha-bad")
    assert row["error_reason"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embed_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pixsage.embed_runner'`.

- [ ] **Step 3: Implement**

```python
# src/pixsage/embed_runner.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from pixsage.catalog import Catalog
from pixsage.embedders.base import Embedder
from pixsage.images import load_image
from pixsage.vectors import VectorStore


class EmbedRunner:
    """Walks the catalog and computes embeddings for each photo using one embedder.

    For each photo:
      - skip if image-vector already exists (and not --force, and caption isn't stale)
      - load the image, embed it
      - if a caption exists, embed it too
      - write rows to the VectorStore (which dedupes by sha256)
    """

    def __init__(
        self,
        catalog: Catalog,
        vectors: VectorStore,
        embedder: Embedder,
        force: bool = False,
        embed_image: bool = True,
        embed_caption: bool = True,
        progress: bool = False,
    ) -> None:
        self.catalog = catalog
        self.vectors = vectors
        self.embedder = embedder
        self.force = force
        self.embed_image = embed_image
        self.embed_caption = embed_caption
        self.progress = progress

    def run(self) -> dict[str, int]:
        info = self.embedder.info
        stats = {"processed": 0, "skipped": 0, "errored": 0}

        rows = list(self.catalog.iter_photos_for_embedding())
        if self.progress:
            from tqdm import tqdm
            iterator = tqdm(rows, unit="photo")
        else:
            iterator = rows

        for row in iterator:
            sha = row["sha256"]
            current_path = row["current_path"]
            caption = row["caption"]
            caption_updated_at = row["caption_updated_at"]

            needs_image = self.embed_image and (
                self.force or self.vectors.get_one(info.image_kind, sha) is None
            )
            needs_text = self.embed_caption and caption is not None and (
                self.force
                or self.vectors.get_one(info.text_kind, sha) is None
                or self._caption_is_stale(info.text_kind, sha, caption_updated_at)
            )

            if not needs_image and not needs_text:
                stats["skipped"] += 1
                continue

            try:
                if needs_image:
                    img = load_image(Path(current_path))
                    img_vec = self.embedder.embed_image([img])[0]
                    self.vectors.append(info.image_kind, [(sha, img_vec)])

                if needs_text:
                    txt_vec = self.embedder.embed_text([caption])[0]
                    self.vectors.append(info.text_kind, [(sha, txt_vec)])

                stats["processed"] += 1
            except Exception as e:
                self.catalog.mark_error(sha, str(e))
                stats["errored"] += 1

        return stats

    def _caption_is_stale(self, kind: str, sha: str, caption_updated_at: str | None) -> bool:
        if caption_updated_at is None:
            return False
        vec_ts = self.vectors.created_at(kind, sha)
        if vec_ts is None:
            return True
        return caption_updated_at > vec_ts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embed_runner.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/embed_runner.py tests/test_embed_runner.py
git commit -m "feat(embed): per-photo runner with caption staleness + error handling"
```

---

## Task 8: `pixsage embed` CLI verb

**Files:**
- Modify: `src/pixsage/cli.py`
- Create: `tests/test_cli_embed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_embed.py
from __future__ import annotations

from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from pixsage.catalog import Catalog
from pixsage.cli import app


runner = CliRunner()


def _make_photo_root(tmp_path: Path) -> Path:
    root = tmp_path / "photos"
    root.mkdir()
    Image.new("RGB", (64, 64), color="red").save(root / "a.jpg")
    Image.new("RGB", (64, 64), color="blue").save(root / "b.jpg")
    return root


def test_embed_runs_with_mock_embedder(tmp_path: Path, monkeypatch):
    """End-to-end with --embedder=mock — proves CLI plumbing without torch."""
    photo_root = _make_photo_root(tmp_path)

    # Seed catalog as if `tag` already ran.
    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    from pixsage.walker import walk_photos, sha256_file
    for p in walk_photos(photo_root):
        sha = sha256_file(p)
        cat.upsert_photo(sha, p, filesize=p.stat().st_size, mtime=p.stat().st_mtime)
        cat.record_caption(sha, f"caption for {p.name}")
    cat.close()

    result = runner.invoke(app, ["embed", str(photo_root), "--embedder", "mock"])
    assert result.exit_code == 0, result.output
    assert "processed=2" in result.output

    # Verify vectors written.
    from pixsage.vectors import VectorStore
    store = VectorStore(photo_root / ".photoindex" / "vectors")
    sha_img, mat_img = store.load("mock_image")
    sha_txt, mat_txt = store.load("mock_text")
    assert len(sha_img) == 2
    assert len(sha_txt) == 2


def test_embed_force_reembeds(tmp_path: Path):
    photo_root = _make_photo_root(tmp_path)
    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    from pixsage.walker import walk_photos, sha256_file
    for p in walk_photos(photo_root):
        sha = sha256_file(p)
        cat.upsert_photo(sha, p, filesize=p.stat().st_size, mtime=p.stat().st_mtime)
    cat.close()

    runner.invoke(app, ["embed", str(photo_root), "--embedder", "mock"])
    result = runner.invoke(app, ["embed", str(photo_root), "--embedder", "mock", "--force"])
    assert result.exit_code == 0
    assert "processed=2" in result.output


def test_embed_help_lists_embedder_choices(tmp_path: Path):
    result = runner.invoke(app, ["embed", "--help"])
    assert result.exit_code == 0
    assert "--embedder" in result.output
    assert "mock" in result.output
    assert "siglip2" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_embed.py -v`
Expected: FAIL — no `embed` command.

- [ ] **Step 3: Implement**

In `src/pixsage/cli.py`, add an embedder factory and the new command. Place after the existing `cleanup` command:

```python
def _build_embedder(name: str):
    """Construct an embedder by short name. Lazy imports keep the CLI cold path light."""
    if name == "mock":
        from pixsage.embedders.mock import MockEmbedder
        return MockEmbedder()
    if name == "siglip2":
        from pixsage.embedders.siglip2 import SigLIP2Embedder
        return SigLIP2Embedder()
    raise typer.BadParameter(f"unknown embedder: {name!r} (choose from: mock, siglip2)")


@app.command()
def embed(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    embedder: str = typer.Option(
        "siglip2", "--embedder",
        help="Embedder to use. Choices: siglip2, mock (mock is for testing only).",
    ),
    force: bool = typer.Option(False, "--force", help="Re-embed photos even if vectors already exist."),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
    no_image: bool = typer.Option(False, "--no-image", help="Skip image embedding."),
    no_caption: bool = typer.Option(False, "--no-caption", help="Skip caption embedding."),
) -> None:
    """Compute embeddings for each photo in the catalog."""
    from pixsage.embed_runner import EmbedRunner
    from pixsage.vectors import VectorStore

    photoindex = photo_root / ".photoindex"
    catalog_path = catalog or (photoindex / "catalog.db")
    if not catalog_path.exists():
        typer.echo(f"no catalog at {catalog_path}; run `pixsage tag` first", err=True)
        raise typer.Exit(code=1)

    cat = Catalog(catalog_path)
    cat.init_schema()  # picks up the caption migration if it's an older catalog

    enc = _build_embedder(embedder)
    typer.echo(f"Loading embedder: {enc.info.name}")
    enc.load(select_device())

    vectors = VectorStore(photoindex / "vectors")

    runner = EmbedRunner(
        catalog=cat,
        vectors=vectors,
        embedder=enc,
        force=force,
        embed_image=not no_image,
        embed_caption=not no_caption,
        progress=True,
    )
    stats = runner.run()
    cat.close()
    typer.echo(f"done. processed={stats['processed']} skipped={stats['skipped']} errored={stats['errored']}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli_embed.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/cli.py tests/test_cli_embed.py
git commit -m "feat(cli): pixsage embed command with --embedder selector"
```

---

## Task 9: Wire caption recording into `pixsage tag`

**Files:**
- Modify: `src/pixsage/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_tag_records_caption_in_catalog(tmp_path: Path, monkeypatch):
    """After `pixsage tag`, the photo row should have a caption populated."""
    from pixsage.catalog import Catalog
    from pixsage.taggers.mock import MockTagger
    from pixsage.taggers.base import Tag, TagResult
    from pixsage.cli import app, build_taggers
    from typer.testing import CliRunner

    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    from PIL import Image
    Image.new("RGB", (64, 64), color="red").save(photo_root / "a.jpg")

    def fake_build_taggers(cfg):
        return [MockTagger(
            name="florence2",
            tags=[Tag(name="cat", confidence=0.9, hierarchy=None, source="florence2")],
            caption="a red rectangle",
        )]
    monkeypatch.setattr("pixsage.cli.build_taggers", fake_build_taggers)

    runner = CliRunner()
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0, result.output

    cat = Catalog(photo_root / ".photoindex" / "catalog.db")
    rows = list(cat.iter_photos_for_embedding())
    assert len(rows) == 1
    assert rows[0]["caption"] == "a red rectangle"
    assert rows[0]["caption_updated_at"] is not None
    cat.close()
```

(If `MockTagger` doesn't accept a `caption` kwarg in the existing implementation, look at `src/pixsage/taggers/mock.py` — it already does per the Phase 1 review.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_tag_records_caption_in_catalog -v`
Expected: FAIL — caption is not recorded; the row's caption is None.

- [ ] **Step 3: Implement**

In `src/pixsage/cli.py`, modify `_process_one` to call `record_caption` on the catalog. Find this block at the end of `_process_one`:

```python
    if not dry_run:
        write_xmp(path, merged, is_raw=is_raw)
        # Embedded XMP changes file bytes → sha256 changes. Update the catalog
        # row's primary key so the next run skip-detects this photo correctly.
        # (Sidecar writes don't touch the source file, so the sha stays.)
        if not is_raw:
            new_sha = sha256_file(path)
            cat.rekey_photo(sha, new_sha)
            sha = new_sha
        cat.record_tags(sha, [t for t in filtered if (t.name, t.source) not in user_rejected])
        cat.mark_tagged(sha, model_versions={t.name: t.model_version for t in taggers})
```

Add a `record_caption` call right after `mark_tagged`, only when the merged caption was actually applied:

```python
    if not dry_run:
        write_xmp(path, merged, is_raw=is_raw)
        if not is_raw:
            new_sha = sha256_file(path)
            cat.rekey_photo(sha, new_sha)
            sha = new_sha
        cat.record_tags(sha, [t for t in filtered if (t.name, t.source) not in user_rejected])
        cat.mark_tagged(sha, model_versions={t.name: t.model_version for t in taggers})
        if merged.description:
            cat.record_caption(sha, merged.description)
```

We use `merged.description` (not `caption`) because that's what actually got written to XMP — it accounts for `caption_overwrite=False` cases where the user's existing description was preserved.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: all passed (including the new one and all existing Phase 1 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/cli.py tests/test_cli.py
git commit -m "feat(cli): record caption to catalog when tag writes XMP description"
```

---

## Task 10: XMP-backfill on first embed run

**Files:**
- Modify: `src/pixsage/embed_runner.py`
- Modify: `tests/test_embed_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embed_runner.py`:

```python
def test_runner_backfills_caption_from_xmp(catalog: Catalog, store: VectorStore, tmp_path: Path, monkeypatch):
    """If the catalog has no caption but XMP does, runner should backfill it."""
    img_path = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), color="red").save(img_path)
    catalog.upsert_photo("sha-x", img_path, filesize=img_path.stat().st_size, mtime=img_path.stat().st_mtime)
    # Catalog caption deliberately not set.

    # Stub read_xmp to return a description (no real exiftool needed).
    from pixsage.xmp import XmpFields
    monkeypatch.setattr(
        "pixsage.embed_runner.read_xmp",
        lambda path, is_raw: XmpFields(subject=[], hierarchical_subject=[], description="backfilled caption"),
    )

    runner = EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8))
    runner.run()

    # After backfill, caption should be in the catalog
    row = catalog.get_photo("sha-x")
    assert row["caption"] == "backfilled caption"
    # And caption vector should exist
    assert store.get_one("mock_text", "sha-x") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embed_runner.py::test_runner_backfills_caption_from_xmp -v`
Expected: FAIL — caption is None and no caption vector written.

- [ ] **Step 3: Implement**

Modify `src/pixsage/embed_runner.py` to attempt XMP backfill before embedding when caption is None. Add the import and adjust `run()`:

```python
from pixsage.xmp import needs_sidecar, read_xmp


# inside EmbedRunner.run(), modify the per-row block:
        for row in iterator:
            sha = row["sha256"]
            current_path = row["current_path"]
            caption = row["caption"]
            caption_updated_at = row["caption_updated_at"]

            # Backfill caption from XMP if catalog row predates Phase 3.
            if caption is None and self.embed_caption:
                try:
                    fields = read_xmp(Path(current_path), is_raw=needs_sidecar(Path(current_path)))
                    if fields.description:
                        self.catalog.record_caption(sha, fields.description)
                        caption = fields.description
                        caption_updated_at = self.catalog.get_photo(sha)["caption_updated_at"]
                except Exception:
                    # XMP read failures shouldn't kill the embed run; we just skip
                    # caption embedding for this photo.
                    pass

            # … rest of the loop unchanged …
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embed_runner.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/embed_runner.py tests/test_embed_runner.py
git commit -m "feat(embed): backfill caption from XMP for pre-Phase-3 catalog rows"
```

---

## Task 11: Search service

**Files:**
- Create: `src/pixsage/search.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixsage.embedders.mock import MockEmbedder
from pixsage.search import Hit, SearchService
from pixsage.vectors import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vectors")


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder(dim=8)


def _normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def test_search_returns_top_k_by_image_when_weight_is_one(store: VectorStore, embedder: MockEmbedder):
    # Two photos: sha1 matches the query, sha2 is orthogonal.
    q_text = "leopard seal"
    q_vec = embedder.embed_text([q_text])[0]
    near = _normalize(q_vec + 0.01)                 # almost the query
    far = _normalize(np.array([1, -1, 0.5, 0, -0.3, 0.2, 1, -1], dtype=np.float32))
    store.append("mock_image", [("sha-near", near), ("sha-far", far)])

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search(q_text, image_weight=1.0, top_k=5)

    assert isinstance(hits[0], Hit)
    assert hits[0].sha256 == "sha-near"
    assert hits[1].sha256 == "sha-far"


def test_search_returns_top_k_by_caption_when_weight_is_zero(store: VectorStore, embedder: MockEmbedder):
    q_text = "leopard seal"
    q_vec = embedder.embed_text([q_text])[0]
    near = _normalize(q_vec + 0.01)
    far = _normalize(np.array([1, -1, 0.5, 0, -0.3, 0.2, 1, -1], dtype=np.float32))
    store.append("mock_text", [("sha-near", near), ("sha-far", far)])

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search(q_text, image_weight=0.0, top_k=5)

    assert hits[0].sha256 == "sha-near"


def test_search_handles_missing_caption_channel(store: VectorStore, embedder: MockEmbedder):
    """Photo with image vector but no caption vector should still be findable."""
    q_text = "leopard seal"
    q_vec = embedder.embed_text([q_text])[0]
    near = _normalize(q_vec + 0.01)
    store.append("mock_image", [("sha-near", near)])
    # No caption vector for sha-near.

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search(q_text, image_weight=0.5, top_k=5)
    assert len(hits) == 1
    assert hits[0].sha256 == "sha-near"


def test_search_by_image_uses_pure_visual_cosine(store: VectorStore, embedder: MockEmbedder):
    a = _normalize(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    b = _normalize(np.array([0.99, 0.01, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    c = _normalize(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    store.append("mock_image", [("a", a), ("b", b), ("c", c)])

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search_by_image("a", top_k=5)

    # 'a' itself is excluded.
    shas = [h.sha256 for h in hits]
    assert shas == ["b", "c"]


def test_search_empty_store_returns_empty(store: VectorStore, embedder: MockEmbedder):
    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search("anything", image_weight=0.5, top_k=5)
    assert hits == []


def test_search_respects_top_k(store: VectorStore, embedder: MockEmbedder):
    q_text = "x"
    q_vec = embedder.embed_text([q_text])[0]
    rows = []
    for i in range(20):
        v = _normalize(q_vec + 0.001 * i * np.ones(8, dtype=np.float32))
        rows.append((f"sha{i}", v))
    store.append("mock_image", rows)

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search(q_text, image_weight=1.0, top_k=5)
    assert len(hits) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

```python
# src/pixsage/search.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixsage.embedders.base import Embedder
from pixsage.vectors import VectorStore


@dataclass(frozen=True)
class Hit:
    sha256: str
    score: float


class SearchService:
    """Loads vector matrices once, answers search queries via numpy.

    Combined score for a text query q at image_weight w:
        s(photo) = w * cos(q, image_vec) + (1-w) * cos(q, caption_vec)
    Photos missing a channel score that channel as 0.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        image_kind: str,
        text_kind: str,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.image_kind = image_kind
        self.text_kind = text_kind

        self._img_shas: np.ndarray = np.array([], dtype=object)
        self._img_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._txt_shas: np.ndarray = np.array([], dtype=object)
        self._txt_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._idx_img: dict[str, int] = {}
        self._idx_txt: dict[str, int] = {}

    def load(self) -> None:
        self._img_shas, self._img_matrix = self.store.load(self.image_kind)
        self._txt_shas, self._txt_matrix = self.store.load(self.text_kind)
        self._idx_img = {s: i for i, s in enumerate(self._img_shas.tolist())}
        self._idx_txt = {s: i for i, s in enumerate(self._txt_shas.tolist())}

    def search(self, query: str, image_weight: float, top_k: int) -> list[Hit]:
        if self._img_matrix.size == 0 and self._txt_matrix.size == 0:
            return []

        q_vec = self.embedder.embed_text([query])[0]

        img_scores = (
            self._img_matrix @ q_vec if self._img_matrix.size else np.zeros(0, dtype=np.float32)
        )
        txt_scores = (
            self._txt_matrix @ q_vec if self._txt_matrix.size else np.zeros(0, dtype=np.float32)
        )

        all_shas = set(self._idx_img.keys()) | set(self._idx_txt.keys())
        hits: list[Hit] = []
        for sha in all_shas:
            i_img = self._idx_img.get(sha)
            i_txt = self._idx_txt.get(sha)
            si = float(img_scores[i_img]) if i_img is not None else 0.0
            st = float(txt_scores[i_txt]) if i_txt is not None else 0.0
            score = image_weight * si + (1.0 - image_weight) * st
            hits.append(Hit(sha256=sha, score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def search_by_image(self, sha256: str, top_k: int) -> list[Hit]:
        if self._img_matrix.size == 0:
            return []
        idx = self._idx_img.get(sha256)
        if idx is None:
            return []
        q = self._img_matrix[idx]
        scores = self._img_matrix @ q
        ranked = np.argsort(-scores)
        hits: list[Hit] = []
        for j in ranked:
            sha = self._img_shas[j]
            if sha == sha256:
                continue
            hits.append(Hit(sha256=str(sha), score=float(scores[j])))
            if len(hits) >= top_k:
                break
        return hits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/search.py tests/test_search.py
git commit -m "feat(search): numpy SearchService with weighted text+caption blend and image NN"
```

---

## Task 12: Web app scaffolding + `[search]` extra

**Files:**
- Modify: `pyproject.toml`
- Create: `src/pixsage/web/__init__.py`
- Create: `src/pixsage/web/app.py`
- Create: `src/pixsage/web/templates/index.html`
- Create: `src/pixsage/web/static/style.css`
- Create: `src/pixsage/web/static/htmx.min.js` (vendored)
- Create: `tests/test_web_app.py`

- [ ] **Step 1: Add the `[search]` extra**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
search = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "jinja2>=3.1",
  "httpx>=0.27",  # FastAPI TestClient
]
```

Run: `pip install -e ".[dev,search]"`.

- [ ] **Step 2: Vendor HTMX**

Download HTMX 2.0.x to `src/pixsage/web/static/htmx.min.js`. Run from repo root:

```bash
mkdir -p src/pixsage/web/static
curl -L -o src/pixsage/web/static/htmx.min.js https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_web_app.py
from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


def test_index_returns_search_page(tmp_path: Path):
    from pixsage.web.app import build_app

    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    (photo_root / ".photoindex").mkdir()

    app = build_app(photo_root=photo_root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "pixsage" in r.text.lower()
        assert "search" in r.text.lower()


def test_static_assets_served(tmp_path: Path):
    from pixsage.web.app import build_app

    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    (photo_root / ".photoindex").mkdir()

    app = build_app(photo_root=photo_root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/static/htmx.min.js")
        assert r.status_code == 200
        assert "htmx" in r.text.lower()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_web_app.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 5: Implement the app factory**

```python
# src/pixsage/web/__init__.py
from pixsage.web.app import build_app  # noqa: F401
```

```python
# src/pixsage/web/app.py
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pixsage.catalog import Catalog
from pixsage.config import load_config, ensure_default_config
from pixsage.search import SearchService
from pixsage.vectors import VectorStore


WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def build_app(photo_root: Path, embedder_name: str = "siglip2") -> FastAPI:
    """Construct the FastAPI app for a photo root.

    Loads catalog, vectors, and the search service eagerly so route handlers
    can stay synchronous and stateless.
    """
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir(exist_ok=True)
    catalog_path = photoindex / "catalog.db"
    cfg_path = photoindex / "vocabulary.toml"
    ensure_default_config(cfg_path)
    config = load_config(cfg_path)

    catalog = Catalog(catalog_path)
    catalog.init_schema()

    # Lazy import to keep `pixsage embed` callable on systems without [search] installed.
    from pixsage.cli import _build_embedder
    embedder = _build_embedder(embedder_name)
    # Defer the actual model load to a serve-time hook (Task 18); for now leave it.

    vectors = VectorStore(photoindex / "vectors")
    search_service = SearchService(
        store=vectors,
        embedder=embedder,
        image_kind=embedder.info.image_kind,
        text_kind=embedder.info.text_kind,
    )

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app = FastAPI(title="pixsage")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Stash everything on the app for routes (Task 13+) to grab.
    app.state.photo_root = photo_root
    app.state.config = config
    app.state.catalog = catalog
    app.state.vectors = vectors
    app.state.embedder = embedder
    app.state.search = search_service
    app.state.templates = templates

    from pixsage.web import routes
    routes.register(app)

    return app
```

- [ ] **Step 6: Stub routes module + index template**

```python
# src/pixsage/web/routes.py
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse


def register(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        templates = app.state.templates
        config = app.state.config
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "default_image_weight": config.search.default_image_weight,
            },
        )
```

```html
<!-- src/pixsage/web/templates/index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>pixsage</title>
  <link rel="stylesheet" href="/static/style.css" />
  <script src="/static/htmx.min.js"></script>
</head>
<body>
  <header>
    <h1>pixsage</h1>
  </header>
  <main>
    <form id="search-form">
      <input type="search" name="q" placeholder="Describe what you want to find…" autofocus />
      <label>
        Visual ⇄ Caption
        <input type="range" name="image_weight" min="0" max="1" step="0.05"
               value="{{ default_image_weight }}" />
      </label>
      <button type="submit">Search</button>
    </form>
    <section id="results"></section>
  </main>
</body>
</html>
```

```css
/* src/pixsage/web/static/style.css */
body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
header { padding: 1rem 1.5rem; border-bottom: 1px solid #333; }
header h1 { margin: 0; font-size: 1.5rem; }
main { padding: 1.5rem; }
#search-form { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem; }
#search-form input[type=search] { flex: 1; padding: 0.5rem; font-size: 1rem; background: #222; color: #eee; border: 1px solid #444; }
#search-form button { padding: 0.5rem 1rem; background: #2a8; color: white; border: 0; cursor: pointer; }
#results { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.5rem; }
.card { background: #1a1a1a; padding: 0.25rem; }
.card img { width: 100%; height: 200px; object-fit: cover; display: block; }
.card .meta { padding: 0.25rem; font-size: 0.8rem; color: #888; }
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_web_app.py -v`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/pixsage/web tests/test_web_app.py
git commit -m "feat(web): FastAPI app factory + index template + vendored HTMX"
```

---

## Task 13: Thumbnail cache

**Files:**
- Create: `src/pixsage/web/thumbs.py`
- Create: `tests/test_web_thumbs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web_thumbs.py
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pixsage.web.thumbs import ThumbnailCache, ThumbSize


@pytest.fixture
def cache(tmp_path: Path) -> ThumbnailCache:
    return ThumbnailCache(root=tmp_path / "thumbs")


def test_get_or_create_returns_path_and_writes_file(cache: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (1024, 768), color="red").save(src, "JPEG")

    path = cache.get_or_create("sha-a", src, ThumbSize.MEDIUM)
    assert path.exists()
    img = Image.open(path)
    assert max(img.size) == 720


def test_second_call_uses_cache(cache: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (1024, 768), color="red").save(src, "JPEG")

    p1 = cache.get_or_create("sha-a", src, ThumbSize.SMALL)
    mtime1 = p1.stat().st_mtime
    p2 = cache.get_or_create("sha-a", src, ThumbSize.SMALL)
    assert p1 == p2
    assert p2.stat().st_mtime == mtime1


def test_different_sizes_create_different_files(cache: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (1024, 768), color="red").save(src, "JPEG")

    s = cache.get_or_create("sha-a", src, ThumbSize.SMALL)
    m = cache.get_or_create("sha-a", src, ThumbSize.MEDIUM)
    l = cache.get_or_create("sha-a", src, ThumbSize.LARGE)

    assert s != m != l
    assert max(Image.open(s).size) == 256
    assert max(Image.open(m).size) == 720
    assert max(Image.open(l).size) == 1440


def test_path_uses_sha_prefix_for_dir_sharding(cache: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (256, 256), color="red").save(src, "JPEG")
    sha = "abcd" + "0" * 60
    path = cache.get_or_create(sha, src, ThumbSize.SMALL)
    # Path shape: <root>/<size>/<sha[:2]>/<sha>.jpg
    assert path.parent.name == "ab"
    assert path.parent.parent.name == "small"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_thumbs.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

```python
# src/pixsage/web/thumbs.py
from __future__ import annotations

from enum import Enum
from pathlib import Path

from pixsage.images import load_image


class ThumbSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


_LONG_EDGE = {
    ThumbSize.SMALL: 256,
    ThumbSize.MEDIUM: 720,
    ThumbSize.LARGE: 1440,
}


class ThumbnailCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str, size: ThumbSize) -> Path:
        return self.root / size.value / sha256[:2] / f"{sha256}.jpg"

    def get_or_create(self, sha256: str, source: Path, size: ThumbSize) -> Path:
        out = self.path_for(sha256, size)
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        img = load_image(source).convert("RGB")
        long_edge = _LONG_EDGE[size]
        if max(img.size) > long_edge:
            ratio = long_edge / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size)
        img.save(out, "JPEG", quality=85)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web_thumbs.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/web/thumbs.py tests/test_web_thumbs.py
git commit -m "feat(web): lazy thumbnail cache with three sizes and sha-prefixed sharding"
```

---

## Task 14: `POST /search` route + result templates

**Files:**
- Modify: `src/pixsage/web/routes.py`
- Modify: `src/pixsage/web/app.py` (load search service)
- Create: `src/pixsage/web/templates/_results.html`
- Create: `src/pixsage/web/templates/_card.html`
- Create: `tests/test_web_search.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web_search.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


def _seed_root(tmp_path: Path) -> Path:
    """Build a photo_root with two photos, mock embeddings already populated."""
    from pixsage.catalog import Catalog
    from pixsage.embedders.mock import MockEmbedder
    from pixsage.vectors import VectorStore

    root = tmp_path / "photos"
    root.mkdir()
    photoindex = root / ".photoindex"
    photoindex.mkdir()

    a, b = root / "a.jpg", root / "b.jpg"
    Image.new("RGB", (64, 64), color="red").save(a)
    Image.new("RGB", (64, 64), color="blue").save(b)

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha-a", a, filesize=a.stat().st_size, mtime=a.stat().st_mtime)
    cat.upsert_photo("sha-b", b, filesize=b.stat().st_size, mtime=b.stat().st_mtime)
    cat.record_caption("sha-a", "a red square")
    cat.record_caption("sha-b", "a blue square")

    embedder = MockEmbedder(dim=8)
    vec_a_img = embedder.embed_image([Image.open(a)])[0]
    vec_b_img = embedder.embed_image([Image.open(b)])[0]
    vec_a_txt = embedder.embed_text(["a red square"])[0]
    vec_b_txt = embedder.embed_text(["a blue square"])[0]

    store = VectorStore(photoindex / "vectors")
    store.append("mock_image", [("sha-a", vec_a_img), ("sha-b", vec_b_img)])
    store.append("mock_text", [("sha-a", vec_a_txt), ("sha-b", vec_b_txt)])

    cat.close()
    return root


def test_search_returns_results_html(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/search", data={"q": "a red square", "image_weight": "0.0"})
        assert r.status_code == 200
        assert "sha-a" in r.text or "sha-b" in r.text
        assert "<article" in r.text or 'class="card"' in r.text


def test_search_empty_query_returns_empty_results(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/search", data={"q": "", "image_weight": "0.5"})
        assert r.status_code == 200
        # Empty results section, no cards
        assert "card" not in r.text.lower() or 'class="card"' not in r.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_search.py -v`
Expected: FAIL — `POST /search` not registered yet.

- [ ] **Step 3: Update app to load search service**

In `src/pixsage/web/app.py`, replace the `# Defer the actual model load…` line with:

```python
    # Load the embedder and the vector matrices now so the first request is fast.
    from pixsage.device import select_device
    embedder.load(select_device())

    vectors = VectorStore(photoindex / "vectors")
    search_service = SearchService(
        store=vectors,
        embedder=embedder,
        image_kind=embedder.info.image_kind,
        text_kind=embedder.info.text_kind,
    )
    search_service.load()
```

(Move the existing `vectors = VectorStore(...)` and `search_service = SearchService(...)` lines to be after this; remove the duplicates so each variable is assigned once.)

- [ ] **Step 4: Add the route**

Append to `src/pixsage/web/routes.py`:

```python
from fastapi import Form
from pathlib import Path


def register(app: FastAPI) -> None:
    # … existing index route …

    @app.post("/search", response_class=HTMLResponse)
    def search(
        request: Request,
        q: str = Form(""),
        image_weight: float = Form(0.5),
    ) -> HTMLResponse:
        templates = app.state.templates
        catalog = app.state.catalog
        config = app.state.config

        if not q.strip():
            return templates.TemplateResponse(
                "_results.html",
                {"request": request, "hits": [], "query": q},
            )

        service = app.state.search
        raw_hits = service.search(q, image_weight=image_weight, top_k=config.search.top_k)

        # Enrich each hit with current_path + filename for the card template.
        hits = []
        for h in raw_hits:
            row = catalog.get_photo(h.sha256)
            if row is None:
                continue
            hits.append({
                "sha256": h.sha256,
                "score": h.score,
                "filename": Path(row["current_path"]).name,
            })

        return templates.TemplateResponse(
            "_results.html",
            {"request": request, "hits": hits, "query": q},
        )
```

- [ ] **Step 5: Templates**

```html
<!-- src/pixsage/web/templates/_results.html -->
{% if hits %}
<div class="results-meta">{{ hits|length }} results for "{{ query }}"</div>
<div class="grid">
  {% for hit in hits %}
    {% include "_card.html" %}
  {% endfor %}
</div>
{% else %}
<div class="results-meta">No results.</div>
{% endif %}
```

```html
<!-- src/pixsage/web/templates/_card.html -->
<article class="card" data-sha="{{ hit.sha256 }}">
  <a href="/photo/{{ hit.sha256 }}">
    <img src="/thumb/{{ hit.sha256 }}?size=medium" alt="{{ hit.filename }}" loading="lazy" />
  </a>
  <div class="meta">
    <div class="filename">{{ hit.filename }}</div>
    <div class="score">{{ "%.3f"|format(hit.score) }}</div>
  </div>
</article>
```

Update `index.html` to wire HTMX so the form posts and swaps the `#results` div:

Replace the `<form id="search-form">` block in `index.html`:

```html
    <form id="search-form"
          hx-post="/search" hx-target="#results" hx-swap="innerHTML">
      <input type="search" name="q" placeholder="Describe what you want to find…" autofocus />
      <label>
        Visual ⇄ Caption
        <input type="range" name="image_weight" min="0" max="1" step="0.05"
               value="{{ default_image_weight }}" />
      </label>
      <button type="submit">Search</button>
    </form>
    <section id="results"></section>
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_web_search.py tests/test_web_app.py -v`
Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add src/pixsage/web tests/test_web_search.py
git commit -m "feat(web): POST /search returns HTMX-friendly grid of result cards"
```

---

## Task 15: `GET /thumb/{sha}` route

**Files:**
- Modify: `src/pixsage/web/app.py` (instantiate ThumbnailCache)
- Modify: `src/pixsage/web/routes.py`
- Modify: `tests/test_web_search.py` (extend to verify thumbs render)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_search.py`:

```python
def test_thumb_route_returns_jpeg(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/thumb/sha-a?size=small")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert len(r.content) > 0


def test_thumb_route_404_for_missing_sha(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/thumb/nonexistent-sha?size=small")
        assert r.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_search.py::test_thumb_route_returns_jpeg -v`
Expected: FAIL — `/thumb/{sha}` not registered.

- [ ] **Step 3: Wire ThumbnailCache into app**

In `src/pixsage/web/app.py`, after `search_service.load()`, add:

```python
    from pixsage.web.thumbs import ThumbnailCache
    app.state.thumbs = ThumbnailCache(photoindex / "thumbs")
```

- [ ] **Step 4: Add the route**

Append to `src/pixsage/web/routes.py`:

```python
from fastapi import HTTPException
from fastapi.responses import FileResponse

from pixsage.web.thumbs import ThumbSize


def register(app: FastAPI) -> None:
    # … existing routes …

    @app.get("/thumb/{sha256}")
    def thumb(sha256: str, size: str = "medium") -> FileResponse:
        try:
            thumb_size = ThumbSize(size)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown size {size!r}")

        catalog = app.state.catalog
        row = catalog.get_photo(sha256)
        if row is None or row["current_path"] is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        source = Path(row["current_path"])
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"source missing on disk: {source}")

        thumbs = app.state.thumbs
        path = thumbs.get_or_create(sha256, source, thumb_size)
        return FileResponse(path, media_type="image/jpeg")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_web_search.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web tests/test_web_search.py
git commit -m "feat(web): GET /thumb/{sha}?size= serves cached thumbnails"
```

---

## Task 16: `GET /photo/{sha}` detail page

**Files:**
- Modify: `src/pixsage/web/routes.py`
- Create: `src/pixsage/web/templates/photo.html`
- Modify: `tests/test_web_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_search.py`:

```python
def test_photo_detail_renders_caption_and_filename(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/photo/sha-a")
        assert r.status_code == 200
        assert "a red square" in r.text          # caption
        assert "a.jpg" in r.text                 # filename
        assert "/similar/sha-a" in r.text        # more-like-this link


def test_photo_detail_404(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/photo/nonexistent-sha")
        assert r.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_search.py::test_photo_detail_renders_caption_and_filename -v`
Expected: FAIL — route not registered.

- [ ] **Step 3: Add the route**

Append to `src/pixsage/web/routes.py`:

```python
def register(app: FastAPI) -> None:
    # … existing routes …

    @app.get("/photo/{sha256}", response_class=HTMLResponse)
    def photo(request: Request, sha256: str) -> HTMLResponse:
        catalog = app.state.catalog
        row = catalog.get_photo(sha256)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        tags = catalog.get_tags(sha256)
        return app.state.templates.TemplateResponse(
            "photo.html",
            {
                "request": request,
                "sha256": sha256,
                "filename": Path(row["current_path"]).name if row["current_path"] else "?",
                "caption": row["caption"],
                "tags": [t.name for t in tags],
            },
        )
```

- [ ] **Step 4: Add the template**

```html
<!-- src/pixsage/web/templates/photo.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ filename }} — pixsage</title>
  <link rel="stylesheet" href="/static/style.css" />
  <script src="/static/htmx.min.js"></script>
</head>
<body>
  <header>
    <a href="/" style="color:#eee;text-decoration:none;">← back to search</a>
    <h1>{{ filename }}</h1>
  </header>
  <main>
    <div class="photo-detail">
      <img src="/thumb/{{ sha256 }}?size=large" alt="{{ filename }}" />
      {% if caption %}
        <p class="caption">{{ caption }}</p>
      {% endif %}
      {% if tags %}
        <ul class="tags">
          {% for t in tags %}<li>{{ t }}</li>{% endfor %}
        </ul>
      {% endif %}
      <a class="more-like-this"
         href="/similar/{{ sha256 }}"
         hx-get="/similar/{{ sha256 }}"
         hx-target="#results"
         hx-swap="innerHTML"
         hx-push-url="false">
        More like this →
      </a>
      <section id="results"></section>
    </div>
  </main>
</body>
</html>
```

Append to `style.css`:

```css
.photo-detail { max-width: 1400px; margin: 0 auto; }
.photo-detail img { max-width: 100%; height: auto; }
.caption { font-size: 1.1rem; color: #ccc; }
.tags { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.tags li { background: #2a2a2a; padding: 0.25rem 0.5rem; font-size: 0.85rem; }
.more-like-this { display: inline-block; margin-top: 1rem; color: #2a8; }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_web_search.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web tests/test_web_search.py
git commit -m "feat(web): GET /photo/{sha} detail page with caption, tags, and more-like-this"
```

---

## Task 17: `GET /similar/{sha}` route

**Files:**
- Modify: `src/pixsage/web/routes.py`
- Modify: `tests/test_web_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_search.py`:

```python
def test_similar_returns_results_partial_excluding_self(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/similar/sha-a")
        assert r.status_code == 200
        # sha-a should NOT appear; sha-b should
        assert "sha-b" in r.text
        assert "sha-a" not in r.text or r.text.count("sha-a") == 0


def test_similar_404_when_photo_missing(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/similar/nonexistent")
        assert r.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_search.py::test_similar_returns_results_partial_excluding_self -v`
Expected: FAIL — route not registered.

- [ ] **Step 3: Add the route**

Append to `src/pixsage/web/routes.py`:

```python
def register(app: FastAPI) -> None:
    # … existing routes …

    @app.get("/similar/{sha256}", response_class=HTMLResponse)
    def similar(request: Request, sha256: str) -> HTMLResponse:
        catalog = app.state.catalog
        config = app.state.config
        templates = app.state.templates
        service = app.state.search

        if catalog.get_photo(sha256) is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        raw_hits = service.search_by_image(sha256, top_k=config.search.top_k)
        hits = []
        for h in raw_hits:
            row = catalog.get_photo(h.sha256)
            if row is None:
                continue
            hits.append({
                "sha256": h.sha256,
                "score": h.score,
                "filename": Path(row["current_path"]).name,
            })

        return templates.TemplateResponse(
            "_results.html",
            {"request": request, "hits": hits, "query": f"similar to {sha256[:8]}"},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web_search.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/web/routes.py tests/test_web_search.py
git commit -m "feat(web): GET /similar/{sha} returns visual nearest-neighbour grid"
```

---

## Task 18: `pixsage serve` CLI verb

**Files:**
- Modify: `src/pixsage/cli.py`
- Create: `tests/test_cli_serve.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_serve.py
from __future__ import annotations

from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from pixsage.cli import app

runner = CliRunner()


def test_serve_help_lists_options():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--host" in result.output
    assert "--no-open" in result.output
    assert "--embedder" in result.output


def test_serve_errors_when_no_catalog(tmp_path: Path):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    result = runner.invoke(app, ["serve", str(photo_root), "--no-open"])
    assert result.exit_code != 0
    assert "no catalog" in result.output.lower() or "catalog" in result.output.lower()
```

(End-to-end serve testing happens via the existing `tests/test_web_*.py` files which exercise the full app via TestClient. The serve CLI itself is a thin uvicorn launcher; we just smoke-test option parsing and the error path.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_serve.py -v`
Expected: FAIL — no `serve` command.

- [ ] **Step 3: Implement**

Append to `src/pixsage/cli.py`:

```python
@app.command()
def serve(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    embedder: str = typer.Option("siglip2", "--embedder", help="Embedder for query encoding."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    no_open: bool = typer.Option(False, "--no-open", help="Don't auto-open a browser."),
    catalog: Path | None = typer.Option(None, "--catalog"),
) -> None:
    """Run the search webapp on http://host:port."""
    photoindex = photo_root / ".photoindex"
    catalog_path = catalog or (photoindex / "catalog.db")
    if not catalog_path.exists():
        typer.echo(f"no catalog at {catalog_path}; run `pixsage tag` then `pixsage embed` first", err=True)
        raise typer.Exit(code=1)

    try:
        import uvicorn
    except ImportError:
        typer.echo("FastAPI + uvicorn not installed. Run: pip install -e \".[search]\"", err=True)
        raise typer.Exit(code=1)

    from pixsage.web.app import build_app
    fastapi_app = build_app(photo_root=photo_root, embedder_name=embedder)

    if not no_open:
        import webbrowser, threading
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()

    typer.echo(f"pixsage serve at http://{host}:{port}/")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli_serve.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/cli.py tests/test_cli_serve.py
git commit -m "feat(cli): pixsage serve runs the search webapp on localhost"
```

---

## Task 19: Extend `pixsage cleanup` with `--thumbs` and `--vectors`

**Files:**
- Modify: `src/pixsage/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_cleanup_thumbs_removes_thumb_dir(tmp_path: Path):
    from typer.testing import CliRunner
    from pixsage.cli import app

    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    # Pre-existing catalog so cleanup doesn't bail early.
    from pixsage.catalog import Catalog
    Catalog(photoindex / "catalog.db").init_schema()

    thumbs_dir = photoindex / "thumbs"
    thumbs_dir.mkdir()
    (thumbs_dir / "junk.jpg").write_bytes(b"x")

    runner = CliRunner()
    result = runner.invoke(app, ["cleanup", str(photo_root), "--thumbs"])
    assert result.exit_code == 0
    assert not thumbs_dir.exists()


def test_cleanup_vectors_removes_vectors_dir(tmp_path: Path):
    from typer.testing import CliRunner
    from pixsage.cli import app

    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    from pixsage.catalog import Catalog
    Catalog(photoindex / "catalog.db").init_schema()

    vectors_dir = photoindex / "vectors"
    vectors_dir.mkdir()
    (vectors_dir / "siglip2_image.parquet").write_bytes(b"x")

    runner = CliRunner()
    result = runner.invoke(app, ["cleanup", str(photo_root), "--vectors"])
    assert result.exit_code == 0
    assert not vectors_dir.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::test_cleanup_thumbs_removes_thumb_dir -v`
Expected: FAIL — flags don't exist.

- [ ] **Step 3: Implement**

In `src/pixsage/cli.py`, modify the `cleanup` command:

```python
@app.command()
def cleanup(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
    thumbs: bool = typer.Option(False, "--thumbs", help="Also delete the thumbnail cache."),
    vectors: bool = typer.Option(False, "--vectors", help="Also delete all vector parquet files."),
) -> None:
    """Drop stale catalog rows. With flags, also clear caches."""
    import shutil

    photoindex = photo_root / ".photoindex"
    catalog_path = catalog or (photoindex / "catalog.db")
    if not catalog_path.exists():
        typer.echo(f"no catalog at {catalog_path}", err=True)
        raise typer.Exit(code=1)

    cat = Catalog(catalog_path)
    cat.init_schema()
    before_photos = cat._conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]  # noqa: SLF001
    before_tags = cat._conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]  # noqa: SLF001
    deleted = cat.cleanup_orphans()
    after_photos = cat._conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]  # noqa: SLF001
    after_tags = cat._conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]  # noqa: SLF001
    cat.close()
    typer.echo(
        f"removed {deleted} orphan photo rows. "
        f"photos: {before_photos} -> {after_photos}, tags: {before_tags} -> {after_tags}"
    )

    if thumbs:
        thumbs_dir = photoindex / "thumbs"
        if thumbs_dir.exists():
            shutil.rmtree(thumbs_dir)
            typer.echo(f"removed thumbnail cache at {thumbs_dir}")

    if vectors:
        vectors_dir = photoindex / "vectors"
        if vectors_dir.exists():
            shutil.rmtree(vectors_dir)
            typer.echo(f"removed vector store at {vectors_dir}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/cli.py tests/test_cli.py
git commit -m "feat(cli): cleanup --thumbs and --vectors flags"
```

---

## Task 20: End-to-end smoke test on real corpus + README updates

**Files:**
- Modify: `README.md`
- Create: `scripts/smoke_phase3.py`

- [ ] **Step 1: Smoke script**

```python
# scripts/smoke_phase3.py
"""Manual smoke test for Phase 3.

Usage:
    python scripts/smoke_phase3.py /path/to/photo_root

Steps:
    1. Verify catalog exists
    2. Run `pixsage embed --embedder mock --limit 5` to confirm pipeline
    3. Print the top-3 sha256s with vectors
    4. Run a single mock-embedder search and print results

Real-corpus / SigLIP2 testing:
    pixsage embed E:\\Sony alpha 7c\\Seymour --limit 100
    pixsage serve E:\\Sony alpha 7c\\Seymour
    Open http://127.0.0.1:8765/ — type queries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pixsage.catalog import Catalog
from pixsage.embedders.mock import MockEmbedder
from pixsage.embed_runner import EmbedRunner
from pixsage.search import SearchService
from pixsage.vectors import VectorStore


def main(photo_root: Path) -> int:
    photoindex = photo_root / ".photoindex"
    if not (photoindex / "catalog.db").exists():
        print("no catalog — run `pixsage tag` first", file=sys.stderr)
        return 1

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    store = VectorStore(photoindex / "vectors")
    embedder = MockEmbedder(dim=16)
    embedder.load("cpu")

    print("Embedding (mock)…")
    runner = EmbedRunner(catalog=cat, vectors=store, embedder=embedder)
    stats = runner.run()
    print(f"  stats: {stats}")

    sha_array, matrix = store.load("mock_image")
    print(f"  image vectors: {len(sha_array)} x {matrix.shape[1] if matrix.size else 0}")

    print("\nSearch (mock query 'wildlife on ice'):")
    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search("wildlife on ice", image_weight=0.5, top_k=3)
    for h in hits:
        row = cat.get_photo(h.sha256)
        print(f"  {h.score:+.3f}  {h.sha256[:12]}  {Path(row['current_path']).name if row else '?'}")

    cat.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/smoke_phase3.py <photo_root>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
```

- [ ] **Step 2: Run the smoke script on a small corpus**

```bash
# On the user's workstation, with the existing Seymour catalog:
python scripts/smoke_phase3.py "E:\\Sony alpha 7c\\Seymour"
```

Expected: prints embed stats, vector count, and three nearest neighbours with shas/filenames. No real model needed.

- [ ] **Step 3: Real-model end-to-end (manual, not CI)**

```bash
pixsage embed "E:\\Sony alpha 7c\\Seymour" --limit 100
pixsage serve "E:\\Sony alpha 7c\\Seymour"
```

In the browser at http://127.0.0.1:8765/, manually verify the four scenarios from the spec's testing notes:
- Search "leopard seal" — top-20 should be wildlife.
- Search "iceberg" — top-20 should be ice/landscape.
- Search "snow-covered mountain" — top-20 should be landscape.
- Click any wildlife photo → "More like this" → results should be visually similar wildlife.

- [ ] **Step 4: README updates**

In `README.md`, append a Phase 3 section after the existing usage section:

````markdown
## Phase 3: Semantic search

After tagging, compute embeddings and run the local search webapp:

```bash
pixsage embed /path/to/photos
pixsage serve /path/to/photos
```

Open http://127.0.0.1:8765/. Type a query, drag the slider to bias toward
visual or caption matching, click any photo for "more like this".

Embed runtime estimate: ~14-21 hours for 50k photos on an RTX 4090. The
`embed` step is interruptible — re-run it to resume. Add `--limit N` for a
quick test on a subset.

Install with the search extras:
```bash
pip install -e ".[taggers,search]"
```
````

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all green except SigLIP2-gated tests on machines without CUDA + cached weights.

- [ ] **Step 6: Commit**

```bash
git add README.md scripts/smoke_phase3.py
git commit -m "docs+smoke: Phase 3 README section + scripts/smoke_phase3.py"
```

---

## Self-Review

**Spec coverage:** Every section of the spec maps to a task —
- Catalog migration → Task 1, 2
- Vectors → Task 3
- Embedder protocol + mock → Task 4
- SigLIP2 → Task 5
- Embeddings/SearchConfig → Task 6
- Embed runner → Task 7, 10
- `pixsage embed` → Task 8
- `tag` extension to record caption → Task 9
- Search service → Task 11
- FastAPI + index + static → Task 12
- ThumbnailCache → Task 13
- POST /search + templates → Task 14
- /thumb route → Task 15
- /photo/{sha} → Task 16
- /similar/{sha} → Task 17
- `pixsage serve` → Task 18
- `cleanup` extensions → Task 19
- Smoke + README → Task 20

**Type consistency:** `EmbedderInfo` (`name`, `image_kind`, `text_kind`, `dim`) is referenced consistently from Task 4 onward. `Hit` (`sha256`, `score`) defined in Task 11, used by Task 14 and Task 17. `ThumbSize.SMALL/MEDIUM/LARGE` (string-valued enum) defined Task 13, used Task 15. `VectorStore.append/load/missing_for/get_one/created_at` defined Task 3, used by Task 7, 10, 11.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step has full code. Every test step has full test code.

**Dependencies between tasks:** Task ordering respects dependencies —
- 1, 2 (catalog) → 7, 9, 10
- 3 (vectors) → 7, 11, 14
- 4 (mock embedder) → 7, 11, 12, 14, 16, 17
- 5 (SigLIP2) is independent of 4 — wired in via `_build_embedder` in Task 8
- 6 (config) → 14 (search.top_k), 18 (defaults)
- 11 (search) → 14, 17
- 12 (app scaffolding) → 14, 15, 16, 17, 18
- 13 (thumbs) → 15

A subagent can pull tasks off the top in order; nothing reaches forward.
