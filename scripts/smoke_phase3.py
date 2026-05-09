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
