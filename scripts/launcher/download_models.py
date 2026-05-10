"""Pre-stage SigLIP2 + MiniLM into the runtime's HF cache.

After this runs, `HF_HOME=<out>` is enough for transformers and
sentence-transformers to find both models offline. The on-disk layout is:

    <out>/hub/
        models--google--siglip2-so400m-patch14-384/
            snapshots/<rev>/...
        models--sentence-transformers--all-MiniLM-L6-v2/
            snapshots/<rev>/...

Total size: ~1.8 GB (will drop to ~280 MB once the text-tower-only optimization
in the follow-up plan ships).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPOS = [
    "google/siglip2-so400m-patch14-384",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def download_models(out_dir: Path) -> None:
    cache_dir = out_dir / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repo_id in REPOS:
        print(f"Downloading {repo_id} -> {cache_dir}")
        snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-stage pixsage runtime models.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory (will create <out>/hub/).")
    args = parser.parse_args()

    download_models(args.out)
    print(f"\nModels staged at: {args.out}")
    print(f"To use: HF_HOME={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
