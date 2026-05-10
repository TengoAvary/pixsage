"""CLI demo: print a summary of an exported `.photoindex/`.

For programmatic use in notebooks/analysis scripts, import directly:

    from pixsage.analysis import load_export
    e = load_export("/path/to/unpacked/.photoindex")
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(photoindex_arg: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from pixsage.analysis import load_export  # noqa: PLC0415

    photoindex = Path(photoindex_arg)
    e = load_export(photoindex)

    n = len(e.shas)
    print(f"{photoindex}: {n} photos")
    print(f"  {sum(1 for s in e.shas if s in e.captions)} with captions")
    print(f"  {sum(1 for s in e.shas if e.tags[s])} with tags")
    print(f"  {sum(1 for s in e.shas if s in e.image_vecs)} with image vectors")
    print(f"  {sum(1 for s in e.shas if s in e.caption_vecs)} with caption vectors")
    print(f"  {sum(1 for s in e.shas if e.geo_predictions[s])} with geo predictions")

    if e.image_vecs:
        sample_dim = next(iter(e.image_vecs.values())).shape[0]
        print(f"  image vector dim: {sample_dim}")
    if e.caption_vecs:
        sample_dim = next(iter(e.caption_vecs.values())).shape[0]
        print(f"  caption vector dim: {sample_dim}")

    shas, mats = e.aligned_matrices(require=("image_vec",))
    if len(shas):
        print()
        print(f"Aligned image matrix: shape={mats['image'].shape}, dtype={mats['image'].dtype}")
    shas, mats = e.aligned_matrices(require=("image_vec", "caption_vec", "geo"))
    if len(shas):
        print(
            f"Aligned image+caption+geo: {len(shas)} photos with all three "
            f"(image {mats['image'].shape}, caption {mats['caption'].shape}, "
            f"geo_top1 {mats['geo_top1'].shape})"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/load_export.py <photoindex_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
