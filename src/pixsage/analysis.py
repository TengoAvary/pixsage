"""Read an exported `.photoindex/` for offline analysis.

The photo files themselves don't need to exist — clustering / aggregation
work runs off the catalog DB (paths, captions, tags, geo predictions) and
the parquet vector files. This module is the canonical loader for that
read path so analysis scripts don't reinvent the SQL/parquet boilerplate.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pixsage.geolocators.base import GeoPrediction


@dataclass
class Export:
    """Everything in a `.photoindex/` directory, keyed by sha256.

    Photos that lack a particular kind of data (no caption, no vectors, no
    geo predictions) are simply absent from the relevant dict — values are
    not zero-filled or NaN-padded. Use `aligned_matrices()` to materialize
    a matrix view across photos that DO have a given combination of fields.
    """

    photoindex: Path
    shas: list[str]                                  # canonical photo order (sha256-sorted)
    filenames: dict[str, str]
    paths: dict[str, str]                            # absolute paths from the source machine
    captions: dict[str, str]                         # only photos with a non-None caption
    tags: dict[str, list[str]]                       # excludes user_rejected; [] if untagged
    image_vecs: dict[str, np.ndarray]                # (image_dim,) float32 each
    caption_vecs: dict[str, np.ndarray]              # (caption_dim,) float32 each
    geo_predictions: dict[str, list[GeoPrediction]]  # top-K per photo (empty list if absent)

    def aligned_matrices(
        self, require: tuple[str, ...] = ("image_vec",)
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Materialize a matrix view across photos that have ALL required fields.

        require: any subset of ("image_vec", "caption_vec", "geo").
        Returns (shas, mats) where shas is a (M,) object array and mats is a dict
        keyed by "image" / "caption" / "geo_top1" depending on what was required.
        Each matrix is in the same row order as `shas`.
        """
        valid = list(self.shas)
        if "image_vec" in require:
            valid = [s for s in valid if s in self.image_vecs]
        if "caption_vec" in require:
            valid = [s for s in valid if s in self.caption_vecs]
        if "geo" in require:
            valid = [s for s in valid if self.geo_predictions.get(s)]

        shas = np.array(valid, dtype=object)
        mats: dict[str, np.ndarray] = {}
        if "image_vec" in require:
            mats["image"] = (
                np.stack([self.image_vecs[s] for s in valid])
                if valid else np.zeros((0, 0), dtype=np.float32)
            )
        if "caption_vec" in require:
            mats["caption"] = (
                np.stack([self.caption_vecs[s] for s in valid])
                if valid else np.zeros((0, 0), dtype=np.float32)
            )
        if "geo" in require:
            mats["geo_top1"] = (
                np.array(
                    [
                        (
                            self.geo_predictions[s][0].latitude,
                            self.geo_predictions[s][0].longitude,
                            self.geo_predictions[s][0].score,
                        )
                        for s in valid
                    ],
                    dtype=np.float32,
                )
                if valid else np.zeros((0, 3), dtype=np.float32)
            )
        return shas, mats


def load_export(photoindex: Path) -> Export:
    """Load a `.photoindex/` directory. Source photos do not need to exist."""
    photoindex = Path(photoindex)
    catalog_path = photoindex / "catalog.db"
    if not catalog_path.exists():
        raise FileNotFoundError(f"no catalog.db at {photoindex}")

    con = sqlite3.connect(catalog_path)
    con.row_factory = sqlite3.Row
    try:
        shas: list[str] = []
        filenames: dict[str, str] = {}
        paths: dict[str, str] = {}
        captions: dict[str, str] = {}
        for r in con.execute(
            "SELECT sha256, filename, current_path, caption FROM photos ORDER BY sha256"
        ):
            sha = r["sha256"]
            shas.append(sha)
            filenames[sha] = r["filename"]
            paths[sha] = r["current_path"]
            if r["caption"]:
                captions[sha] = r["caption"]

        tags: dict[str, list[str]] = {sha: [] for sha in shas}
        for r in con.execute(
            "SELECT sha256, tag FROM tags WHERE user_rejected = 0 ORDER BY sha256, tag"
        ):
            if r["sha256"] in tags:
                tags[r["sha256"]].append(r["tag"])

        geo_predictions: dict[str, list[GeoPrediction]] = {sha: [] for sha in shas}
        # geo_predictions table only exists in catalogs that picked up the Phase 4
        # schema migration. Older catalogs predate it — treat absence as "no
        # predictions" rather than an error.
        try:
            cur = con.execute(
                "SELECT sha256, latitude, longitude, score FROM geo_predictions "
                "ORDER BY sha256, rank"
            )
            for r in cur:
                if r["sha256"] in geo_predictions:
                    geo_predictions[r["sha256"]].append(
                        GeoPrediction(
                            latitude=r["latitude"],
                            longitude=r["longitude"],
                            score=r["score"],
                        )
                    )
        except sqlite3.OperationalError:
            pass
    finally:
        con.close()

    image_vecs = _load_parquet(photoindex / "vectors" / "siglip2_image.parquet")
    caption_vecs = _load_parquet(photoindex / "vectors" / "minilm_caption.parquet")

    return Export(
        photoindex=photoindex,
        shas=shas,
        filenames=filenames,
        paths=paths,
        captions=captions,
        tags=tags,
        image_vecs=image_vecs,
        caption_vecs=caption_vecs,
        geo_predictions=geo_predictions,
    )


def _load_parquet(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    table = pq.read_table(path)
    shas = table.column("sha256").to_pylist()
    vecs = table.column("vector").to_pylist()
    return {sha: np.array(v, dtype=np.float32) for sha, v in zip(shas, vecs)}
