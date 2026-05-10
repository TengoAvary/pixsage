from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pixsage.analysis import load_export
from pixsage.catalog import Catalog
from pixsage.geolocators.base import GeoPrediction
from pixsage.taggers.base import Tag


def _seed(tmp_path: Path) -> Path:
    """Build a tiny `.photoindex/` with two photos — one fully populated, one tag-only."""
    photoindex = tmp_path / ".photoindex"
    photoindex.mkdir()

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha-a", tmp_path / "a.jpg", filesize=100, mtime=1.0)
    cat.upsert_photo("sha-b", tmp_path / "b.jpg", filesize=200, mtime=2.0)
    cat.record_caption("sha-a", "a leopard seal")
    cat.record_tags("sha-a", [
        Tag(name="seal", confidence=0.9, hierarchy=None, source="ram"),
        Tag(name="ice", confidence=0.7, hierarchy=None, source="ram"),
    ])
    cat.record_tags("sha-b", [
        Tag(name="rock", confidence=0.8, hierarchy=None, source="ram"),
    ])
    cat.record_geo_predictions("sha-a", "geoclip", [
        GeoPrediction(latitude=-64.0, longitude=-57.0, score=0.5),
        GeoPrediction(latitude=-65.5, longitude=-60.0, score=0.2),
    ])
    cat.close()

    vec_dir = photoindex / "vectors"
    vec_dir.mkdir()
    pq.write_table(
        pa.table({
            "sha256": ["sha-a", "sha-b"],
            "vector": pa.array([[0.1] * 4, [0.2] * 4], type=pa.list_(pa.float32())),
            "created_at": ["2026-01-01", "2026-01-01"],
        }),
        vec_dir / "siglip2_image.parquet",
    )
    pq.write_table(
        pa.table({
            "sha256": ["sha-a"],
            "vector": pa.array([[0.3] * 3], type=pa.list_(pa.float32())),
            "created_at": ["2026-01-01"],
        }),
        vec_dir / "minilm_caption.parquet",
    )

    return photoindex


def test_load_returns_all_data(tmp_path: Path):
    e = load_export(_seed(tmp_path))

    assert e.shas == sorted(["sha-a", "sha-b"])
    assert e.filenames["sha-a"] == "a.jpg"
    assert e.captions == {"sha-a": "a leopard seal"}
    assert sorted(e.tags["sha-a"]) == ["ice", "seal"]
    assert e.tags["sha-b"] == ["rock"]
    assert e.image_vecs["sha-a"].shape == (4,)
    assert e.image_vecs["sha-b"].shape == (4,)
    assert e.caption_vecs["sha-a"].shape == (3,)
    assert "sha-b" not in e.caption_vecs
    assert len(e.geo_predictions["sha-a"]) == 2
    assert e.geo_predictions["sha-a"][0].latitude == -64.0
    assert e.geo_predictions["sha-b"] == []


def test_load_handles_missing_vectors_dir(tmp_path: Path):
    """Catalog-only export (no embed run) should still load."""
    photoindex = tmp_path / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    cat.close()

    e = load_export(photoindex)
    assert e.shas == ["sha1"]
    assert e.image_vecs == {}
    assert e.caption_vecs == {}
    assert e.geo_predictions == {"sha1": []}


def test_load_missing_catalog_raises(tmp_path: Path):
    photoindex = tmp_path / ".photoindex"
    photoindex.mkdir()
    with pytest.raises(FileNotFoundError):
        load_export(photoindex)


def test_aligned_matrices_intersects_required(tmp_path: Path):
    e = load_export(_seed(tmp_path))

    shas, mats = e.aligned_matrices(require=("image_vec",))
    assert len(shas) == 2
    assert mats["image"].shape == (2, 4)

    shas, mats = e.aligned_matrices(require=("image_vec", "caption_vec"))
    assert list(shas) == ["sha-a"]
    assert mats["image"].shape == (1, 4)
    assert mats["caption"].shape == (1, 3)

    shas, mats = e.aligned_matrices(require=("image_vec", "geo"))
    assert list(shas) == ["sha-a"]
    assert mats["geo_top1"].shape == (1, 3)
    np.testing.assert_allclose(mats["geo_top1"][0], [-64.0, -57.0, 0.5], atol=1e-5)


def test_aligned_matrices_handles_empty(tmp_path: Path):
    photoindex = tmp_path / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.close()

    e = load_export(photoindex)
    shas, mats = e.aligned_matrices(require=("image_vec",))
    assert len(shas) == 0


def test_load_excludes_user_rejected_tags(tmp_path: Path):
    photoindex = tmp_path / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    cat.record_tags("sha1", [
        Tag(name="kept", confidence=0.9, hierarchy=None, source="ram"),
        Tag(name="rejected", confidence=0.9, hierarchy=None, source="ram"),
    ])
    cat.flag_user_rejections("sha1", surviving_xmp_tags={"kept"})
    cat.close()

    e = load_export(photoindex)
    assert e.tags["sha1"] == ["kept"]
