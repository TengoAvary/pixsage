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
