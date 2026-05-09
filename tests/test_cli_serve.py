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
