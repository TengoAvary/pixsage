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


def test_serve_help_mentions_registry():
    """Multi-catalog serve exposes a --registry override and accepts no path."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--registry" in result.output


def test_serve_rejects_nonexistent_path(tmp_path: Path):
    """Passing an explicit photo_root that doesn't exist still errors."""
    missing = tmp_path / "nope"
    result = runner.invoke(app, ["serve", str(missing), "--no-open"])
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower() or "no such" in result.output.lower()
