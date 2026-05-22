from __future__ import annotations

import os
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


def test_serve_sets_hf_offline_before_loading_models(monkeypatch, tmp_path: Path):
    """serve forces HF offline mode so model loads skip hub ETag round-trips —
    a ~10s-per-launch network cost when weights are already cached. The env
    vars must be set before build_app (which imports transformers) runs."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    seen: dict[str, str | None] = {}

    def fake_build_app(**kwargs):
        seen["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        seen["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE")
        raise SystemExit(0)  # bail before uvicorn.run

    monkeypatch.setattr("pixsage.web.app.build_app", fake_build_app)
    runner.invoke(app, ["serve", "--no-open", "--registry", str(tmp_path / "r.json")])

    assert seen["HF_HUB_OFFLINE"] == "1"
    assert seen["TRANSFORMERS_OFFLINE"] == "1"


def test_serve_respects_user_hf_offline_override(monkeypatch, tmp_path: Path):
    """setdefault: a user who explicitly sets HF_HUB_OFFLINE=0 (to refresh
    models) is not overridden by serve."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    seen: dict[str, str | None] = {}

    def fake_build_app(**kwargs):
        seen["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        raise SystemExit(0)

    monkeypatch.setattr("pixsage.web.app.build_app", fake_build_app)
    runner.invoke(app, ["serve", "--no-open", "--registry", str(tmp_path / "r.json")])

    assert seen["HF_HUB_OFFLINE"] == "0"
