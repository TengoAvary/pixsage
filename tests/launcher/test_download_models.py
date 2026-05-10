from pathlib import Path
from unittest.mock import patch

import pytest


def test_download_models_invokes_snapshot_download_for_each_repo(tmp_path: Path) -> None:
    from scripts.launcher.download_models import REPOS, download_models

    captured: list[dict] = []

    def fake_snapshot(repo_id: str, **kwargs) -> str:
        captured.append({"repo_id": repo_id, **kwargs})
        # Return a fake local path; snapshot_download contract is to return the
        # path to the downloaded snapshot.
        return str(tmp_path / repo_id.replace("/", "--"))

    with patch("scripts.launcher.download_models.snapshot_download", side_effect=fake_snapshot):
        download_models(out_dir=tmp_path / "models")

    repo_ids = {c["repo_id"] for c in captured}
    assert repo_ids == set(REPOS)
    for c in captured:
        # All calls should target <out>/models/hub
        assert c["cache_dir"] == str(tmp_path / "models" / "hub")


def test_repos_contains_siglip2_and_minilm() -> None:
    from scripts.launcher.download_models import REPOS

    assert "google/siglip2-so400m-patch14-384" in REPOS
    assert "sentence-transformers/all-MiniLM-L6-v2" in REPOS
