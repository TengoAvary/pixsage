import pytest

from scripts.launcher.pbs_targets import PBS_RELEASE, TARGETS, get_target


def test_pbs_release_is_pinned() -> None:
    assert PBS_RELEASE  # non-empty string like "20260508"
    assert PBS_RELEASE.isdigit()
    assert len(PBS_RELEASE) == 8  # YYYYMMDD


def test_targets_have_expected_keys() -> None:
    assert set(TARGETS.keys()) == {"windows-x64", "macos-arm64", "macos-x86_64"}


def test_macos_x86_64_target_uses_intel_triple() -> None:
    target = get_target("macos-x86_64")
    assert "x86_64-apple-darwin" in target.tarball_url
    assert target.python_relpath == "python/bin/python3"


def test_get_target_returns_known() -> None:
    target = get_target("windows-x64")
    assert target.tarball_url.startswith("https://github.com/astral-sh/python-build-standalone/")
    assert target.tarball_url.endswith(".tar.gz")
    assert target.python_relpath  # e.g. "python/install/python.exe" or "python/bin/python3"
    assert target.expected_python_version.startswith("3.12.")


def test_get_target_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_target("linux-x64")
