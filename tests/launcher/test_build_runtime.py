from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.launcher.build_runtime import (
    download_pbs_tarball,
    extract_pbs_tarball,
    verify_python_binary,
)


def test_download_uses_cache_when_present(tmp_path: Path) -> None:
    """If the tarball is already in the cache dir, no download happens."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fake_tarball = cache_dir / "fake.tar.gz"
    fake_tarball.write_bytes(b"already-here")

    # Patch urllib.request.urlretrieve to fail loudly if called.
    with patch("scripts.launcher.build_runtime.urlretrieve") as m:
        result = download_pbs_tarball(
            url="https://example.com/fake.tar.gz",
            cache_dir=cache_dir,
        )
    m.assert_not_called()
    assert result == fake_tarball


def test_download_fetches_when_cache_miss(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def fake_urlretrieve(url: str, dest: str) -> tuple[str, object]:
        Path(dest).write_bytes(b"downloaded")
        return (dest, None)

    with patch("scripts.launcher.build_runtime.urlretrieve", side_effect=fake_urlretrieve) as m:
        result = download_pbs_tarball(
            url="https://example.com/foo.tar.gz",
            cache_dir=cache_dir,
        )
    m.assert_called_once()
    assert result.exists()
    assert result.read_bytes() == b"downloaded"


def test_extract_unpacks_into_runtime_dir(tmp_path: Path) -> None:
    """Extract a tiny tarball and verify the runtime dir has expected contents."""
    import tarfile
    import io

    # Build a minimal tarball: contains python/foo.txt
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="python/foo.txt")
        payload = b"hello"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    tarball = tmp_path / "tiny.tar.gz"
    tarball.write_bytes(buf.getvalue())

    runtime_dir = tmp_path / "runtime"
    extract_pbs_tarball(tarball, runtime_dir)
    assert (runtime_dir / "python" / "foo.txt").read_bytes() == b"hello"


def test_verify_python_binary_runs_python_v(tmp_path: Path) -> None:
    """Uses the host python as a stand-in to verify the version-check logic."""
    import sys

    host_python = Path(sys.executable)
    expected_version = ".".join(str(p) for p in sys.version_info[:3])
    # verify_python_binary should not raise when versions match
    verify_python_binary(host_python, expected_version)


def test_verify_python_binary_rejects_wrong_version(tmp_path: Path) -> None:
    import sys

    host_python = Path(sys.executable)
    with pytest.raises(RuntimeError, match="version mismatch"):
        verify_python_binary(host_python, "9.99.999")


def test_pip_install_invokes_target_pip(tmp_path: Path) -> None:
    """pip_install should call the runtime's python with `-m pip install --target`."""
    from scripts.launcher.build_runtime import pip_install

    fake_python = tmp_path / "python.exe"
    fake_python.write_bytes(b"")  # presence-only

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        # Mimic CompletedProcess
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    with patch("scripts.launcher.build_runtime.subprocess.run", side_effect=fake_run):
        pip_install(
            python_exe=fake_python,
            target_dir=tmp_path / "site-packages",
            project_dir=tmp_path / "project",
            extras="serve",
        )
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == str(fake_python)
    assert cmd[1:5] == ["-m", "pip", "install", "--target"]
    assert cmd[5] == str(tmp_path / "site-packages")
    # The project + extras spec is the last arg
    assert cmd[-1].endswith("[serve]")
