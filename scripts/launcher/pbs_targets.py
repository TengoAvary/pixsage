"""python-build-standalone target table.

Pinned versions: update PBS_RELEASE + PYTHON_VERSION here when bumping.
Verify URLs by checking https://github.com/astral-sh/python-build-standalone/releases/tag/<PBS_RELEASE>
"""
from __future__ import annotations

from dataclasses import dataclass

# python-build-standalone release tag (YYYYMMDD format).
# Verified to contain 3.12.x install_only_stripped tarballs for both targets below.
PBS_RELEASE = "20260508"
PYTHON_VERSION = "3.12.13"


@dataclass(frozen=True)
class PBSTarget:
    name: str  # human-readable target id, e.g. "windows-x64"
    tarball_url: str
    # Path inside the extracted tarball where the python binary lives,
    # relative to the runtime root we extract into.
    python_relpath: str
    expected_python_version: str


def _pbs_url(filename: str) -> str:
    return (
        f"https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{PBS_RELEASE}/{filename}"
    )


TARGETS: dict[str, PBSTarget] = {
    "windows-x64": PBSTarget(
        name="windows-x64",
        tarball_url=_pbs_url(
            f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
        ),
        # The install_only_stripped layout puts python.exe directly under python/
        python_relpath="python/python.exe",
        expected_python_version=PYTHON_VERSION,
    ),
    "macos-arm64": PBSTarget(
        name="macos-arm64",
        tarball_url=_pbs_url(
            f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-aarch64-apple-darwin-install_only_stripped.tar.gz"
        ),
        python_relpath="python/bin/python3",
        expected_python_version=PYTHON_VERSION,
    ),
}


def get_target(name: str) -> PBSTarget:
    if name not in TARGETS:
        raise KeyError(
            f"unknown target {name!r}; known: {sorted(TARGETS.keys())}"
        )
    return TARGETS[name]
