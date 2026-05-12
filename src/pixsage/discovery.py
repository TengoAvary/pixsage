"""Find pixsage catalogs (`.photoindex/` directories) by walking mounted drives.

Used at serve startup to detect newly-plugged-in drives. The walk is bounded
(BFS, max depth, time budget) and stops descending into directories that
already contain `.photoindex/` — those subtrees are owned by their catalog.
"""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable


log = logging.getLogger(__name__)


# Directory names we never descend into. Cuts walk time and avoids false
# positives inside dev trees / OS metadata.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".cache",
    ".Trash", ".Trashes", "System Volume Information",
    ".fseventsd", ".Spotlight-V100", ".TemporaryItems",
})


def list_mounted_roots() -> list[Path]:
    """Return likely roots for `walk_for_photoindex`.

    Mac:  /Volumes/* (excluding the boot volume) + ~/
    Win:  every live drive letter
    Linux: /media/*, /mnt/*, ~/  (best-effort)
    """
    roots: list[Path] = []
    home = Path.home()

    if sys.platform == "darwin":
        volumes = Path("/Volumes")
        if volumes.exists():
            for v in volumes.iterdir():
                roots.append(v)
        roots.append(home)
    elif sys.platform == "win32":
        import string
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                roots.append(drive)
    else:
        for parent in (Path("/media"), Path("/mnt")):
            if parent.exists():
                for v in parent.iterdir():
                    roots.append(v)
        roots.append(home)

    return roots


def walk_for_photoindex(
    roots: Iterable[Path],
    *,
    max_depth: int = 6,
    time_budget_s: float = 5.0,
) -> list[Path]:
    """BFS each root; return absolute paths of every `.photoindex/` found.

    Stop descending into any directory that itself contains `.photoindex/`
    (no nested catalogs). Skip directories whose name is in SKIP_DIRS or
    begins with `.` (other than `.photoindex` itself, which is the find).
    Bounded by max_depth from each root and time_budget_s across the whole
    walk.
    """
    found: list[Path] = []
    deadline = time.monotonic() + time_budget_s

    for root in roots:
        if not root.exists():
            continue
        # (path, depth) queue
        queue: deque[tuple[Path, int]] = deque([(root, 0)])
        while queue:
            if time.monotonic() > deadline:
                log.warning("walk_for_photoindex hit time budget; partial results")
                return found
            current, depth = queue.popleft()

            try:
                children = list(current.iterdir())
            except (PermissionError, OSError) as e:
                log.debug("skipping %s: %s", current, e)
                continue

            # Check this dir for a .photoindex child first. If we find one,
            # add it and do NOT descend further from `current`.
            photoindex_here = None
            for child in children:
                if child.name == ".photoindex" and child.is_dir():
                    photoindex_here = child
                    break
            if photoindex_here is not None:
                found.append(photoindex_here.resolve())
                continue

            # No catalog here — keep walking, but respect depth budget.
            if depth >= max_depth:
                continue
            for child in children:
                if not child.is_dir():
                    continue
                if child.name in SKIP_DIRS or child.name.startswith("."):
                    continue
                queue.append((child, depth + 1))

    return found
