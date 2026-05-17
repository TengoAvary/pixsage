"""Find pixsage catalogs (.photoindex/ directories) by walking a user-chosen root.

Called by POST /catalogs/add-scan. The walk is bounded by max_depth and a time
budget (BFS), skipping SKIP_DIRS and dotfiles, and stops descending into
directories that already contain .photoindex/.
"""
from __future__ import annotations

import logging
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


def safe_is_dir(p: "Path") -> bool:
    """Path.is_dir() that treats un-stattable paths as non-dirs.

    Path.is_dir() swallows generic OSError but NOT PermissionError
    (EACCES) — SIP-protected files under a system volume would otherwise
    raise. Used by the walker and the folder-browser endpoint.
    """
    try:
        return p.is_dir()
    except OSError:
        return False


def walk_for_photoindex(
    roots: Iterable[Path],
    *,
    max_depth: int = 6,
    time_budget_s: float = 15.0,
) -> list[Path]:
    """BFS each root; return absolute paths of every `.photoindex/` found.

    Stop descending into any directory that itself contains `.photoindex/`
    (no nested catalogs). Skip directories whose name is in SKIP_DIRS or
    begins with `.` (other than `.photoindex` itself, which is the find).
    Bounded by max_depth from each root and time_budget_s across the whole
    walk.
    """
    def _is_dir(p: Path) -> bool:
        # Path.is_dir() swallows generic OSError but NOT PermissionError
        # (EACCES) — SIP-protected files under a mounted system volume
        # (e.g. /Volumes/Macintosh HD/usr/sbin/*) would otherwise abort
        # the whole walk. Treat anything we can't stat as "not a dir".
        try:
            return p.is_dir()
        except OSError:
            return False

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
                if child.name == ".photoindex" and _is_dir(child):
                    photoindex_here = child
                    break
            if photoindex_here is not None:
                found.append(photoindex_here.resolve())
                continue

            # No catalog here — keep walking, but respect depth budget.
            if depth >= max_depth:
                continue
            for child in children:
                if not _is_dir(child):
                    continue
                if child.name in SKIP_DIRS or child.name.startswith("."):
                    continue
                queue.append((child, depth + 1))

    return found
