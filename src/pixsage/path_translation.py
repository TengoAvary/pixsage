from __future__ import annotations

from pathlib import Path, PureWindowsPath, PurePosixPath


class PathResolver:
    """Translate stored catalog paths onto the runtime filesystem.

    The catalog's `meta.photo_root_at_embed` records the root used at embed
    time (e.g. r"E:\\Sony alpha 7c" on Windows). At serve time the actual
    files may live somewhere else — different drive letter, different OS,
    different mount point. Resolver substitutes the prefix and falls back
    to the verbatim stored path when the substitution doesn't exist.
    """

    def __init__(self, stored_root: str | None, runtime_root: Path) -> None:
        self._stored_root = stored_root
        self._runtime_root = Path(runtime_root)

    def resolve(self, stored_path: str) -> Path:
        """Return a Path pointing at the file on the current filesystem.

        - If `stored_root` is None (legacy catalog with no anchor), pass through
          as a native Path.
        - If `stored_path` starts with `stored_root`, swap the prefix for
          `runtime_root` and return the result if that file exists.
        - Otherwise return the stored path verbatim. Caller is responsible for
          checking `.exists()` and surfacing 404-style errors.
        """
        if self._stored_root is None:
            return Path(stored_path)

        translated = self._try_translate(stored_path)
        if translated is not None and translated.exists():
            return translated

        # Last resort: maybe the stored path happens to exist verbatim
        # (e.g. drive layout matches across machines).
        verbatim = Path(stored_path)
        if verbatim.exists():
            return verbatim

        # Nothing exists. Return the translated guess (better diagnostics)
        # if we have one, else verbatim.
        return translated if translated is not None else verbatim

    def _try_translate(self, stored_path: str) -> Path | None:
        # The stored path was written by str(Path(...)) on the embed-time OS.
        # On Windows that means backslashes; on POSIX, forward slashes. We
        # use the relevant Pure*Path to compute the relative subpath.
        for pure_cls in (PureWindowsPath, PurePosixPath):
            try:
                stored = pure_cls(stored_path)
                root = pure_cls(self._stored_root)  # type: ignore[arg-type]
                # is_relative_to was added in 3.9
                if stored.parts[: len(root.parts)] == root.parts:
                    relative_parts = stored.parts[len(root.parts) :]
                    return self._runtime_root.joinpath(*relative_parts)
            except (ValueError, IndexError):
                continue
        return None
