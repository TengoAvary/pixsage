# Launcher Plan 1: Path Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pixsage serve` resolve photo paths correctly regardless of which machine, drive letter, or OS the catalog is being served on. Foundation for the clickable launcher (Plan 3).

**Architecture:** Catalog gains a `meta` key/value table that records the `photo_root` used at embed time. At serve startup, a `PathResolver` is constructed from the stored root + the runtime `photo_root` and used at every site that resolves `row["current_path"]` to disk. Resolver does prefix substitution; if substitution produces a path that doesn't exist, falls back to the verbatim stored path.

**Tech Stack:** Python 3.11+, SQLite, FastAPI, pytest. No new third-party deps.

**Companion plans (queued, not in this plan):**
- Plan 2: portable runtime build pipeline (`build_runtime.py`, `download_models.py`).
- Plan 3: native launcher (Rust crate) + folder staging.

---

## File Structure

**Create:**
- `src/pixsage/path_translation.py` — `PathResolver` class. Pure logic, no I/O.
- `tests/test_meta_table.py` — catalog `meta` table tests.
- `tests/test_path_translation.py` — resolver unit tests.
- `tests/test_serve_path_translation.py` — end-to-end test that a catalog built with `E:\foo` paths is correctly served from a different root.

**Modify:**
- `src/pixsage/catalog.py` — add `meta` schema, `set_meta` / `get_meta` / `set_photo_root_if_unset` methods.
- `src/pixsage/cli.py` — `tag`, `embed`, `geolocate` write `photo_root_at_embed` after `init_schema`.
- `src/pixsage/web/app.py` — construct `PathResolver` and stash on `app.state`.
- `src/pixsage/web/routes.py` — replace direct `Path(row["current_path"])` reads with resolver calls.
- `src/pixsage/web/clusters.py` (if it has similar reads) — same treatment.

---

### Task 1: Add `meta` table schema to catalog

**Files:**
- Modify: `src/pixsage/catalog.py` (add schema + class methods)
- Test: `tests/test_meta_table.py` (create)

**Background:** Catalog needs a small key/value store for catalog-scoped metadata that doesn't fit elsewhere. First key: `photo_root_at_embed`. Future keys (catalog format version, last embedder name, etc.) will reuse it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_meta_table.py`:

```python
from pathlib import Path

import pytest

from pixsage.catalog import Catalog


def test_meta_set_and_get(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "test.db")
    cat.init_schema()
    cat.set_meta("photo_root_at_embed", r"E:\Sony alpha 7c")
    assert cat.get_meta("photo_root_at_embed") == r"E:\Sony alpha 7c"
    cat.close()


def test_meta_get_missing_returns_none(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "test.db")
    cat.init_schema()
    assert cat.get_meta("does_not_exist") is None
    cat.close()


def test_meta_overwrites_existing_key(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "test.db")
    cat.init_schema()
    cat.set_meta("k", "v1")
    cat.set_meta("k", "v2")
    assert cat.get_meta("k") == "v2"
    cat.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_meta_table.py -v`
Expected: FAIL with `AttributeError: 'Catalog' object has no attribute 'set_meta'`

- [ ] **Step 3: Add the schema constant**

In `src/pixsage/catalog.py`, after `SCHEMA_USER_LOCATIONS` (around line 83), add:

```python
SCHEMA_META = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""
```

- [ ] **Step 4: Wire schema creation into `init_schema`**

In `src/pixsage/catalog.py`, in `Catalog.init_schema` (around line 98–106), add `SCHEMA_META` to the executescript calls:

```python
def init_schema(self) -> None:
    with self._conn:
        self._conn.executescript(SCHEMA_PHOTOS)
        self._conn.executescript(SCHEMA_TAGS)
        self._conn.executescript(SCHEMA_RUNS)
        self._conn.executescript(SCHEMA_GEO_PREDICTIONS)
        self._conn.executescript(SCHEMA_USER_LOCATIONS)
        self._conn.executescript(SCHEMA_META)
        self._migrate_add_caption_columns()
```

- [ ] **Step 5: Add `set_meta` and `get_meta` methods**

In `src/pixsage/catalog.py`, add as methods on the `Catalog` class (place near other small helpers, around line 140 area — near `mark_tagged`):

```python
def set_meta(self, key: str, value: str) -> None:
    with self._conn:
        self._conn.execute(
            """
            INSERT INTO meta (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, _now()),
        )

def get_meta(self, key: str) -> str | None:
    cur = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
    row = cur.fetchone()
    return row["value"] if row else None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_meta_table.py -v`
Expected: 3 passed

- [ ] **Step 7: Run full test suite to confirm no regressions**

Run: `pytest -x`
Expected: all existing tests pass (177 passing, 1 skipped, 1 xfailed at start of plan)

- [ ] **Step 8: Commit**

```bash
git add src/pixsage/catalog.py tests/test_meta_table.py
git commit -m "feat(catalog): add meta key/value table

Foundation for storing catalog-scoped metadata. First consumer
(next commit) will record photo_root_at_embed for path translation
across machines."
```

---

### Task 2: `set_photo_root_if_unset` + write from CLI verbs

**Files:**
- Modify: `src/pixsage/catalog.py` (add idempotent setter)
- Modify: `src/pixsage/cli.py` (call after `init_schema` in tag, embed, geolocate)
- Test: `tests/test_meta_table.py` (extend)

**Background:** Each pipeline verb knows the `photo_root` it was invoked with. The first one to run for a fresh catalog should record it; subsequent runs should leave it alone (so re-tagging with a slightly different `--catalog` path doesn't clobber the original embed root).

- [ ] **Step 1: Add the failing test**

Append to `tests/test_meta_table.py`:

```python
def test_set_photo_root_if_unset_writes_when_empty(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "test.db")
    cat.init_schema()
    cat.set_photo_root_if_unset(Path(r"E:\Sony alpha 7c"))
    assert cat.get_meta("photo_root_at_embed") == str(Path(r"E:\Sony alpha 7c"))
    cat.close()


def test_set_photo_root_if_unset_preserves_existing(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "test.db")
    cat.init_schema()
    cat.set_photo_root_if_unset(Path(r"E:\original"))
    cat.set_photo_root_if_unset(Path(r"F:\different"))
    assert cat.get_meta("photo_root_at_embed") == str(Path(r"E:\original"))
    cat.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_meta_table.py::test_set_photo_root_if_unset_writes_when_empty -v`
Expected: FAIL with `AttributeError: 'Catalog' object has no attribute 'set_photo_root_if_unset'`

- [ ] **Step 3: Add `set_photo_root_if_unset` method**

In `src/pixsage/catalog.py`, after the `set_meta` / `get_meta` block:

```python
def set_photo_root_if_unset(self, photo_root: Path) -> None:
    """Record the photo_root the catalog was built against, if not already set.
    Subsequent calls with different roots are no-ops — the FIRST recorded root
    wins. This anchor is what `PathResolver` uses to translate stored paths
    onto a different machine at serve time."""
    if self.get_meta("photo_root_at_embed") is None:
        self.set_meta("photo_root_at_embed", str(photo_root))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_meta_table.py -v`
Expected: 5 passed

- [ ] **Step 5: Wire into CLI verbs**

In `src/pixsage/cli.py`, find the three places where `Catalog(...)` is constructed and `init_schema()` is called for a verb that takes `photo_root` — currently `tag`, `embed`, and `geolocate`. After each `cat.init_schema()` add `cat.set_photo_root_if_unset(photo_root)`.

Concretely, look for these call sites:
- The `tag` command (around line 80-200): catalog construction. Add the call right after `init_schema`.
- The `embed` command (around line 325-326): after `cat.init_schema()  # picks up the caption migration if it's an older catalog`, add `cat.set_photo_root_if_unset(photo_root)`.
- The `geolocate` command: same pattern.

(Use grep to find the exact lines: `grep -n "init_schema()" src/pixsage/cli.py`.)

- [ ] **Step 6: Run full test suite**

Run: `pytest -x`
Expected: all green. Existing CLI tests should still pass; the new code is additive.

- [ ] **Step 7: Commit**

```bash
git add src/pixsage/catalog.py src/pixsage/cli.py tests/test_meta_table.py
git commit -m "feat(catalog): record photo_root_at_embed on first pipeline run

tag/embed/geolocate now anchor the catalog to the photo_root they were
invoked with. First-write-wins semantics so re-runs from a different
working dir don't clobber the embed-time anchor."
```

---

### Task 3: `PathResolver` module

**Files:**
- Create: `src/pixsage/path_translation.py`
- Test: `tests/test_path_translation.py` (create)

**Background:** Pure-logic class that translates a stored absolute path to a runtime absolute path. No I/O at construction; one stat call per resolve to handle the missing-translated-file fallback.

- [ ] **Step 1: Write the failing test**

Create `tests/test_path_translation.py`:

```python
from pathlib import Path, PureWindowsPath

import pytest

from pixsage.path_translation import PathResolver


def test_resolver_no_translation_when_roots_match(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"x")
    resolver = PathResolver(stored_root=str(tmp_path), runtime_root=tmp_path)
    assert resolver.resolve(str(target)) == target


def test_resolver_substitutes_prefix_when_translated_exists(tmp_path: Path) -> None:
    new_root = tmp_path / "new"
    new_root.mkdir()
    target = new_root / "sub" / "photo.jpg"
    target.parent.mkdir()
    target.write_bytes(b"x")

    # Stored path uses a fictional Windows root; runtime root is different.
    stored_path = r"E:\fakeroot\sub\photo.jpg"
    resolver = PathResolver(stored_root=r"E:\fakeroot", runtime_root=new_root)
    resolved = resolver.resolve(stored_path)
    assert resolved == target


def test_resolver_falls_back_to_stored_path_when_translated_missing(tmp_path: Path) -> None:
    # Translated path won't exist (we never create it). Stored path also doesn't exist
    # but resolver should still return the stored path, leaving the caller to detect
    # the missing file via downstream Path.exists() checks.
    stored_path = r"E:\fakeroot\sub\photo.jpg"
    resolver = PathResolver(stored_root=r"E:\fakeroot", runtime_root=tmp_path / "empty")
    resolved = resolver.resolve(stored_path)
    assert str(resolved).endswith("photo.jpg")


def test_resolver_handles_no_stored_root(tmp_path: Path) -> None:
    """If photo_root_at_embed was never set (legacy catalog), resolver passes
    paths through unchanged."""
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"x")
    resolver = PathResolver(stored_root=None, runtime_root=tmp_path)
    assert resolver.resolve(str(target)) == target


def test_resolver_handles_unix_to_windows_translation(tmp_path: Path) -> None:
    """Catalog made on Windows (E:\foo\bar.jpg), served on Unix
    (/Volumes/whatever/bar.jpg). Both use forward-slash and backslash, mixed."""
    target = tmp_path / "bar.jpg"
    target.write_bytes(b"x")
    stored_path = r"E:\foo\bar.jpg"
    resolver = PathResolver(stored_root=r"E:\foo", runtime_root=tmp_path)
    assert resolver.resolve(stored_path) == target
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_path_translation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pixsage.path_translation'`

- [ ] **Step 3: Implement the resolver**

Create `src/pixsage/path_translation.py`:

```python
from __future__ import annotations

from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Union


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_path_translation.py -v`
Expected: 5 passed

- [ ] **Step 5: Run full test suite**

Run: `pytest -x`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/path_translation.py tests/test_path_translation.py
git commit -m "feat(path_translation): PathResolver for cross-machine catalog serving

Pure-logic prefix-substitution. Handles cross-OS (Windows-stored,
served from Unix) via PureWindowsPath/PurePosixPath probing. Falls
back to stored path verbatim when translation produces a missing file."
```

---

### Task 4: Wire `PathResolver` into `build_app`

**Files:**
- Modify: `src/pixsage/web/app.py`
- Test: `tests/test_serve_path_translation.py` (create — placeholder for now; full e2e test in Task 6)

**Background:** Construct one resolver per app instance. Stash on `app.state.path_resolver`. Routes pull from there.

- [ ] **Step 1: Write the test stub**

Create `tests/test_serve_path_translation.py`:

```python
from pathlib import Path

import pytest


def test_app_state_has_path_resolver(tmp_path: Path) -> None:
    """build_app constructs a PathResolver from the catalog meta and
    runtime photo_root, exposed on app.state.path_resolver."""
    from pixsage.catalog import Catalog
    from pixsage.web.app import build_app

    photo_root = tmp_path / "drive" / "Sony alpha 7c"
    photo_root.mkdir(parents=True)

    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    cat.set_photo_root_if_unset(Path(r"E:\Sony alpha 7c"))
    cat.close()

    app = build_app(photo_root=photo_root, embedder_name="mock")
    resolver = app.state.path_resolver
    # Translation: stored prefix E:\Sony alpha 7c → runtime tmp_path/drive/Sony alpha 7c
    target = photo_root / "DSC_1234.ARW"
    target.write_bytes(b"raw")
    resolved = resolver.resolve(r"E:\Sony alpha 7c\DSC_1234.ARW")
    assert resolved == target
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_serve_path_translation.py -v`
Expected: FAIL with `AttributeError` on `app.state.path_resolver` (or similar).

- [ ] **Step 3: Add resolver construction in `build_app`**

In `src/pixsage/web/app.py`, after `catalog = Catalog(catalog_path); catalog.init_schema()` (around line 44–45), add:

```python
    from pixsage.path_translation import PathResolver
    stored_root = catalog.get_meta("photo_root_at_embed")
    path_resolver = PathResolver(stored_root=stored_root, runtime_root=photo_root)
```

Then after `app.state.thumbs = app_thumbs` (around line 77), add:

```python
    app.state.path_resolver = path_resolver
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_serve_path_translation.py::test_app_state_has_path_resolver -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest -x`
Expected: all green. (`mock` embedder is the standard test-mode embedder; no model load.)

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web/app.py tests/test_serve_path_translation.py
git commit -m "feat(web): construct PathResolver at app startup

Reads catalog.meta['photo_root_at_embed']; resolver lives on
app.state.path_resolver for routes to consume."
```

---

### Task 5: Routes use `PathResolver` for filesystem reads

**Files:**
- Modify: `src/pixsage/web/routes.py` (replace `Path(row["current_path"])` reads that touch disk)
- Test: extend `tests/test_serve_path_translation.py`

**Background:** Two classes of `current_path` reads in routes.py:
- **Display-only** (formatting `Path(row["current_path"]).name` for filename strings): leave untouched. Filename derives correctly regardless of root.
- **Filesystem-touching** (`source.exists()`, `FileResponse(source)`, thumbnail generation): MUST go through resolver.

The grep from earlier showed lines 78, 81, 229, 231 in `routes.py` are the filesystem-touching ones (the thumbnail and download routes). Lines 60, 102, 130 are display-only.

- [ ] **Step 1: Write a failing end-to-end test**

Append to `tests/test_serve_path_translation.py`:

```python
import io

from fastapi.testclient import TestClient
from PIL import Image


def _make_jpeg(path: Path, color: str = "red") -> None:
    img = Image.new("RGB", (32, 32), color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG")


def test_thumb_route_resolves_translated_path(tmp_path: Path) -> None:
    """Catalog has a current_path of E:\\Sony alpha 7c\\DSC_0001.JPG,
    file actually lives at tmp_path/drive/Sony alpha 7c/DSC_0001.JPG.
    /grid/thumb/<sha> should serve it."""
    from pixsage.catalog import Catalog
    from pixsage.web.app import build_app

    photo_root = tmp_path / "drive" / "Sony alpha 7c"
    photo_root.mkdir(parents=True)
    real = photo_root / "DSC_0001.JPG"
    _make_jpeg(real)

    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    cat.set_photo_root_if_unset(Path(r"E:\Sony alpha 7c"))
    # Insert photo with the FAKE Windows path; then the resolver has work to do.
    cat._conn.execute(
        "INSERT INTO photos (sha256, current_path, filename, filesize, mtime, added_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        ("abc123", r"E:\Sony alpha 7c\DSC_0001.JPG", "DSC_0001.JPG", real.stat().st_size, real.stat().st_mtime),
    )
    cat._conn.commit()
    cat.close()

    app = build_app(photo_root=photo_root, embedder_name="mock")
    client = TestClient(app)
    r = client.get("/grid/thumb/abc123?size=small")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_serve_path_translation.py::test_thumb_route_resolves_translated_path -v`
Expected: FAIL — likely 404 because `Path(r"E:\Sony alpha 7c\DSC_0001.JPG").exists()` is False on the test machine.

- [ ] **Step 3: Update the thumb route**

In `src/pixsage/web/routes.py`, find the thumb route (around line 70-87):

```python
        catalog = app.state.catalog
        row = catalog.get_photo(sha256)
        if row is None or row["current_path"] is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        source = Path(row["current_path"])
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"source missing on disk: {source}")
```

Replace `source = Path(row["current_path"])` with:

```python
        resolver = app.state.path_resolver
        source = resolver.resolve(row["current_path"])
```

The subsequent `if not source.exists():` line is unchanged — resolver already tries hard to find an existing path, so a False here means the file is genuinely missing.

- [ ] **Step 4: Update other filesystem-touching sites**

Use `grep -n 'Path(row\["current_path"\])' src/pixsage/web/routes.py` to find every site. Apply the same replacement (`Path(row["current_path"])` → `app.state.path_resolver.resolve(row["current_path"])`) ONLY for sites that subsequently call `.exists()`, `.read_bytes()`, `FileResponse(...)`, or pass the path into thumbnail generation.

DO NOT change sites that only do `Path(row["current_path"]).name` — those are filename strings for display and work fine without translation.

Specifically (line numbers from grep at start of plan, may have drifted):
- Line ~81 in thumb route: change to resolver call (above).
- Line ~229–231 in download/full-image route: change to resolver call.

Leave alone: lines ~60, ~102, ~130 (display-only `.name` reads).

- [ ] **Step 5: Check `clusters.py` and similar**

Run: `grep -rn 'Path(row\["current_path"\])\|Path(.*current_path)' src/pixsage/`
For every match outside routes.py that touches the filesystem, apply the same change. Display-only reads stay as-is.

If `clusters.py` doesn't have any FS-touching reads, leave it. (It probably operates on shas + vectors + lat/lon, not file bytes.)

- [ ] **Step 6: Run new test**

Run: `pytest tests/test_serve_path_translation.py -v`
Expected: all pass.

- [ ] **Step 7: Run full test suite**

Run: `pytest -x`
Expected: all green. Pay particular attention to existing route tests — they should still pass because in tests, `photo_root` matches the catalog's stored root (resolver is a no-op).

- [ ] **Step 8: Commit**

```bash
git add src/pixsage/web/routes.py tests/test_serve_path_translation.py
git commit -m "feat(web): resolve catalog paths via PathResolver

Thumbnail + full-image routes now translate current_path through
the resolver. Display-only .name reads left untouched."
```

---

### Task 6: Forward-compatibility — legacy catalogs (no `photo_root_at_embed`)

**Files:**
- Test: extend `tests/test_serve_path_translation.py`

**Background:** A catalog built before this plan was implemented won't have `photo_root_at_embed` set. `PathResolver(stored_root=None, ...)` already passes paths through unchanged (Task 3, test_resolver_handles_no_stored_root). Add an end-to-end test confirming serving a legacy catalog still works.

- [ ] **Step 1: Write the test**

Append to `tests/test_serve_path_translation.py`:

```python
def test_serves_legacy_catalog_without_photo_root_meta(tmp_path: Path) -> None:
    """A catalog created before Plan 1 has no meta.photo_root_at_embed.
    Resolver receives stored_root=None and passes paths through verbatim,
    so as long as the file lives at its current_path on this machine,
    serving works."""
    from pixsage.catalog import Catalog
    from pixsage.web.app import build_app

    photo_root = tmp_path / "Sony alpha 7c"
    photo_root.mkdir()
    real = photo_root / "DSC_0001.JPG"
    _make_jpeg(real, "blue")

    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    # Deliberately do NOT call set_photo_root_if_unset — simulate legacy.
    cat._conn.execute(
        "INSERT INTO photos (sha256, current_path, filename, filesize, mtime, added_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        ("legacy1", str(real), real.name, real.stat().st_size, real.stat().st_mtime),
    )
    cat._conn.commit()
    cat.close()

    app = build_app(photo_root=photo_root, embedder_name="mock")
    client = TestClient(app)
    r = client.get("/grid/thumb/legacy1?size=small")
    assert r.status_code == 200, r.text
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_serve_path_translation.py::test_serves_legacy_catalog_without_photo_root_meta -v`
Expected: PASS (no implementation change needed — resolver already handles `stored_root=None`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_serve_path_translation.py
git commit -m "test(serve): legacy catalogs (no photo_root meta) still serve

Verifies resolver's stored_root=None code path is exercised end-to-end
through build_app + thumbnail route."
```

---

### Task 7: Verification + journal-worthy summary

- [ ] **Step 1: Full test suite**

Run: `pytest -v`
Expected: existing 177 passing + new tests (8 in test_meta_table, 5 in test_path_translation, ≥3 in test_serve_path_translation) ≈ 193 passed, 1 skipped, 1 xfailed.

- [ ] **Step 2: Smoke-test the actual photographer's catalog**

Run a manual integration check against the live photographer catalog:

```bash
pixsage serve "E:\Sony alpha 7c"  # replace with whatever drive letter is currently mounted
```

(If the E: drive is unplugged, skip this step and note in the commit message that manual-smoke deferred.)

Expected: webapp loads, search works, thumbnails appear (drive letter agreement means resolver is a no-op).

- [ ] **Step 3: Update the project journal entry**

Run `/journal` at the end of the implementation session. Highlight:
- Path translation foundation laid; pixsage serve is now portable across machines
- Plan 2 (runtime build) and Plan 3 (native launcher) queued
- Migration story for old catalogs: handled by `set_photo_root_if_unset` no-op pattern (legacy catalogs have `stored_root=None` and resolver passes through; once any verb runs against the catalog with explicit `photo_root`, the anchor is set)

---

## Self-review

**Spec coverage:**
- §"Path-translation layer in pixsage serve" ✅ (Tasks 3-5)
- §"meta table storing the photo_root used at embed time" ✅ (Tasks 1-2)
- §"Apply at every catalog read that surfaces a path to the browser" ✅ (Task 5 + grep for missed sites)
- §"If the substitution doesn't produce a file that exists, fall through to trying the catalog's stored path verbatim" ✅ (Task 3, Step 3 implementation + test)
- §"Surface an in-app warning if neither resolves" — partially: the route returns 404, but no banner UI. That's fine for Plan 1; UI banner is a follow-up if photographer flags it.
- §runtime build, native launcher, folder staging — explicitly out of scope for Plan 1 (queued as Plan 2 and 3).

**Placeholder scan:** none.

**Type consistency:** `PathResolver(stored_root: str | None, runtime_root: Path)` is consistent across Tasks 3, 4, 5.

**Method-name consistency:** `set_photo_root_if_unset`, `set_meta`, `get_meta`, `path_resolver` (snake_case attribute), `resolve` — consistent throughout.
