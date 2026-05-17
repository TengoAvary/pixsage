# Directory-picker catalog discovery — design

Date: 2026-05-17
Status: Approved (brainstorming) — ready for implementation plan

## Context

`pixsage serve` currently auto-discovers photo catalogs by walking every
mounted volume at startup via `discovery.list_mounted_roots()` +
`walk_for_photoindex()` (called from `web/app.py:build_app()` and the
`POST /catalogs/rescan` route). On macOS `list_mounted_roots()` returns
every `/Volumes/*` entry — including the boot volume `/Volumes/Macintosh
HD` — so every launch crawls the entire system volume. This:

- Caused a hard crash: `Path.is_dir()` propagates `PermissionError` on
  SIP-protected files (`/usr/sbin/*`), aborting the whole walk
  (separately hot-fixed in commit `3bfa464`; the underlying "why are we
  walking the system volume at all" problem remains).
- Burns the full 15s walk time-budget on irrelevant system paths on
  *every* startup, even though discovery's only purpose is to populate a
  registry that already persists across runs.

The registry (`catalogs.json`) already remembers every catalog ever
added, with enable/disable, rename, and remove already wired through the
web UI. Discovery is just one (buggy, expensive) way to populate it.

**Goal:** replace automatic volume-walking with an explicit,
user-driven flow: the user browses to and picks a directory in the web
UI; the server walks *only that picked subtree* once, registers every
`.photoindex/` catalog it finds, and persists them. No filesystem
crawling at startup. Indexing of raw photos remains a separate CLI
concern (`pixsage tag` / `pixsage embed`) — out of scope.

## Decisions (from brainstorming)

1. "Add a directory" = register **already-indexed** catalogs. A catalog
   is a directory containing `.photoindex/` (with `catalog.db`).
2. Picking a directory **walks its entire subtree** and bulk-registers
   every `.photoindex/` found (not just the picked dir itself).
3. Picker is an **in-page server-rendered folder browser** (no native
   OS dialog; browsers cannot return server-side paths).
4. **Automatic** volume-walk discovery is **removed entirely**. The
   walker itself (`walk_for_photoindex`) is **kept** and reused, scoped
   to the user-picked root.
5. Empty pick (no `.photoindex/` anywhere in subtree) → inform the user,
   add nothing. No index-on-add.

## Design

### 1. Remove automatic discovery; keep the walker

- `src/pixsage/discovery.py`: delete `list_mounted_roots()`. **Keep**
  `walk_for_photoindex()` and `SKIP_DIRS` (now only ever called with a
  single user-picked root). The `PermissionError`-safe `_is_dir`
  wrapper added in `3bfa464` stays.
- `src/pixsage/web/app.py` `build_app()`: remove the
  `if not skip_discovery: walk_for_photoindex(list_mounted_roots())`
  block (lines ~87–92) and the `skip_discovery` parameter. Startup
  performs only `registry.refresh_availability()` (Section 2) — zero
  filesystem walking beyond `exists()` on already-registered paths.

### 2. Registry: split reconciliation

`registry.refresh_from_discovery(discovered_paths)` currently does two
things: (a) for every existing entry, set `available =
exists(photoindex_path)` and bump `last_seen`; (b) add any
`discovered_paths` not already registered. Split:

- **`refresh_availability()`** — behavior (a) only. No args, no adds.
  Called by `build_app()` at startup and by the availability-refresh
  route. This is the no-walk "is the drive plugged in?" check.
- **Bulk-add for the picker flow** — the browse-add route walks the
  picked root with `walk_for_photoindex([picked_dir])` and calls
  `registry.add()` for each found `.photoindex/` not already registered
  (label = parent dir name, `enabled=True`), then `registry.save()`.
  `add()` already exists; dedupe via existing
  `find_by_photoindex_path()`.
- `refresh_from_discovery()` is removed once both call sites
  (`app.py`, `routes.py /catalogs/rescan`) are migrated.

### 3. Folder-browser endpoint + modal

- **`GET /catalogs/browse?path=<dir>`** → JSON:
  `{ path, parent, entries: [{name, path, has_photoindex}], roots: [...] }`
  - `entries`: immediate child directories of `path`, via the
    `PermissionError`-safe `_is_dir`; `has_photoindex` true if that
    child directly contains `.photoindex/`.
  - `roots`: quick-links — the user home dir, plus `/Volumes/*` entries
    obtained with a **single non-recursive `iterdir()`** (cheap; this is
    listing mount points, not the recursive walk that was the bug).
  - Default `path` when omitted = user home.
  - Reject non-existent / non-directory paths with a 400 + message.
- **`POST /catalogs/add-scan`** (form param `path`): run
  `walk_for_photoindex([path])`; bulk-register finds per Section 2;
  return a summary message: `"Added N catalog(s)"`, `"N added, M already
  registered"`, or `"No indexed catalogs found under <path> — nothing
  added"`. Then reload newly-enabled+available catalogs into
  `MultiSearchService` (reuse existing `_load_catalog_into_multi()`).
- `templates/_catalogs.html`: replace the paste-path text input with an
  **"Add catalog…"** button opening a modal that drives
  `/catalogs/browse` (breadcrumb from `parent`, clickable folder list,
  up/into navigation, `.photoindex/` hint badge) and confirms the
  current folder via `/catalogs/add-scan`. Show the returned summary.
- Replace **`POST /catalogs/rescan`** (volume discovery) with **`POST
  /catalogs/refresh`** → `registry.refresh_availability()` +
  reconcile `MultiSearchService` load/unload state (same reconcile
  logic the old rescan used, minus the walk). Rename the UI button
  "Rescan drives" → "Refresh availability".
- The existing `POST /catalogs/add` may be kept (single explicit
  pre-indexed path) or folded into `add-scan` (a single-catalog subtree
  is just N=1). Implementation plan to choose; folding is preferred for
  one code path.

### 4. CLI / back-compat

- `pixsage serve <path>` unchanged: an explicit `photo_root` still
  auto-registers that one directory at startup (single explicit path,
  no walk). `--registry`, `--host`, `--port`, `--no-open` unchanged.

### 5. Test migration (explicit — wide but mechanical)

- `skip_discovery=True` is passed in ~10 test files
  (`test_web_app.py`, `test_web_search.py`, `test_web_catalogs.py`,
  `test_web_app_multi.py`, `test_serve_path_translation.py`,
  `test_web_clusters.py`). Removing the param requires deleting that
  kwarg at every call site (mechanical; default behavior is now
  "no walk" anyway).
- `tests/test_discovery.py`: keep (walker still exists); remove any
  `list_mounted_roots` cases.
- `tests/test_registry.py`: replace `refresh_from_discovery` cases with
  `refresh_availability` + explicit `add()` cases.
- `tests/test_web_catalogs.py`: the rescan tests monkeypatch
  `list_mounted_roots` (lines ~209–234) — rewrite for `/catalogs/browse`
  + `/catalogs/add-scan` + `/catalogs/refresh`.

## Testing strategy

- **Unit — registry:** `refresh_availability` flips available/offline by
  path existence, bumps `last_seen`, never adds entries. Bulk-add
  dedupes already-registered paths.
- **Unit — browse endpoint:** lists child dirs; `parent` traversal;
  permission-denied child dir is skipped not fatal; `has_photoindex`
  flag correct; bad path → 400.
- **Unit — add-scan:** temp tree with 2 nested `.photoindex/` → both
  registered; empty subtree → none added, correct message; re-scan of
  same tree → "already registered", no dupes.
- **Integration:** `build_app()` startup performs zero recursive walk
  (assert via spy/monkeypatch that `walk_for_photoindex` is not called
  at startup); registered-but-offline catalog shows offline without a
  walk.
- **Web:** modal renders, navigates into/out of dirs, posts add-scan and
  shows summary; "Refresh availability" toggles an offline catalog back
  online when its path reappears.
- Full `pytest` green after the test migration in Section 5.

## Critical files

- `src/pixsage/discovery.py` — delete `list_mounted_roots`; keep walker.
- `src/pixsage/registry.py` — split `refresh_from_discovery` →
  `refresh_availability` + reuse `add`.
- `src/pixsage/web/app.py` — drop startup walk + `skip_discovery`.
- `src/pixsage/web/routes.py` — add `/catalogs/browse`,
  `/catalogs/add-scan`; replace `/catalogs/rescan` →
  `/catalogs/refresh`; reuse `_load_catalog_into_multi()`.
- `src/pixsage/web/templates/_catalogs.html` — folder-browser modal.
- `tests/` — migration per Section 5.

## Out of scope

- Indexing raw photos from the web UI (driving `tag`/`embed`) — separate
  project.
- Native OS folder dialogs.
- Any change to search, embedding, or `.photoindex/` internals.
