# Multi-catalog search — design

**Date:** 2026-05-12
**Status:** approved
**Scope:** Replace the per-folder-launcher single-catalog model with an app-on-laptop multi-catalog model. The app remembers catalogs across sessions, shows availability per drive, lets the user toggle which catalogs participate in search, and merges results across all enabled catalogs into one ranked list.

## Goal

The photographer plugs in a drive and opens **one** pixsage app on his laptop (installed once). The app shows every catalog it has ever seen — currently-available ones (drive plugged in) are green and toggleable; offline ones (drive not plugged in) are greyed out but still in the list. Searching runs across every enabled-and-available catalog at once. Adding a new catalog is paste-a-path in the browser; no Terminal required.

This replaces the current "double-click a launcher inside an indexed folder" model. The drive carries only `.photoindex/` data (no shell scripts staged into folders); the app lives on the laptop.

## Non-goals

- Cross-catalog visual similarity ("more like this" across catalogs). v1 keeps "more like this" within the source catalog. Cross-catalog visual is a follow-up if it turns out to matter.
- Cross-machine registry sync. Each user's laptop has its own registry; nothing flows between machines.
- Catalog merging / consolidation. Catalogs stay independent on disk.
- Auto-detect when a drive is plugged in *while the app is running*. The user clicks "Rescan drives" to pick up newly-mounted drives. (Filesystem watching is a v2 polish.)
- Native folder picker. v1 uses paste-a-path; browser folder pickers are flaky across platforms and add no value over a path input for this use case.
- Filesystem-watching, push notifications, or any background process beyond `serve`.

## User experience

### Day-one install

`install_runtime --target macos-x86_64` (or `windows-x64`, etc.) does what it does today — builds the runtime, downloads models — and adds one new step: drop a single laptop-level launcher at a discoverable location.

- Mac: `~/Applications/Pixsage Search.command`
- Win: `%USERPROFILE%\Desktop\Pixsage Search.bat` (Start Menu is also acceptable; user choice)

The launcher invokes `python -m pixsage serve` with **no path argument**.

### Daily use

1. Photographer plugs in his drive.
2. Double-clicks `Pixsage Search` on his laptop.
3. App opens in the browser at `http://127.0.0.1:8765/`. On startup it:
   - Reads `catalogs.json`. For each registered catalog, checks whether its `photoindex_path` exists. Marks each available / offline.
   - Walks mounted drives for `.photoindex/` directories not yet in the registry. Any new ones are auto-added (toggled on).
4. The catalog panel is collapsed by default if at least one catalog is enabled and available; auto-expanded if zero catalogs are usable.
5. He searches. Results are merged across every enabled-and-available catalog whose embedder signature matches the query encoder, ranked by score.

### First-time empty state

If `catalogs.json` doesn't exist and discovery turns up nothing (no drive plugged in), the app shows the catalog panel expanded with empty state copy: *"No catalogs yet. Plug in a drive that contains a `.photoindex/` folder and click Rescan drives, or paste a path to add one manually."*

### Adding a catalog manually

Catalog panel → **Add catalog…** button → inline form with a single text input ("Paste a path to a folder containing `.photoindex/`"). Server-side validation:
- Path exists.
- Path contains `.photoindex/catalog.db`.
- `catalog.db` opens cleanly and has expected schema.

On valid input: assigns a ULID, reads photo count + embedder signatures, adds to registry, toggled on. The panel refreshes.

On invalid input: inline error explaining what's missing (e.g. "No `.photoindex/catalog.db` at that path").

### Removing a catalog

Row → `⋯` → **Remove from registry**. Removes the entry from `catalogs.json`. Does **not** touch the catalog files on disk — the catalog can be re-added later (manually or via rescan) and will get a fresh registry entry.

### Renaming a catalog's label

Row → `⋯` → **Rename**. Label is a free-text string stored only in the registry (not in the catalog itself). Default label on auto-add: the parent directory's name (e.g. "Sony alpha 7c").

## Architecture

### Components

#### 1. Registry

JSON file at the runtime root (same dir as the installed Python + models):
- Mac: `~/Library/Application Support/pixsage/catalogs.json`
- Win: `%LOCALAPPDATA%\pixsage\catalogs.json`

Schema:

```json
{
  "version": 1,
  "catalogs": [
    {
      "id": "01HQR4XXXXXXXXXXXXXXXXXXXX",
      "photoindex_path": "/Volumes/Sony alpha 7c/.photoindex",
      "label": "Sony α7c",
      "enabled": true,
      "first_seen": "2026-05-12T14:00:00Z",
      "last_seen": "2026-05-12T14:32:00Z",
      "image_embedder_signature": "siglip2-so400m-patch14-384@v1",
      "caption_embedder_signature": "minilm-L6-v2@v2"
    }
  ]
}
```

- **id** — stable ULID. Persisted to the catalog's own `meta` table as `registry_id` on first-sight so we can re-find a catalog by id even if it moves on disk. If a catalog ever appears at a new path with a `registry_id` already in the registry, we update its `photoindex_path` in-place rather than duplicating the entry.
- **photoindex_path** — absolute path to the `.photoindex/` dir. (Photo root translation for serving paths is handled per-catalog by the existing PathResolver from the Plan 1 work; not the registry's job.)
- **embedder_signatures** — derived from each catalog's `meta` table. The catalog already records embedder names per vector via the existing `image_kind` / `text_kind` columns; we synthesize a stable signature string at registration time.

The registry is owned by `serve` (single writer). Concurrent serve processes are not a supported scenario in v1.

#### 2. Discovery

A `scripts.launcher.discovery` module (or `pixsage.registry.discovery` — TBD by implementer) that walks mounted-drive roots looking for `.photoindex/`.

Walk algorithm:
- Roots:
  - Mac: enumerate `/Volumes/*` plus `~/`
  - Win: enumerate live drive letters (`A:\..Z:\`) that respond to `Path.exists()`
  - Linux: `/media/*`, `/mnt/*`, `~/` (lower priority; not tested in v1)
- BFS from each root.
- Stop descending into any directory that itself contains `.photoindex/` (that subtree is "owned" by that catalog — no nested catalogs).
- Skip dotfiles, `node_modules`, `.git`, system dirs (`System Volume Information` on Windows, `.Trashes` on Mac, etc.).
- Bounded by **depth ≤ 6** and **5 seconds per root**; whichever hits first.

The walk returns a list of absolute `.photoindex/` paths. Each is fed through the registry: if `registry_id` is in the catalog's meta and known → mark available (or update `photoindex_path` if it moved); if new → auto-add with toggled-on.

#### 3. Multi-catalog SearchService

The existing `SearchService` stays as-is — one catalog, one set of matrices, one embedder. We add a new `MultiSearchService` (or `SearchOrchestrator` — implementer's call) that owns `{catalog_id → SearchService}`.

**Text query:**

```python
def search(query: str, image_weight: float, top_k: int) -> list[Hit]:
    # 1. Encode the query once with the orchestrator's primary encoder
    # 2. For each enabled-and-available catalog:
    #    - skip if signature doesn't match the primary encoder
    #    - else call its SearchService.search(...) for top_k
    # 3. Merge all returned hits, sort by score, take top_k.
    # 4. Each hit carries catalog_id for UI badging.
```

**Image-similarity query (`/similar/{sha256}`):**

The source photo lives in one catalog. v1: look up which catalog owns this sha (by querying each enabled-and-available catalog's `photos` table) and call its `SearchService.search_by_image(sha)` — single-catalog result set. The route becomes `/similar/{catalog_id}/{sha256}` (vs the current `/similar/{sha256}`) so the source catalog is unambiguous and the handler doesn't have to guess. The `/photo/{sha256}` route grows the same treatment: `/photo/{catalog_id}/{sha256}`. Both are internal routes (not user-bookmarked); no backward-compat shim needed.

**Encoder compatibility:**

Each catalog reports two signatures. The query encoder per channel is the *orchestrator's* configured encoder (today: SigLIP2 for the visual channel, MiniLM for the caption channel). Mismatched catalogs are silently skipped for the channel they don't match — they may still be searchable on the other channel. The UI surfaces a small hint next to each catalog row showing which channels are searchable.

**Score normalization:**

Cosine similarities are directly comparable across catalogs that share the same encoder per channel. Final mixed score is `(1 - image_weight) * caption_score + image_weight * image_score` — the existing per-catalog logic, but evaluated separately per catalog and merged.

#### 4. Catalog manager UI

Lives at `/` as a collapsible panel above the existing search form.

**Visible row state:**

| Field | Source | Notes |
|---|---|---|
| status dot | path-exists check + encoder-compat | green = available + compat; yellow = available + partial compat; grey = offline |
| label | registry | editable inline |
| photo count | catalog.photos COUNT | cached at registration; refreshed on availability check |
| path | registry | small/muted; clickable to copy |
| toggle | registry.enabled | persists on click |
| row actions (`⋯`) | — | rename, remove from registry |

**Panel footer:**

- `Add catalog…` button → inline path-input form.
- `Rescan drives` button → re-runs discovery + availability checks.
- (Both run as POSTs that return updated panel HTML; the panel uses no JS framework — same plain server-rendered approach we just adopted for search.)

**Result attribution in the grid:**

When more than one catalog is enabled, each result card grows a small badge showing the source catalog's label. Single-catalog case: no badge (uncluttered).

#### 5. Installed laptop launcher

A new step in `install_runtime`:

- After the runtime is built and models are downloaded, write a `Pixsage Search.command` (Mac) or `Pixsage Search.bat` (Win) to a laptop-discoverable location.
- The launcher invokes `python -m pixsage serve` with no path argument and the standard env (PYTHONNOUSERSITE=1, HF_HOME pointing at the runtime hub dir, etc.).
- Idempotent: re-running `install_runtime` overwrites the launcher with the current template.

The existing per-folder `stage-launchers` CLI is kept (for testing convenience) but stops being the recommended path. README's Phase 5 section gets rewritten around the laptop-level model.

### Data flow

```
User clicks Pixsage Search.command
  └─> python -m pixsage serve  (no path)
        └─> registry.load()                  → list of CatalogEntry (with availability state)
        └─> discovery.scan_mounted_drives()  → list of newly-discovered .photoindex paths
        └─> registry.merge(new)              → auto-add new ones, toggled on
        └─> MultiSearchService(registry)     → loads per-catalog SearchService for each enabled+available
        └─> FastAPI starts; opens browser

User clicks search
  └─> GET /?q=...
        └─> MultiSearchService.search(q, image_weight, top_k)
              └─> per-catalog SearchService.search(...)
              └─> merge by score, take top_k
        └─> render results grid with per-row catalog_id badge

User clicks "More like this"
  └─> GET /similar/<catalog_id>/<sha256>
        └─> resolve catalog_id → SearchService → search_by_image(sha)
        └─> render results (single-catalog, no badge needed)

User clicks toggle on catalog row
  └─> POST /catalogs/<id>/toggle
        └─> registry.update(id, enabled=not enabled)
        └─> MultiSearchService.reload()  (cheap — just adds/removes a per-catalog SearchService from the dict)
        └─> render updated panel

User clicks Rescan drives
  └─> POST /catalogs/rescan
        └─> registry.refresh_availability()
        └─> discovery.scan_mounted_drives()
        └─> registry.merge(new)
        └─> render updated panel
```

### Error handling

- **Registry file missing or corrupt** → treat as empty registry; back up the broken file to `catalogs.json.broken-<timestamp>` and proceed.
- **Registered path no longer exists** → mark offline; keep the entry. (Drive unmounted, folder renamed, etc.) Don't auto-remove.
- **Schema mismatch when opening a catalog** → mark unusable, render with red dot + "schema too old/new — re-embed required". Don't crash the panel.
- **Add-catalog path doesn't contain `.photoindex/catalog.db`** → inline form error.
- **Discovery walk hits a permission error** → log and skip that branch; don't fail the rescan.

## Testing strategy

### Unit

- `registry.py`: load / save / merge / update; corruption recovery; ULID stability.
- `discovery.py`: walk semantics with a synthetic filesystem fixture (nested `.photoindex/` dirs, depth limits, time limits).
- `multi_search.py`: query merging across mocked per-catalog SearchServices; encoder-compat filtering.

### Integration

- `test_web_catalogs.py`: end-to-end through FastAPI test client — start with a fixture registry, hit `/`, assert panel renders correctly; POST `/catalogs/add`, assert registry mutated and panel updated; POST `/catalogs/<id>/toggle`, same.
- `test_web_search.py` (existing): grows multi-catalog cases — search with two catalogs enabled returns mixed results; toggle one off, search again, results restricted.

### Manual / gated smoke

- The existing `PIXSAGE_LAUNCHER_SMOKE=1` test grows a multi-catalog scenario: build runtime, install laptop launcher, point at two synthetic catalogs, verify both appear in the panel and search returns from both.

## Open implementation questions

(Captured for the implementation plan to resolve, not blocking design approval.)

1. **Module placement:** registry / discovery / multi-search live in `pixsage.registry`, `pixsage.discovery`, `pixsage.search.multi`? Or under `scripts.launcher`? Lean toward `pixsage.*` since these are runtime concerns, not install concerns.
2. **Registry schema versioning:** v1 has `version: 1` field. Migration story when v2 ships is TBD; for now just bail on unknown versions with a clear error.
3. **Launcher install location on Windows:** Desktop, Start Menu, or both? Desktop is most discoverable for a non-technical user; Start Menu is more "appropriate" but requires more user discovery. Leaning Desktop for v1.
4. **`stage-launchers` deprecation path:** keep functional; add a deprecation note to CLI help. Remove in a future cleanup pass if no one uses it after the new model lands.

## Estimated effort

~1-2 days of focused work split as:

- Registry + discovery + tests: ~half day
- MultiSearchService refactor + tests: ~half day
- Catalog manager UI + routes + tests: ~half day
- Installer change + README rewrite + smoke test extension: ~half day

To be confirmed when the implementation plan lands.
