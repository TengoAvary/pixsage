# Phase 1 — Tier 1 MVP Design

Status: design, 2026-05-09
Source: `photo-corpus-pipeline-spec.md` (Phase 1 of the master spec)

## Goal

Auto-tag a photographer's full corpus with keywords and captions that appear natively in Lightroom, without disrupting their existing catalog or workflow. Provide a CLI that the photographer can point at their photo root and walk away from. Re-runnable, resumable, and respectful of manual edits.

This phase delivers a useful product on its own. It also lays a forward-compatible foundation (catalog schema, file identification by sha256, vocabulary config) that Phases 2–5 extend.

## Scope

In scope:
- CLI (`pixsage tag`) walking a photo root, identifying images by sha256.
- Two taggers (Florence-2 and RAM++) running on each photo.
- Static vocabulary config (TOML) for confidence thresholds, exclusions, hierarchy overrides.
- XMP writer (sidecar for raws, embedded for JPEG/TIFF/HEIC/DNG) using exiftool.
- Minimal SQLite catalog tracking sha256, tags, model versions, and user-rejected tags.
- Idempotent re-runs with `--force`, sample mode for vocabulary tuning.

Explicitly out of scope (deferred to later phases):
- Embeddings, captions beyond a single sentence, geo estimation, similarity graph, clustering (Phase 2).
- Web app and visualizations (Phase 4).
- Vocabulary review UI / batch tag editing UI (Phase 5).
- Disk cache for decoded image data (Phase 1.5 if profiling shows need).
- pHash / EXIF triple identification (Phase 2).

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  pixsage tag <photo_root> [--force] [--sample N] [--catalog X]    │
│                          [--config <vocabulary.toml>]             │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Walker  │───▶│  Image   │───▶│  Taggers │───▶│   XMP    │
│ + sha256 │    │  Loader  │    │ Fl2/RAM++│    │  Writer  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
      │              ▲                │                │
      │              │                ▼                │
      │              │         ┌──────────┐            │
      │              │         │  Vocab   │            │
      │              │         │  Filter  │◀── vocabulary.toml
      │              │         └──────────┘            │
      ▼              │                                 ▼
┌─────────────────────┴───────────────────────────────────┐
│  SQLite catalog: <photo_root>/.photoindex/catalog.db    │
│  photos, tags (with user_rejected), runs                │
└─────────────────────────────────────────────────────────┘
```

Six logical components, each independently testable. The catalog is the only stateful element; everything else is a pure transform.

## Components

### Walker (`walker.py`)

- Walks `<photo_root>` for image files by extension (raws + JPEG/TIFF/HEIC/DNG).
- Computes sha256 of file bytes (streaming, 1 MiB chunks).
- Upserts into `photos(sha256, current_path, filename, filesize, mtime, last_seen_at)`.
- Decides per-photo whether to skip — skip if `last_tagged_at` is set AND `model_versions` matches the current run AND `--force` is not set.
- For `--sample N`: deterministically samples N photos by sorting sha256s and taking first N.

### Image loader (`images.py`)

- Two paths:
  - **Raw files** (`.arw`, `.cr3`, `.nef`, `.dng`*, `.raf`, `.orf`, `.rw2`): use `rawpy.imread().extract_thumb()` to get the embedded preview JPEG. Decode that with Pillow.
  - **Non-raw files** (`.jpg`, `.jpeg`, `.tif`, `.tiff`, `.heic`, `.png`): Pillow with `pillow-heif` registered for HEIC.
- Resize to 1024px on the long edge using Pillow's `LANCZOS`. Convert to RGB.
- Returns a PIL Image. No on-disk caching.
- *DNG note: most DNGs work via rawpy. If we hit DNGs that don't (e.g., linear DNGs without thumbnails), fall back to Pillow.

### Taggers (`taggers/`)

Abstract base in `taggers/base.py`:

```python
@dataclass
class Tag:
    name: str
    confidence: float
    hierarchy: str | None  # e.g., "Wildlife|Bird|Penguin"
    source: str            # "florence2" | "ram++"

class Tagger(Protocol):
    name: str
    model_version: str
    def load(self, device: str) -> None: ...
    def tag(self, image: PIL.Image.Image) -> tuple[list[Tag], str | None]: ...
    # Returns (tags, optional_caption)
```

**`Florence2Tagger`** (`taggers/florence2.py`):
- Loads `microsoft/Florence-2-large` via `transformers.AutoModelForCausalLM`.
- Lazy-loads on first `.tag()` call.
- Runs two prompts: `<MORE_DETAILED_CAPTION>` (returns sentence → used as `dc:description`), `<DENSE_REGION_CAPTION>` (returns objects/regions → flat tags). The dense-region caption is the *sole* source of Florence-2 tags; the caption sentence is used only for `dc:description` and is never mined for tags.
- Florence-2 doesn't natively output confidences per tag; we synthesize a confidence of `1.0` for every region-captioned tag. The vocabulary filter still applies a configurable threshold so adjustments later (e.g., adding `<OD>` outputs at lower synthesized confidence) compose naturally.
- Produces `Tag` objects with `source="florence2"`. `hierarchy` set when the tag matches a `hierarchy_overrides` entry; otherwise `None`.

**`RamPlusPlusTagger`** (`taggers/ramplusplus.py`):
- Loads RAM++ (`xinyu1205/recognize-anything`) — install path via git URL until/unless a PyPI release exists. (Treat as an open question; see "Open items" below.)
- Returns flat tags with confidence scores.
- Produces `Tag` objects with `source="ram++"` and `hierarchy=None` unless a hierarchy override matches.

Both taggers always run on the GPU when available — `device.py` selects CUDA → MPS → CPU. CPU is a compatibility fallback, not a perf target.

### Vocabulary filter (`vocabulary.py`)

Reads `vocabulary.toml`. Schema:

```toml
[florence2]
enabled = true
confidence_threshold = 0.6
exclude = ["photograph", "image", "picture"]

[ram_plus_plus]
enabled = true
confidence_threshold = 0.5
exclude = []

[hierarchy_overrides]
# flat tag -> hierarchical path
"penguin"    = "Wildlife|Bird|Penguin"
"glacier"    = "Scene|Mountain|Glacier"
"golden hour" = "Time|Golden Hour"

[caption]
enabled = true        # write Florence-2's <MORE_DETAILED_CAPTION> to dc:description
overwrite = false     # do not overwrite an existing dc:description
```

Filter steps:
1. If a tagger is `enabled = false`, drop all its tags.
2. Drop tags below the per-tagger `confidence_threshold`.
3. Drop tags in the per-tagger `exclude` list (case-insensitive, exact match on `name`).
4. For each remaining tag, if `name.lower()` matches a `hierarchy_overrides` key, set `tag.hierarchy` to the override.

The full schema is validated via `pydantic` at config-load time so misspellings fail fast.

### XMP writer (`xmp.py`)

Wraps `exiftool` via `subprocess`. Per photo:

1. Read existing XMP fields with one exiftool call: `dc:subject`, `lr:hierarchicalSubject`, `dc:description`. (For raws without a sidecar, treat as empty.)
2. Look up previously-auto-applied tags for this `sha256` in `tags` (where `user_rejected = 0`).
3. **User-rejection detection:** any tag in our DB record but missing from current XMP `dc:subject` is flagged: `UPDATE tags SET user_rejected = 1`. We do not re-add user-rejected tags on this run.
4. Compute the union to write:
   - `dc:subject` ← (existing dc:subject) ∪ (new auto-tags this run, name field) ∪ (per-source marker tags `auto-tagged-florence2`, `auto-tagged-ram` if either tagger contributed any tags this run).
   - `lr:hierarchicalSubject` ← (existing lr:hierarchicalSubject) ∪ (new auto-tags' hierarchy, where set).
   - `dc:description` ← caption from Florence-2, only if config.caption.enabled and (existing description is empty OR config.caption.overwrite is true).
5. Write with one exiftool call:
   - Raws → `<path>.xmp` sidecar (using `-o <path>.xmp`).
   - Non-raws (JPEG/TIFF/HEIC/DNG) → embedded XMP in the file.

We never delete tags from `dc:subject`. The merge is strictly additive (modulo user-rejected suppression).

### Catalog (`catalog.py`)

SQLite via `sqlite3` stdlib. Schema (Phase 1 subset of the full spec schema):

```sql
CREATE TABLE photos (
  sha256 TEXT PRIMARY KEY,
  current_path TEXT,
  filename TEXT,
  filesize INTEGER,
  mtime REAL,
  last_tagged_at TEXT,
  model_versions TEXT,        -- JSON, e.g. {"florence2": "v1.0", "ram++": "v1.2"}
  added_at TEXT,
  last_seen_at TEXT,
  error_reason TEXT
);

CREATE TABLE tags (
  sha256 TEXT NOT NULL,
  tag TEXT NOT NULL,
  source TEXT NOT NULL,        -- 'florence2' | 'ram++'
  confidence REAL,
  hierarchy TEXT,              -- nullable
  user_rejected INTEGER NOT NULL DEFAULT 0,
  applied_at TEXT,
  PRIMARY KEY (sha256, tag, source),
  FOREIGN KEY (sha256) REFERENCES photos(sha256) ON DELETE CASCADE
);

CREATE TABLE runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT,
  finished_at TEXT,
  photos_processed INTEGER,
  photos_skipped INTEGER,
  photos_errored INTEGER,
  config_hash TEXT,
  model_versions TEXT
);

CREATE INDEX idx_tags_sha256 ON tags(sha256);
CREATE INDEX idx_tags_source ON tags(source);
```

Phase 2 will add `phash`, `gps_*`, `caption`, `cluster_id`, `umap_*`, etc. without disturbing Phase 1's writes.

### CLI (`cli.py`)

`typer` app, single command:

```
pixsage tag <photo_root>
    [--force]                  # re-tag everything, ignore last_tagged_at
    [--sample N]               # tag a deterministic sample of N photos
    [--catalog PATH]           # default: <photo_root>/.photoindex/catalog.db
    [--config PATH]            # default: <photo_root>/.photoindex/vocabulary.toml
                               #          (created with defaults on first run)
    [--limit N]                # cap number of photos processed this run
    [--dry-run]                # walk + tag, but skip XMP write and catalog write
```

On first run, if no `vocabulary.toml` exists at the resolved config path, write a default one with sensible thresholds and an empty exclusion list. Photographer edits in place.

The CLI:
1. Resolves catalog and config paths (creates `.photoindex/` if missing).
2. Loads + validates config.
3. Computes `config_hash` (sha256 of canonicalized config) — recorded in `runs.config_hash`.
4. Initializes catalog, opens taggers (lazy load on first tag call).
5. Walks `<photo_root>`, deciding skip/process per photo.
6. For each processed photo: load image → tag → filter → merge with catalog state → write XMP → update catalog. Commits per-photo so an interrupted run resumes cleanly.
7. tqdm progress bar with rate, ETA, and rolling counts of {tagged, skipped, errored}.

## Data flow per photo

```
file path
  │
  ▼
sha256(file bytes) ─────────────────────────────▶ photos.sha256
  │
  ▼
catalog lookup
  ├─ already tagged + model_versions match + no --force ─▶ skip, update last_seen_at
  ▼
image loader
  │   raw   ──▶ rawpy.extract_thumb() ──▶ PIL.Image
  │   else  ──▶ Pillow.open()           ──▶ PIL.Image
  │   resize long-edge to 1024px (LANCZOS), convert RGB
  ▼
Florence2Tagger.tag(img)  →  RamPlusPlusTagger.tag(img)
  │   sequential GPU calls; both taggers loaded once and reused across photos
  ▼
vocabulary.filter(tags) — per-source threshold + exclusions + hierarchy overrides
  ▼
catalog.read_tags(sha256) ──▶ identify user_rejected tags from prior runs
  ▼
xmp.write_merged(path, new_tags, caption, marker_tags)
  │   raw   ──▶ exiftool writes <path>.xmp sidecar
  │   else  ──▶ exiftool writes embedded XMP
  ▼
catalog: upsert tags rows, set last_tagged_at + model_versions
```

Each tagger lazy-loads its model on its first `.tag()` call and reuses it for the rest of the run. Per-photo loop is `for photo in walker: process(photo)` — no inter-photo batching in Phase 1. (A 4090 chews through Florence-2 + RAM++ at usable rates per-image; batching is a Phase 1.5 optimization once we have profiling data.)

## Error handling

| Failure mode | Behavior |
|---|---|
| File can't be decoded (corrupt, unknown format) | Log warning; set `photos.error_reason`; skip; continue run |
| GPU OOM during inference | Catch; retry image at 768px; then 512px; on third failure log + skip |
| exiftool subprocess error | Log error; set `photos.error_reason`; do not update catalog tags for that photo (no half-state) |
| SQLite locked | Retry with exponential backoff (single-writer expected; defensive) |
| User Ctrl-C | Per-photo commits mean the next run resumes cleanly with no lost state |
| Model download fails on first run | Fail fast with explicit message: HF cache path, network requirement, offline-mode notes |
| No GPU available | Log warning, fall back to CPU; warn that runs will be very slow but proceed |

`error_reason` lets the photographer (and Phase 2 code) target retries at errored photos.

## Testing

- **Unit:**
  - `vocabulary.filter` — config + raw tags → expected filtered tags (per-source thresholds, exclusions, hierarchy overrides).
  - `catalog` — upsert idempotency, `user_rejected` flagging, `last_tagged_at` semantics.
  - `xmp.merge` — pure logic test with a fake exiftool wrapper: existing tags + new auto-tags + previously-rejected → expected merged set.
  - `walker.sha256` — known input → known hash.
- **Component:**
  - `images.load` — fixture dir with one of each format (.jpg, .heic, plus a real raw file from a common camera if obtainable). Asserts RGB Image, ≤1024px long edge.
- **Integration (no GPU required):**
  - End-to-end on 5 fixture JPEGs with mock taggers returning fixed tag sets. Asserts embedded XMP has expected tags, catalog has expected rows. Run twice → second run skips all. Run with `--force` → tags re-merged correctly. Manually edit one fixture's XMP to remove a tag → re-run with `--force` → that tag is `user_rejected = 1` and not re-added.
  - `--sample N` reproducibility: same seed → same photos.
- **GPU smoke test (manual, not CI):**
  - Real Florence-2 + RAM++ load on the 4090, run on 1 image, sanity-check tag output.
- **Test fixtures:** 5–10 small CC-licensed JPEGs in `tests/fixtures/images/`. One HEIC. One real raw file (e.g., a small Sony or Canon raw) if obtainable; the raw test is skipped via `pytest.importorskip("rawpy")` and a fixture-presence check otherwise.

## Project layout

```
pixsage/
  pyproject.toml
  README.md
  src/pixsage/
    __init__.py
    cli.py
    config.py            # vocabulary.toml loader + pydantic models
    walker.py            # file walking + sha256
    images.py            # raw + non-raw decode, resize
    device.py            # CUDA → MPS → CPU detection
    catalog.py           # SQLite schema + ops
    xmp.py               # exiftool wrapper, XMP merge
    vocabulary.py        # tag filter
    taggers/
      __init__.py
      base.py
      florence2.py
      ramplusplus.py
  tests/
    fixtures/
      images/
      vocabulary.toml
    test_walker.py
    test_images.py
    test_vocabulary.py
    test_catalog.py
    test_xmp.py
    test_e2e.py
  docs/superpowers/specs/
    2026-05-09-tier1-mvp-design.md  # this file
```

## Dependencies

- `python >= 3.11` (stdlib `tomllib`)
- `torch`, `transformers` — Florence-2
- RAM++ — `recognize-anything` (install via git URL until/unless PyPI release; verify package import name in implementation)
- `pillow`, `pillow-heif`
- `rawpy`
- `pydantic` (config validation)
- `typer` (CLI)
- `tqdm` (progress)
- exiftool binary — runtime dependency, not Python package. README documents `winget install ExifTool` / `brew install exiftool`.

No SQLAlchemy in Phase 1 — schema is small and stable; stdlib `sqlite3` keeps dependencies lean.

## Success criteria

A photographer with a corpus of N photos can:

1. `pip install pixsage` (or run from source).
2. Install exiftool per OS instructions.
3. `pixsage tag /path/to/photos` — walks away.
4. On return, every photo has an XMP (sidecar for raws, embedded for everything else) populated with auto-tags from Florence-2 + RAM++, plus a caption.
5. Open Lightroom → Library → select photos → Metadata → Read Metadata from File. Keywords appear in the keyword panel, hierarchical tags render as nested.
6. Build a Lightroom smart collection on `auto-tagged-florence2` to inspect just the auto-tagged subset.
7. Edit `<photo_root>/.photoindex/vocabulary.toml` to ban tags that came out noisy.
8. `pixsage tag /path/to/photos --force --sample 50` to re-tag a sample with the new vocabulary; verify noisy tags are gone.
9. Remove an auto-tag from one photo manually in Lightroom (writes back to XMP via "Save Metadata to File" or auto-write). Re-run `--force`; verify the removed tag stays removed.

## Open items (resolved before/during implementation)

- **RAM++ install path.** Confirm exact pip-installable name and import path; if it must come from git, pin a commit hash in `pyproject.toml`.
- **Florence-2 effective threshold.** Florence-2 region tags all get synthesized confidence `1.0`, so the `florence2.confidence_threshold` is essentially an on/off switch in Phase 1. If we later add a second Florence-2 source (e.g., `<OD>` outputs at lower synthesized confidence), the threshold becomes meaningful — design accommodates this without schema changes.
- **HEIC on Windows.** `pillow-heif` should work on Windows but verify on the development machine before relying on it.
- **DNG variants.** A small set of DNG variants don't expose embedded thumbnails. If we hit one in testing, fall back to Pillow direct decode and add a test fixture.

## Forward compatibility with Phase 2

Phase 2 will add:
- pHash + EXIF triple identification → new columns on `photos`.
- Embeddings (SigLIP2 / GeoCLIP), captions (separate from the Florence-2 short caption), geo estimates → new tables / parquet sidecars.
- UMAP/HDBSCAN clustering → `cluster_id`, `umap_x`, `umap_y` on `photos`.

Phase 1's schema is a strict subset of the full spec schema. Migrations in Phase 2 are additive only.
