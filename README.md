# pixsage

Auto-tag a photographer's photo corpus with keywords and captions that appear natively in Lightroom.

This is the Tier 1 MVP from `docs/superpowers/specs/2026-05-09-tier1-mvp-design.md`. It runs Florence-2 + RAM++ over each image, writes XMP keywords (sidecar for raws, embedded for JPEG/HEIC/TIFF/DNG), and tracks state in a SQLite catalog so re-runs are incremental and respect manual edits.

## Install

```bash
pip install -e ".[taggers]"
```

You also need **exiftool** on PATH:
- Windows: `winget install OliverBetz.ExifTool` or download from https://exiftool.org/
- macOS: `brew install exiftool`
- Linux: `apt install libimage-exiftool-perl`

The first `pixsage tag` run will download Florence-2 weights (~3 GB) from Hugging Face automatically.

RAM++ needs a manually downloaded checkpoint:

```bash
# Linux/macOS
curl -L -o ~/.cache/pixsage/ram_plus_swin_large_14m.pth \
  https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
export PIXSAGE_RAM_CKPT=~/.cache/pixsage/ram_plus_swin_large_14m.pth
```

```powershell
# Windows
$dir = "$env:USERPROFILE\.cache\pixsage"; New-Item -ItemType Directory -Path $dir -Force | Out-Null
Invoke-WebRequest `
  -Uri "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth" `
  -OutFile "$dir\ram_plus_swin_large_14m.pth"
$env:PIXSAGE_RAM_CKPT = "$dir\ram_plus_swin_large_14m.pth"
```

The checkpoint is ~2.9 GB.

> **RAM++ is required, not optional.** With the default vocabulary
> (`florence2.tags_enabled = false`) RAM++ is the *only* keyword source. If
> the checkpoint is missing or `PIXSAGE_RAM_CKPT` is unset, `pixsage tag`
> **hard-crashes at model load** with `checkpoint url or path is invalid` —
> there is no Florence-2-only fallback. Captions alone (no keywords) is not a
> useful "tag" result.
>
> **`PIXSAGE_RAM_CKPT` is session-scoped.** `export` / `$env:` only sets it
> for the current shell. Background jobs, new terminals, scheduled runs, and
> orchestration scripts will *not* inherit it and will crash as above. Either
> persist it once —
> `setx PIXSAGE_RAM_CKPT "%USERPROFILE%\.cache\pixsage\ram_plus_swin_large_14m.pth"`
> (Windows, new shells only) — **or** set it at the top of your run script
> (see [Live monitoring](#live-monitoring) for the canonical pattern). The
> default search path if the env var is unset is just
> `ram_plus_swin_large_14m.pth` in the *current working directory*, which is
> almost never where the checkpoint actually is.

**Note for Windows users:** Florence-2's HF modeling file imports `flash_attn`, which has no Windows wheels. The pixsage wrapper registers a stub before loading and uses the eager attention implementation, so this works out of the box — you do not need to install flash_attn yourself.

## Quick start

```bash
pixsage tag /path/to/photos
```

Outputs:
- XMP sidecars next to raws (e.g., `DSC_0001.ARW` → `DSC_0001.xmp`, Lightroom convention).
- Embedded XMP in JPEG/HEIC/TIFF/DNG.
- A catalog at `/path/to/photos/.photoindex/catalog.db`.
- A vocabulary config at `/path/to/photos/.photoindex/vocabulary.toml` (created on first run with sensible defaults).

## In Lightroom

Enable **Catalog Settings → Metadata → Automatically write changes into XMP** once. Or, on demand: select photos → Metadata → **Read Metadata from File**.

Auto-applied keywords appear in the Keyword List. Hierarchical keywords show as nested. Tag-source attribution is recorded in the catalog DB (`<photo_root>/.photoindex/catalog.db`) — query the `tags` table on its `source` column to inspect Florence-2 vs RAM++ contributions.

## Tuning the vocabulary

Edit `<photo_root>/.photoindex/vocabulary.toml`. The default looks like this:

```toml
[florence2]
enabled = true
tags_enabled = false   # caption-only by default — see explanation below
confidence_threshold = 0.5
exclude = ["photograph", "image", "picture"]

[ram_plus_plus]
enabled = true
tags_enabled = true
confidence_threshold = 0.4
exclude = []

[hierarchy_overrides]
"penguin" = "Wildlife|Bird|Penguin"
```

**Why Florence-2 is caption-only by default.** Florence-2's caption (the long descriptive sentence in `dc:description`) is great. But its region/object outputs as *tags* tend to be multi-word phrases like `traditional Dutch houses along canal in Bruges, Belgium` that don't compose with Lightroom's exact-match keyword filtering. RAM++ is the cleaner tag source — its 4585-tag vocabulary is curated for the keyword use case. Set `florence2.tags_enabled = true` if you want the region phrases too.

Re-run with `--force` (and optionally `--sample 50` first) to re-tag with the new vocabulary. **Note:** `--force` *merges* new tags with the existing XMP — it never deletes prior auto-tags. If you've improved the model or want a clean slate, use `--rewrite` (described below).

## When you've improved the code or vocabulary and want a do-over

`--force` re-runs the taggers but merges with whatever XMP is already there. After several iterations you'll accumulate stale tags from old runs.

`--rewrite` is the do-over flag. It:

1. Reads each photo's current XMP.
2. Looks up which tags pixsage previously applied (from the catalog).
3. Removes those — plus any legacy `auto-tagged-*` source markers from older pixsage versions — from `dc:subject`. Your manually-added keywords stay.
4. Wipes the matching catalog rows so `user_rejected` flags reset.
5. Runs the (now-improved) taggers and writes fresh tags.
6. Always overwrites `dc:description` if `caption.enabled`.

```bash
pixsage tag /path/to/photos --rewrite           # full corpus
pixsage tag /path/to/photos --rewrite --sample 50  # tune on a sample first
```

`--rewrite` implies `--force`. It does NOT touch raws (sidecar XMPs are written from scratch each run already).

## Common flags

| Flag | Default | Purpose |
|---|---|---|
| `--force` | off | Re-tag photos even if already tagged at current model versions; merges with existing XMP |
| `--rewrite` | off | Wipe previously-applied auto-tags before re-tagging. Implies `--force` |
| `--sample N` | 0 (no sampling) | Tag a deterministic sample of N photos. Good for vocabulary tuning |
| `--catalog PATH` | `<photo_root>/.photoindex/catalog.db` | Override catalog location |
| `--config PATH` | `<photo_root>/.photoindex/vocabulary.toml` | Override config location |
| `--limit N` | 0 (no limit) | Stop after N photos processed this run |
| `--dry-run` | off | Run pipeline but skip XMP writes and catalog tag updates |
| `--skip-extensions` | (none) | Comma-separated extensions to exclude, e.g. `.jpg,.jpeg`. Useful when raws + JPGs coexist and you only want to tag one |
| `--only-extensions` | (none) | Only process these extensions, e.g. `.arw,.cr3`. Mutually exclusive with `--skip-extensions` |

## Demo corpus

When you don't have access to the photographer's photos, fetch a small public corpus to validate the end-to-end pipeline:

```bash
python scripts/fetch_demo_corpus.py
pixsage tag tests/demo_corpus
exiftool -XMP-dc:Subject -XMP-dc:Description tests/demo_corpus/*.jpg
```

This downloads ~22 photos from picsum.photos into `tests/demo_corpus/` (gitignored). Use it for "does the pipeline work on real images" testing. For tag-quality validation, point pixsage at the photographer's actual photos.

Add your own URLs to `tests/demo_corpus_urls.txt` to expand the corpus.

## Manual smoke test

After installation, verify the pipeline end-to-end:

```bash
mkdir /tmp/pixsage_smoke
cp <a-real-photo.jpg> /tmp/pixsage_smoke/
pixsage tag /tmp/pixsage_smoke
exiftool -XMP-dc:Subject -XMP-dc:Description /tmp/pixsage_smoke/*.jpg
```

You should see keywords matching the contents of the photo in `Subject`, and a generated caption in `Description`.

## Phase 3: Semantic search

After tagging, compute embeddings and run the local search webapp:

```bash
pip install -e ".[taggers,search]"  # one-time, includes sentence-transformers
pixsage embed /path/to/photos       # embed images + captions, ~9 photos/sec on a 4090
pixsage serve /path/to/photos       # local webapp, auto-opens browser
```

Open http://127.0.0.1:8765/. Type a query, drag the slider (Caption ⇄ Visual)
to bias the blend, click any photo for "more like this".

**Two channels, two encoders.**
- *Visual:* SigLIP2-so400m for image embedding and text→image cross-modal queries (~3 GB, downloads on first `embed` run).
- *Caption:* sentence-transformers/all-MiniLM-L6-v2 for caption indexing and text→text semantic queries (~80 MB, downloads on first run). SigLIP2's text encoder isn't suited for text→text retrieval; MiniLM is.

**Runtime.** Embed: ~9 photos/sec on an RTX 4090, so 50k photos ≈ 90 min.
The step is interruptible — re-run to resume. Add `--limit N` for a subset.

**Search latency.** GPU not required for `serve`. SigLIP2 query encoding
runs at ~130 ms/query on CPU; MiniLM at ~5 ms; matmul against vector matrices
is sub-millisecond. So `embed` benefits from a GPU; `serve` is fine on a
laptop with the pre-computed `.photoindex/` directory copied over.
```

## Phase 4 (in progress): geolocation + offline analysis

`pixsage geolocate` runs [GeoCLIP](https://github.com/VicenteVivan/geo-clip)
over each catalogued photo and stores the top-K (lat, lon, probability)
predictions in the catalog. Per-photo predictions on out-of-distribution
content (e.g. Antarctic wildlife) are noisy on their own — the intended use
is cluster-level aggregation: combine GeoCLIP predictions with the SigLIP2
similarity network and look for clusters where predictions cohere on a
location, then surface uncertain clusters to the user for HITL labelling.

```bash
pip install -e ".[taggers,search,geo]"   # adds the geoclip dep
pixsage geolocate /path/to/photos        # top-K=5 by default
```

Predictions land in the `geo_predictions` table (`sha256, model, rank, lat,
lon, score, created_at`) — they travel with `catalog.db` and don't need the
source photos to read back.

### Offline analysis workflow

The clustering / aggregation work runs on a separate machine. The
`.photoindex/` directory is the portable artifact: catalog DB, vector
parquets, vocabulary config — everything except the source photos.

```bash
# On the photographer's drive (the machine with photos + GPU):
pixsage tag       /e/Photos/Antarctica
pixsage embed     /e/Photos/Antarctica
pixsage geolocate /e/Photos/Antarctica
pixsage export    /e/Photos/Antarctica --out /e/exports/antarctica.zip

# On the analysis machine: unzip, point pixsage at the unpacked .photoindex/,
# or read catalog.db + vectors/*.parquet directly with sqlite3 / pyarrow.
```

`export` skips the regenerable `thumbs/` cache by default. Pass
`--include-thumbs` if you want the `serve` UI to render fast on the analysis
machine without re-decoding raws.

On the analysis machine, `pixsage.analysis.load_export()` is the canonical
read path: returns an `Export` dataclass with sha-keyed dicts of paths,
captions, tags, image/caption vectors, and geo predictions, plus an
`aligned_matrices()` helper that materializes the intersection of photos
that have any required combination of fields. `python scripts/load_export.py
<photoindex>` prints a summary of an unpacked export.

### Live monitoring

For long full-corpus runs, `scripts/dashboard.py` is a small FastAPI page that
polls the catalog DB + parquet vector files + system stats every 2 seconds:

```bash
pip install -e ".[search,dashboard]"
python scripts/dashboard.py /path/to/photo_root \
    --logdir /path/to/full-run-logs \
    --total-raw-paths 2123 \
    --dupe-rate 0.36 \
    --port 8766
```

Shows: active stage + tqdm tail + per-stage progress bars + throughput and
ETA + CPU / RAM / GPU (via `nvidia-smi`) / disk read MB/s. Open
`http://127.0.0.1:8766/`. If geolocation has run, a live map of GeoCLIP's
top-1 guesses fills in as predictions land.

The dashboard reads `<logdir>/<stage>.log` (one per stage: `tag.log`,
`embed.log`, `geolocate.log`). It does not run the pipeline — an
orchestration script does, redirecting each stage there. Canonical pattern
for a Windows full-corpus run (`scripts/`-adjacent, gitignored as
`.<corpus>-*`):

```powershell
# .myrun-pipeline.ps1
$env:PIXSAGE_RAM_CKPT = "C:\Users\you\.cache\pixsage\ram_plus_swin_large_14m.pth"
$root   = "H:\my-corpus"
$logdir = "C:\path\to\pixsage\.myrun-logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

# cmd /c redirect — NOT `python ... > log` in PowerShell. PowerShell writes
# UTF-16-LE with a BOM; the dashboard's tqdm regex expects ASCII-ish bytes
# and the progress bars will read as blank/garbled otherwise.
cmd /c "python -m pixsage tag `"$root`" > `"$logdir\tag.log`" 2>&1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }   # don't embed a failed tag
cmd /c "python -m pixsage embed `"$root`" > `"$logdir\embed.log`" 2>&1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmd /c "python -m pixsage geolocate `"$root`" --all > `"$logdir\geolocate.log`" 2>&1"
exit $LASTEXITCODE
```

Launch it detached, then start the dashboard pointed at the same `--logdir`:

```powershell
Start-Process powershell -ArgumentList '-NoProfile','-File','.\.myrun-pipeline.ps1' -WindowStyle Hidden
python scripts/dashboard.py H:/my-corpus --logdir C:/path/to/pixsage/.myrun-logs --total-raw-paths <N> --dupe-rate 0.30 --port 8766
```

Three traps, each of which has cost a debugging session:

1. **Set `PIXSAGE_RAM_CKPT` *inside* the script.** A background/detached
   process does not inherit your interactive shell's env — bare
   `python -m pixsage tag` from a launched job crashes at RAM++ load even
   though it "worked yesterday" interactively.
2. **`cmd /c "... > log 2>&1"`, never PowerShell `>`/`Tee-Object`.** The
   UTF-16 BOM breaks the dashboard's progress parsing (tail looks fine,
   every other stat freezes).
3. **`--total-raw-paths` is the recursive candidate count**, not unique
   photos; `--dupe-rate` (raw+jpeg pairs ≈ 0.3) is only used to estimate the
   unique-sha denominator. Both are display-only — wrong values just skew
   the ETA, they don't affect the run.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use synthetic JPEGs and `MockTagger` — no model weights needed. Tests that touch exiftool skip cleanly if it isn't on PATH.

## Experimental: HITL cluster labelling

The webapp has dormant routes (`/explore`, `/cluster/{id}`, `/cluster/{id}/label`)
that surface visual-similarity clusters and let you write a single (lat, lon,
place_name) label to every photo in a cluster at once. Originally built as a
fallback for corpora where GeoCLIP fails (e.g. Antarctic photos predicted in
Greenland), but in practice not part of the photographer's daily workflow.

Off by default. Enable for exploration via:

```python
from pixsage.web.app import build_app
app = build_app(photo_root, experimental_cluster_labelling=True)
```

The supporting infrastructure (`pixsage.clusters`, `pixsage.xmp.write_gps` /
`read_gps`, the `user_locations` catalog table) is production-quality and
useful regardless. If a better location-labelling UX emerges (interactive
map, smarter cluster suggestions) we'd reuse the scaffolding; otherwise the
whole experimental block is one commit to remove.

## Photographer-facing launcher (Phase 5)

One installable app per laptop. The app remembers every catalog it has ever seen, scans for newly-mounted drives on launch, and lets the user toggle which catalogs participate in search. No per-folder launchers; drives carry only `.photoindex/` data.

**One-time setup on his machine** (installs runtime + drops a single `Pixsage Search` launcher; takes ~10 minutes; downloads ~2 GB of model weights):

First, get the pixsage source onto the target machine:

```bash
git clone <pixsage-repo-url> ~/dev/pixsage
cd ~/dev/pixsage
```

The bootstrap python needs only the stdlib.

**Windows prerequisite — enable Developer Mode.** HuggingFace's model cache uses symlinks; on Windows these require either Developer Mode or admin rights. Without this the install crashes mid-download.

> Settings → Privacy & Security → For Developers → Developer Mode → On

Then:

```powershell
# Windows
python -m scripts.launcher.install_runtime --target windows-x64
```

```bash
# macOS (Apple Silicon — M1/M2/M3/M4)
python3 -m scripts.launcher.install_runtime --target macos-arm64
```

```bash
# macOS (Intel)
python3 -m scripts.launcher.install_runtime --target macos-x86_64
```

This puts:
- Runtime under `%LOCALAPPDATA%\pixsage` (Win) or `~/Library/Application Support/pixsage` (Mac).
- A `Pixsage Search` launcher on the user's Desktop (Win) or in `~/Applications/` (Mac).

**Daily use:**
1. Plug in any drive containing one or more `.photoindex/` folders.
2. Double-click `Pixsage Search` on the laptop. Browser opens to the search webapp.
3. The catalog panel above the search box lists every catalog the app has ever seen — available ones (drive plugged in) are green; offline ones (drive not plugged in) are greyed out.
4. Toggle catalogs on/off, rename them, add new ones, or remove ones you no longer want. Use **Rescan drives** to pick up a freshly-plugged-in drive while the app is running.

**To stop:** kill the python process via Task Manager / Activity Monitor, or close the Terminal window that the `.command` opened.

**Per-folder launchers (secondary).** `pixsage stage-launchers <folder>` still drops a per-folder `.bat`/`.command` if you want a folder-specific bookmark, but the laptop-level launcher is the canonical entry point.

Plan 1 (path translation across machines) makes this work even when the drive's letter or mount point differs between Jack's workstation and the photographer's laptop. Catalog records `photo_root_at_embed` and serve translates absolute paths at request time.

## What's still open

- pHash + EXIF triple identification — Phase 2 (deferred; may be skipped if the photographer's workflow doesn't expose the limitation).
- Vocabulary review UI — Phase 5.
- Native single-file launcher binary (replaces the .bat/.command pair) — Plan 3.5. Eliminates the brief terminal flash on Windows and adds a tray icon for clean Quit.
- SigLIP2 text-tower-only extraction — drops the runtime models footprint from ~1.8 GB to ~280 MB. Plan 2.5 follow-up.
