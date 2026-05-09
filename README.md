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

The checkpoint is ~2.9 GB. If you skip this step, RAM++ will fail to load and the pipeline will fall back to Florence-2 only.

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

Auto-applied keywords appear in the Keyword List. Hierarchical keywords show as nested. Each photo also gets a marker tag (`auto-tagged-florence2`, `auto-tagged-ram`) — build a smart collection on these to inspect just the auto-tagged subset.

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
3. Removes those — and our `auto-tagged-*` source markers — from `dc:subject`. Your manually-added keywords stay.
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

You should see keywords matching the contents of the photo, plus `auto-tagged-florence2` (and `auto-tagged-ram` if RAM++ loaded successfully) in `Subject`, and a generated caption in `Description`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use synthetic JPEGs and `MockTagger` — no model weights needed. Tests that touch exiftool skip cleanly if it isn't on PATH.

## What's not in Phase 1

- Embeddings, similarity search, captions beyond a single sentence, geo estimation, clustering — Phase 2.
- Web app — Phase 4.
- pHash / EXIF triple identification — Phase 2.
- Vocabulary review UI — Phase 5.
