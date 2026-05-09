# Phase 3 — Embedded Semantic Search Design

Status: design, 2026-05-09
Source: `photo-corpus-pipeline-spec.md` (Phase 3 + a search-only slice of Phase 4)

## Goal

Make embeddings useful by giving the photographer a semantic search UI on day one. Type "leopard seal hauled out on ice" → relevant photos. Click any photo → "more like this." That's the milestone.

This phase delivers the search product end-to-end on top of Phase 1's catalog and captions. It deliberately omits the rest of Phase 4 (map, UMAP scatter, cluster browser) — those are separable views and earn their keep on their own.

## Scope

In scope:
- CLI (`pixsage embed`) walking the existing catalog, computing SigLIP2 vectors for each photo's image and caption.
- Multi-model storage layout: one parquet file per `vector_kind` keyed by `sha256`, ready to absorb OpenCLIP / GeoCLIP later.
- CLI (`pixsage serve`) running a local FastAPI app on `localhost:<PORT>`.
- Search webapp (Jinja templates + HTMX) with one text-search box, photo grid, photo detail, and "more like this" action.
- Combined ranking: weighted sum of image-channel and caption-channel cosine similarity with a UI slider.
- Lazily-generated, on-disk thumbnail cache.
- Idempotent re-runs of `pixsage embed`; resumable from interruption.

Explicitly out of scope (deferred):
- Map view (Phase 4 — needs EXIF GPS extraction, also tiny).
- UMAP scatter / cluster browser (Phase 4 — needs HDBSCAN, cluster naming).
- Faceted filters (date / tag / folder / has-GPS) layered onto search (Phase 4).
- Authentication, multi-user, remote access (out of scope for v1; localhost-only).
- Phase 2 (pHash + EXIF triple identification + GPS extraction). Still deferred per `project_phase_status.md`.
- React / build pipeline. Phase 3 stays pure-Python with HTMX in templates.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  pixsage embed <photo_root> [--catalog X] [--limit N] [--model M]     │
└───────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────────────────┐
│ Catalog  │───▶│  Image   │───▶│  SigLIP2   │───▶│  Parquet Writer  │
│  Reader  │    │  Loader  │    │  Encoder   │    │   (per kind)     │
└──────────┘    └──────────┘    └────────────┘    └──────────────────┘
       │                              ▲                     │
       │                              │                     ▼
       │                       ┌────────────┐    ┌──────────────────┐
       │                       │  Caption   │    │ .photoindex/     │
       └──────────────────────▶│  Reader    │    │ vectors/         │
                               │ (catalog)  │    │   siglip2_image  │
                               └────────────┘    │     .parquet     │
                                                 │   siglip2_caption│
                                                 │     .parquet     │
                                                 └──────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│  pixsage serve <photo_root> [--catalog X] [--port P]                  │
└───────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI app                                                          │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────────────────┐    │
│  │ Vector     │    │   Search     │    │   Routes (Jinja+HTMX)   │    │
│  │ Index      │───▶│   Service    │───▶│   GET /                 │    │
│  │ (numpy in  │    │ (cosine, RFF │    │   POST /search          │    │
│  │  RAM)      │    │  weighting)  │    │   GET /photo/{sha}      │    │
│  └────────────┘    └──────────────┘    │   GET /similar/{sha}    │    │
│                                        │   GET /thumb/{sha}      │    │
│                                        └─────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

## Components

### `pixsage.embedders.base`
Abstract base mirroring `taggers.base`. One interface so OpenCLIP / GeoCLIP slot in later.

```python
class Embedder(Protocol):
    name: str             # e.g. "siglip2-so400m-patch14-384"
    image_dim: int
    text_dim: int

    def embed_image(self, images: list[PIL.Image]) -> np.ndarray: ...
    def embed_text(self, texts: list[str]) -> np.ndarray: ...
```

Returns L2-normalized float32 vectors. Cosine reduces to dot product downstream.

### `pixsage.embedders.siglip2`
Wraps `google/siglip2-so400m-patch14-384` via `transformers`. Loads in fp16 on CUDA, fp32 on CPU. Lazy import (heavy deps stay out of CLI cold path).

Both encoders sit in one model — embedding a caption alongside its image is the same forward pass with two heads. Caption embed adds ~5% to per-photo time.

### `pixsage.embed_runner`
Per-photo loop, mirroring the tagger runner pattern:
1. Walk catalog, find photos that are either missing the configured `vector_kind`s **or** whose caption text was updated after the existing caption-vector was written (`photos.caption_updated_at > vectors.created_at`).
2. Load image (existing `images.load_image`).
3. Read caption from `photos.caption` column (see Schema migration below). If null, fall back to reading XMP `dc:description` and write it to the catalog (one-time backfill for photos tagged before Phase 3).
4. Run encoder; append to per-kind parquet (replacing prior row for the sha if present — keyed by sha).
5. Per-photo flush. Interruptible and resumable. `--force` re-embeds everything regardless of staleness check.

### Schema migration

Phase 1 stored captions only in XMP `dc:description` and never duplicated them into the catalog. Phase 3 needs cheap caption access for the embed runner and the photo-detail page, so the migration adds two columns to `photos`:

```sql
ALTER TABLE photos ADD COLUMN caption TEXT;
ALTER TABLE photos ADD COLUMN caption_updated_at TEXT;
```

Run idempotently in `Catalog.init_schema()` via `PRAGMA table_info(photos)` introspection. Phase 1's tag CLI gets a small extension to call `catalog.record_caption(sha256, caption)` after writing XMP when `caption is not None` — that's a 3-line addition to `cli.py`. Existing photos backfill lazily on first embed run from XMP.

### `pixsage.vectors`
Storage layer. One parquet file per `vector_kind` at `<photo_root>/.photoindex/vectors/<vector_kind>.parquet`, schema:

```
sha256: str (PK-ish, dedupe on append)
vector: list<float32> [fixed length]
created_at: timestamp
```

Functions: `append(kind, rows)`, `load(kind) -> (sha_array, matrix)`, `missing_for(kind, all_sha) -> set[str]`.

Why parquet (not catalog BLOBs):
- Decouples re-embed from catalog rewrites — re-running an embedder doesn't churn `photos.sha256` rows.
- One file per model is the cleanest unit of "drop in, take out, swap" as embedders evolve.
- Pyarrow + numpy both first-class. Memory-mapped read into a `(N, D)` float32 matrix at app boot.

### `pixsage.search`
Pure-numpy vector search. At server boot, load all `vector_kind` matrices into RAM. 50k photos × 1152 dims × float32 ≈ 230 MB per kind; two kinds ≈ 460 MB. Brute-force cosine on a 4090's CPU is sub-10 ms for 50k.

```python
def search(
    query_text: str,
    image_weight: float,    # 0.0–1.0, slider value
    top_k: int = 60,
) -> list[Hit]: ...

def search_by_image(sha256: str, top_k: int = 60) -> list[Hit]: ...
```

Combined score:

```
score = image_weight  * cos(query_text_vec, photo_image_vec)
      + (1 - image_weight) * cos(query_text_vec, photo_caption_vec)
```

Photos missing one channel (no caption) score that channel as 0 — they remain searchable on the present channel.

`search_by_image` skips text encoding entirely and does pure visual cosine against image vectors.

### `pixsage.web`
FastAPI app, single module to start.
- Routes:
  - `GET /` — search page (empty state)
  - `POST /search` — HTMX request, returns `_results.html` partial: photo grid
  - `GET /photo/{sha256}` — detail page: full photo, caption, tags, "more like this" button
  - `GET /similar/{sha256}` — same partial as `/search` results, but ranked by image-vector neighbors
  - `GET /thumb/{sha256}?size={small|medium|large}` — serves cached thumb
- Templates: `templates/index.html`, `templates/_results.html`, `templates/photo.html`, `templates/_card.html`
- Static: `static/style.css`, `static/htmx.min.js`

Thumbnail generation: lazy. On first request for a sha, decode the source via `images.load_image`, downscale to one of three target sizes (256 / 720 / 1440 long edge), save JPEG quality 85 to `<photo_root>/.photoindex/thumbs/<size>/<sha2>/<sha>.jpg`. Subsequent requests stream the file directly. Eviction is manual (`pixsage cleanup --thumbs` already exists conceptually; we'll extend it).

## Data flow

**Embed run:**
```
catalog.iter_photos_missing("siglip2_image")
  → for each photo:
      image = load_image(path)
      caption = catalog.get_caption(sha256)  # may be None
      img_vec = encoder.embed_image([image])[0]
      cap_vec = encoder.embed_text([caption])[0] if caption else None
      vectors.append("siglip2_image", [(sha, img_vec)])
      if cap_vec is not None:
          vectors.append("siglip2_caption", [(sha, cap_vec)])
```

**Search request:**
```
POST /search {q: "leopard seal", image_weight: 0.6}
  → q_vec = encoder.embed_text([q])[0]                     # (D,)
  → img_scores = image_matrix @ q_vec                      # (N_img,)
  → cap_scores = caption_matrix @ q_vec                    # (N_cap,)
  → for each sha in union(sha_for_image, sha_for_caption):
        s = 0.6 * img_scores[idx_img.get(sha, NAN)] (or 0)
          + 0.4 * cap_scores[idx_cap.get(sha, NAN)] (or 0)
  → argpartition top_k → list of Hit(sha256, score, current_path)
  → render _results.html with thumbnails
```

Caption vectors only exist for a subset of photos. Two parallel index arrays (`sha_for_image`, `sha_for_caption`) plus dict `sha → row` lookups handle the join. Merge happens after each matrix multiply. At 50k photos this is one Python loop over a ~100k-key dict — sub-50 ms with `numpy` arrays as backing.

**More-like-this:**
```
GET /similar/{sha}
  → photo_vec = image_matrix[index_of[sha]]
  → scores = image_matrix @ photo_vec  (purely visual)
  → drop self → top_k → render _results.html
```

## CLI surface

New verbs alongside existing `tag` and `cleanup`:

```
pixsage embed <photo_root>           Compute embeddings for catalog photos
  --catalog PATH                     Override catalog location
  --model NAME                       Default: siglip2-so400m-patch14-384
  --limit N                          Process at most N photos
  --force                            Re-embed photos that already have vectors
  --batch-size N                     Default: 16

pixsage serve <photo_root>           Run search webapp
  --catalog PATH
  --port N                           Default: 8765
  --host HOST                        Default: 127.0.0.1
  --no-open                          Don't auto-open browser

pixsage cleanup                      (existing) — extend with --thumbs --vectors
```

## Configuration

Extend `pixsage.config.Config`:

```toml
[embeddings]
enabled = true

[embeddings.siglip2]
enabled = true
model = "google/siglip2-so400m-patch14-384"
image = true        # embed images
caption = true      # embed Florence-2 captions
batch_size = 16

[search]
default_image_weight = 0.5
top_k = 60
thumb_size_default = "medium"   # 720 long edge
```

Adding OpenCLIP later is a new `[embeddings.openclip]` block with `image = true, caption = false` (or true). The runner picks up any enabled embedder.

## Error handling

- **Image fails to decode:** mark photo with `error_reason` in catalog (existing pattern), skip embedding, continue.
- **Caption missing:** skip caption embed for that photo. Image embed still proceeds. Search still works on image channel.
- **Model load OOM:** clear error message naming the model and available VRAM.
- **Vector dim mismatch on append:** parquet schema enforces — fail loudly with the offending sha and dims.
- **Server start with empty vector store:** show a setup-prompt page ("run `pixsage embed` first").
- **Server start with stale catalog (sha in vectors but not in photos):** log a warning, drop the orphan vectors at boot.

## Testing

Mirror Phase 1's style: unit + integration + a few real-corpus tests gated on the photographer's drive being mounted.

- **embedders/test_siglip2.py**: model load (skipped if no CUDA / no model cache), tiny image + caption embed, dim/dtype/normalization assertions.
- **test_vectors.py**: parquet round-trip, append-dedup, missing-for set semantics.
- **test_embed_runner.py**: end-to-end with a stub embedder (returns deterministic random vectors), 3 photos, asserts vectors and resumability.
- **test_search.py**: synthetic 100-vector matrix, hand-crafted query → known nearest, weight-blend math, missing-caption photos still scored on image channel.
- **web/test_routes.py**: FastAPI TestClient — search GET/POST, photo detail, similar, thumb generation. Mock embedder for query encoding.
- **test_thumbs.py**: thumbnail cache hit / miss / regeneration.

Real-corpus check (manual, not CI): on the Seymour ARW sample, search for "leopard seal," "iceberg," "snow-covered mountain" — eyeball top-20 grids. Same set used in Phase 1's HTML report so we can compare.

## Performance budget

| Stage | 50k photos | Notes |
|---|---|---|
| Embed (image+caption) | ~14-21 hrs overnight | I/O-bound: ARW decode via rawpy is ~1 sec/photo; SigLIP2 fp16 forward pass is ~50 ms/photo on the 4090. Roughly half of Phase 1's 28 hr because no Florence-2 / RAM++. |
| Vector load at server boot | <2 s | Read parquet, mmap into numpy. ~230 MB × N kinds. |
| Single search query | <80 ms end-to-end | SigLIP2 text encode (~40 ms) + two matmuls (~10 ms) + merge + render |
| Thumbnail first generation | ~200 ms (JPEG), ~1.5 s (raw via rawpy) | Cached after first hit; subsequent reads <5 ms |
| VRAM at serve time | ~3 GB | SigLIP2 text encoder loaded for live query embedding. Image encoder discarded after embed run unless `--keep-image-encoder`. |

## Dependencies

Two new optional extras in `pyproject.toml`:

```toml
[project.optional-dependencies]
search = [
    "fastapi >= 0.110",
    "uvicorn[standard] >= 0.27",
    "jinja2 >= 3.1",
    "pyarrow >= 15",
]
# embeddings reuses the existing [taggers] extra (torch, transformers).
```

Install for full Phase 3: `pip install -e ".[taggers,search]"`. SigLIP2 weights cached at `~/.cache/huggingface/` on first run (same path as Florence-2). HTMX served as a single static JS file checked into the repo (no CDN).

## Forward compatibility

- **OpenCLIP / GeoCLIP later:** new `Embedder` subclass, new vector_kind, new parquet, no schema changes. Search service auto-discovers any vector_kind and exposes weight knobs.
- **Phase 4 map / scatter / clusters:** the same vector matrices feed UMAP / HDBSCAN. Map view reads EXIF GPS independently.
- **Phase 4 filters:** sidebar wraps the existing `/search` route with `WHERE` clauses on the catalog tags table joined to the search results — no rework of the vector path.
- **Faiss / sqlite-vec when corpus grows past ~500k:** swap `pixsage.search` internals; keep the public API.

## Open questions

- **Caption-channel quality on tag-only photos** (RAM++ contributes tags but not a sentence caption). Should we also embed the joined-tags string ("ice, mountain, snow, outdoors") as a fallback caption? Probably yes — adds another `vector_kind = siglip2_tags`. Cheap. Will add if first real-corpus test shows visible gaps on landscape photos.
- **Reciprocal Rank Fusion vs weighted sum** for combining image and caption channels. Weighted sum is simpler, transparent, and exposes the slider naturally. Stick with it for v1.
- **Local CLIP tokenizer download path** on Windows — `transformers` defaults to `~/.cache/huggingface`. Confirmed writable; same path Florence-2 already uses.
