from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import typer
from PIL import Image
from tqdm import tqdm

from pixsage.catalog import Catalog
from pixsage.config import Config, ensure_default_config, load_config
from pixsage.device import select_device
from pixsage.images import load_image
from pixsage.taggers.base import Tag, Tagger, TagResult
from pixsage.vocabulary import filter_tags
from pixsage.walker import sample_paths, sha256_file, walk_photos
from pixsage.xmp import CameraGps, XmpFields, merge_xmp, needs_sidecar, read_metadata, write_xmp

app = typer.Typer(help="pixsage — Tier 1 photo auto-tagger")


@app.callback()
def _root() -> None:
    """Force multi-command behavior so `pixsage tag ...` requires the explicit subcommand."""


def build_taggers(config: Config) -> list[Tagger]:
    taggers: list[Tagger] = []
    if config.florence2.enabled:
        from pixsage.taggers.florence2 import Florence2Tagger
        taggers.append(Florence2Tagger())
    if config.ram_plus_plus.enabled:
        # Added in Task 16
        try:
            from pixsage.taggers.ramplusplus import RamPlusPlusTagger
            taggers.append(RamPlusPlusTagger())
        except ImportError:
            pass
    return taggers


def _config_hash(config: Config) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_exts(spec: str) -> set[str]:
    """'jpg, .JPG ,heic' -> {'.jpg', '.heic'}. Lowercases, ensures leading dot."""
    out: set[str] = set()
    for raw in spec.split(","):
        e = raw.strip().lower()
        if not e:
            continue
        out.add(e if e.startswith(".") else f".{e}")
    return out


def _apply_extension_filter(
    paths: list[Path], skip: str | None, only: str | None
) -> list[Path]:
    """Filter walk_photos output by --skip-extensions / --only-extensions."""
    if only:
        keep = _normalize_exts(only)
        return [p for p in paths if p.suffix.lower() in keep]
    if skip:
        drop = _normalize_exts(skip)
        return [p for p in paths if p.suffix.lower() not in drop]
    return paths


@app.command()
def tag(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    force: bool = typer.Option(False, "--force", help="Re-tag photos even if already tagged at current model versions."),
    rewrite: bool = typer.Option(
        False,
        "--rewrite",
        help=(
            "Wipe previously-applied auto-tags from XMP and the catalog before re-tagging. "
            "Acts as if pixsage never ran on these photos. User-applied keywords are preserved. "
            "Implies --force."
        ),
    ),
    sample: int = typer.Option(0, "--sample", min=0, help="If >0, tag only N deterministically sampled photos."),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
    config_path: Path | None = typer.Option(None, "--config", help="Override vocabulary.toml path."),
    limit: int = typer.Option(0, "--limit", min=0, help="Stop after this many photos processed."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run pipeline but skip XMP writes and catalog tag updates."),
    skip_extensions: str | None = typer.Option(
        None,
        "--skip-extensions",
        help="Comma-separated extensions to exclude (e.g. '.jpg,.jpeg'). Mutually exclusive with --only-extensions.",
    ),
    only_extensions: str | None = typer.Option(
        None,
        "--only-extensions",
        help="Only process these extensions (e.g. '.arw,.cr3'). Mutually exclusive with --skip-extensions.",
    ),
) -> None:
    """Tag photos under PHOTO_ROOT with AI-generated keywords and captions; write XMP and update the catalog."""
    if skip_extensions and only_extensions:
        typer.echo("--skip-extensions and --only-extensions are mutually exclusive", err=True)
        raise typer.Exit(code=2)

    photoindex = photo_root / ".photoindex"
    photoindex.mkdir(exist_ok=True)
    catalog_path = catalog or (photoindex / "catalog.db")
    cfg_path = config_path or (photoindex / "vocabulary.toml")
    ensure_default_config(cfg_path)
    config = load_config(cfg_path)

    cat = Catalog(catalog_path)
    cat.init_schema()
    cat.set_photo_root_if_unset(photo_root)

    typer.echo(f"Loading taggers on device: {select_device()}")
    taggers = build_taggers(config)
    for t in taggers:
        t.load(select_device())
    model_versions = {t.name: t.model_version for t in taggers}

    run_id = cat.start_run(config_hash=_config_hash(config), model_versions=model_versions)

    paths = list(walk_photos(photo_root))
    found_total = len(paths)
    paths = _apply_extension_filter(paths, skip=skip_extensions, only=only_extensions)
    if len(paths) != found_total:
        typer.echo(f"Found {found_total} candidate images, {len(paths)} after extension filter.")
    else:
        typer.echo(f"Found {found_total} candidate images.")

    typer.echo("Hashing files…")
    hashes: dict[Path, str] = {p: sha256_file(p) for p in tqdm(paths, unit="file")}

    if sample > 0:
        paths = sample_paths(paths, hashes, n=sample)

    # Group sampled paths by sha so we know which ones share content.
    paths_per_sha: dict[str, list[Path]] = {}
    for p in paths:
        paths_per_sha.setdefault(hashes[p], []).append(p)

    processed = 0
    skipped = 0
    errored = 0
    model_runs = 0
    dupe_writes = 0

    # --rewrite implies --force: we never want to skip a photo we're about to wipe.
    effective_force = force or rewrite

    # Per-sha caches valid for this run only:
    #   sha_to_tags: tagger output reused across dupe paths so the model only
    #     runs once per content
    #   sha_prior_strip: tags-to-strip captured on the first dupe path (before
    #     delete_tags clears the DB) so subsequent dupe paths in --rewrite mode
    #     can still strip the old auto-tags from their XMP.
    sha_to_tags: dict[str, tuple[list[Tag], str | None]] = {}
    sha_prior_strip: dict[str, list[Tag]] = {}
    seen_shas_this_run: set[str] = set()

    # Batch size for GPU tagger throughput. 4 fits comfortably on a 24GB GPU
    # for Florence-2 + RAM++ on the resized images load_image returns.
    import os
    batch_size = int(os.environ.get("PIXSAGE_TAGGER_BATCH", "4"))

    pbar = tqdm(total=len(paths), unit="img")
    i = 0
    stop = False
    while i < len(paths) and not stop:
        chunk = paths[i : i + batch_size]
        i += batch_size

        # Stage A: per-path bookkeeping. Decide which paths in the chunk need
        # work, and which shas need a fresh model run.
        chunk_work: list[tuple[Path, str, bool]] = []  # (path, sha, is_first_for_sha)
        shas_needing_model: list[tuple[str, Path]] = []  # (sha, path-to-load-image-from)
        shas_needing_model_set: set[str] = set()
        for path in chunk:
            sha = hashes[path]
            is_dupe_set = len(paths_per_sha[sha]) > 1
            already_tagged = not effective_force and not cat.needs_tagging(sha, model_versions)

            # Non-dupe paths preserve the original skip-on-rerun semantics. Dupe
            # sets always run through option A so every path gets a sidecar
            # even if the sha was already tagged. Decide before doing any disk
            # work so resumes on external drives skip cheaply.
            if already_tagged and not is_dupe_set:
                skipped += 1
                pbar.update(1)
                continue

            stat = path.stat()
            cat.upsert_photo(sha256=sha, path=path, filesize=stat.st_size, mtime=stat.st_mtime)
            if limit and processed >= limit:
                stop = True
                # Account for the rest of this chunk in the progress bar.
                pbar.update(len(chunk) - (chunk.index(path)))
                break

            is_first_for_sha = sha not in seen_shas_this_run
            seen_shas_this_run.add(sha)

            if is_first_for_sha and sha not in sha_to_tags:
                if already_tagged:
                    # Dupe set with one sha tagged in a prior run: pull the
                    # stored auto-tags + caption from the catalog instead of
                    # re-running the model.
                    sha_to_tags[sha] = _reconstitute_tags_from_catalog(sha, cat)
                else:
                    if sha not in shas_needing_model_set:
                        shas_needing_model.append((sha, path))
                        shas_needing_model_set.add(sha)
            elif not is_first_for_sha:
                dupe_writes += 1

            chunk_work.append((path, sha, is_first_for_sha))

        # Stage B: batched model run for any shas needing fresh tags.
        if shas_needing_model:
            try:
                images = [load_image(p) for _, p in shas_needing_model]
                results = _run_taggers_batch(images, taggers, config)
                for (sha, _), result in zip(shas_needing_model, results):
                    sha_to_tags[sha] = result
                    model_runs += 1
            except Exception as e:  # batch failed wholesale — mark each pending sha
                msg = str(e)
                for sha, src_path in shas_needing_model:
                    cat.mark_error(sha, msg)
                    errored += 1
                    typer.echo(f"  error on {src_path.name}: {e}", err=True)
                # Drop any path that depended on a failed model run.
                chunk_work = [w for w in chunk_work if w[1] in sha_to_tags]

        # Stage C: per-path XMP write + catalog update, in original order.
        for path, sha, is_first_for_sha in chunk_work:
            try:
                filtered_tags, caption = sha_to_tags[sha]
                new_sha, camera_gps = _apply_to_path(
                    path=path,
                    sha=sha,
                    is_raw=needs_sidecar(path),
                    filtered_tags=filtered_tags,
                    caption=caption,
                    is_first_for_sha=is_first_for_sha,
                    taggers=taggers,
                    config=config,
                    cat=cat,
                    dry_run=dry_run,
                    rewrite=rewrite,
                    sha_prior_strip=sha_prior_strip,
                )
                if not dry_run and camera_gps is not None:
                    cat.set_camera_gps(
                        new_sha,
                        latitude=camera_gps.latitude,
                        longitude=camera_gps.longitude,
                        altitude=camera_gps.altitude,
                    )
                if new_sha != sha:
                    sha_to_tags[new_sha] = sha_to_tags[sha]
                    seen_shas_this_run.add(new_sha)
                processed += 1
            except Exception as e:  # broad: log + continue
                cat.mark_error(sha, str(e))
                errored += 1
                typer.echo(f"  error on {path.name}: {e}", err=True)
            pbar.update(1)
    pbar.close()

    cat.finish_run(run_id, processed=processed, skipped=skipped, errored=errored)
    cat.close()
    extras = []
    if model_runs:
        extras.append(f"model_runs={model_runs}")
    if dupe_writes:
        extras.append(f"dupe_writes={dupe_writes}")
    extra_str = f" ({', '.join(extras)})" if extras else ""
    typer.echo(f"done. processed={processed} skipped={skipped} errored={errored}{extra_str}")


@app.command()
def cleanup(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
    thumbs: bool = typer.Option(False, "--thumbs", help="Also delete the thumbnail cache."),
    vectors: bool = typer.Option(False, "--vectors", help="Also delete all vector parquet files."),
) -> None:
    """Drop stale catalog rows left behind by errored writes.

    Each photo file should have exactly one row in the catalog (keyed by
    sha256). When a prior run errored after write_xmp succeeded but before
    rekey_photo committed, the catalog accumulates an extra row per photo.
    This command keeps the most-recently-seen row for each path and drops
    the rest. Tag rows for the dropped photos cascade-delete automatically.

    With --thumbs / --vectors, also wipes the corresponding Phase 3 caches.
    """
    import shutil

    photoindex = photo_root / ".photoindex"
    catalog_path = catalog or (photoindex / "catalog.db")
    if not catalog_path.exists():
        typer.echo(f"no catalog at {catalog_path}", err=True)
        raise typer.Exit(code=1)

    cat = Catalog(catalog_path)
    cat.init_schema()
    before_photos = cat._conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]  # noqa: SLF001
    before_tags = cat._conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]  # noqa: SLF001
    deleted = cat.cleanup_orphans()
    after_photos = cat._conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]  # noqa: SLF001
    after_tags = cat._conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]  # noqa: SLF001
    cat.close()
    typer.echo(
        f"removed {deleted} orphan photo rows. "
        f"photos: {before_photos} -> {after_photos}, tags: {before_tags} -> {after_tags}"
    )

    if thumbs:
        thumbs_dir = photoindex / "thumbs"
        if thumbs_dir.exists():
            shutil.rmtree(thumbs_dir)
            typer.echo(f"removed thumbnail cache at {thumbs_dir}")

    if vectors:
        vectors_dir = photoindex / "vectors"
        if vectors_dir.exists():
            shutil.rmtree(vectors_dir)
            typer.echo(f"removed vector store at {vectors_dir}")


def _build_embedder(name: str):
    """Construct an embedder by short name. Lazy imports keep the CLI cold path light."""
    if name == "mock":
        from pixsage.embedders.mock import MockEmbedder
        return MockEmbedder()
    if name == "siglip2":
        from pixsage.embedders.siglip2 import SigLIP2Embedder
        return SigLIP2Embedder()
    raise typer.BadParameter(f"unknown embedder: {name!r} (choose from: mock, siglip2)")


def _build_geolocator(name: str, top_k: int):
    """Construct a geolocator by short name. Lazy imports keep the CLI cold path light."""
    if name == "mock":
        from pixsage.geolocators.mock import MockGeolocator
        return MockGeolocator(top_k=top_k)
    if name == "geoclip":
        from pixsage.geolocators.geoclip import GeoCLIPGeolocator
        return GeoCLIPGeolocator(top_k=top_k)
    raise typer.BadParameter(f"unknown geolocator: {name!r} (choose from: geoclip, mock)")


@app.command()
def embed(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    embedder: str = typer.Option(
        "siglip2", "--embedder",
        help="Embedder to use. Choices: siglip2, mock (mock is for testing only).",
    ),
    force: bool = typer.Option(False, "--force", help="Re-embed photos even if vectors already exist."),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
    no_image: bool = typer.Option(False, "--no-image", help="Skip image embedding."),
    no_caption: bool = typer.Option(False, "--no-caption", help="Skip caption embedding."),
) -> None:
    """Compute embeddings for each photo in the catalog."""
    from pixsage.embed_runner import EmbedRunner
    from pixsage.vectors import VectorStore

    photoindex = photo_root / ".photoindex"
    catalog_path = catalog or (photoindex / "catalog.db")
    if not catalog_path.exists():
        typer.echo(f"no catalog at {catalog_path}; run `pixsage tag` first", err=True)
        raise typer.Exit(code=1)

    cat = Catalog(catalog_path)
    cat.init_schema()  # picks up the caption migration if it's an older catalog
    cat.set_photo_root_if_unset(photo_root)

    enc = _build_embedder(embedder)
    typer.echo(f"Loading embedder: {enc.info.name}")
    enc.load(select_device())

    vectors = VectorStore(photoindex / "vectors")

    embed_runner = EmbedRunner(
        catalog=cat,
        vectors=vectors,
        embedder=enc,
        force=force,
        embed_image=not no_image,
        embed_caption=not no_caption,
        progress=True,
    )
    stats = embed_runner.run()
    cat.close()
    typer.echo(f"done. processed={stats['processed']} skipped={stats['skipped']} errored={stats['errored']}")


def _run_taggers(path: Path, taggers: list[Tagger], config: Config) -> tuple[list[Tag], str | None]:
    """Run every enabled tagger on the image and return (filtered_tags, caption).
    Pure model work — no catalog or XMP I/O. Result is reused across dupe paths."""
    img = load_image(path)
    raw_tags: list[Tag] = []
    caption: str | None = None
    for t in taggers:
        result = _tag_with_retry(t, img)
        raw_tags.extend(result.tags)
        if caption is None and result.caption:
            caption = result.caption
    return filter_tags(raw_tags, config), caption


def _run_taggers_batch(
    images: list[Image.Image],
    taggers: list[Tagger],
    config: Config,
) -> list[tuple[list[Tag], str | None]]:
    """Batched form of _run_taggers. Each tagger.tag_batch runs once over the
    full image list; results are merged per-image. OOM is handled by halving
    the batch and recursing, falling back to single-image resize retry."""
    n = len(images)
    per_image_tags: list[list[Tag]] = [[] for _ in range(n)]
    per_image_caption: list[str | None] = [None] * n
    for t in taggers:
        results = _tag_batch_with_retry(t, images)
        for idx, r in enumerate(results):
            per_image_tags[idx].extend(r.tags)
            if per_image_caption[idx] is None and r.caption:
                per_image_caption[idx] = r.caption
    return [
        (filter_tags(tags, config), cap)
        for tags, cap in zip(per_image_tags, per_image_caption)
    ]


def _tag_batch_with_retry(tagger: Tagger, images: list[Image.Image]) -> list[TagResult]:
    """Call tagger.tag_batch(images). On OOM, halve and recurse; at batch
    size 1, fall back to _tag_with_retry's resize ladder. Taggers that don't
    implement tag_batch get per-image dispatch via _tag_with_retry."""
    if not images:
        return []
    if not hasattr(tagger, "tag_batch"):
        return [_tag_with_retry(tagger, img) for img in images]
    try:
        return tagger.tag_batch(images)
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" not in msg and "oom" not in msg:
            raise
        if len(images) == 1:
            return [_tag_with_retry(tagger, images[0])]
        mid = len(images) // 2
        return (
            _tag_batch_with_retry(tagger, images[:mid])
            + _tag_batch_with_retry(tagger, images[mid:])
        )


def _reconstitute_tags_from_catalog(sha: str, cat: Catalog) -> tuple[list[Tag], str | None]:
    """Pull the (non-rejected) auto-tags + caption that were stored for this sha
    in a prior run. Used when a dupe set contains a sha already tagged before, so
    new copies can get sidecars without re-running the model."""
    cur = cat._conn.execute(  # noqa: SLF001 — internal SQL is fine for our own helper
        "SELECT tag, confidence, hierarchy, source FROM tags "
        "WHERE sha256 = ? AND user_rejected = 0",
        (sha,),
    )
    tags = [
        Tag(
            name=r["tag"],
            confidence=r["confidence"] or 0.0,
            hierarchy=r["hierarchy"],
            source=r["source"],
        )
        for r in cur
    ]
    row = cat.get_photo(sha)
    caption = row["caption"] if row else None
    return tags, caption


def _apply_to_path(
    path: Path,
    sha: str,
    is_raw: bool,
    filtered_tags: list[Tag],
    caption: str | None,
    is_first_for_sha: bool,
    taggers: list[Tagger],
    config: Config,
    cat: Catalog,
    dry_run: bool,
    rewrite: bool,
    sha_prior_strip: dict[str, list[Tag]],
) -> tuple[str, CameraGps | None]:
    """Per-path: read existing XMP, merge with the cached auto-tags+caption,
    write XMP, update the catalog. Returns (possibly rekeyed sha, camera GPS).

    Only the FIRST path for a sha runs flag_user_rejections (the surviving-XMP
    set is meaningful for the path that previously held our auto-tags; on a
    fresh dupe path with empty XMP we'd otherwise mark every tag rejected).
    """
    existing, camera_gps = read_metadata(path, is_raw=is_raw)

    if rewrite:
        if is_first_for_sha:
            # Capture the prior auto-tags before delete_tags wipes them; dupe
            # paths in this run reuse the captured list to strip their own XMP.
            prior_tags = cat.get_tags(sha)
            sha_prior_strip[sha] = prior_tags
            existing = _strip_auto_artifacts(existing, prior_tags)
            cat.delete_tags(sha)
        else:
            existing = _strip_auto_artifacts(existing, sha_prior_strip.get(sha, []))
    else:
        existing = _strip_legacy_markers(existing)

    if is_first_for_sha:
        cat.flag_user_rejections(sha, surviving_xmp_tags=set(existing.subject))
    user_rejected = cat.get_user_rejected(sha)

    merged = merge_xmp(
        existing=existing,
        new_tags=filtered_tags,
        user_rejected=user_rejected,
        caption=caption if config.caption.enabled else None,
        caption_overwrite=config.caption.overwrite or rewrite,
    )

    if not dry_run:
        write_xmp(path, merged, is_raw=is_raw)
        # Embedded XMP changes file bytes → sha256 changes. Update the catalog
        # row's primary key so the next run skip-detects this photo correctly.
        # (Sidecar writes don't touch the source file, so the sha stays.)
        if not is_raw:
            new_sha = sha256_file(path)
            cat.rekey_photo(sha, new_sha)
            sha = new_sha
        if is_first_for_sha:
            cat.record_tags(
                sha, [t for t in filtered_tags if (t.name, t.source) not in user_rejected]
            )
        cat.mark_tagged(sha, model_versions={t.name: t.model_version for t in taggers})
        if merged.description:
            cat.record_caption(sha, merged.description)

    return sha, camera_gps


def _is_legacy_marker(s: str) -> bool:
    """Detect markers ('auto-tagged-florence2', 'auto-tagged-ram') from older builds."""
    from pixsage.xmp import LEGACY_MARKER_PREFIX
    return s.startswith(LEGACY_MARKER_PREFIX)


def _strip_legacy_markers(existing: XmpFields) -> XmpFields:
    """Drop legacy source markers from existing XMP. Used on every run so XMP
    written by older pixsage versions sheds them naturally."""
    return XmpFields(
        subject=[s for s in existing.subject if not _is_legacy_marker(s)],
        hierarchical_subject=existing.hierarchical_subject,
        description=existing.description,
    )


def _strip_auto_artifacts(existing: XmpFields, prior_tags: list[Tag]) -> XmpFields:
    """Return XmpFields with previously-applied auto tags + legacy markers removed."""
    auto_names = {t.name for t in prior_tags}
    auto_hierarchies = {t.hierarchy for t in prior_tags if t.hierarchy}
    return XmpFields(
        subject=[
            s for s in existing.subject
            if s not in auto_names and not _is_legacy_marker(s)
        ],
        hierarchical_subject=[h for h in existing.hierarchical_subject if h not in auto_hierarchies],
        # Description is overwritten downstream when caption.enabled (we force
        # caption_overwrite=True for --rewrite). Keep the existing string here
        # so the merge sees a consistent starting state regardless.
        description=existing.description,
    )


def _tag_with_retry(tagger: Tagger, image: Image.Image) -> TagResult:
    """Try the tagger; on OOM-like failure, retry at 768 then 512 px long edge."""
    sizes = [None, 768, 512]
    last_err: Exception | None = None
    for fallback in sizes:
        try:
            target = image if fallback is None else _resize_to_long_edge(image, fallback)
            return tagger.tag(target)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "oom" in msg:
                last_err = e
                continue
            raise
    raise last_err if last_err else RuntimeError("tagger failed without exception")


def _resize_to_long_edge(img: Image.Image, target: int) -> Image.Image:
    from pixsage.images import _resize_long_edge
    return _resize_long_edge(img, target)


@app.command()
def geolocate(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    geolocator: str = typer.Option(
        "geoclip", "--geolocator",
        help="Geolocator to use. Choices: geoclip, mock (mock is for testing only).",
    ),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Number of top GPS predictions to store per photo."),
    force: bool = typer.Option(False, "--force", help="Re-predict photos that already have geo predictions for this model."),
    all_photos: bool = typer.Option(
        False, "--all",
        help="Predict on every photo, including those that already have real GPS (EXIF or user-applied). Default skips them as redundant.",
    ),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
) -> None:
    """Predict GPS coordinates for catalogued photos that lack real GPS.

    By default, photos with EXIF GPS or a user-applied location are skipped
    (running GeoCLIP on them produces guesses that are worse than the truth
    already in the file). Pass --all to override.

    Predictions live in the geo_predictions table and travel with the catalog.db,
    so the analysis machine doesn't need the source photos to read them back.
    """
    from pixsage.geo_runner import GeoRunner

    photoindex = photo_root / ".photoindex"
    catalog_path = catalog or (photoindex / "catalog.db")
    if not catalog_path.exists():
        typer.echo(f"no catalog at {catalog_path}; run `pixsage tag` first", err=True)
        raise typer.Exit(code=1)

    cat = Catalog(catalog_path)
    cat.init_schema()  # picks up geo_predictions schema if it's an older catalog
    cat.set_photo_root_if_unset(photo_root)

    geo = _build_geolocator(geolocator, top_k=top_k)
    typer.echo(f"Loading geolocator: {geo.info.name}")
    geo.load(select_device())

    runner = GeoRunner(catalog=cat, geolocator=geo, force=force, progress=True, include_with_camera_gps=all_photos)
    stats = runner.run()
    cat.close()
    typer.echo(f"done. processed={stats['processed']} skipped={stats['skipped']} errored={stats['errored']}")


@app.command(name="backfill-exif-gps")
def backfill_exif_gps(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    force: bool = typer.Option(False, "--force", help="Re-read EXIF for photos that already have stored GPS."),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
) -> None:
    """Populate exif_latitude/longitude/altitude for an already-tagged catalog.

    Use this after upgrading from a pixsage version that didn't extract EXIF
    GPS during `tag`. Iterates every photo in the catalog, reads its EXIF GPS
    via exiftool, and stores the result.

    Photos that already have stored GPS are skipped unless --force is set.
    Photos whose current_path doesn't exist on this machine are reported but
    not flagged as errors (the catalog may have been moved).
    """
    from pixsage.xmp import read_camera_gps

    photoindex = photo_root / ".photoindex"
    catalog_path = catalog or (photoindex / "catalog.db")
    if not catalog_path.exists():
        typer.echo(f"no catalog at {catalog_path}; run `pixsage tag` first", err=True)
        raise typer.Exit(code=1)

    cat = Catalog(catalog_path)
    cat.init_schema()

    checked = 0
    with_gps = 0
    skipped = 0
    missing = 0
    errored = 0

    cur = cat._conn.execute("SELECT sha256, current_path FROM photos")
    rows = cur.fetchall()
    for row in rows:
        sha = row["sha256"]
        path = Path(row["current_path"])

        if not force and cat.get_camera_gps(sha) is not None:
            skipped += 1
            continue

        if not path.exists():
            missing += 1
            continue

        checked += 1
        try:
            gps = read_camera_gps(path)
        except Exception as e:
            errored += 1
            typer.echo(f"  error on {path.name}: {e}", err=True)
            continue

        if gps is not None:
            cat.set_camera_gps(sha, latitude=gps.latitude, longitude=gps.longitude, altitude=gps.altitude)
            with_gps += 1

    cat.close()
    typer.echo(
        f"done. checked: {checked} with gps: {with_gps} "
        f"no gps: {checked - with_gps} skipped: {skipped} "
        f"missing: {missing} errored: {errored}"
    )


@app.command()
def export(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    out: Path = typer.Option(..., "--out", help="Path to the output zip (e.g. catalog-export.zip)."),
    include_thumbs: bool = typer.Option(
        False, "--include-thumbs",
        help="Also bundle the thumbnail cache. Off by default — thumbs regenerate from photos and bloat the export.",
    ),
) -> None:
    """Bundle the .photoindex/ directory into a portable zip for offline analysis.

    The catalog.db, vector parquet files, vocabulary.toml, and (optionally)
    thumbnails are included. Source photos are not — analysis on the receiving
    machine works directly off the catalog/vectors and never opens the originals.
    """
    import zipfile

    photoindex = photo_root / ".photoindex"
    if not photoindex.exists():
        typer.echo(f"no .photoindex at {photoindex}", err=True)
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in photoindex.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(photoindex)
            if not include_thumbs and rel.parts and rel.parts[0] == "thumbs":
                continue
            zf.write(p, arcname=str(rel))
            file_count += 1

    size_mb = out.stat().st_size / (1024 * 1024)
    typer.echo(f"wrote {out} ({file_count} files, {size_mb:.1f} MB)")


@app.command()
def serve(
    photo_root: Path | None = typer.Argument(
        None,
        exists=False,  # may be omitted; presence checked below
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Optional. Auto-registers this folder's catalog on startup.",
    ),
    embedder: str = typer.Option("siglip2", "--embedder", help="Embedder for query encoding."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    no_open: bool = typer.Option(False, "--no-open", help="Don't auto-open a browser."),
    registry: Path | None = typer.Option(None, "--registry", help="Override registry path."),
) -> None:
    """Run the multi-catalog search webapp on http://host:port.

    With no arguments, reads the registry, scans mounted drives, and shows
    every catalog it knows about. Pass a folder to register it on startup.
    """
    if photo_root is not None and not photo_root.exists():
        typer.echo(f"path does not exist: {photo_root}", err=True)
        raise typer.Exit(code=1)

    try:
        import uvicorn
    except ImportError:
        typer.echo("FastAPI + uvicorn not installed. Run: pip install -e \".[search]\"", err=True)
        raise typer.Exit(code=1)

    # Serve only ever loads already-downloaded models (install_runtime /
    # `pixsage embed` fetched them). Force HF offline so each from_pretrained
    # skips its hub ETag round-trip — those network checks add ~10s to every
    # launch even when weights are fully cached. Must be set before transformers
    # is imported (via build_app). setdefault lets a user override to refresh.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from pixsage.web.app import build_app
    fastapi_app = build_app(
        photo_root=photo_root,
        registry_path=registry,
        embedder_name=embedder,
    )

    if not no_open:
        import webbrowser, threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}/")).start()

    typer.echo(f"pixsage serve at http://{host}:{port}/")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


@app.command(name="run")
def run(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    dashboard_port: int = typer.Option(8766, "--dashboard-port", help="Port for the progress dashboard."),
    no_dashboard: bool = typer.Option(False, "--no-dashboard", help="Skip launching the progress dashboard."),
) -> None:
    """Run tag + embed on PHOTO_ROOT with a live progress dashboard.

    Logs go to <PHOTO_ROOT>/.photoindex/logs/{tag,embed}.log. Dashboard
    serves at http://127.0.0.1:<port>/ until the pipeline finishes, then
    is torn down automatically.

    Geolocate is intentionally not part of `run` — GeoCLIP has proven
    near-useless on portfolio photography. Run `pixsage geolocate`
    separately if you need it.
    """
    import subprocess
    import sys
    import time

    photoindex = photo_root / ".photoindex"
    photoindex.mkdir(exist_ok=True)
    logdir = photoindex / "logs"
    logdir.mkdir(exist_ok=True)

    dashboard_proc: subprocess.Popen | None = None
    if not no_dashboard:
        dashboard_script = _find_dashboard_script()
        if dashboard_script is None:
            typer.echo(
                "warning: dashboard script not found (only available from a "
                "source checkout); continuing without dashboard",
                err=True,
            )
        else:
            dashboard_log_path = logdir / "dashboard.log"
            # Binary-mode log; child writes utf-8 directly, avoiding parent
            # encoding interference on Windows.
            dashboard_log = open(dashboard_log_path, "wb")
            dashboard_proc = subprocess.Popen(
                [
                    sys.executable, str(dashboard_script), str(photo_root),
                    "--logdir", str(logdir),
                    "--port", str(dashboard_port),
                ],
                stdout=dashboard_log,
                stderr=subprocess.STDOUT,
            )
            # Give uvicorn ~1s to bind the port (or crash). If it died,
            # surface the failure but continue with the pipeline.
            time.sleep(1.0)
            if dashboard_proc.poll() is not None:
                typer.echo(
                    f"warning: dashboard exited (code {dashboard_proc.returncode}); "
                    f"see {dashboard_log_path}",
                    err=True,
                )
                dashboard_proc = None
            else:
                typer.echo(f"dashboard: http://127.0.0.1:{dashboard_port}/")

    try:
        for name in ("tag", "embed"):
            log_path = logdir / f"{name}.log"
            typer.echo(f"[{name}] -> {log_path}")
            # Binary mode + utf-8 from the child sidesteps PowerShell's
            # UTF-16-BOM redirect trap that breaks dashboard tqdm parsing.
            with open(log_path, "wb") as logf:
                stage_proc = subprocess.Popen(
                    [sys.executable, "-m", "pixsage", name, str(photo_root)],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
                exit_code = stage_proc.wait()
            if exit_code != 0:
                typer.echo(
                    f"[{name}] failed (exit {exit_code}); see {log_path}",
                    err=True,
                )
                raise typer.Exit(code=exit_code)
            typer.echo(f"[{name}] {_stage_summary(log_path)}")
    finally:
        if dashboard_proc is not None and dashboard_proc.poll() is None:
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                dashboard_proc.kill()


def _find_dashboard_script() -> Path | None:
    """Locate scripts/dashboard.py relative to the pixsage package.

    Returns the absolute path when running from a source checkout (editable
    install or repo clone). Returns None for wheel-only installs where
    scripts/ doesn't ship — caller falls back to dashboard-less operation.
    """
    import pixsage
    repo_root = Path(pixsage.__file__).resolve().parent.parent.parent
    candidate = repo_root / "scripts" / "dashboard.py"
    return candidate if candidate.is_file() else None


def _stage_summary(log_path: Path) -> str:
    """Pull the final `done.` line from a stage log; fall back to 'done'."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "done"
    text = text.replace("\r", "\n")
    for line in reversed(text.splitlines()):
        if line.startswith("done."):
            return line
    return "done"


@app.command(name="stage-launchers")
def stage_launchers(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
) -> None:
    """Drop `Pixsage Search.bat` + `Pixsage Search.command` into an indexed folder.

    Run once per folder after `pixsage embed` so the photographer can launch
    the search webapp by double-clicking the launcher in Explorer / Finder.

    Requires the pixsage runtime to already be installed at the canonical
    local path (%LOCALAPPDATA%\\pixsage on Windows, ~/Library/Application
    Support/pixsage on Mac). See `scripts/launcher/install_runtime.py`.
    """
    from scripts.launcher.stage_folder import stage_folder
    stage_folder(photo_root)
    typer.echo(f"Staged launchers in: {photo_root}")


if __name__ == "__main__":
    app()
