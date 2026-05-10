from __future__ import annotations

import hashlib
import json
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
from pixsage.xmp import XmpFields, merge_xmp, needs_sidecar, read_xmp, write_xmp

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

    for path in tqdm(paths, unit="img"):
        sha = hashes[path]
        stat = path.stat()
        cat.upsert_photo(sha256=sha, path=path, filesize=stat.st_size, mtime=stat.st_mtime)

        is_dupe_set = len(paths_per_sha[sha]) > 1
        already_tagged = not effective_force and not cat.needs_tagging(sha, model_versions)

        # Non-dupe paths preserve the original skip-on-rerun semantics. Dupe
        # sets always run through option A so every path gets a sidecar even
        # if the sha was already tagged.
        if already_tagged and not is_dupe_set:
            skipped += 1
            continue
        if limit and processed >= limit:
            break

        is_first_for_sha = sha not in seen_shas_this_run
        seen_shas_this_run.add(sha)

        try:
            if is_first_for_sha and sha not in sha_to_tags:
                if already_tagged:
                    # Dupe set with one sha tagged in a prior run: pull the
                    # stored auto-tags + caption from the catalog instead of
                    # re-running the model.
                    sha_to_tags[sha] = _reconstitute_tags_from_catalog(sha, cat)
                else:
                    sha_to_tags[sha] = _run_taggers(path, taggers, config)
                    model_runs += 1
            elif not is_first_for_sha:
                dupe_writes += 1

            filtered_tags, caption = sha_to_tags[sha]
            new_sha = _apply_to_path(
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
            if new_sha != sha:
                sha_to_tags[new_sha] = sha_to_tags[sha]
                seen_shas_this_run.add(new_sha)
            processed += 1
        except Exception as e:  # broad: log + continue
            cat.mark_error(sha, str(e))
            errored += 1
            typer.echo(f"  error on {path.name}: {e}", err=True)

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
) -> str:
    """Per-path: read existing XMP, merge with the cached auto-tags+caption,
    write XMP, update the catalog. Returns the (possibly rekeyed) sha.

    Only the FIRST path for a sha runs flag_user_rejections (the surviving-XMP
    set is meaningful for the path that previously held our auto-tags; on a
    fresh dupe path with empty XMP we'd otherwise mark every tag rejected).
    """
    existing = read_xmp(path, is_raw=is_raw)

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

    return sha


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
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
) -> None:
    """Predict GPS coordinates for each catalogued photo and store the top-K in the catalog.

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

    geo = _build_geolocator(geolocator, top_k=top_k)
    typer.echo(f"Loading geolocator: {geo.info.name}")
    geo.load(select_device())

    runner = GeoRunner(catalog=cat, geolocator=geo, force=force, progress=True)
    stats = runner.run()
    cat.close()
    typer.echo(f"done. processed={stats['processed']} skipped={stats['skipped']} errored={stats['errored']}")


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
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    embedder: str = typer.Option("siglip2", "--embedder", help="Embedder for query encoding."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    no_open: bool = typer.Option(False, "--no-open", help="Don't auto-open a browser."),
    catalog: Path | None = typer.Option(None, "--catalog"),
) -> None:
    """Run the search webapp on http://host:port."""
    photoindex = photo_root / ".photoindex"
    catalog_path = catalog or (photoindex / "catalog.db")
    if not catalog_path.exists():
        typer.echo(f"no catalog at {catalog_path}; run `pixsage tag` then `pixsage embed` first", err=True)
        raise typer.Exit(code=1)

    try:
        import uvicorn
    except ImportError:
        typer.echo("FastAPI + uvicorn not installed. Run: pip install -e \".[search]\"", err=True)
        raise typer.Exit(code=1)

    from pixsage.web.app import build_app
    fastapi_app = build_app(photo_root=photo_root, embedder_name=embedder, catalog_path=catalog_path)

    if not no_open:
        import webbrowser, threading
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()

    typer.echo(f"pixsage serve at http://{host}:{port}/")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
