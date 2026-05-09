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

    processed = 0
    skipped = 0
    errored = 0

    # --rewrite implies --force: we never want to skip a photo we're about to wipe.
    effective_force = force or rewrite

    for path in tqdm(paths, unit="img"):
        sha = hashes[path]
        stat = path.stat()
        cat.upsert_photo(sha256=sha, path=path, filesize=stat.st_size, mtime=stat.st_mtime)
        if not effective_force and not cat.needs_tagging(sha, model_versions):
            skipped += 1
            continue
        if limit and processed >= limit:
            break
        try:
            _process_one(
                path=path,
                sha=sha,
                is_raw=needs_sidecar(path),
                taggers=taggers,
                config=config,
                cat=cat,
                dry_run=dry_run,
                rewrite=rewrite,
            )
            processed += 1
        except Exception as e:  # broad: log + continue
            cat.mark_error(sha, str(e))
            errored += 1
            typer.echo(f"  error on {path.name}: {e}", err=True)

    cat.finish_run(run_id, processed=processed, skipped=skipped, errored=errored)
    cat.close()
    typer.echo(f"done. processed={processed} skipped={skipped} errored={errored}")


@app.command()
def cleanup(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
) -> None:
    """Drop stale catalog rows left behind by errored writes.

    Each photo file should have exactly one row in the catalog (keyed by
    sha256). When a prior run errored after write_xmp succeeded but before
    rekey_photo committed, the catalog accumulates an extra row per photo.
    This command keeps the most-recently-seen row for each path and drops
    the rest. Tag rows for the dropped photos cascade-delete automatically.
    """
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


def _build_embedder(name: str):
    """Construct an embedder by short name. Lazy imports keep the CLI cold path light."""
    if name == "mock":
        from pixsage.embedders.mock import MockEmbedder
        return MockEmbedder()
    if name == "siglip2":
        from pixsage.embedders.siglip2 import SigLIP2Embedder
        return SigLIP2Embedder()
    raise typer.BadParameter(f"unknown embedder: {name!r} (choose from: mock, siglip2)")


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


def _process_one(
    path: Path,
    sha: str,
    is_raw: bool,
    taggers: list[Tagger],
    config: Config,
    cat: Catalog,
    dry_run: bool,
    rewrite: bool = False,
) -> None:
    img = load_image(path)
    raw_tags: list[Tag] = []
    caption: str | None = None

    for t in taggers:
        result = _tag_with_retry(t, img)
        raw_tags.extend(result.tags)
        if caption is None and result.caption:
            caption = result.caption

    filtered = filter_tags(raw_tags, config)

    existing = read_xmp(path, is_raw=is_raw)

    if rewrite:
        # Strip every auto-tag this photo previously got from us (and any
        # legacy source markers from older builds) before merging. User-applied
        # keywords stay. Also wipe the DB tag rows so user_rejected resets —
        # the user asked for a clean slate.
        existing = _strip_auto_artifacts(existing, cat.get_tags(sha))
        cat.delete_tags(sha)
    else:
        # Strip legacy source markers ("auto-tagged-florence2" / "auto-tagged-ram")
        # written by an older pixsage version so they fade out of XMP on the
        # next ordinary --force run. Doesn't touch user-applied keywords.
        existing = _strip_legacy_markers(existing)

    cat.flag_user_rejections(sha, surviving_xmp_tags=set(existing.subject))
    user_rejected = cat.get_user_rejected(sha)

    merged = merge_xmp(
        existing=existing,
        new_tags=filtered,
        user_rejected=user_rejected,
        caption=caption if config.caption.enabled else None,
        # --rewrite always replaces the description so config improvements take effect.
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
        cat.record_tags(sha, [t for t in filtered if (t.name, t.source) not in user_rejected])
        cat.mark_tagged(sha, model_versions={t.name: t.model_version for t in taggers})
        if merged.description:
            cat.record_caption(sha, merged.description)


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
    fastapi_app = build_app(photo_root=photo_root, embedder_name=embedder)

    if not no_open:
        import webbrowser, threading
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()

    typer.echo(f"pixsage serve at http://{host}:{port}/")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
