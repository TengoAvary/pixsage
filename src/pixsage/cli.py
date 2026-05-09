from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer
from tqdm import tqdm

from pixsage.catalog import Catalog
from pixsage.config import Config, ensure_default_config, load_config
from pixsage.device import select_device
from pixsage.images import load_image
from pixsage.taggers.base import Tag, Tagger, TagResult
from pixsage.vocabulary import filter_tags
from pixsage.walker import sample_paths, sha256_file, walk_photos
from pixsage.xmp import XmpFields, merge_xmp, read_xmp, write_xmp

app = typer.Typer(help="pixsage — Tier 1 photo auto-tagger")


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


def _is_raw(path: Path) -> bool:
    from pixsage.images import RAW_EXTENSIONS
    return path.suffix.lower() in RAW_EXTENSIONS


@app.command()
def tag(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    force: bool = typer.Option(False, "--force", help="Re-tag photos even if already tagged at current model versions."),
    sample: int = typer.Option(0, "--sample", min=0, help="If >0, tag only N deterministically sampled photos."),
    catalog: Path | None = typer.Option(None, "--catalog", help="Override catalog DB path."),
    config_path: Path | None = typer.Option(None, "--config", help="Override vocabulary.toml path."),
    limit: int = typer.Option(0, "--limit", min=0, help="Stop after this many photos processed."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run pipeline but skip XMP writes and catalog tag updates."),
) -> None:
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
    typer.echo(f"Found {len(paths)} candidate images.")

    typer.echo("Hashing files…")
    hashes: dict[Path, str] = {p: sha256_file(p) for p in tqdm(paths, unit="file")}

    if sample > 0:
        paths = sample_paths(paths, hashes, n=sample)

    processed = 0
    skipped = 0
    errored = 0

    for path in tqdm(paths, unit="img"):
        sha = hashes[path]
        stat = path.stat()
        cat.upsert_photo(sha256=sha, path=path, filesize=stat.st_size, mtime=stat.st_mtime)
        if not force and not cat.needs_tagging(sha, model_versions):
            skipped += 1
            continue
        if limit and processed >= limit:
            break
        try:
            _process_one(path=path, sha=sha, is_raw=_is_raw(path), taggers=taggers, config=config, cat=cat, dry_run=dry_run)
            processed += 1
        except Exception as e:  # broad: log + continue
            cat.mark_error(sha, str(e))
            errored += 1
            typer.echo(f"  error on {path.name}: {e}", err=True)

    cat.finish_run(run_id, processed=processed, skipped=skipped, errored=errored)
    cat.close()
    typer.echo(f"done. processed={processed} skipped={skipped} errored={errored}")


def _process_one(
    path: Path,
    sha: str,
    is_raw: bool,
    taggers: list[Tagger],
    config: Config,
    cat: Catalog,
    dry_run: bool,
) -> None:
    img = load_image(path)

    raw_tags: list[Tag] = []
    caption: str | None = None
    sources_with_tags: set[str] = set()
    for t in taggers:
        result: TagResult = t.tag(img)
        raw_tags.extend(result.tags)
        if result.tags:
            sources_with_tags.add(t.name)
        if caption is None and result.caption:
            caption = result.caption

    filtered = filter_tags(raw_tags, config)
    sources_with_filtered = {t.source for t in filtered}

    existing = read_xmp(path, is_raw=is_raw)
    cat.flag_user_rejections(sha, surviving_xmp_tags=set(existing.subject))
    user_rejected = cat.get_user_rejected(sha)

    merged = merge_xmp(
        existing=existing,
        new_tags=filtered,
        previously_applied=cat.get_previously_applied(sha),
        user_rejected=user_rejected,
        caption=caption if config.caption.enabled else None,
        caption_overwrite=config.caption.overwrite,
        sources_with_tags=sources_with_filtered,
    )

    if not dry_run:
        write_xmp(path, merged, is_raw=is_raw)
        cat.record_tags(sha, [t for t in filtered if (t.name, t.source) not in user_rejected])
        cat.mark_tagged(sha, model_versions={t.name: t.model_version for t in taggers})


if __name__ == "__main__":
    app()
