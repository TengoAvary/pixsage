from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pixsage.catalog import Catalog
from pixsage.config import Config, DEFAULT_CONFIG_TOML, load_config, ensure_default_config
from pixsage.multi_search import MultiSearchService
from pixsage.path_translation import PathResolver
from pixsage.registry import (
    DEFAULT_CAPTION_SIGNATURE,
    DEFAULT_IMAGE_SIGNATURE,
    Registry,
    derive_signatures,
)
from pixsage.search import SearchService
from pixsage.vectors import VectorStore


WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def default_registry_path() -> Path:
    """Canonical location for catalogs.json — same dir as the installed runtime."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "pixsage" / "catalogs.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "pixsage" / "catalogs.json"
    return Path.home() / ".local" / "share" / "pixsage" / "catalogs.json"


def _default_config() -> Config:
    """Parse the in-tree DEFAULT_CONFIG_TOML into a Config object.

    Used when no catalogs are loaded yet, so the empty-state page can still
    render with sensible search defaults (top_k, default_image_weight, ...).
    """
    return Config.model_validate(tomllib.loads(DEFAULT_CONFIG_TOML))


def build_app(
    photo_root: Path | None = None,
    registry_path: Path | None = None,
    embedder_name: str = "siglip2",
    *,
    catalog_path: Path | None = None,
    experimental_cluster_labelling: bool = False,
    skip_discovery: bool = False,
) -> FastAPI:
    """Construct the FastAPI app for multi-catalog search.

    Args:
        photo_root: Optional. If given, ensures its .photoindex/ is in the
            registry (backward compat with the per-folder launcher model).
        registry_path: Override for the catalogs.json location.
        embedder_name: Which embedder to use for query encoding.
        catalog_path: Deprecated single-catalog override, accepted for
            backward compat with the old `serve --catalog` flag. Ignored if
            registry-driven (a non-None registry_path always wins).
        experimental_cluster_labelling: Off by default. See routes.py.
        skip_discovery: If True, don't scan mounted drives on startup.
            Useful in tests to avoid touching /Volumes/.
    """
    registry_path = registry_path or default_registry_path()
    registry = Registry(registry_path)
    registry.load()

    # Auto-register photo_root if given.
    if photo_root is not None:
        pi = photo_root / ".photoindex"
        pi.mkdir(parents=True, exist_ok=True)
        if registry.find_by_photoindex_path(str(pi.resolve())) is None:
            img_sig, cap_sig = derive_signatures(pi)
            registry.add(
                photoindex_path=str(pi.resolve()),
                label=photo_root.name,
                image_embedder_signature=img_sig,
                caption_embedder_signature=cap_sig,
            )

    # Discovery + availability reconciliation.
    if not skip_discovery:
        from pixsage.discovery import list_mounted_roots, walk_for_photoindex
        discovered = walk_for_photoindex(list_mounted_roots())
        registry.refresh_from_discovery(discovered)
    else:
        registry.refresh_from_discovery(discovered_paths=[])
    registry.save()

    # Build the embedder once (shared by all SearchServices).
    from pixsage.cli import _build_embedder
    from pixsage.device import select_device
    embedder = _build_embedder(embedder_name)
    embedder.load(select_device())

    # Build per-catalog SearchService for each enabled+available entry.
    multi = MultiSearchService()
    catalogs: dict[str, Catalog] = {}
    resolvers: dict[str, PathResolver] = {}
    thumbs: dict[str, object] = {}
    photoindex_paths: dict[str, Path] = {}
    from pixsage.web.thumbs import ThumbnailCache

    for entry in registry.entries():
        if not (entry.enabled and entry.available):
            continue
        photoindex = Path(entry.photoindex_path)
        catalog = Catalog(photoindex / "catalog.db")
        catalog.init_schema()
        catalogs[entry.id] = catalog
        photoindex_paths[entry.id] = photoindex

        stored_root = catalog.get_meta("photo_root_at_embed")
        resolvers[entry.id] = PathResolver(
            stored_root=stored_root,
            runtime_root=photoindex.parent,
        )
        thumbs[entry.id] = ThumbnailCache(photoindex / "thumbs")

        vectors = VectorStore(photoindex / "vectors")
        service = SearchService(
            store=vectors,
            embedder=embedder,
            image_kind=embedder.info.image_kind,
            text_kind=embedder.info.text_kind,
        )
        service.load()
        multi.add_catalog(
            catalog_id=entry.id,
            service=service,
            image_sig=entry.image_embedder_signature or DEFAULT_IMAGE_SIGNATURE,
            caption_sig=entry.caption_embedder_signature or DEFAULT_CAPTION_SIGNATURE,
        )

    # Resolve a config — first loaded catalog's vocabulary.toml wins; fall back
    # to the in-tree default if no catalogs are loaded.
    if catalogs:
        first_id = next(iter(catalogs))
        first_pi = photoindex_paths[first_id]
        cfg_path = first_pi / "vocabulary.toml"
        ensure_default_config(cfg_path)
        config = load_config(cfg_path)
    else:
        config = _default_config()

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app = FastAPI(title="pixsage")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Multi-catalog state — Task 9 will rewrite the routes to consume these.
    app.state.registry = registry
    app.state.registry_path = registry_path
    app.state.multi_search = multi
    app.state.embedder = embedder
    app.state.catalogs = catalogs  # dict {catalog_id: Catalog}
    app.state.path_resolvers = resolvers  # dict {catalog_id: PathResolver}
    app.state.thumbs_by_catalog = thumbs  # dict {catalog_id: ThumbnailCache}
    app.state.photoindex_paths = photoindex_paths  # dict {catalog_id: Path}
    app.state.config = config
    app.state.templates = templates

    # Backward-compat shim: existing routes (pre-Task 9) read scalars like
    # app.state.catalog, app.state.search, app.state.thumbs,
    # app.state.path_resolver, app.state.photo_root. When exactly one catalog
    # is loaded we expose them so the existing route handlers and the
    # test_web_search.py suite keep working until Task 9 lands.
    if len(catalogs) == 1:
        only_id = next(iter(catalogs))
        app.state.catalog = catalogs[only_id]
        app.state.thumbs = thumbs[only_id]
        app.state.path_resolver = resolvers[only_id]
        app.state.photo_root = photoindex_paths[only_id].parent
        # MultiSearchService stores the underlying per-catalog SearchService.
        app.state.search = multi._catalogs[only_id].service
        app.state.vectors = VectorStore(photoindex_paths[only_id] / "vectors")
    else:
        app.state.catalog = None
        app.state.thumbs = None
        app.state.path_resolver = None
        app.state.photo_root = None
        app.state.search = None
        app.state.vectors = None

    from pixsage.web import routes
    routes.register(app, experimental_cluster_labelling=experimental_cluster_labelling)

    return app
