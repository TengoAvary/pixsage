from __future__ import annotations

import os
import sys
import threading
import tomllib
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pixsage.config import Config, DEFAULT_CONFIG_TOML, load_config, ensure_default_config
from pixsage.multi_search import MultiSearchService
from pixsage.registry import (
    Registry,
    derive_signatures,
)
from pixsage.web.loader import BackendLoader


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
    defer_load: bool = False,
    experimental_cluster_labelling: bool = False,
) -> FastAPI:
    """Construct the FastAPI app for multi-catalog search.

    Args:
        photo_root: Optional. If given, ensures its .photoindex/ is in the
            registry (backward compat with the per-folder launcher model).
        registry_path: Override for the catalogs.json location.
        embedder_name: Which embedder to use for query encoding.
        defer_load: If True, load the embedder + catalog vectors in a background
            thread and return immediately (server answers a loading screen while
            it warms up). If False (default), load synchronously so the returned
            app is already ready — preserves behavior for tests and other callers.
        experimental_cluster_labelling: Off by default. See routes.py.
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

    # No startup discovery walk — catalogs enter the registry only via the
    # folder-browser picker (POST /catalogs/add-scan) or an explicit
    # photo_root arg. Startup only re-checks which registered paths exist.
    registry.refresh_availability()
    registry.save()

    # --- Synchronous half: cheap setup, returns near-instantly. ---
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app = FastAPI(title="pixsage")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    loader = BackendLoader(["Loading search model…", "Loading catalog vectors…"])
    app.state.loader = loader
    app.state.registry = registry
    app.state.registry_path = registry_path
    app.state.multi_search = MultiSearchService()
    app.state.embedder = None
    app.state.catalogs = {}                # {catalog_id: Catalog}
    app.state.path_resolvers = {}          # {catalog_id: PathResolver}
    app.state.thumbs_by_catalog = {}       # {catalog_id: ThumbnailCache}
    app.state.photoindex_paths = {}        # {catalog_id: Path}
    app.state.config = _default_config()   # replaced by load_fn once catalogs load
    app.state.templates = templates

    from pixsage.web import routes
    routes.register(app, experimental_cluster_labelling=experimental_cluster_labelling)

    # --- Slow half: embedder + per-catalog services. Run inline or threaded. ---
    def load_fn(ldr: BackendLoader) -> None:
        from pixsage.cli import _build_embedder
        from pixsage.device import select_device
        from pixsage.web.routes import _load_catalog_into_multi

        ldr.start_phase(0)
        embedder = _build_embedder(embedder_name)
        embedder.load(select_device())
        app.state.embedder = embedder
        ldr.finish_phase(0)

        ldr.start_phase(1)
        # Safe to populate app.state incrementally: routes gate on loader.status == "ready", so no request reads these dicts until the final flip.
        for entry in registry.entries():
            if not (entry.enabled and entry.available):
                continue
            _load_catalog_into_multi(app, entry)
        if app.state.catalogs:
            first_id = next(iter(app.state.catalogs))
            cfg_path = app.state.photoindex_paths[first_id] / "vocabulary.toml"
            ensure_default_config(cfg_path)
            app.state.config = load_config(cfg_path)
        ldr.finish_phase(1)

    if defer_load:
        threading.Thread(target=loader.run, args=(load_fn,), daemon=True).start()
    else:
        loader.run(load_fn)

    return app
