from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _serve_extra() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return data["project"]["optional-dependencies"]["serve"]


def test_serve_extra_includes_rawpy() -> None:
    """The serve runtime MUST ship rawpy.

    `pixsage serve` renders thumbnails via pixsage.images.load_image, which
    routes RAW extensions (.arw/.cr2/.nef/...) through rawpy. A catalog of
    Sony .ARW files served without rawpy raises ModuleNotFoundError on every
    RAW /thumb request (HTTP 500, broken images). rawpy is therefore a
    serve-time dependency, not tag/embed-only — do not drop it from [serve]
    to "keep the runtime minimal".
    """
    serve = _serve_extra()
    assert any(dep.lower().startswith("rawpy") for dep in serve), (
        f"rawpy missing from [project.optional-dependencies].serve: {serve}"
    )
