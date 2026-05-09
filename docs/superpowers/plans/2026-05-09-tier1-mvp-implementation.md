# Tier 1 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working `pixsage tag` CLI that walks a photographer's photo root, runs Florence-2 + RAM++ tagging, writes XMP keywords/captions Lightroom can read, and tracks state in a forward-compatible SQLite catalog.

**Architecture:** Six logical components (walker, image loader, taggers, vocabulary filter, XMP writer, catalog) wired together by a CLI orchestrator. Bottom-up build: utilities → data layer → I/O → orchestrator → real models. Mock taggers let us prove the end-to-end pipeline before pulling in `torch` + Florence-2 + RAM++.

**Tech Stack:** Python ≥3.11, PyTorch, transformers (Florence-2), recognize-anything (RAM++), Pillow + pillow-heif, rawpy, pydantic, typer, tqdm, SQLite (stdlib), exiftool (runtime binary). Test framework: pytest.

**Spec:** `docs/superpowers/specs/2026-05-09-tier1-mvp-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, entrypoint, ruff config |
| `src/pixsage/__init__.py` | Package marker, version |
| `src/pixsage/device.py` | CUDA → MPS → CPU detection |
| `src/pixsage/config.py` | TOML loader, pydantic schema, default config generation |
| `src/pixsage/vocabulary.py` | Pure-function tag filter (threshold, exclude, hierarchy override) |
| `src/pixsage/walker.py` | Directory walk + sha256 + sample selection |
| `src/pixsage/images.py` | Decode JPEG/HEIC/raw, resize to 1024px |
| `src/pixsage/catalog.py` | SQLite schema, upserts, user-rejection tracking |
| `src/pixsage/xmp.py` | Pure merge logic + exiftool subprocess wrapper |
| `src/pixsage/taggers/__init__.py` | Re-exports base types |
| `src/pixsage/taggers/base.py` | `Tag` dataclass + `Tagger` protocol |
| `src/pixsage/taggers/mock.py` | Test-only deterministic tagger |
| `src/pixsage/taggers/florence2.py` | Florence-2 wrapper |
| `src/pixsage/taggers/ramplusplus.py` | RAM++ wrapper |
| `src/pixsage/cli.py` | typer app, orchestrates the pipeline |
| `tests/test_device.py` | Device selection unit tests |
| `tests/test_config.py` | Config loader/validator unit tests |
| `tests/test_vocabulary.py` | Filter unit tests |
| `tests/test_walker.py` | Walker + sha256 unit tests |
| `tests/test_images.py` | Image loader unit tests |
| `tests/test_catalog.py` | Catalog ops unit tests |
| `tests/test_xmp.py` | XMP merge + exiftool integration tests |
| `tests/test_taggers_mock.py` | Mock tagger sanity tests |
| `tests/test_cli.py` | End-to-end CLI tests (mocked taggers) |
| `tests/conftest.py` | Shared fixtures (synthetic image factory, tmp catalog) |
| `tests/demo_corpus_urls.txt` | Curated URL list for the demo corpus fetcher |
| `scripts/fetch_demo_corpus.py` | Downloads ~20 public photos for integration testing |
| `README.md` | Install, exiftool prereq, usage, smoke test instructions |

The taggers package isolates the heavy dependencies — early tasks don't need to install `torch` or model weights.

---

## Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `src/pixsage/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pixsage"
version = "0.1.0"
description = "Photographer's photo corpus pipeline — Tier 1 MVP."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
  "pydantic>=2.5",
  "typer>=0.12",
  "tqdm>=4.66",
  "pillow>=10.0",
  "pillow-heif>=0.16",
]

[project.optional-dependencies]
taggers = [
  "torch>=2.2",
  "transformers>=4.40",
  "rawpy>=0.21",
  "recognize-anything @ git+https://github.com/xinyu1205/recognize-anything.git",
]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.4",
]

[project.scripts]
pixsage = "pixsage.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/pixsage"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create `src/pixsage/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create `tests/__init__.py` (empty)**

```python
```

- [ ] **Step 4: Create `tests/conftest.py` with shared fixtures**

```python
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_jpeg(tmp_path: Path):
    """Factory: write a synthetic JPEG to tmp_path/<name>.jpg, return the path."""
    def _make(name: str = "img.jpg", size: tuple[int, int] = (800, 600), color: str = "red") -> Path:
        path = tmp_path / name
        img = Image.new("RGB", size, color=color)
        img.save(path, format="JPEG", quality=85)
        return path
    return _make


@pytest.fixture
def photo_root(tmp_path: Path) -> Path:
    """An empty photo root with a .photoindex/ subdirectory."""
    root = tmp_path / "photos"
    root.mkdir()
    (root / ".photoindex").mkdir()
    return root
```

- [ ] **Step 5: Install editable + dev deps**

```bash
pip install -e ".[dev]"
```

Expected: clean install, no errors. `pixsage` command is on PATH but doesn't do anything yet.

- [ ] **Step 6: Verify pytest discovers no tests yet**

```bash
pytest
```

Expected: `no tests ran in ...s`. Exit code 5 (no tests collected) is fine.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/pixsage/__init__.py tests/__init__.py tests/conftest.py
git commit -m "scaffold: pyproject.toml, package layout, shared test fixtures"
```

---

## Task 2: Device selection

**Files:**
- Create: `src/pixsage/device.py`
- Create: `tests/test_device.py`

- [ ] **Step 1: Write the failing test**

`tests/test_device.py`:

```python
from __future__ import annotations

from unittest.mock import patch

from pixsage.device import select_device


def test_select_device_prefers_cuda():
    with patch("pixsage.device._cuda_available", return_value=True), \
         patch("pixsage.device._mps_available", return_value=True):
        assert select_device() == "cuda"


def test_select_device_falls_back_to_mps():
    with patch("pixsage.device._cuda_available", return_value=False), \
         patch("pixsage.device._mps_available", return_value=True):
        assert select_device() == "mps"


def test_select_device_falls_back_to_cpu():
    with patch("pixsage.device._cuda_available", return_value=False), \
         patch("pixsage.device._mps_available", return_value=False):
        assert select_device() == "cpu"
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_device.py -v
```

Expected: `ModuleNotFoundError: No module named 'pixsage.device'`

- [ ] **Step 3: Implement `src/pixsage/device.py`**

```python
from __future__ import annotations


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def select_device() -> str:
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_device.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/device.py tests/test_device.py
git commit -m "feat(device): CUDA → MPS → CPU selection"
```

---

## Task 3: Config schema + loader

**Files:**
- Create: `src/pixsage/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from pixsage.config import Config, ensure_default_config, load_config


def test_load_config_minimal(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    p.write_text("""
[florence2]
enabled = true
confidence_threshold = 0.5
exclude = []

[ram_plus_plus]
enabled = true
confidence_threshold = 0.4
exclude = []
""")
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.florence2.enabled is True
    assert cfg.florence2.confidence_threshold == 0.5
    assert cfg.ram_plus_plus.confidence_threshold == 0.4


def test_load_config_with_hierarchy_overrides(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    p.write_text("""
[florence2]
enabled = true
confidence_threshold = 0.5
exclude = ["x"]

[ram_plus_plus]
enabled = false
confidence_threshold = 0.4
exclude = []

[hierarchy_overrides]
"penguin" = "Wildlife|Bird|Penguin"
""")
    cfg = load_config(p)
    assert cfg.ram_plus_plus.enabled is False
    assert cfg.hierarchy_overrides == {"penguin": "Wildlife|Bird|Penguin"}


def test_load_config_invalid_threshold_raises(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    p.write_text("""
[florence2]
enabled = true
confidence_threshold = "high"
exclude = []

[ram_plus_plus]
enabled = true
confidence_threshold = 0.5
exclude = []
""")
    with pytest.raises(ValueError):
        load_config(p)


def test_ensure_default_config_creates_file(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    assert not p.exists()
    ensure_default_config(p)
    assert p.exists()
    cfg = load_config(p)
    assert cfg.florence2.enabled is True
    assert cfg.ram_plus_plus.enabled is True


def test_ensure_default_config_does_not_overwrite(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    p.write_text('[florence2]\nenabled = false\nconfidence_threshold = 0.9\nexclude = []\n\n[ram_plus_plus]\nenabled = false\nconfidence_threshold = 0.9\nexclude = []\n')
    ensure_default_config(p)
    cfg = load_config(p)
    assert cfg.florence2.enabled is False  # untouched
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_config.py -v
```

Expected: ImportError on `pixsage.config`.

- [ ] **Step 3: Implement `src/pixsage/config.py`**

```python
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class TaggerConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    exclude: list[str] = Field(default_factory=list)


class CaptionConfig(BaseModel):
    enabled: bool = True
    overwrite: bool = False


class Config(BaseModel):
    florence2: TaggerConfig
    ram_plus_plus: TaggerConfig
    hierarchy_overrides: dict[str, str] = Field(default_factory=dict)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)


DEFAULT_CONFIG_TOML = """\
# pixsage vocabulary configuration. Edit and re-run `pixsage tag --force` to apply.

[florence2]
enabled = true
confidence_threshold = 0.5
exclude = ["photograph", "image", "picture"]

[ram_plus_plus]
enabled = true
confidence_threshold = 0.4
exclude = []

[hierarchy_overrides]
# flat tag (lowercase) = "Top|Mid|Leaf"
# example:
# "penguin" = "Wildlife|Bird|Penguin"

[caption]
enabled = true
overwrite = false
"""


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        data = tomllib.load(f)
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid config at {path}: {e}") from e


def ensure_default_config(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/config.py tests/test_config.py
git commit -m "feat(config): pydantic schema, TOML loader, default config writer"
```

---

## Task 4: Vocabulary filter

**Files:**
- Create: `src/pixsage/vocabulary.py`
- Create: `tests/test_vocabulary.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_vocabulary.py`:

```python
from __future__ import annotations

from pixsage.config import CaptionConfig, Config, TaggerConfig
from pixsage.taggers.base import Tag
from pixsage.vocabulary import filter_tags


def make_config(
    fl_enabled=True, fl_threshold=0.5, fl_exclude=None,
    ram_enabled=True, ram_threshold=0.4, ram_exclude=None,
    hierarchy_overrides=None,
):
    return Config(
        florence2=TaggerConfig(enabled=fl_enabled, confidence_threshold=fl_threshold, exclude=fl_exclude or []),
        ram_plus_plus=TaggerConfig(enabled=ram_enabled, confidence_threshold=ram_threshold, exclude=ram_exclude or []),
        hierarchy_overrides=hierarchy_overrides or {},
        caption=CaptionConfig(),
    )


def test_filter_drops_below_threshold():
    cfg = make_config(fl_threshold=0.6)
    tags = [
        Tag(name="penguin", confidence=0.7, hierarchy=None, source="florence2"),
        Tag(name="ice", confidence=0.5, hierarchy=None, source="florence2"),
    ]
    out = filter_tags(tags, cfg)
    assert [t.name for t in out] == ["penguin"]


def test_filter_drops_excluded_case_insensitive():
    cfg = make_config(fl_exclude=["Photograph"])
    tags = [
        Tag(name="photograph", confidence=1.0, hierarchy=None, source="florence2"),
        Tag(name="penguin", confidence=1.0, hierarchy=None, source="florence2"),
    ]
    out = filter_tags(tags, cfg)
    assert [t.name for t in out] == ["penguin"]


def test_filter_disables_tagger():
    cfg = make_config(ram_enabled=False)
    tags = [
        Tag(name="penguin", confidence=1.0, hierarchy=None, source="florence2"),
        Tag(name="bird", confidence=1.0, hierarchy=None, source="ram++"),
    ]
    out = filter_tags(tags, cfg)
    assert [t.source for t in out] == ["florence2"]


def test_filter_applies_hierarchy_override():
    cfg = make_config(hierarchy_overrides={"penguin": "Wildlife|Bird|Penguin"})
    tags = [
        Tag(name="Penguin", confidence=1.0, hierarchy=None, source="florence2"),
    ]
    out = filter_tags(tags, cfg)
    assert out[0].hierarchy == "Wildlife|Bird|Penguin"


def test_filter_preserves_existing_hierarchy_when_no_override():
    cfg = make_config()
    tags = [
        Tag(name="penguin", confidence=1.0, hierarchy="Wildlife|Bird|Penguin", source="florence2"),
    ]
    out = filter_tags(tags, cfg)
    assert out[0].hierarchy == "Wildlife|Bird|Penguin"
```

- [ ] **Step 2: Stub `src/pixsage/taggers/__init__.py` and `taggers/base.py` minimally**

(We need `Tag` to import in tests now; full `Tagger` protocol comes in Task 11.)

`src/pixsage/taggers/__init__.py`:

```python
from pixsage.taggers.base import Tag

__all__ = ["Tag"]
```

`src/pixsage/taggers/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tag:
    name: str
    confidence: float
    hierarchy: str | None
    source: str  # "florence2" | "ram++"
```

- [ ] **Step 3: Run test, verify fail**

```bash
pytest tests/test_vocabulary.py -v
```

Expected: ImportError on `pixsage.vocabulary`.

- [ ] **Step 4: Implement `src/pixsage/vocabulary.py`**

```python
from __future__ import annotations

from pixsage.config import Config, TaggerConfig
from pixsage.taggers.base import Tag


def filter_tags(tags: list[Tag], config: Config) -> list[Tag]:
    out: list[Tag] = []
    overrides = {k.lower(): v for k, v in config.hierarchy_overrides.items()}
    for tag in tags:
        cfg = _config_for_source(tag.source, config)
        if cfg is None or not cfg.enabled:
            continue
        if tag.confidence < cfg.confidence_threshold:
            continue
        if any(tag.name.lower() == ex.lower() for ex in cfg.exclude):
            continue
        hierarchy = overrides.get(tag.name.lower(), tag.hierarchy)
        out.append(Tag(name=tag.name, confidence=tag.confidence, hierarchy=hierarchy, source=tag.source))
    return out


def _config_for_source(source: str, config: Config) -> TaggerConfig | None:
    if source == "florence2":
        return config.florence2
    if source == "ram++":
        return config.ram_plus_plus
    return None
```

- [ ] **Step 5: Run test, verify pass**

```bash
pytest tests/test_vocabulary.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/vocabulary.py src/pixsage/taggers/__init__.py src/pixsage/taggers/base.py tests/test_vocabulary.py
git commit -m "feat(vocabulary): per-source filter with thresholds, exclusions, hierarchy overrides"
```

---

## Task 5: Walker + sha256

**Files:**
- Create: `src/pixsage/walker.py`
- Create: `tests/test_walker.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_walker.py`:

```python
from __future__ import annotations

from pathlib import Path

from pixsage.walker import IMAGE_EXTENSIONS, sample_paths, sha256_file, walk_photos


def test_sha256_file_known_input(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    # known sha256("hello world")
    assert sha256_file(p) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_walk_photos_finds_images(tmp_path: Path, make_jpeg):
    make_jpeg("a.jpg")
    make_jpeg("b.JPG")
    (tmp_path / "notes.txt").write_text("hi")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.tif").write_bytes(b"\x00")  # not a real tif but extension matches
    found = sorted(p.name.lower() for p in walk_photos(tmp_path))
    assert found == ["a.jpg", "b.jpg", "c.tif"]


def test_walk_photos_skips_photoindex_dir(tmp_path: Path, make_jpeg):
    make_jpeg("ok.jpg")
    idx = tmp_path / ".photoindex"
    idx.mkdir()
    # Even if a stray jpeg ends up under .photoindex/, we ignore it.
    Path(idx / "ignored.jpg").write_bytes(b"\xff\xd8\xff")
    found = [p.name for p in walk_photos(tmp_path)]
    assert found == ["ok.jpg"]


def test_image_extensions_includes_common_raws_and_jpegs():
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".heic" in IMAGE_EXTENSIONS
    assert ".arw" in IMAGE_EXTENSIONS
    assert ".cr3" in IMAGE_EXTENSIONS
    assert ".nef" in IMAGE_EXTENSIONS
    assert ".dng" in IMAGE_EXTENSIONS


def test_sample_paths_deterministic(tmp_path: Path):
    paths = [tmp_path / f"{i:03d}.jpg" for i in range(20)]
    hashes = {p: f"{i:064x}" for i, p in enumerate(paths)}
    sampled1 = sample_paths(paths, hashes, n=5)
    sampled2 = sample_paths(paths, hashes, n=5)
    assert sampled1 == sampled2
    assert len(sampled1) == 5


def test_sample_paths_caps_at_total(tmp_path: Path):
    paths = [tmp_path / f"{i}.jpg" for i in range(3)]
    hashes = {p: f"{i:064x}" for i, p in enumerate(paths)}
    assert len(sample_paths(paths, hashes, n=99)) == 3
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_walker.py -v
```

Expected: ImportError on `pixsage.walker`.

- [ ] **Step 3: Implement `src/pixsage/walker.py`**

```python
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg",
    ".tif", ".tiff",
    ".heic", ".heif",
    ".png",
    # raws
    ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".dng",
})

PHOTOINDEX_DIR = ".photoindex"

CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def walk_photos(root: Path) -> Iterator[Path]:
    """Yield every image file under root, skipping .photoindex/."""
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if PHOTOINDEX_DIR in p.parts:
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def sample_paths(paths: list[Path], hashes: dict[Path, str], n: int) -> list[Path]:
    """Deterministic sample: sort by sha256, take first n."""
    return sorted(paths, key=lambda p: hashes[p])[:n]
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_walker.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/walker.py tests/test_walker.py
git commit -m "feat(walker): file walking, sha256 streaming hash, deterministic sampling"
```

---

## Task 6: Image loader (non-raw)

**Files:**
- Create: `src/pixsage/images.py`
- Create: `tests/test_images.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_images.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pixsage.images import LONG_EDGE_TARGET, load_image


def test_load_jpeg_returns_rgb(make_jpeg):
    p = make_jpeg("a.jpg", size=(2000, 1500), color="green")
    img = load_image(p)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_image_resizes_long_edge(make_jpeg):
    p = make_jpeg("big.jpg", size=(4000, 1000))
    img = load_image(p)
    assert max(img.size) == LONG_EDGE_TARGET


def test_load_image_preserves_aspect_ratio(make_jpeg):
    p = make_jpeg("wide.jpg", size=(2000, 500))
    img = load_image(p)
    w, h = img.size
    assert abs(w / h - 2000 / 500) < 0.02


def test_load_image_does_not_upscale(make_jpeg):
    p = make_jpeg("small.jpg", size=(400, 300))
    img = load_image(p)
    assert img.size == (400, 300)


def test_load_image_unknown_extension_raises(tmp_path: Path):
    p = tmp_path / "x.xyz"
    p.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(p)
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_images.py -v
```

Expected: ImportError on `pixsage.images`.

- [ ] **Step 3: Implement `src/pixsage/images.py`**

```python
from __future__ import annotations

from pathlib import Path

from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

LONG_EDGE_TARGET = 1024

NON_RAW_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif", ".png",
})

RAW_EXTENSIONS: frozenset[str] = frozenset({
    ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".dng",
})


def load_image(path: Path) -> Image.Image:
    ext = path.suffix.lower()
    if ext in NON_RAW_EXTENSIONS:
        img = Image.open(path)
    elif ext in RAW_EXTENSIONS:
        img = _load_raw(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    img = img.convert("RGB")
    return _resize_long_edge(img, LONG_EDGE_TARGET)


def _load_raw(path: Path) -> Image.Image:
    # Implemented in Task 7.
    raise NotImplementedError("Raw loading not yet implemented")


def _resize_long_edge(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= target:
        return img
    scale = target / long_edge
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_images.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/images.py tests/test_images.py
git commit -m "feat(images): non-raw decode + long-edge resize to 1024px"
```

---

## Task 7: Image loader (raw via rawpy)

**Files:**
- Modify: `src/pixsage/images.py`
- Modify: `tests/test_images.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_images.py`:

```python
RAW_FIXTURE = Path(__file__).parent / "fixtures" / "images" / "sample.arw"


@pytest.mark.skipif(not RAW_FIXTURE.exists(), reason="raw fixture not present")
def test_load_raw_returns_rgb():
    img = load_image(RAW_FIXTURE)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert max(img.size) <= LONG_EDGE_TARGET
```

The path is `tests/fixtures/images/sample.arw`. If you have access to a small Sony or Canon raw, drop it there. Otherwise the test is skipped.

- [ ] **Step 2: Run test, verify it skips (or fails if rawpy not installed)**

```bash
pip install -e ".[taggers]"   # installs torch + rawpy + others
pytest tests/test_images.py -v
```

Expected: `test_load_raw_returns_rgb SKIPPED` (no fixture) — other tests still pass.

- [ ] **Step 3: Replace the `_load_raw` stub**

In `src/pixsage/images.py`, replace `_load_raw`:

```python
def _load_raw(path: Path) -> Image.Image:
    import rawpy
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except rawpy.LibRawNoThumbnailError:
            # fall back: develop the raw (slow, but ensures we can always load something)
            rgb = raw.postprocess(no_auto_bright=True, output_bps=8)
            return Image.fromarray(rgb, mode="RGB")
    if thumb.format == rawpy.ThumbFormat.JPEG:
        from io import BytesIO
        return Image.open(BytesIO(thumb.data))
    # rawpy.ThumbFormat.BITMAP
    return Image.fromarray(thumb.data, mode="RGB")
```

- [ ] **Step 4: Run test, verify pass or skip**

```bash
pytest tests/test_images.py -v
```

Expected: all non-raw tests pass; raw test skipped if no fixture, passes if fixture present.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/images.py tests/test_images.py
git commit -m "feat(images): raw decode via rawpy embedded thumbnail (with develop fallback)"
```

---

## Task 8: Catalog — photos table

**Files:**
- Create: `src/pixsage/catalog.py`
- Create: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_catalog.py`:

```python
from __future__ import annotations

from pathlib import Path

from pixsage.catalog import Catalog


def test_catalog_init_creates_schema(tmp_path: Path):
    db_path = tmp_path / "catalog.db"
    cat = Catalog(db_path)
    cat.init_schema()
    cat.close()
    assert db_path.exists()


def test_upsert_photo_inserts_row(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    cat.upsert_photo(sha256="a" * 64, path=tmp_path / "x.jpg", filesize=100, mtime=1.0)
    row = cat.get_photo("a" * 64)
    assert row is not None
    assert row["filename"] == "x.jpg"
    assert row["filesize"] == 100
    assert row["last_tagged_at"] is None
    assert row["model_versions"] is None
    cat.close()


def test_upsert_photo_updates_last_seen(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "b" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=100, mtime=1.0)
    first_seen = cat.get_photo(sha)["last_seen_at"]
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=100, mtime=2.0)
    second_seen = cat.get_photo(sha)["last_seen_at"]
    assert second_seen >= first_seen
    cat.close()


def test_mark_tagged_records_versions(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "c" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=10, mtime=1.0)
    cat.mark_tagged(sha, model_versions={"florence2": "1.0", "ram++": "1.0"})
    row = cat.get_photo(sha)
    assert row["last_tagged_at"] is not None
    assert "florence2" in row["model_versions"]
    cat.close()


def test_needs_tagging_logic(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "d" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=10, mtime=1.0)
    versions = {"florence2": "1.0", "ram++": "1.0"}
    assert cat.needs_tagging(sha, versions) is True
    cat.mark_tagged(sha, versions)
    assert cat.needs_tagging(sha, versions) is False
    assert cat.needs_tagging(sha, {"florence2": "2.0", "ram++": "1.0"}) is True
    cat.close()
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_catalog.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/pixsage/catalog.py` (photos table only for now)**

```python
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_PHOTOS = """
CREATE TABLE IF NOT EXISTS photos (
  sha256 TEXT PRIMARY KEY,
  current_path TEXT,
  filename TEXT,
  filesize INTEGER,
  mtime REAL,
  last_tagged_at TEXT,
  model_versions TEXT,
  added_at TEXT,
  last_seen_at TEXT,
  error_reason TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA_PHOTOS)

    def close(self) -> None:
        self._conn.close()

    def upsert_photo(self, sha256: str, path: Path, filesize: int, mtime: float) -> None:
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO photos (sha256, current_path, filename, filesize, mtime, added_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    current_path = excluded.current_path,
                    filename = excluded.filename,
                    filesize = excluded.filesize,
                    mtime = excluded.mtime,
                    last_seen_at = excluded.last_seen_at
                """,
                (sha256, str(path), path.name, filesize, mtime, now, now),
            )

    def get_photo(self, sha256: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM photos WHERE sha256 = ?", (sha256,))
        row = cur.fetchone()
        return dict(row) if row else None

    def mark_tagged(self, sha256: str, model_versions: dict[str, str]) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE photos SET last_tagged_at = ?, model_versions = ?, error_reason = NULL WHERE sha256 = ?",
                (_now(), json.dumps(model_versions, sort_keys=True), sha256),
            )

    def mark_error(self, sha256: str, reason: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE photos SET error_reason = ? WHERE sha256 = ?",
                (reason, sha256),
            )

    def needs_tagging(self, sha256: str, model_versions: dict[str, str]) -> bool:
        row = self.get_photo(sha256)
        if row is None:
            return True
        if row["last_tagged_at"] is None:
            return True
        if row["model_versions"] is None:
            return True
        existing = json.loads(row["model_versions"])
        return existing != model_versions
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_catalog.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): photos table, upsert, needs_tagging, mark_tagged/error"
```

---

## Task 9: Catalog — tags table + user-rejection

**Files:**
- Modify: `src/pixsage/catalog.py`
- Modify: `tests/test_catalog.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_catalog.py`:

```python
from pixsage.taggers.base import Tag


def test_record_tags_inserts_rows(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "e" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=1, mtime=1.0)
    tags = [
        Tag("penguin", 1.0, "Wildlife|Bird|Penguin", "florence2"),
        Tag("bird", 0.9, None, "ram++"),
    ]
    cat.record_tags(sha, tags)
    stored = cat.get_tags(sha)
    assert {t.name for t in stored} == {"penguin", "bird"}
    cat.close()


def test_record_tags_idempotent(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "f" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=1, mtime=1.0)
    tags = [Tag("penguin", 1.0, None, "florence2")]
    cat.record_tags(sha, tags)
    cat.record_tags(sha, tags)
    assert len(cat.get_tags(sha)) == 1
    cat.close()


def test_user_rejected_flagging(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "0" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=1, mtime=1.0)
    cat.record_tags(sha, [
        Tag("penguin", 1.0, None, "florence2"),
        Tag("ice", 0.8, None, "florence2"),
    ])
    # Pretend the user removed "ice" from XMP. We pass the surviving set:
    cat.flag_user_rejections(sha, surviving_xmp_tags={"penguin"})
    rejected = cat.get_user_rejected(sha)
    assert rejected == {("ice", "florence2")}
    not_rejected = {t.name for t in cat.get_tags(sha) if not cat.is_user_rejected(sha, t.name, t.source)}
    assert not_rejected == {"penguin"}
    cat.close()


def test_user_rejected_persists_across_record(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "1" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=1, mtime=1.0)
    cat.record_tags(sha, [Tag("ice", 1.0, None, "florence2")])
    cat.flag_user_rejections(sha, surviving_xmp_tags=set())
    # Re-record: should NOT clear the rejection flag.
    cat.record_tags(sha, [Tag("ice", 1.0, None, "florence2")])
    assert cat.is_user_rejected(sha, "ice", "florence2") is True
    cat.close()
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_catalog.py::test_record_tags_inserts_rows -v
```

Expected: AttributeError on `record_tags`.

- [ ] **Step 3: Extend `src/pixsage/catalog.py`**

Add this schema script and append it:

```python
SCHEMA_TAGS = """
CREATE TABLE IF NOT EXISTS tags (
  sha256 TEXT NOT NULL,
  tag TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL,
  hierarchy TEXT,
  user_rejected INTEGER NOT NULL DEFAULT 0,
  applied_at TEXT,
  PRIMARY KEY (sha256, tag, source),
  FOREIGN KEY (sha256) REFERENCES photos(sha256) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tags_sha256 ON tags(sha256);
CREATE INDEX IF NOT EXISTS idx_tags_source ON tags(source);
"""
```

Update `init_schema` to run both:

```python
def init_schema(self) -> None:
    with self._conn:
        self._conn.executescript(SCHEMA_PHOTOS)
        self._conn.executescript(SCHEMA_TAGS)
```

Add tag methods to `Catalog`:

```python
    def record_tags(self, sha256: str, tags: list["Tag"]) -> None:
        from pixsage.taggers.base import Tag  # local import to keep catalog import-light
        now = _now()
        with self._conn:
            for t in tags:
                self._conn.execute(
                    """
                    INSERT INTO tags (sha256, tag, source, confidence, hierarchy, applied_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sha256, tag, source) DO UPDATE SET
                        confidence = excluded.confidence,
                        hierarchy = excluded.hierarchy,
                        applied_at = excluded.applied_at
                    """,
                    (sha256, t.name, t.source, t.confidence, t.hierarchy, now),
                )

    def get_tags(self, sha256: str) -> list["Tag"]:
        from pixsage.taggers.base import Tag
        cur = self._conn.execute(
            "SELECT tag, confidence, hierarchy, source FROM tags WHERE sha256 = ?",
            (sha256,),
        )
        return [Tag(name=r["tag"], confidence=r["confidence"] or 0.0, hierarchy=r["hierarchy"], source=r["source"]) for r in cur]

    def get_previously_applied(self, sha256: str) -> set[tuple[str, str]]:
        cur = self._conn.execute(
            "SELECT tag, source FROM tags WHERE sha256 = ?",
            (sha256,),
        )
        return {(r["tag"], r["source"]) for r in cur}

    def flag_user_rejections(self, sha256: str, surviving_xmp_tags: set[str]) -> None:
        """Any tag we previously applied that's NOT in surviving_xmp_tags becomes user_rejected."""
        with self._conn:
            cur = self._conn.execute(
                "SELECT tag, source FROM tags WHERE sha256 = ?",
                (sha256,),
            )
            for r in cur.fetchall():
                if r["tag"] not in surviving_xmp_tags:
                    self._conn.execute(
                        "UPDATE tags SET user_rejected = 1 WHERE sha256 = ? AND tag = ? AND source = ?",
                        (sha256, r["tag"], r["source"]),
                    )

    def is_user_rejected(self, sha256: str, tag: str, source: str) -> bool:
        cur = self._conn.execute(
            "SELECT user_rejected FROM tags WHERE sha256 = ? AND tag = ? AND source = ?",
            (sha256, tag, source),
        )
        row = cur.fetchone()
        return bool(row and row["user_rejected"])

    def get_user_rejected(self, sha256: str) -> set[tuple[str, str]]:
        cur = self._conn.execute(
            "SELECT tag, source FROM tags WHERE sha256 = ? AND user_rejected = 1",
            (sha256,),
        )
        return {(r["tag"], r["source"]) for r in cur}
```

- [ ] **Step 4: Run all catalog tests**

```bash
pytest tests/test_catalog.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): tags table with user-rejection tracking"
```

---

## Task 10: Catalog — runs table

**Files:**
- Modify: `src/pixsage/catalog.py`
- Modify: `tests/test_catalog.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_catalog.py`:

```python
def test_runs_table_records_run(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    run_id = cat.start_run(config_hash="abc", model_versions={"florence2": "1.0"})
    assert isinstance(run_id, int)
    cat.finish_run(run_id, processed=5, skipped=2, errored=0)
    runs = cat.list_runs()
    assert len(runs) == 1
    assert runs[0]["photos_processed"] == 5
    assert runs[0]["photos_errored"] == 0
    cat.close()
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_catalog.py::test_runs_table_records_run -v
```

Expected: AttributeError.

- [ ] **Step 3: Extend `catalog.py` — runs schema + methods**

Add the schema:

```python
SCHEMA_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT,
  finished_at TEXT,
  photos_processed INTEGER,
  photos_skipped INTEGER,
  photos_errored INTEGER,
  config_hash TEXT,
  model_versions TEXT
);
"""
```

Update `init_schema`:

```python
def init_schema(self) -> None:
    with self._conn:
        self._conn.executescript(SCHEMA_PHOTOS)
        self._conn.executescript(SCHEMA_TAGS)
        self._conn.executescript(SCHEMA_RUNS)
```

Add methods:

```python
    def start_run(self, config_hash: str, model_versions: dict[str, str]) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs (started_at, config_hash, model_versions) VALUES (?, ?, ?)",
                (_now(), config_hash, json.dumps(model_versions, sort_keys=True)),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, processed: int, skipped: int, errored: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, photos_processed = ?, photos_skipped = ?, photos_errored = ? WHERE run_id = ?",
                (_now(), processed, skipped, errored, run_id),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        cur = self._conn.execute("SELECT * FROM runs ORDER BY run_id")
        return [dict(r) for r in cur]
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_catalog.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): runs table for per-run statistics"
```

---

## Task 11: Tagger protocol + mock tagger

**Files:**
- Modify: `src/pixsage/taggers/base.py`
- Create: `src/pixsage/taggers/mock.py`
- Create: `tests/test_taggers_mock.py`

- [ ] **Step 1: Add the `Tagger` protocol to `taggers/base.py`**

Replace `src/pixsage/taggers/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class Tag:
    name: str
    confidence: float
    hierarchy: str | None
    source: str  # "florence2" | "ram++"


@dataclass(frozen=True)
class TagResult:
    tags: list[Tag]
    caption: str | None  # only Florence-2 produces a caption in Phase 1


class Tagger(Protocol):
    name: str
    model_version: str

    def load(self, device: str) -> None: ...
    def tag(self, image: Image.Image) -> TagResult: ...
```

- [ ] **Step 2: Update `taggers/__init__.py`**

```python
from pixsage.taggers.base import Tag, Tagger, TagResult

__all__ = ["Tag", "Tagger", "TagResult"]
```

- [ ] **Step 3: Write the failing tests**

`tests/test_taggers_mock.py`:

```python
from __future__ import annotations

from PIL import Image

from pixsage.taggers.mock import MockTagger


def test_mock_tagger_returns_configured_tags():
    tagger = MockTagger(
        name="florence2",
        model_version="mock-1",
        tags_per_call=[("penguin", 1.0), ("ice", 0.9)],
        caption="A penguin on ice.",
    )
    tagger.load("cpu")
    img = Image.new("RGB", (10, 10))
    result = tagger.tag(img)
    assert {t.name for t in result.tags} == {"penguin", "ice"}
    assert all(t.source == "florence2" for t in result.tags)
    assert result.caption == "A penguin on ice."


def test_mock_tagger_no_caption():
    tagger = MockTagger(name="ram++", model_version="mock-1", tags_per_call=[("bird", 0.8)])
    tagger.load("cpu")
    result = tagger.tag(Image.new("RGB", (10, 10)))
    assert result.caption is None
    assert result.tags[0].source == "ram++"
```

- [ ] **Step 4: Run test, verify fail**

```bash
pytest tests/test_taggers_mock.py -v
```

Expected: ImportError.

- [ ] **Step 5: Implement `src/pixsage/taggers/mock.py`**

```python
from __future__ import annotations

from PIL import Image

from pixsage.taggers.base import Tag, TagResult


class MockTagger:
    """Deterministic tagger for testing the orchestrator without loading real models."""

    def __init__(
        self,
        name: str,
        model_version: str,
        tags_per_call: list[tuple[str, float]],
        caption: str | None = None,
    ):
        self.name = name
        self.model_version = model_version
        self._tags = tags_per_call
        self._caption = caption
        self._loaded = False

    def load(self, device: str) -> None:
        self._loaded = True

    def tag(self, image: Image.Image) -> TagResult:
        if not self._loaded:
            raise RuntimeError("MockTagger.load() not called")
        tags = [Tag(name=n, confidence=c, hierarchy=None, source=self.name) for n, c in self._tags]
        return TagResult(tags=tags, caption=self._caption)
```

- [ ] **Step 6: Run tests, verify pass**

```bash
pytest tests/test_taggers_mock.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/pixsage/taggers/base.py src/pixsage/taggers/mock.py src/pixsage/taggers/__init__.py tests/test_taggers_mock.py
git commit -m "feat(taggers): Tagger protocol, TagResult, MockTagger for tests"
```

---

## Task 12: XMP merge logic (pure)

**Files:**
- Create: `src/pixsage/xmp.py`
- Create: `tests/test_xmp.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_xmp.py`:

```python
from __future__ import annotations

from pixsage.taggers.base import Tag, TagResult
from pixsage.xmp import XmpFields, merge_xmp


def test_merge_adds_new_auto_tags():
    existing = XmpFields(subject=["antarctica"], hierarchical_subject=[], description=None)
    new = [
        Tag("penguin", 1.0, "Wildlife|Bird|Penguin", "florence2"),
    ]
    merged = merge_xmp(
        existing=existing,
        new_tags=new,
        previously_applied={("penguin", "florence2")},  # already in our DB
        user_rejected=set(),
        caption="A penguin.",
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert "penguin" in merged.subject
    assert "antarctica" in merged.subject
    assert "auto-tagged-florence2" in merged.subject
    assert "Wildlife|Bird|Penguin" in merged.hierarchical_subject
    assert merged.description == "A penguin."


def test_merge_preserves_user_keywords():
    existing = XmpFields(subject=["my keyword", "another"], hierarchical_subject=[], description=None)
    merged = merge_xmp(
        existing=existing,
        new_tags=[Tag("penguin", 1.0, None, "florence2")],
        previously_applied=set(),
        user_rejected=set(),
        caption=None,
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert "my keyword" in merged.subject
    assert "another" in merged.subject
    assert "penguin" in merged.subject


def test_merge_skips_user_rejected_tags():
    existing = XmpFields(subject=[], hierarchical_subject=[], description=None)
    merged = merge_xmp(
        existing=existing,
        new_tags=[
            Tag("penguin", 1.0, None, "florence2"),
            Tag("ice", 0.9, None, "florence2"),
        ],
        previously_applied={("penguin", "florence2"), ("ice", "florence2")},
        user_rejected={("ice", "florence2")},
        caption=None,
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert "penguin" in merged.subject
    assert "ice" not in merged.subject


def test_merge_does_not_overwrite_existing_description():
    existing = XmpFields(subject=[], hierarchical_subject=[], description="Photographer's caption")
    merged = merge_xmp(
        existing=existing,
        new_tags=[Tag("penguin", 1.0, None, "florence2")],
        previously_applied=set(),
        user_rejected=set(),
        caption="Auto caption",
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert merged.description == "Photographer's caption"


def test_merge_overwrites_when_configured():
    existing = XmpFields(subject=[], hierarchical_subject=[], description="old")
    merged = merge_xmp(
        existing=existing,
        new_tags=[],
        previously_applied=set(),
        user_rejected=set(),
        caption="new",
        caption_overwrite=True,
        sources_with_tags=set(),
    )
    assert merged.description == "new"


def test_merge_marker_tags_per_source():
    existing = XmpFields(subject=[], hierarchical_subject=[], description=None)
    merged = merge_xmp(
        existing=existing,
        new_tags=[
            Tag("penguin", 1.0, None, "florence2"),
            Tag("bird", 0.9, None, "ram++"),
        ],
        previously_applied=set(),
        user_rejected=set(),
        caption=None,
        caption_overwrite=False,
        sources_with_tags={"florence2", "ram++"},
    )
    assert "auto-tagged-florence2" in merged.subject
    assert "auto-tagged-ram" in merged.subject


def test_merge_no_marker_tag_when_source_has_no_new_tags():
    existing = XmpFields(subject=[], hierarchical_subject=[], description=None)
    merged = merge_xmp(
        existing=existing,
        new_tags=[Tag("penguin", 1.0, None, "florence2")],
        previously_applied=set(),
        user_rejected=set(),
        caption=None,
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert "auto-tagged-ram" not in merged.subject
```

- [ ] **Step 2: Run tests, verify fail**

```bash
pytest tests/test_xmp.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement merge logic in `src/pixsage/xmp.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from pixsage.taggers.base import Tag

MARKER_PREFIX = "auto-tagged-"


@dataclass(frozen=True)
class XmpFields:
    subject: list[str]
    hierarchical_subject: list[str]
    description: str | None


def _marker(source: str) -> str:
    short = source.replace("++", "")  # ram++ -> ram
    return f"{MARKER_PREFIX}{short}"


def merge_xmp(
    existing: XmpFields,
    new_tags: list[Tag],
    previously_applied: set[tuple[str, str]],
    user_rejected: set[tuple[str, str]],
    caption: str | None,
    caption_overwrite: bool,
    sources_with_tags: set[str],
) -> XmpFields:
    # Filter user-rejected from new tags.
    keepable = [t for t in new_tags if (t.name, t.source) not in user_rejected]

    # Subject = existing ∪ keepable ∪ markers.
    subject_set = list(dict.fromkeys(existing.subject))  # de-dupe, preserve order
    for t in keepable:
        if t.name not in subject_set:
            subject_set.append(t.name)
    for src in sorted(sources_with_tags):
        # Only emit a marker if at least one tag from that source survived.
        if any(t.source == src and (t.name, t.source) not in user_rejected for t in new_tags):
            m = _marker(src)
            if m not in subject_set:
                subject_set.append(m)

    # Hierarchical subject.
    hier = list(dict.fromkeys(existing.hierarchical_subject))
    for t in keepable:
        if t.hierarchy and t.hierarchy not in hier:
            hier.append(t.hierarchy)

    # Description.
    if caption is not None and (caption_overwrite or not existing.description):
        description = caption
    else:
        description = existing.description

    return XmpFields(subject=subject_set, hierarchical_subject=hier, description=description)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_xmp.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/xmp.py tests/test_xmp.py
git commit -m "feat(xmp): pure merge logic for keywords, hierarchy, caption, marker tags"
```

---

## Task 13: exiftool subprocess wrapper

**Files:**
- Modify: `src/pixsage/xmp.py`
- Modify: `tests/test_xmp.py`

This task requires `exiftool` on PATH. Verify before starting:

```bash
exiftool -ver
```

Expected: prints a version like `12.99`. If not installed:
- Windows: `winget install OliverBetz.ExifTool` or download from https://exiftool.org/
- macOS: `brew install exiftool`

- [ ] **Step 1: Add integration tests for exiftool round-trip**

Append to `tests/test_xmp.py`:

```python
import shutil
from pathlib import Path

import pytest

from pixsage.xmp import read_xmp, write_xmp

EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")


@needs_exiftool
def test_write_and_read_jpeg(make_jpeg):
    p = make_jpeg("rt.jpg")
    fields = XmpFields(
        subject=["penguin", "ice"],
        hierarchical_subject=["Wildlife|Bird|Penguin"],
        description="A penguin on ice.",
    )
    write_xmp(p, fields, is_raw=False)
    got = read_xmp(p, is_raw=False)
    assert set(got.subject) >= {"penguin", "ice"}
    assert "Wildlife|Bird|Penguin" in got.hierarchical_subject
    assert got.description == "A penguin on ice."


@needs_exiftool
def test_write_raw_uses_sidecar(tmp_path: Path):
    # We don't need a real raw — exiftool will create a sidecar even from a fake path
    # as long as we tell it to write to <path>.xmp explicitly.
    fake_raw = tmp_path / "fake.arw"
    fake_raw.write_bytes(b"\x00")  # contents irrelevant; exiftool only reads/writes the sidecar
    fields = XmpFields(subject=["penguin"], hierarchical_subject=[], description=None)
    write_xmp(fake_raw, fields, is_raw=True)
    sidecar = tmp_path / "fake.xmp"
    assert sidecar.exists()
    got = read_xmp(fake_raw, is_raw=True)
    assert "penguin" in got.subject


@needs_exiftool
def test_read_xmp_returns_empty_when_no_sidecar(tmp_path: Path):
    p = tmp_path / "no_sidecar.arw"
    p.write_bytes(b"\x00")
    fields = read_xmp(p, is_raw=True)
    assert fields.subject == []
    assert fields.hierarchical_subject == []
    assert fields.description is None
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_xmp.py -k "write_and_read_jpeg or write_raw_uses_sidecar or read_xmp_returns_empty" -v
```

Expected: AttributeError on `write_xmp` / `read_xmp`.

- [ ] **Step 3: Add exiftool wrapper to `src/pixsage/xmp.py`**

Append to `src/pixsage/xmp.py`:

```python
import json
import shutil
import subprocess
from pathlib import Path

EXIFTOOL = shutil.which("exiftool") or "exiftool"


def _sidecar_path(raw_path: Path) -> Path:
    """Lightroom sidecar convention: DSC_0001.ARW -> DSC_0001.xmp."""
    return raw_path.with_suffix(".xmp")


def read_xmp(path: Path, is_raw: bool) -> XmpFields:
    target = _sidecar_path(path) if is_raw else path
    if is_raw and not target.exists():
        return XmpFields(subject=[], hierarchical_subject=[], description=None)
    cmd = [
        EXIFTOOL,
        "-json",
        "-XMP-dc:Subject",
        "-XMP-lr:HierarchicalSubject",
        "-XMP-dc:Description",
        str(target),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"exiftool read failed: {e.stderr}") from e
    data = json.loads(result.stdout) if result.stdout.strip() else [{}]
    if not data:
        return XmpFields(subject=[], hierarchical_subject=[], description=None)
    record = data[0]
    return XmpFields(
        subject=_to_list(record.get("Subject")),
        hierarchical_subject=_to_list(record.get("HierarchicalSubject")),
        description=record.get("Description"),
    )


def _to_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def write_xmp(path: Path, fields: XmpFields, is_raw: bool) -> None:
    target = _sidecar_path(path) if is_raw else path
    args = [EXIFTOOL, "-overwrite_original", "-charset", "utf8"]
    args += [f"-XMP-dc:Subject={s}" for s in []]  # placeholder; we'll clear+set below
    # Clear and re-set to ensure exact set semantics for these fields:
    args = [
        EXIFTOOL,
        "-overwrite_original",
        "-charset", "utf8",
        "-XMP-dc:Subject=",
        "-XMP-lr:HierarchicalSubject=",
    ]
    for s in fields.subject:
        args.append(f"-XMP-dc:Subject+={s}")
    for h in fields.hierarchical_subject:
        args.append(f"-XMP-lr:HierarchicalSubject+={h}")
    if fields.description is not None:
        args.append(f"-XMP-dc:Description={fields.description}")
    if is_raw:
        # Write to sidecar explicitly. exiftool's -o flag creates a new file.
        if target.exists():
            args.append(str(target))
        else:
            # Initialize sidecar by writing to it directly.
            args.append("-o")
            args.append(str(target))
    else:
        args.append(str(path))
    try:
        subprocess.run(args, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"exiftool write failed: {e.stderr}") from e
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_xmp.py -v
```

Expected: all tests pass (exiftool tests skipped only if exiftool isn't installed).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/xmp.py tests/test_xmp.py
git commit -m "feat(xmp): exiftool subprocess wrapper for embedded + sidecar I/O"
```

---

## Task 14: CLI orchestrator (with mock taggers)

**Files:**
- Create: `src/pixsage/cli.py`
- Create: `tests/test_cli.py`

This task wires every component together using `MockTagger`. Real models come in Tasks 15–16.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pixsage.catalog import Catalog
from pixsage.cli import app
from pixsage.taggers.mock import MockTagger
from pixsage.xmp import read_xmp


runner = CliRunner()


@pytest.fixture(autouse=True)
def use_mock_taggers(monkeypatch):
    def fake_build_taggers(_config):
        return [
            MockTagger(name="florence2", model_version="mock-1", tags_per_call=[("penguin", 1.0)], caption="A penguin."),
            MockTagger(name="ram++", model_version="mock-1", tags_per_call=[("bird", 0.9)]),
        ]
    monkeypatch.setattr("pixsage.cli.build_taggers", fake_build_taggers)


def test_tag_writes_xmp_and_catalog(tmp_path: Path, make_jpeg):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg")
    a.rename(photo_root / "a.jpg")
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0, result.stdout
    fields = read_xmp(photo_root / "a.jpg", is_raw=False)
    assert "penguin" in fields.subject
    assert "bird" in fields.subject
    assert "auto-tagged-florence2" in fields.subject
    assert "auto-tagged-ram" in fields.subject
    assert fields.description == "A penguin."

    db = photo_root / ".photoindex" / "catalog.db"
    assert db.exists()
    cat = Catalog(db)
    cat.init_schema()
    runs = cat.list_runs()
    assert len(runs) == 1
    assert runs[0]["photos_processed"] == 1
    cat.close()


def test_tag_skip_already_tagged(tmp_path: Path, make_jpeg):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg")
    a.rename(photo_root / "a.jpg")
    runner.invoke(app, ["tag", str(photo_root)])
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0
    cat = Catalog(photo_root / ".photoindex" / "catalog.db")
    cat.init_schema()
    runs = cat.list_runs()
    assert len(runs) == 2
    assert runs[1]["photos_processed"] == 0
    assert runs[1]["photos_skipped"] == 1
    cat.close()


def test_force_retag(tmp_path: Path, make_jpeg):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg")
    a.rename(photo_root / "a.jpg")
    runner.invoke(app, ["tag", str(photo_root)])
    result = runner.invoke(app, ["tag", str(photo_root), "--force"])
    assert result.exit_code == 0
    cat = Catalog(photo_root / ".photoindex" / "catalog.db")
    cat.init_schema()
    runs = cat.list_runs()
    assert runs[1]["photos_processed"] == 1
    cat.close()


def test_user_rejection_persists(tmp_path: Path, make_jpeg):
    """Remove an auto tag from XMP, --force re-run, expect tag stays removed."""
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg")
    a.rename(photo_root / "a.jpg")
    runner.invoke(app, ["tag", str(photo_root)])
    # Remove "penguin" from XMP, leaving "bird".
    fields = read_xmp(photo_root / "a.jpg", is_raw=False)
    fields_minus = type(fields)(
        subject=[s for s in fields.subject if s != "penguin"],
        hierarchical_subject=fields.hierarchical_subject,
        description=fields.description,
    )
    from pixsage.xmp import write_xmp
    write_xmp(photo_root / "a.jpg", fields_minus, is_raw=False)
    runner.invoke(app, ["tag", str(photo_root), "--force"])
    fields_after = read_xmp(photo_root / "a.jpg", is_raw=False)
    assert "penguin" not in fields_after.subject
    assert "bird" in fields_after.subject


def test_sample_n(tmp_path: Path, make_jpeg):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    for i in range(5):
        p = make_jpeg(f"{i}.jpg")
        p.rename(photo_root / f"{i}.jpg")
    result = runner.invoke(app, ["tag", str(photo_root), "--sample", "2"])
    assert result.exit_code == 0
    cat = Catalog(photo_root / ".photoindex" / "catalog.db")
    cat.init_schema()
    runs = cat.list_runs()
    assert runs[0]["photos_processed"] == 2
    cat.close()
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_cli.py -v
```

Expected: ImportError on `pixsage.cli`.

- [ ] **Step 3: Implement `src/pixsage/cli.py`**

```python
from __future__ import annotations

import hashlib
import json
import sys
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
    """Production tagger factory. Tests monkeypatch this; Tasks 15–16 replace it with real models."""
    raise NotImplementedError("Tests must monkeypatch pixsage.cli.build_taggers; real impl arrives in Tasks 15–16.")


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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 5 passed (skipped if exiftool isn't installed — those tests round-trip through the real exiftool).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/cli.py tests/test_cli.py
git commit -m "feat(cli): orchestrator wiring walker, taggers, vocab filter, XMP, catalog"
```

---

## Task 15: Florence-2 tagger

**Files:**
- Create: `src/pixsage/taggers/florence2.py`
- Modify: `src/pixsage/cli.py`
- Modify: `tests/test_cli.py` (no change beyond ensuring monkey-patch still routes through `build_taggers`)

This task downloads model weights (~3 GB) on first use. Network required.

- [ ] **Step 1: Implement `src/pixsage/taggers/florence2.py`**

```python
from __future__ import annotations

from PIL import Image

from pixsage.taggers.base import Tag, TagResult

MODEL_ID = "microsoft/Florence-2-large"
MODEL_VERSION = MODEL_ID  # use HF model id; encodes the version


class Florence2Tagger:
    name = "florence2"
    model_version = MODEL_VERSION

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self._device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        self._model.eval()

    def tag(self, image: Image.Image) -> TagResult:
        caption = self._run_prompt(image, "<MORE_DETAILED_CAPTION>")
        regions = self._run_prompt(image, "<DENSE_REGION_CAPTION>")
        # The dense-region output is a dict with "labels" or text per region.
        # Florence-2 returns it as {"<DENSE_REGION_CAPTION>": {"bboxes": [...], "labels": [...]}} or similar.
        labels = self._extract_labels(regions)
        tags = [Tag(name=lbl.strip(), confidence=1.0, hierarchy=None, source="florence2") for lbl in labels if lbl.strip()]
        # De-dupe by lower-cased name, preserve order:
        seen = set()
        unique_tags: list[Tag] = []
        for t in tags:
            key = t.name.lower()
            if key not in seen:
                seen.add(key)
                unique_tags.append(t)
        caption_text = caption.get("<MORE_DETAILED_CAPTION>") if isinstance(caption, dict) else str(caption) if caption else None
        return TagResult(tags=unique_tags, caption=caption_text)

    def _run_prompt(self, image: Image.Image, task: str):
        import torch
        inputs = self._processor(text=task, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(text, task=task, image_size=(image.width, image.height))

    def _extract_labels(self, regions) -> list[str]:
        if isinstance(regions, dict):
            payload = regions.get("<DENSE_REGION_CAPTION>", regions)
            if isinstance(payload, dict):
                return list(payload.get("labels", []))
        return []
```

- [ ] **Step 2: Wire into `cli.build_taggers`**

In `src/pixsage/cli.py`, replace the `NotImplementedError` body:

```python
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
```

- [ ] **Step 3: Manual smoke test**

This step is **not part of CI** — run it on the GPU box with a single test image.

```bash
mkdir -p /tmp/pixsage_smoke
cp <some-real-photo.jpg> /tmp/pixsage_smoke/
pixsage tag /tmp/pixsage_smoke --catalog /tmp/pixsage_smoke/.photoindex/catalog.db
exiftool -XMP-dc:Subject -XMP-dc:Description /tmp/pixsage_smoke/<photo>.jpg
```

Expected: keywords like the actual contents of the photo, plus `auto-tagged-florence2`. Caption is a sentence describing the image.

- [ ] **Step 4: Confirm existing tests still pass**

```bash
pytest tests/test_cli.py -v
```

Expected: still 5 passed (the monkeypatch in `test_cli.py` replaces `build_taggers` so we don't pull Florence-2 in CI).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/taggers/florence2.py src/pixsage/cli.py
git commit -m "feat(taggers): Florence-2 wrapper using DENSE_REGION_CAPTION + MORE_DETAILED_CAPTION"
```

---

## Task 16: RAM++ tagger

**Files:**
- Create: `src/pixsage/taggers/ramplusplus.py`

This task depends on the `recognize-anything` package being installable. If installation fails, the issue is upstream — log it as an open follow-up rather than work around it.

- [ ] **Step 1: Verify the package is importable**

```bash
python -c "from ram.models import ram_plus; print('ok')"
```

Expected: `ok`. If not, follow the README at https://github.com/xinyu1205/recognize-anything for setup. The package exposes `ram_plus` as the model factory.

- [ ] **Step 2: Implement `src/pixsage/taggers/ramplusplus.py`**

```python
from __future__ import annotations

from PIL import Image

from pixsage.taggers.base import Tag, TagResult

MODEL_VERSION = "ram_plus_swin_large_14m"


class RamPlusPlusTagger:
    name = "ram++"
    model_version = MODEL_VERSION

    def __init__(self):
        self._model = None
        self._transform = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from ram import inference_ram_openset as inference  # noqa: F401  (re-export check)
        from ram.models import ram_plus
        from ram import get_transform

        self._device = device
        # Image size 384 is the standard RAM++ training resolution.
        self._transform = get_transform(image_size=384)
        # Load the public checkpoint. Users may need to download the .pth and pass via env.
        import os
        ckpt = os.environ.get("PIXSAGE_RAM_CKPT", "ram_plus_swin_large_14m.pth")
        model = ram_plus(pretrained=ckpt, image_size=384, vit="swin_l")
        model.eval()
        self._model = model.to(device)

    def tag(self, image: Image.Image) -> TagResult:
        import torch
        from ram import inference_ram

        x = self._transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            tags_string, _ = inference_ram(x, self._model)
        # `inference_ram` returns a comma-or-pipe-separated string of English tags.
        # We split on " | " (RAM++'s separator) and strip.
        labels = [s.strip() for s in tags_string.split("|") if s.strip()]
        # RAM++ does not surface per-tag confidences via this entrypoint;
        # we synthesize confidence 1.0 (filter handles thresholding).
        tags = [Tag(name=lbl, confidence=1.0, hierarchy=None, source="ram++") for lbl in labels]
        return TagResult(tags=tags, caption=None)
```

- [ ] **Step 3: Manual smoke test**

```bash
PIXSAGE_RAM_CKPT=/path/to/ram_plus_swin_large_14m.pth pixsage tag /tmp/pixsage_smoke --force
exiftool -XMP-dc:Subject /tmp/pixsage_smoke/<photo>.jpg
```

Expected: noticeably more granular tags than Florence-2 alone, plus `auto-tagged-ram` marker.

- [ ] **Step 4: Confirm existing tests still pass**

```bash
pytest -v
```

Expected: all green; CI tests don't pull RAM++.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/taggers/ramplusplus.py
git commit -m "feat(taggers): RAM++ wrapper via recognize-anything"
```

---

## Task 17: Error-handling polish

**Files:**
- Modify: `src/pixsage/cli.py`
- Modify: `src/pixsage/images.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add a test for OOM-style retry behavior**

Append to `tests/test_cli.py`:

```python
def test_oom_retry_falls_back_to_smaller_size(tmp_path: Path, make_jpeg, monkeypatch):
    """Simulate OOM on first image-tag call; verify pipeline retries at smaller size."""
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg", size=(1500, 1000))
    a.rename(photo_root / "a.jpg")

    call_log: list[tuple[int, int]] = []

    class FlakyTagger:
        name = "florence2"
        model_version = "mock-1"
        def load(self, device): pass
        def tag(self, image):
            call_log.append(image.size)
            if len(call_log) == 1:
                raise RuntimeError("CUDA out of memory")
            from pixsage.taggers.base import Tag, TagResult
            return TagResult(tags=[Tag("ok", 1.0, None, "florence2")], caption=None)

    def fake_build(_cfg):
        return [FlakyTagger()]

    monkeypatch.setattr("pixsage.cli.build_taggers", fake_build)
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0, result.stdout
    assert len(call_log) >= 2
    assert max(call_log[1]) < max(call_log[0])  # second call used a smaller image
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_cli.py::test_oom_retry_falls_back_to_smaller_size -v
```

Expected: fails — current pipeline doesn't retry.

- [ ] **Step 3: Add OOM retry in `_process_one`**

Replace the body of `_process_one` in `src/pixsage/cli.py`:

```python
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
        result = _tag_with_retry(t, img)
        raw_tags.extend(result.tags)
        if result.tags:
            sources_with_tags.add(t.name)
        if caption is None and result.caption:
            caption = result.caption

    filtered = filter_tags(raw_tags, config)
    sources_with_filtered = {tag.source for tag in filtered}

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


def _tag_with_retry(tagger: Tagger, image: Image.Image) -> TagResult:
    """Try the tagger; on OOM-like failure, retry at 768 then 512."""
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
```

Also add this import at the top of `cli.py`:

```python
from PIL import Image
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/cli.py tests/test_cli.py
git commit -m "feat(cli): OOM retry — fall back to 768 then 512 px on tagger failure"
```

---

## Task 18: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
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

The first `pixsage tag` run will download Florence-2 weights (~3 GB) from Hugging Face. RAM++ may require manually downloading a checkpoint; set `PIXSAGE_RAM_CKPT=/path/to/ram_plus_swin_large_14m.pth` if you've placed it outside the working directory.

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

Edit `<photo_root>/.photoindex/vocabulary.toml`. The format:

```toml
[florence2]
enabled = true
confidence_threshold = 0.5
exclude = ["photograph", "image", "picture"]

[ram_plus_plus]
enabled = true
confidence_threshold = 0.4
exclude = []

[hierarchy_overrides]
"penguin" = "Wildlife|Bird|Penguin"
```

Re-run with `--force` (and optionally `--sample 50` first) to re-tag with the new vocabulary.

## Common flags

| Flag | Default | Purpose |
|---|---|---|
| `--force` | off | Re-tag photos even if already tagged at current model versions |
| `--sample N` | 0 (no sampling) | Tag a deterministic sample of N photos. Good for vocabulary tuning |
| `--catalog PATH` | `<photo_root>/.photoindex/catalog.db` | Override catalog location |
| `--config PATH` | `<photo_root>/.photoindex/vocabulary.toml` | Override config location |
| `--limit N` | 0 (no limit) | Stop after N photos processed this run |
| `--dry-run` | off | Run pipeline but skip XMP writes and catalog tag updates |

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with install, usage, vocabulary tuning, smoke test"
```

---

## Task 19: Demo corpus + integration runbook

**Files:**
- Create: `scripts/fetch_demo_corpus.py`
- Create: `tests/demo_corpus_urls.txt`
- Modify: `.gitignore`
- Modify: `README.md`

The photographer's actual photos are the gold-standard integration test, but we don't have them yet. This task gives us a small reproducible public corpus to validate that the pipeline produces sensible output on real images. Tag-quality validation against real photos comes after Phase 1 ships.

We use [Lorem Picsum](https://picsum.photos) — stable IDs map to specific real photos curated from Unsplash. ~20 IDs spaced across the catalog give us varied subjects (landscapes, portraits, urban, animals, food, abstract).

- [ ] **Step 1: Add demo corpus dir to `.gitignore`**

Append to `.gitignore`:

```
# Demo corpus (downloaded by scripts/fetch_demo_corpus.py)
tests/demo_corpus/
```

- [ ] **Step 2: Create `tests/demo_corpus_urls.txt`**

```
# Curated Lorem Picsum IDs covering varied subjects.
# Each line: a URL. Lines starting with # and blank lines are ignored.
# Each photo is downloaded as <id>.jpg into tests/demo_corpus/.
# Add your own URLs here as needed — any direct-link image URL works.
https://picsum.photos/id/10/2000/1333
https://picsum.photos/id/12/2000/1333
https://picsum.photos/id/20/2000/1333
https://picsum.photos/id/27/2000/1333
https://picsum.photos/id/40/2000/1333
https://picsum.photos/id/65/2000/1333
https://picsum.photos/id/91/2000/1333
https://picsum.photos/id/110/2000/1333
https://picsum.photos/id/128/2000/1333
https://picsum.photos/id/164/2000/1333
https://picsum.photos/id/177/2000/1333
https://picsum.photos/id/200/2000/1333
https://picsum.photos/id/237/2000/1333
https://picsum.photos/id/250/2000/1333
https://picsum.photos/id/293/2000/1333
https://picsum.photos/id/314/2000/1333
https://picsum.photos/id/342/2000/1333
https://picsum.photos/id/433/2000/1333
https://picsum.photos/id/535/2000/1333
https://picsum.photos/id/659/2000/1333
https://picsum.photos/id/823/2000/1333
https://picsum.photos/id/1000/2000/1333
```

- [ ] **Step 3: Create `scripts/fetch_demo_corpus.py`**

```python
"""Download a small public test corpus into tests/demo_corpus/.

Idempotent: skips files already present.
Each picsum URL is saved as <id>.jpg; other URLs use the trailing path component.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "tests" / "demo_corpus"
URLS_FILE = ROOT / "tests" / "demo_corpus_urls.txt"


def target_filename(url: str) -> str:
    parts = urlparse(url).path.strip("/").split("/")
    if "id" in parts:  # picsum.photos URLs
        idx = parts.index("id")
        if idx + 1 < len(parts):
            return f"{parts[idx + 1]}.jpg"
    return parts[-1] or "img.jpg"


def main() -> int:
    if not URLS_FILE.exists():
        print(f"Missing URL list: {URLS_FILE}", file=sys.stderr)
        return 1
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    urls = [
        line.strip()
        for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    downloaded = 0
    skipped = 0
    failed = 0
    for url in urls:
        target = CORPUS_DIR / target_filename(url)
        if target.exists():
            skipped += 1
            continue
        print(f"Downloading {url} -> {target.name}")
        try:
            urllib.request.urlretrieve(url, target)
            downloaded += 1
        except Exception as e:  # noqa: BLE001  (any failure is reportable)
            print(f"  failed: {e}", file=sys.stderr)
            failed += 1
    print(f"done. downloaded={downloaded} skipped={skipped} failed={failed} total={len(urls)}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add a "Demo corpus" section to `README.md`**

Insert this section between "## Manual smoke test" and "## Tests":

```markdown
## Demo corpus

When you don't have access to the photographer's photos, fetch a small public corpus to validate the end-to-end pipeline:

\`\`\`bash
python scripts/fetch_demo_corpus.py
pixsage tag tests/demo_corpus
exiftool -XMP-dc:Subject -XMP-dc:Description tests/demo_corpus/*.jpg
\`\`\`

This downloads ~20 photos from picsum.photos into `tests/demo_corpus/` (gitignored). Use it for "does the pipeline work on real images" testing. For tag-quality validation, point pixsage at the photographer's actual photos.

Add your own URLs to `tests/demo_corpus_urls.txt` to expand the corpus.
```

(In the actual README, replace `\`\`\`` with triple backticks — the escaping above is just so this plan renders correctly.)

- [ ] **Step 5: Verify the script runs**

```bash
python scripts/fetch_demo_corpus.py
ls tests/demo_corpus
```

Expected: ~22 JPEG files. If picsum.photos is unreachable, the script prints failures but doesn't raise.

- [ ] **Step 6: Run a real end-to-end smoke test**

After Florence-2 + RAM++ are wired (Tasks 15–16):

```bash
pixsage tag tests/demo_corpus
exiftool -XMP-dc:Subject -XMP-dc:Description tests/demo_corpus/10.jpg
```

Expected: tags that plausibly describe the photo content; caption is a sentence; markers `auto-tagged-florence2` and `auto-tagged-ram` present. Inspect a handful of files to gut-check tag quality.

- [ ] **Step 7: Commit**

```bash
git add scripts/fetch_demo_corpus.py tests/demo_corpus_urls.txt .gitignore README.md
git commit -m "feat(test): demo corpus fetcher + integration runbook"
```

---

## Task 20: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest
```

Expected: all tests pass (or skip cleanly on missing exiftool / raw fixture).

- [ ] **Step 2: Lint**

```bash
ruff check src tests
```

Expected: no issues. Fix any reported.

- [ ] **Step 3: Manual end-to-end smoke test on the demo corpus**

On the GPU box (assumes `python scripts/fetch_demo_corpus.py` has populated `tests/demo_corpus/`):

```bash
pixsage tag tests/demo_corpus
```

- Verify the run completes without errors.
- Verify embedded XMP on JPEGs has plausible keywords (`exiftool -XMP-dc:Subject -XMP-dc:Description tests/demo_corpus/*.jpg`). Spot-check 3–5 photos for tag quality.
- Re-run `pixsage tag tests/demo_corpus`. Output reports `processed=0 skipped=N`.
- Edit a tag from one photo's XMP (`exiftool -XMP-dc:Subject-=<tag> tests/demo_corpus/<id>.jpg`). Re-run with `--force`. Verify that tag stays removed (user-rejection behavior).
- Edit `vocabulary.toml` in `tests/demo_corpus/.photoindex/`, add an exclusion. Re-run with `--force`. Verify the excluded tag is gone everywhere.

- [ ] **Step 4: Manual smoke test on real raws (when available)**

When a folder of the photographer's actual photos is accessible (USB drive, network share, etc.) — likely after Phase 1 ships:

```bash
pixsage tag /path/to/photographer_sample --sample 50
```

- Verify XMP sidecars exist for raws (e.g., `DSC_0001.ARW` → `DSC_0001.xmp`).
- Open a raw in Lightroom (after Read Metadata from File). Keywords appear in the Keyword List; hierarchical keywords render as nested.
- Build a Lightroom smart collection on `auto-tagged-florence2`. Inspect the auto-tagged subset.
- Sit with the photographer for 15 minutes; ask which tags are wrong, missing, or noise. Use that feedback to iterate `vocabulary.toml`.

- [ ] **Step 5: Final commit (if any local changes)**

```bash
git status
# if anything dirty, fix + commit
```

---

## Summary of what gets built

After all 20 tasks:

- A `pixsage tag` CLI installable via `pip install -e ".[taggers]"` (or `[dev]` for tests).
- 6 components (walker, image loader, taggers, vocabulary filter, XMP writer, catalog) with isolated unit tests.
- End-to-end CLI tests using `MockTagger` — no GPU or model weights needed in CI.
- Idempotent re-runs with `--force` re-tag behavior and user-rejection persistence.
- A SQLite catalog whose schema is forward-compatible with Phase 2.
- README covering install, usage, vocabulary tuning, and smoke testing.

Phase 2 will extend the catalog (pHash, EXIF triple, embeddings, geo, clusters) without touching the Phase 1 schema.
