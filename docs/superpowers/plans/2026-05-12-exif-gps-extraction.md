# EXIF GPS Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract camera EXIF GPS during `pixsage tag`, store it in the catalog, and skip GeoCLIP for photos that already have real GPS. Plus a `backfill-exif-gps` command for already-tagged catalogs.

**Architecture:**
- Three new nullable columns on `photos`: `exif_latitude`, `exif_longitude`, `exif_altitude` (all `REAL`).
- New `xmp.read_camera_gps(path)` reads EXIF tags from the file (never the sidecar). Returns `None` if absent or the (0,0) sentinel.
- New `xmp.read_metadata(path, is_raw)` returns `(XmpFields, CameraGps | None)` in one call. For embedded-XMP files (DNG, JPEG, HEIC, raws-without-sidecar) it fetches both via a single exiftool subprocess; for raw+sidecar it makes two (one for XMP from the sidecar, one for EXIF from the raw).
- `_apply_to_path` in the tag loop switches from `read_xmp` to `read_metadata` and writes any detected GPS to the catalog via `set_camera_gps`.
- `iter_photos_for_geolocation` filters out rows with `exif_latitude IS NOT NULL` or with an entry in `user_locations`. `geolocate --all` overrides.
- New CLI `pixsage backfill-exif-gps <photo_root>` walks the catalog and populates the new columns from each file's EXIF without re-running the taggers.

**Tech Stack:** SQLite + exiftool (already used), typer CLI, pytest. No new dependencies.

---

## File Structure

**Modify:**
- `src/pixsage/xmp.py` — add `CameraGps` dataclass, `read_camera_gps`, `read_metadata`.
- `src/pixsage/catalog.py` — schema migration; `set_camera_gps`, `get_camera_gps`; update `iter_photos_for_geolocation`.
- `src/pixsage/cli.py` — `_apply_to_path` calls `read_metadata`; per-photo loop stores GPS; `geolocate` gains `--all`; new `backfill_exif_gps` command.
- `src/pixsage/geo_runner.py` — pass `include_with_camera_gps` through to the catalog iterator.

**Create:**
- `tests/test_xmp_camera_gps.py` — new tests for `read_camera_gps` and `read_metadata`.
- `tests/test_catalog_camera_gps.py` — new tests for catalog GPS methods + iterator filtering.
- `tests/test_cli_backfill_gps.py` — tests for the backfill CLI.

**Touch (test updates):**
- `tests/test_geo_runner.py` — assert default skip of photos with EXIF GPS; assert `include_with_camera_gps=True` overrides.

---

## Task 1: Schema migration — add EXIF GPS columns to `photos`

**Files:**
- Modify: `src/pixsage/catalog.py:106-122` (the `init_schema` + `_migrate_add_caption_columns` block)
- Test: `tests/test_catalog_camera_gps.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_catalog_camera_gps.py`:

```python
from __future__ import annotations

from pathlib import Path

from pixsage.catalog import Catalog


def test_init_schema_adds_exif_gps_columns(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    cur = cat._conn.execute("PRAGMA table_info(photos)")
    columns = {row["name"] for row in cur.fetchall()}
    assert "exif_latitude" in columns
    assert "exif_longitude" in columns
    assert "exif_altitude" in columns
    cat.close()


def test_init_schema_is_idempotent_for_exif_columns(tmp_path: Path):
    db = tmp_path / "catalog.db"
    Catalog(db).init_schema()
    # Second init must not raise "duplicate column" error.
    Catalog(db).init_schema()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_catalog_camera_gps.py -v
```

Expected: FAIL — `exif_latitude` not in columns.

- [ ] **Step 3: Extend the migration**

In `src/pixsage/catalog.py`, rename `_migrate_add_caption_columns` → `_migrate_add_columns` and add the three GPS columns. Replace the existing method body:

```python
    def _migrate_add_columns(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(photos)")
        existing = {row["name"] for row in cur.fetchall()}
        additions = [
            ("caption", "TEXT"),
            ("caption_updated_at", "TEXT"),
            ("exif_latitude", "REAL"),
            ("exif_longitude", "REAL"),
            ("exif_altitude", "REAL"),
        ]
        for name, type_ in additions:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE photos ADD COLUMN {name} {type_}")
```

Update `init_schema` to call the renamed method (replace `self._migrate_add_caption_columns()` with `self._migrate_add_columns()` on line 114).

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_catalog_camera_gps.py -v
```

Expected: PASS (both tests).

Also run the full catalog test suite to confirm nothing regressed:

```
pytest tests/test_catalog.py tests/test_catalog_caption.py tests/test_catalog_geo.py -v
```

Expected: PASS for all.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/catalog.py tests/test_catalog_camera_gps.py
git commit -m "feat(catalog): add exif_latitude/longitude/altitude columns to photos"
```

---

## Task 2: Catalog methods — `set_camera_gps`, `get_camera_gps`

**Files:**
- Modify: `src/pixsage/catalog.py` (add methods after `record_caption`, around line 192)
- Test: `tests/test_catalog_camera_gps.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog_camera_gps.py`:

```python
def test_set_and_get_camera_gps_roundtrip(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\x00")
    cat.upsert_photo("sha1", img, filesize=1, mtime=0.0)

    cat.set_camera_gps("sha1", latitude=-52.5, longitude=-60.4, altitude=30.5)
    got = cat.get_camera_gps("sha1")
    assert got == {"latitude": -52.5, "longitude": -60.4, "altitude": 30.5}
    cat.close()


def test_get_camera_gps_returns_none_when_absent(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\x00")
    cat.upsert_photo("sha1", img, filesize=1, mtime=0.0)
    assert cat.get_camera_gps("sha1") is None
    cat.close()


def test_set_camera_gps_overwrites(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\x00")
    cat.upsert_photo("sha1", img, filesize=1, mtime=0.0)

    cat.set_camera_gps("sha1", latitude=1.0, longitude=2.0, altitude=None)
    cat.set_camera_gps("sha1", latitude=10.0, longitude=20.0, altitude=100.0)
    got = cat.get_camera_gps("sha1")
    assert got == {"latitude": 10.0, "longitude": 20.0, "altitude": 100.0}
    cat.close()


def test_set_camera_gps_accepts_no_altitude(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\x00")
    cat.upsert_photo("sha1", img, filesize=1, mtime=0.0)

    cat.set_camera_gps("sha1", latitude=1.0, longitude=2.0, altitude=None)
    got = cat.get_camera_gps("sha1")
    assert got == {"latitude": 1.0, "longitude": 2.0, "altitude": None}
    cat.close()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_catalog_camera_gps.py -v
```

Expected: FAIL — `Catalog` has no attribute `set_camera_gps`.

- [ ] **Step 3: Implement the methods**

In `src/pixsage/catalog.py`, add after `record_caption` (around line 192, before `iter_photos_for_embedding`):

```python
    def set_camera_gps(
        self,
        sha256: str,
        latitude: float,
        longitude: float,
        altitude: float | None,
    ) -> None:
        """Record the camera-recorded EXIF GPS for a photo.

        Distinct from `user_locations` (HITL-applied) and `geo_predictions`
        (GeoCLIP guesses) — this is the original signal from the camera.
        """
        with self._conn:
            self._conn.execute(
                """
                UPDATE photos
                   SET exif_latitude = ?, exif_longitude = ?, exif_altitude = ?
                 WHERE sha256 = ?
                """,
                (latitude, longitude, altitude, sha256),
            )

    def get_camera_gps(self, sha256: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT exif_latitude, exif_longitude, exif_altitude FROM photos WHERE sha256 = ?",
            (sha256,),
        )
        row = cur.fetchone()
        if row is None or row["exif_latitude"] is None:
            return None
        return {
            "latitude": float(row["exif_latitude"]),
            "longitude": float(row["exif_longitude"]),
            "altitude": float(row["exif_altitude"]) if row["exif_altitude"] is not None else None,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_catalog_camera_gps.py -v
```

Expected: PASS (all 6 tests including Task 1's).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/catalog.py tests/test_catalog_camera_gps.py
git commit -m "feat(catalog): set/get_camera_gps for EXIF GPS storage"
```

---

## Task 3: Iterator skips photos with EXIF GPS or user location by default

**Files:**
- Modify: `src/pixsage/catalog.py:450-465` (`iter_photos_for_geolocation`)
- Test: `tests/test_catalog_camera_gps.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog_camera_gps.py`:

```python
def test_iter_geolocation_skips_photos_with_camera_gps(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    for sha in ("sha-has-gps", "sha-no-gps"):
        p = tmp_path / f"{sha}.jpg"
        p.write_bytes(b"\x00")
        cat.upsert_photo(sha, p, filesize=1, mtime=0.0)
    cat.set_camera_gps("sha-has-gps", latitude=10.0, longitude=20.0, altitude=None)

    yielded = {r["sha256"] for r in cat.iter_photos_for_geolocation()}
    assert yielded == {"sha-no-gps"}
    cat.close()


def test_iter_geolocation_skips_photos_with_user_location(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    for sha in ("sha-user-loc", "sha-no-loc"):
        p = tmp_path / f"{sha}.jpg"
        p.write_bytes(b"\x00")
        cat.upsert_photo(sha, p, filesize=1, mtime=0.0)
    cat.record_user_location("sha-user-loc", 10.0, 20.0, "Test", "manual")

    yielded = {r["sha256"] for r in cat.iter_photos_for_geolocation()}
    assert yielded == {"sha-no-loc"}
    cat.close()


def test_iter_geolocation_include_with_camera_gps_returns_all(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    for sha in ("sha-has-gps", "sha-no-gps"):
        p = tmp_path / f"{sha}.jpg"
        p.write_bytes(b"\x00")
        cat.upsert_photo(sha, p, filesize=1, mtime=0.0)
    cat.set_camera_gps("sha-has-gps", latitude=10.0, longitude=20.0, altitude=None)

    yielded = {r["sha256"] for r in cat.iter_photos_for_geolocation(include_with_camera_gps=True)}
    assert yielded == {"sha-has-gps", "sha-no-gps"}
    cat.close()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_catalog_camera_gps.py -v
```

Expected: FAIL — the first two new tests fail because the iterator currently returns all rows; the third fails because `include_with_camera_gps` is not a parameter.

- [ ] **Step 3: Update the iterator**

In `src/pixsage/catalog.py`, replace the `iter_photos_for_geolocation` method (lines 450-465) with:

```python
    def iter_photos_for_geolocation(
        self,
        include_errored: bool = False,
        include_with_camera_gps: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield {sha256, current_path} for photos the geolocator should consider.

        By default skips photos that already have real GPS (EXIF or HITL-applied).
        Pass `include_with_camera_gps=True` (mapped from `geolocate --all`) to run
        predictions on every photo regardless of existing GPS — useful for
        benchmarking GeoCLIP against ground truth.

        Errored photos are excluded by default; pass `include_errored=True` to
        retry them (mapped from `--force`).
        """
        clauses: list[str] = []
        if not include_errored:
            clauses.append("error_reason IS NULL")
        if not include_with_camera_gps:
            clauses.append("exif_latitude IS NULL")
            clauses.append("sha256 NOT IN (SELECT sha256 FROM user_locations)")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._conn.execute(f"SELECT sha256, current_path FROM photos{where}")
        for row in cur:
            yield dict(row)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_catalog_camera_gps.py tests/test_geo_runner.py tests/test_catalog_geo.py -v
```

Expected: PASS for all. (The existing `test_geo_runner` tests use photos without `exif_latitude`, so they still surface in the iterator.)

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/catalog.py tests/test_catalog_camera_gps.py
git commit -m "feat(catalog): iter_photos_for_geolocation skips photos with real GPS"
```

---

## Task 4: `xmp.read_camera_gps` reads EXIF GPS from a file

**Files:**
- Modify: `src/pixsage/xmp.py` (add `CameraGps` dataclass and `read_camera_gps` function)
- Test: `tests/test_xmp_camera_gps.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_xmp_camera_gps.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from pixsage.xmp import CameraGps, read_camera_gps

EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")


def _inject_exif_gps(
    path: Path,
    lat: float,
    lon: float,
    alt: float | None = None,
) -> None:
    """Write EXIF (not XMP) GPS tags into an existing image via exiftool."""
    lat_ref = "N" if lat >= 0 else "S"
    lon_ref = "E" if lon >= 0 else "W"
    args = [
        EXIFTOOL,
        "-overwrite_original",
        f"-EXIF:GPSLatitude={abs(lat)}",
        f"-EXIF:GPSLatitudeRef={lat_ref}",
        f"-EXIF:GPSLongitude={abs(lon)}",
        f"-EXIF:GPSLongitudeRef={lon_ref}",
    ]
    if alt is not None:
        args.extend([f"-EXIF:GPSAltitude={alt}", "-EXIF:GPSAltitudeRef=0"])
    args.append(str(path))
    subprocess.run(args, check=True, capture_output=True)


@needs_exiftool
def test_read_camera_gps_jpeg_with_full_gps(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    _inject_exif_gps(p, lat=-52.4873, lon=-60.4447, alt=30.5)

    got = read_camera_gps(p)
    assert got is not None
    assert isinstance(got, CameraGps)
    assert abs(got.latitude - -52.4873) < 1e-3
    assert abs(got.longitude - -60.4447) < 1e-3
    assert got.altitude is not None
    assert abs(got.altitude - 30.5) < 1e-1


@needs_exiftool
def test_read_camera_gps_returns_none_when_absent(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    assert read_camera_gps(p) is None


@needs_exiftool
def test_read_camera_gps_filters_zero_zero_sentinel(tmp_path: Path):
    """(0,0) is the 'no fix' bug from some old devices — return None."""
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    _inject_exif_gps(p, lat=0.0, lon=0.0, alt=None)
    assert read_camera_gps(p) is None


@needs_exiftool
def test_read_camera_gps_without_altitude(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    _inject_exif_gps(p, lat=51.5, lon=-0.1, alt=None)

    got = read_camera_gps(p)
    assert got is not None
    assert abs(got.latitude - 51.5) < 1e-3
    assert abs(got.longitude - -0.1) < 1e-3
    assert got.altitude is None


@needs_exiftool
def test_read_camera_gps_southern_hemisphere(tmp_path: Path):
    """Confirm S/W refs round-trip into signed decimal."""
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    _inject_exif_gps(p, lat=-64.28, lon=-56.74, alt=None)

    got = read_camera_gps(p)
    assert got is not None
    assert got.latitude < 0
    assert got.longitude < 0
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_xmp_camera_gps.py -v
```

Expected: FAIL — `CameraGps`/`read_camera_gps` not defined.

- [ ] **Step 3: Implement `CameraGps` and `read_camera_gps`**

In `src/pixsage/xmp.py`, add near the top (after the `XmpFields` dataclass, around line 50):

```python
@dataclass(frozen=True)
class CameraGps:
    latitude: float
    longitude: float
    altitude: float | None
```

Add the `dataclass` import at the top of the file if not already present:

```python
from dataclasses import dataclass
```

(Already imported — file already has `XmpFields` as `@dataclass`. Confirm by reading the top of the file; if `dataclass` is imported, do nothing.)

Then add the function after `read_gps` (around line 180), so the EXIF / XMP gps reads are visually adjacent:

```python
def read_camera_gps(path: Path) -> CameraGps | None:
    """Read EXIF GPS tags directly from a file (never a sidecar).

    Returns None when GPS is absent or when latitude AND longitude are both
    near zero (the (0,0) "no fix" sentinel seen in some old devices). The `#`
    suffix on each tag forces signed decimal output, applying the
    GPSLatitudeRef/LongitudeRef letters automatically.
    """
    cmd = [
        EXIFTOOL,
        "-json",
        "-coordFormat", "%+.10f",
        "-EXIF:GPSLatitude",
        "-EXIF:GPSLongitude",
        "-EXIF:GPSAltitude#",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    except subprocess.CalledProcessError:
        return None
    data = json.loads(result.stdout) if result.stdout.strip() else [{}]
    if not data or "GPSLatitude" not in data[0] or "GPSLongitude" not in data[0]:
        return None
    rec = data[0]
    lat = float(rec["GPSLatitude"])
    lon = float(rec["GPSLongitude"])
    if abs(lat) < 0.01 and abs(lon) < 0.01:
        return None
    alt_raw = rec.get("GPSAltitude")
    altitude = float(alt_raw) if alt_raw is not None else None
    return CameraGps(latitude=lat, longitude=lon, altitude=altitude)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_xmp_camera_gps.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/xmp.py tests/test_xmp_camera_gps.py
git commit -m "feat(xmp): read_camera_gps extracts EXIF GPS from photo files"
```

---

## Task 5: `xmp.read_metadata` returns XMP fields + camera GPS in one call

**Files:**
- Modify: `src/pixsage/xmp.py` (add `read_metadata`)
- Test: `tests/test_xmp_camera_gps.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_xmp_camera_gps.py`:

```python
from pixsage.xmp import read_metadata, write_xmp, XmpFields


@needs_exiftool
def test_read_metadata_jpeg_returns_both_xmp_and_gps(tmp_path: Path):
    """For an embedded-XMP file, one subprocess fetches both XMP and EXIF GPS."""
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    write_xmp(p, XmpFields(subject=["foo"], hierarchical_subject=[], description="hello"), is_raw=False)
    _inject_exif_gps(p, lat=10.0, lon=20.0, alt=100.0)

    xmp_fields, gps = read_metadata(p, is_raw=False)
    assert xmp_fields.subject == ["foo"]
    assert xmp_fields.description == "hello"
    assert gps is not None
    assert abs(gps.latitude - 10.0) < 1e-3
    assert abs(gps.longitude - 20.0) < 1e-3


@needs_exiftool
def test_read_metadata_jpeg_without_gps(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    write_xmp(p, XmpFields(subject=["foo"], hierarchical_subject=[], description=None), is_raw=False)

    xmp_fields, gps = read_metadata(p, is_raw=False)
    assert xmp_fields.subject == ["foo"]
    assert gps is None


@needs_exiftool
def test_read_metadata_raw_sidecar(tmp_path: Path):
    """For sidecar raws, XMP comes from the sidecar and GPS from the raw file.

    Uses a placeholder raw + empty sidecar (the same pattern as
    test_xmp_gps.test_write_gps_updates_existing_sidecar_for_raw_path).
    """
    raw = tmp_path / "DSC0001.arw"
    # Write a valid JPEG masquerading as .arw so exiftool can read EXIF GPS off it.
    Image.new("RGB", (64, 64), color="red").save(raw, format="JPEG")
    _inject_exif_gps(raw, lat=-1.0, lon=2.0, alt=None)

    sidecar = raw.with_suffix(".xmp")
    write_xmp(raw, XmpFields(subject=["bar"], hierarchical_subject=[], description="x"), is_raw=True)
    assert sidecar.exists()

    xmp_fields, gps = read_metadata(raw, is_raw=True)
    assert xmp_fields.subject == ["bar"]
    assert xmp_fields.description == "x"
    assert gps is not None
    assert abs(gps.latitude - -1.0) < 1e-3
    assert abs(gps.longitude - 2.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_xmp_camera_gps.py -v -k read_metadata
```

Expected: FAIL — `read_metadata` not defined.

- [ ] **Step 3: Implement `read_metadata`**

In `src/pixsage/xmp.py`, add after `read_camera_gps`:

```python
def read_metadata(path: Path, is_raw: bool) -> tuple[XmpFields, CameraGps | None]:
    """Read XMP fields + camera EXIF GPS from a photo.

    For embedded-XMP files (DNG, JPEG, HEIC, raws without a sidecar present),
    a single exiftool subprocess fetches both — the EXIF and XMP IFDs live in
    the same file, so we just ask for both tag families at once.

    For raw+sidecar files, the XMP fields live in the .xmp sidecar but the
    EXIF GPS lives in the raw — two subprocesses.
    """
    if is_raw:
        # XMP from sidecar (existing behaviour); GPS from the raw separately.
        xmp = read_xmp(path, is_raw=True)
        gps = read_camera_gps(path)
        return xmp, gps

    # Embedded path: one subprocess covers both.
    cmd = [
        EXIFTOOL,
        "-json",
        "-coordFormat", "%+.10f",
        "-XMP-dc:Subject",
        "-XMP-lr:HierarchicalSubject",
        "-XMP-dc:Description",
        "-EXIF:GPSLatitude",
        "-EXIF:GPSLongitude",
        "-EXIF:GPSAltitude#",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"exiftool read failed: {e.stderr}") from e
    data = json.loads(result.stdout) if result.stdout.strip() else [{}]
    rec = data[0] if data else {}
    xmp = XmpFields(
        subject=_to_list(rec.get("Subject")),
        hierarchical_subject=_to_list(rec.get("HierarchicalSubject")),
        description=rec.get("Description"),
    )
    if "GPSLatitude" not in rec or "GPSLongitude" not in rec:
        return xmp, None
    lat = float(rec["GPSLatitude"])
    lon = float(rec["GPSLongitude"])
    if abs(lat) < 0.01 and abs(lon) < 0.01:
        return xmp, None
    alt_raw = rec.get("GPSAltitude")
    altitude = float(alt_raw) if alt_raw is not None else None
    return xmp, CameraGps(latitude=lat, longitude=lon, altitude=altitude)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_xmp_camera_gps.py -v
```

Expected: PASS (all 8 tests across Tasks 4+5).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/xmp.py tests/test_xmp_camera_gps.py
git commit -m "feat(xmp): read_metadata bundles XMP + camera GPS in one call"
```

---

## Task 6: Tag loop populates `exif_latitude/longitude/altitude` via `read_metadata`

**Files:**
- Modify: `src/pixsage/cli.py:18` (import), `cli.py:163-220` (the per-photo loop), `cli.py:387-450` (`_apply_to_path`)
- Test: `tests/test_xmp_camera_gps.py` (integration test of the tag flow's GPS write)

The tag loop's `_apply_to_path` currently calls `read_xmp(path, is_raw=is_raw)` at line 408. We replace that with `read_metadata`, then write any detected GPS into the catalog. The catalog handle is already in scope inside the per-photo loop (it's the `cat` variable from the surrounding function at line 163-220) — we add the `set_camera_gps` call there, NOT inside `_apply_to_path` (which would mean threading `cat` through that function unnecessarily).

Strategy: change `_apply_to_path` to return `(new_sha, camera_gps)` instead of just `new_sha`. The caller in the per-photo loop writes the GPS to the catalog after the path apply succeeds.

- [ ] **Step 1: Write the failing test**

Direct unit test of `_apply_to_path` (more reliable than a CLI runner test for this; no Florence-2/RAM++ dependency tangles). Append to `tests/test_xmp_camera_gps.py`:

```python
@needs_exiftool
def test_apply_to_path_returns_camera_gps(tmp_path: Path):
    from pixsage.cli import _apply_to_path
    from pixsage.catalog import Catalog
    from pixsage.config import Config
    from pixsage.taggers.base import Tag

    img = tmp_path / "shot.jpg"
    Image.new("RGB", (64, 64), color="red").save(img)
    _inject_exif_gps(img, lat=-52.5, lon=-60.4, alt=30.0)

    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha1", img, filesize=img.stat().st_size, mtime=img.stat().st_mtime)

    new_sha, gps = _apply_to_path(
        path=img,
        sha="sha1",
        is_raw=False,
        filtered_tags=[Tag(name="foo", confidence=0.9, hierarchy=None, source="mock")],
        caption="hello",
        is_first_for_sha=True,
        taggers=[],
        config=Config(),
        cat=cat,
        dry_run=False,
        rewrite=False,
        sha_prior_strip={},
    )
    assert gps is not None
    assert abs(gps.latitude - -52.5) < 1e-3
    assert abs(gps.longitude - -60.4) < 1e-3
    cat.close()


@needs_exiftool
def test_apply_to_path_returns_none_gps_for_file_without_exif(tmp_path: Path):
    from pixsage.cli import _apply_to_path
    from pixsage.catalog import Catalog
    from pixsage.config import Config
    from pixsage.taggers.base import Tag

    img = tmp_path / "shot.jpg"
    Image.new("RGB", (64, 64), color="red").save(img)
    # No _inject_exif_gps call — file has no GPS.

    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha1", img, filesize=img.stat().st_size, mtime=img.stat().st_mtime)

    new_sha, gps = _apply_to_path(
        path=img,
        sha="sha1",
        is_raw=False,
        filtered_tags=[Tag(name="foo", confidence=0.9, hierarchy=None, source="mock")],
        caption=None,
        is_first_for_sha=True,
        taggers=[],
        config=Config(),
        cat=cat,
        dry_run=False,
        rewrite=False,
        sha_prior_strip={},
    )
    assert gps is None
    cat.close()
```

(If `Config()` requires arguments — check `src/pixsage/config.py` — use whatever default-construction pattern the existing CLI tests use; see `tests/test_cli.py` for examples.)

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_xmp_camera_gps.py -v -k camera_gps
```

Expected: FAIL — `_apply_to_path` currently returns a single sha, not a tuple. Or the catalog columns stay NULL because nothing populates them yet.

- [ ] **Step 3: Update `_apply_to_path` and the per-photo loop**

In `src/pixsage/cli.py`:

**3a.** Update the import on line 18 to include `read_metadata` and `CameraGps`:

```python
from pixsage.xmp import CameraGps, XmpFields, merge_xmp, needs_sidecar, read_metadata, write_xmp
```

**3b.** Replace the `read_xmp` call in `_apply_to_path` (line 408). Change:

```python
    existing = read_xmp(path, is_raw=is_raw)
```

to:

```python
    existing, camera_gps = read_metadata(path, is_raw=is_raw)
```

**3c.** Change the return type of `_apply_to_path`. The signature change: from `-> str` to `-> tuple[str, CameraGps | None]`. Locate every `return ... # new_sha` statement inside `_apply_to_path` (there's at least one returning the new sha) and change each to `return new_sha, camera_gps`. Read the current function body fully before editing — there may be multiple early returns; each gets the same treatment (returning the GPS read at the top).

**3d.** Update the caller in the per-photo loop (around lines 197-214). Replace:

```python
            new_sha = _apply_to_path(
                path=path,
                ...
            )
            if new_sha != sha:
                sha_to_tags[new_sha] = sha_to_tags[sha]
                seen_shas_this_run.add(new_sha)
            processed += 1
```

with:

```python
            new_sha, camera_gps = _apply_to_path(
                path=path,
                ...
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
```

Notes:
- We write to `new_sha` (not `sha`) because a fresh embedded-XMP write changes file bytes and thus the sha256 (Phase 1 gotcha). The `rekey_photo` inside `_apply_to_path` has already migrated the photo row, so `cat.set_camera_gps(new_sha, ...)` lands on the right row.
- The `not dry_run` guard mirrors the existing pattern — `--dry-run` skips XMP writes and catalog tag updates, so it should skip GPS writes too.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_xmp_camera_gps.py tests/test_cli.py tests/test_cli_dupe_paths.py -v
```

Expected: PASS for the new test and all existing CLI tag tests (no regression).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/cli.py tests/test_xmp_camera_gps.py
git commit -m "feat(tag): extract and store camera EXIF GPS during tag run"
```

---

## Task 7: `geolocate --all` flag overrides the EXIF-GPS skip

**Files:**
- Modify: `src/pixsage/geo_runner.py` (accept `include_with_camera_gps` and pass it through)
- Modify: `src/pixsage/cli.py:511-545` (the `geolocate` command — add `--all` flag)
- Test: `tests/test_geo_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_geo_runner.py`:

```python
def test_runner_skips_photos_with_camera_gps_by_default(catalog: Catalog, tmp_path: Path):
    _seed_photo(catalog, "sha-no-gps", tmp_path, name="a.jpg")
    img_with_gps = _seed_photo(catalog, "sha-has-gps", tmp_path, name="b.jpg")
    catalog.set_camera_gps("sha-has-gps", latitude=10.0, longitude=20.0, altitude=None)

    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=2)).run()

    assert catalog.get_geo_predictions("sha-no-gps", "mock") != []
    assert catalog.get_geo_predictions("sha-has-gps", "mock") == []


def test_runner_predict_all_includes_photos_with_camera_gps(catalog: Catalog, tmp_path: Path):
    _seed_photo(catalog, "sha-has-gps", tmp_path, name="b.jpg")
    catalog.set_camera_gps("sha-has-gps", latitude=10.0, longitude=20.0, altitude=None)

    GeoRunner(
        catalog=catalog,
        geolocator=MockGeolocator(top_k=2),
        include_with_camera_gps=True,
    ).run()

    assert catalog.get_geo_predictions("sha-has-gps", "mock") != []
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_geo_runner.py -v -k camera_gps
```

Expected: FAIL — `GeoRunner.__init__` does not accept `include_with_camera_gps`; even without it, the second test would still predict on the EXIF-GPS photo (since the iterator skip change happens via the runner config flag).

- [ ] **Step 3: Plumb the flag through the runner**

In `src/pixsage/geo_runner.py`, update `__init__` and `run`:

```python
class GeoRunner:
    def __init__(
        self,
        catalog: Catalog,
        geolocator: Geolocator,
        force: bool = False,
        progress: bool = False,
        include_with_camera_gps: bool = False,
    ) -> None:
        self.catalog = catalog
        self.geolocator = geolocator
        self.force = force
        self.progress = progress
        self.include_with_camera_gps = include_with_camera_gps

    def run(self) -> dict[str, int]:
        info = self.geolocator.info
        stats = {"processed": 0, "skipped": 0, "errored": 0}

        rows = list(self.catalog.iter_photos_for_geolocation(
            include_errored=self.force,
            include_with_camera_gps=self.include_with_camera_gps,
        ))
        # ... rest unchanged
```

- [ ] **Step 4: Wire `--all` into the CLI**

In `src/pixsage/cli.py`, update the `geolocate` function signature (line 511-520):

```python
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
```

And in the runner instantiation (line 542):

```python
    runner = GeoRunner(
        catalog=cat,
        geolocator=geo,
        force=force,
        progress=True,
        include_with_camera_gps=all_photos,
    )
```

Also update the docstring (lines 521-525):

```python
    """Predict GPS coordinates for catalogued photos that lack real GPS.

    By default, photos with EXIF GPS or a user-applied location are skipped
    (running GeoCLIP on them produces guesses that are worse than the truth
    already in the file). Pass --all to override.

    Predictions live in the geo_predictions table and travel with the catalog.db,
    so the analysis machine doesn't need the source photos to read them back.
    """
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_geo_runner.py tests/test_cli_geolocate.py -v
```

Expected: PASS for new tests and no regression on existing ones.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/geo_runner.py src/pixsage/cli.py tests/test_geo_runner.py
git commit -m "feat(geolocate): skip photos with EXIF/user GPS by default; --all overrides"
```

---

## Task 8: `pixsage backfill-exif-gps` CLI for existing catalogs

**Files:**
- Modify: `src/pixsage/cli.py` (add new `backfill_exif_gps` command after `geolocate`)
- Test: `tests/test_cli_backfill_gps.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_backfill_gps.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from pixsage.catalog import Catalog
from pixsage.cli import app

EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")


def _inject_exif_gps(path: Path, lat: float, lon: float) -> None:
    args = [
        EXIFTOOL,
        "-overwrite_original",
        f"-EXIF:GPSLatitude={abs(lat)}",
        f"-EXIF:GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
        f"-EXIF:GPSLongitude={abs(lon)}",
        f"-EXIF:GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
        str(path),
    ]
    subprocess.run(args, check=True, capture_output=True)


@needs_exiftool
def test_backfill_populates_camera_gps_for_existing_catalog(tmp_path: Path):
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()

    # Simulate a previous tag run: catalog has rows but no exif_latitude yet.
    img1 = photo_root / "a.jpg"
    img2 = photo_root / "b.jpg"
    Image.new("RGB", (64, 64), color="red").save(img1)
    Image.new("RGB", (64, 64), color="blue").save(img2)
    _inject_exif_gps(img1, lat=-52.5, lon=-60.4)
    # img2 deliberately has no GPS.

    cat.upsert_photo("sha-a", img1, filesize=img1.stat().st_size, mtime=img1.stat().st_mtime)
    cat.upsert_photo("sha-b", img2, filesize=img2.stat().st_size, mtime=img2.stat().st_mtime)
    cat.close()

    runner = CliRunner()
    result = runner.invoke(app, ["backfill-exif-gps", str(photo_root)])
    assert result.exit_code == 0, result.output
    assert "checked: 2" in result.output.lower()
    assert "with gps: 1" in result.output.lower()

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    assert cat.get_camera_gps("sha-a") is not None
    assert cat.get_camera_gps("sha-b") is None
    cat.close()


@needs_exiftool
def test_backfill_skips_already_populated_rows(tmp_path: Path):
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()

    img = photo_root / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(img)
    _inject_exif_gps(img, lat=-52.5, lon=-60.4)
    cat.upsert_photo("sha-a", img, filesize=img.stat().st_size, mtime=img.stat().st_mtime)
    cat.set_camera_gps("sha-a", latitude=99.0, longitude=99.0, altitude=None)
    cat.close()

    runner = CliRunner()
    result = runner.invoke(app, ["backfill-exif-gps", str(photo_root)])
    assert result.exit_code == 0, result.output
    assert "skipped: 1" in result.output.lower()

    # The pre-existing (99, 99) value should be untouched.
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    got = cat.get_camera_gps("sha-a")
    assert got is not None
    assert got["latitude"] == 99.0
    cat.close()


@needs_exiftool
def test_backfill_force_refreshes_populated_rows(tmp_path: Path):
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()

    img = photo_root / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(img)
    _inject_exif_gps(img, lat=-52.5, lon=-60.4)
    cat.upsert_photo("sha-a", img, filesize=img.stat().st_size, mtime=img.stat().st_mtime)
    cat.set_camera_gps("sha-a", latitude=99.0, longitude=99.0, altitude=None)
    cat.close()

    runner = CliRunner()
    result = runner.invoke(app, ["backfill-exif-gps", str(photo_root), "--force"])
    assert result.exit_code == 0, result.output

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    got = cat.get_camera_gps("sha-a")
    assert got is not None
    assert abs(got["latitude"] - -52.5) < 1e-3
    cat.close()


@needs_exiftool
def test_backfill_handles_missing_files_gracefully(tmp_path: Path):
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()

    # Reference a path that doesn't exist (simulates portable catalog).
    cat.upsert_photo(
        "sha-missing",
        Path("/nonexistent/path.jpg"),
        filesize=0,
        mtime=0.0,
    )
    cat.close()

    runner = CliRunner()
    result = runner.invoke(app, ["backfill-exif-gps", str(photo_root)])
    assert result.exit_code == 0, result.output
    assert "missing: 1" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_cli_backfill_gps.py -v
```

Expected: FAIL — no `backfill-exif-gps` subcommand.

- [ ] **Step 3: Implement the command**

In `src/pixsage/cli.py`, add after the `geolocate` command (before `export`, around line 547):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_cli_backfill_gps.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/cli.py tests/test_cli_backfill_gps.py
git commit -m "feat(cli): backfill-exif-gps for catalogs predating EXIF extraction"
```

---

## Task 9: Full test sweep + smoke-run on a real corpus

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full test suite**

```
pytest -x
```

Expected: PASS for everything previously passing; the new test files add ~17 tests; total should go from 214 → ~231 passing.

- [ ] **Step 2: Smoke-test against the iPhone corpus**

```
python -m pixsage backfill-exif-gps "E:\iphone 15 pro"
```

Expected output (numbers ~ish): `checked: 4523 with gps: ~4500 no gps: ~20 skipped: 0 missing: 0 errored: 0`. Wall-clock ~1-3 min (exiftool metadata-only walk).

Sanity-check a few rows:

```
python -c "import sqlite3; c=sqlite3.connect(r'E:\iphone 15 pro\.photoindex\catalog.db'); print('with-gps:', c.execute('SELECT COUNT(*) FROM photos WHERE exif_latitude IS NOT NULL').fetchone()[0]); print('total:', c.execute('SELECT COUNT(*) FROM photos').fetchone()[0])"
```

- [ ] **Step 3: Smoke-test that `geolocate` now skips the EXIF-GPS photos**

```
python -m pixsage geolocate "E:\iphone 15 pro" --geolocator mock
```

Expected: `processed=~20 skipped=0 errored=0` (only the ~20 GPS-less photos get the mock geolocator). Without our changes it would have processed all 4,523.

- [ ] **Step 4: Smoke-test `--all` override**

```
python -m pixsage geolocate "E:\iphone 15 pro" --geolocator mock --all
```

Expected: `processed=~4500 skipped=0` (predictions overwrite for everything because mock isn't real GeoCLIP — confirms the iterator now returns all rows when `--all` is set).

- [ ] **Step 5: No commit needed for this task — it's pure verification.**

If smoke tests pass, the feature is shipped end-to-end.

---

## Optional follow-up (NOT part of this plan, but worth flagging)

- **`/map` route in serve**: now that catalog has real GPS for 4,523 photos, the natural next deliverable is a `/map` route in the webapp that renders the same Leaflet view we built in `scripts/build_iphone_map.py`. Reuses the same tile setup. ~1-2 hours of work, separable PR.
- **Pruning the now-redundant GeoCLIP predictions**: ~22,500 rows in `geo_predictions` for photos that now have real EXIF GPS. Could add a `--prune-redundant` flag to backfill. Tiny disk cost; left as-is for now.
- **Update photographer-handoff doc**: when the photographer's next corpus is GPS-bearing, the workflow becomes `tag → embed → (skip geolocate, EXIF already populated)`. Worth a one-line note in `docs/photographer-handoff.md`.
