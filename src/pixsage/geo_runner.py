from __future__ import annotations

import sys
from pathlib import Path

from pixsage.catalog import Catalog
from pixsage.geolocators.base import Geolocator
from pixsage.images import load_image


class GeoRunner:
    """Walks the catalog and computes top-K geolocation predictions per photo.

    For each photo:
      - skip if predictions for this geolocator already exist (and not --force)
      - load the image, run the geolocator
      - record top-K predictions in the catalog (replaces any prior ones for the
        same model)
    """

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
        if self.progress:
            from tqdm import tqdm
            iterator = tqdm(rows, unit="photo")
        else:
            iterator = rows

        for row in iterator:
            sha = row["sha256"]
            current_path = row["current_path"]

            if not self.force and self.catalog.get_geo_predictions(sha, info.name):
                stats["skipped"] += 1
                continue

            try:
                img = load_image(Path(current_path))
                preds = self.geolocator.predict([img])[0]
                self.catalog.record_geo_predictions(sha, info.name, preds)
                self.catalog.clear_error(sha)
                stats["processed"] += 1
            except Exception as e:
                self.catalog.mark_error(sha, str(e))
                stats["errored"] += 1
                msg = f"  error on {Path(current_path).name}: {e}"
                if self.progress:
                    from tqdm import tqdm
                    tqdm.write(msg, file=sys.stderr)
                else:
                    sys.stderr.write(msg + "\n")

        return stats
