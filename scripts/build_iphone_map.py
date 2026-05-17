"""Turn an exiftool GPS JSON dump into a self-contained Leaflet map HTML.

Reads a JSON array of records with GPSLatitude / GPSLongitude (plus optional
FileName / Directory / DateTimeOriginal) and writes a single static HTML file
with a satellite/streets toggle and one marker per geotagged photo.

Originally built for the iPhone 15 Pro corpus; paths/title are now CLI args
so it works for any corpus. Defaults preserve the original no-arg behaviour.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
html,body,#map{height:100%;margin:0;font-family:system-ui,sans-serif}
.info{position:absolute;top:8px;right:8px;background:#fff;padding:8px 10px;border-radius:6px;
      box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:1000;font-size:13px}
.popup b{display:block;margin-bottom:2px}
.popup .meta{color:#666;font-size:11px}
</style></head>
<body>
<div id="map"></div>
<div class="info"><b>__TITLE__</b><br>__N__ photos with GPS</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const POINTS = __POINTS__;
const map = L.map('map');
const sat = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 19, attribution: 'Imagery &copy; Esri'}
);
const streets = L.tileLayer(
  'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap, &copy; CARTO', subdomains: 'abcd'}
);
sat.addTo(map);
L.control.layers({'Satellite': sat, 'Streets': streets}, null, {position: 'topleft'}).addTo(map);
const layer = L.layerGroup();
const bounds = L.latLngBounds();
for (const p of POINTS) {
  const m = L.circleMarker([p.lat, p.lon], {
    radius: 4, color: '#ff3b30', weight: 1, fillColor: '#ff3b30', fillOpacity: 0.7
  });
  m.bindPopup(
    '<div class="popup"><b>' + p.name + '</b>' +
    '<span class="meta">' + p.dir + '<br>' + p.when + '<br>' +
    p.lat.toFixed(5) + ', ' + p.lon.toFixed(5) + '</span></div>'
  );
  layer.addLayer(m);
  bounds.extend([p.lat, p.lon]);
}
layer.addTo(map);
map.fitBounds(bounds, {padding: [30, 30]});
</script>
</body></html>
"""


def build_map(src: Path, out: Path, title: str, strip_prefix: str) -> int:
    records = json.loads(src.read_text(encoding="utf-8-sig"))
    points = []
    for r in records:
        lat = r.get("GPSLatitude")
        lon = r.get("GPSLongitude")
        if lat is None or lon is None:
            continue
        directory = r.get("Directory") or ""
        if strip_prefix:
            directory = directory.replace(strip_prefix, "")
        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "name": r.get("FileName", ""),
            "dir": directory,
            "when": r.get("DateTimeOriginal", ""),
        })

    html = (
        HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__N__", f"{len(points):,}")
        .replace("__POINTS__", json.dumps(points, separators=(",", ":")))
    )
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(points):,} points, {out.stat().st_size / 1024:.0f} KB)")
    return len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / ".iphone-gps.json",
        help="exiftool GPS JSON dump (default: .iphone-gps.json)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output HTML path (default: <input stem>.html, e.g. .iphone-map.html)",
    )
    parser.add_argument(
        "--title", default="Photo locations",
        help="map title shown in the corner badge and <title>",
    )
    parser.add_argument(
        "--strip-prefix", default="",
        help="leading Directory string to strip from popups "
             "(e.g. 'E:/iphone 15 pro/')",
    )
    args = parser.parse_args()

    out = args.output
    if out is None:
        stem = args.input.stem or "map"
        out = args.input.with_name(f"{stem}-map.html")
    build_map(args.input, out, args.title, args.strip_prefix)


if __name__ == "__main__":
    main()
