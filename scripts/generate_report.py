"""Generate a single-file HTML report for a tagged photo directory.

Usage:
    python scripts/generate_report.py <photo_root> [output.html]

Walks the photo root, reads XMP via exiftool, embeds 512px JPEG thumbnails
as base64 data URLs, and writes a self-contained report HTML you can open
in any browser. No server, no dependencies beyond what's already installed.
"""
from __future__ import annotations

import base64
import html
import io
import sys
from pathlib import Path

from PIL import Image

from pixsage.images import LONG_EDGE_TARGET, load_image
from pixsage.walker import walk_photos
from pixsage.xmp import read_xmp


THUMB_LONG_EDGE = 512
MARKER_PREFIX = "auto-tagged-"


def thumb_data_url(path: Path) -> str:
    """Generate a 512px-long-edge JPEG thumbnail and encode as a data URL."""
    img = load_image(path)
    if max(img.size) > THUMB_LONG_EDGE:
        scale = THUMB_LONG_EDGE / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def split_tags(subject: list[str]) -> tuple[list[str], list[str]]:
    """Separate the auto-tagged-X markers from real keyword tags."""
    real, markers = [], []
    for t in subject:
        if t.startswith(MARKER_PREFIX):
            markers.append(t)
        else:
            real.append(t)
    return real, markers


HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pixsage report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         background: #0e0e10; color: #e8e8ea; margin: 0; padding: 24px;
         line-height: 1.45; }
  h1 { margin: 0 0 4px; font-size: 22px; }
  .meta { color: #888; font-size: 13px; margin-bottom: 24px; }
  .photo { display: grid; grid-template-columns: 280px 1fr; gap: 24px;
           padding: 16px; background: #16171b; border-radius: 8px;
           margin-bottom: 16px; align-items: start; }
  .photo img { width: 100%; height: auto; border-radius: 4px; display: block; }
  .name { font-weight: 600; font-size: 15px; margin-bottom: 8px; color: #e8e8ea; }
  .tags { margin-bottom: 12px; }
  .tag { display: inline-block; background: #2c3340; color: #d8e0ee;
         padding: 3px 10px; border-radius: 12px; font-size: 12px;
         margin: 2px 4px 2px 0; }
  .marker { background: #1a3a2a; color: #80d0a0; }
  .empty { color: #666; font-style: italic; font-size: 13px; }
  .desc { color: #b8b8bd; font-size: 13px; line-height: 1.5;
          padding: 8px 0 0; border-top: 1px solid #25262a; margin-top: 12px; }
  .desc .label { color: #666; font-size: 11px; text-transform: uppercase;
                 letter-spacing: 0.5px; margin-bottom: 4px; }
</style>
</head>
<body>
"""

HTML_FOOT = "</body></html>\n"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/generate_report.py <photo_root> [output.html]", file=sys.stderr)
        return 1
    photo_root = Path(sys.argv[1]).resolve()
    if not photo_root.is_dir():
        print(f"not a directory: {photo_root}", file=sys.stderr)
        return 1
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else photo_root / "pixsage-report.html"

    paths = list(walk_photos(photo_root))
    print(f"Found {len(paths)} images. Building report...")

    parts = [HTML_HEAD]
    parts.append(f"<h1>{html.escape(photo_root.name)}</h1>")
    parts.append(f'<div class="meta">{len(paths)} photos · {photo_root}</div>')

    tagged = 0
    for path in paths:
        try:
            url = thumb_data_url(path)
            fields = read_xmp(path, is_raw=path.suffix.lower() not in {".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif", ".png", ".dng"})
        except Exception as e:  # noqa: BLE001
            print(f"  skip {path.name}: {e}", file=sys.stderr)
            continue

        real_tags, markers = split_tags(fields.subject)
        if real_tags:
            tagged += 1

        parts.append('<div class="photo">')
        parts.append(f'<img src="{url}" alt="{html.escape(path.name)}">')
        parts.append('<div>')
        parts.append(f'<div class="name">{html.escape(path.name)}</div>')
        parts.append('<div class="tags">')
        if real_tags:
            for t in real_tags:
                parts.append(f'<span class="tag">{html.escape(t)}</span>')
        else:
            parts.append('<span class="empty">no tags</span>')
        for m in markers:
            parts.append(f'<span class="tag marker">{html.escape(m)}</span>')
        parts.append('</div>')
        if fields.description:
            parts.append('<div class="desc">')
            parts.append('<div class="label">Caption</div>')
            parts.append(html.escape(fields.description))
            parts.append('</div>')
        parts.append('</div></div>')

    parts.append(HTML_FOOT)
    output.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {output} ({tagged}/{len(paths)} tagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
