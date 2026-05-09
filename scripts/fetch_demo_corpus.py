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
