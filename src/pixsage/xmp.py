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


import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

EXIFTOOL = shutil.which("exiftool") or "exiftool"

SIDECAR_EXTENSIONS: frozenset[str] = frozenset({
    ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2",  # NOT .dng — DNG uses embedded XMP
})


def needs_sidecar(path: Path) -> bool:
    """Return True if this file should use an XMP sidecar (not embedded XMP)."""
    return path.suffix.lower() in SIDECAR_EXTENSIONS


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
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
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
    # Repeated `-tag=val` REPLACES list-typed XMP fields atomically; an empty
    # `-tag=` clears the field. The `+=` operator only appends and does not
    # interact with a leading `-tag=` clear (exiftool merges both ops).
    args = [
        EXIFTOOL,
        "-overwrite_original",
        "-charset", "utf8",
    ]
    if fields.subject:
        args.extend(f"-XMP-dc:Subject={s}" for s in fields.subject)
    else:
        args.append("-XMP-dc:Subject=")
    if fields.hierarchical_subject:
        args.extend(f"-XMP-lr:HierarchicalSubject={h}" for h in fields.hierarchical_subject)
    else:
        args.append("-XMP-lr:HierarchicalSubject=")
    if fields.description is not None:
        args.append(f"-XMP-dc:Description={fields.description}")
    if is_raw:
        # Write/update sidecar at target path.
        if target.exists():
            args.append(str(target))
        else:
            args.append(str(path))   # input: the raw file
            args.append("-o")
            args.append(str(target))  # output: the new sidecar
    else:
        args.append(str(path))
    try:
        subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"exiftool write failed: {e.stderr}") from e
