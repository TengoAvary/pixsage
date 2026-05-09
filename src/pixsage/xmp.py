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
