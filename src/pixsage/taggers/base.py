from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tag:
    name: str
    confidence: float
    hierarchy: str | None
    source: str  # "florence2" | "ram++"
