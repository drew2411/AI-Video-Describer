"""Entity-level memory for characters, locations, and objects.

This is a placeholder module where we will implement data structures
and helpers to track recurring entities across a movie, supporting
non-redundant, context-aware AD generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Entity:
    """Simple record for an entity mentioned in AD or subtitles."""

    name: str
    type: str  # e.g. "person", "location", "object"
    description: str = ""
    first_time: float | None = None
    last_time: float | None = None
    times_mentioned: int = 0
    aliases: List[str] = field(default_factory=list)

