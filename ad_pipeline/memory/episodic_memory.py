"""Episodic (event-level) memory for the AD pipeline.

Stores a time-ordered history of shots/events and associated AD, which
can later be used to provide long-range context and avoid repetition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EpisodeEvent:
    """Minimal representation of an AD-relevant event."""

    shot_id: int
    start_time: float
    end_time: float
    ad_text: str
    extra: Dict | None = None


class EpisodicMemory:
    """In-memory store of recent events.

    This is intentionally simple for now; it can grow into a more
    sophisticated, windowed or hierarchical memory later.
    """

    def __init__(self) -> None:
        self.events: List[EpisodeEvent] = []

    def add_event(self, event: EpisodeEvent) -> None:
        self.events.append(event)

