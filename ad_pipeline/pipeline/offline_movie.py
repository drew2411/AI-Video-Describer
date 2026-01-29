"""Offline movie-level pipeline entrypoints.

This module is the future home for the `process_movie` logic currently
implemented in `core/mad_ad_generator_final.py`.
"""

from __future__ import annotations

from typing import Optional


def process_movie(
    movie_id: str,
    max_shots: Optional[int] = None,
    verify_only: bool = False,
) -> None:
    """Placeholder offline pipeline.

    This mirrors the legacy CLI arguments so that we can gradually move
    logic here without changing external behavior.
    """

    raise NotImplementedError(
        "process_movie() has not been migrated from the legacy script yet."
    )

