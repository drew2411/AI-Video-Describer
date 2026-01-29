"""Prompt templates and shot-type classification for AD generation.

During the refactor, the logic from `core/mad_ad_generator_final.py`
responsible for classifying shot types and constructing prompts will be
moved into this module.
"""

from __future__ import annotations

from typing import List


def classify_shot_type(context_ads: List[str], subtitle_text: str) -> str:
    """Placeholder shot-type classifier.

    The legacy implementation distinguishes special cases such as logo
    or credits shots. For now, always return "normal".
    """

    # TODO: port the real heuristic from the legacy script.
    return "normal"

