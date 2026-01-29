"""Audio description generation entrypoints.

This module will wrap LLM calls and use prompts defined in
:mod:`ad_pipeline.generation.prompts` to generate AD for a given shot
and its context.
"""

from __future__ import annotations

from typing import Dict, List

from groq import Groq

from .prompts import classify_shot_type


def generate_ad(
    client: Groq,
    shot_info: Dict,
    context_ads: List[str],
    context_sources: List[str],
    subtitle_text: str = "",
    movie_name: str = "",
    movie_synopsis: str = "",
) -> str:
    """Placeholder AD generator.

    The signature mirrors the legacy function so we can gradually move
    its implementation here. For now this returns an empty string.
    """

    _ = classify_shot_type(context_ads, subtitle_text)
    # TODO: port full prompt + Groq call here.
    return ""

