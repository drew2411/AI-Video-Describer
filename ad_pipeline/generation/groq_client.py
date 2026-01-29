"""Groq LLM client helpers for AD generation.

The legacy pipeline used a Groq client created inside
`core/mad_ad_generator_final.py`. That logic will be migrated here.
"""

from __future__ import annotations

import os

from groq import Groq


def init_groq_from_env() -> Groq:
    """Initialize a Groq client using environment variables.

    Expects a `GROQ_API_KEY` environment variable to be set.
    """

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    return Groq(api_key=api_key)

