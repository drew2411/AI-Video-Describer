"""Environment-aware configuration helpers.

This module is intended to layer environment variables (e.g. from `.env`)
on top of the defaults defined in :mod:`ad_pipeline.config.base_config`.

For now it provides a simple helper for obtaining the effective config
object; more sophisticated logic can be added as needed.
"""

from __future__ import annotations

import os
from dataclasses import replace

from .base_config import BaseConfig, DEFAULT_CONFIG


def load_config_from_env(base: BaseConfig | None = None) -> BaseConfig:
    """Return a configuration instance with environment overrides applied.

    Currently supports a minimal set of overrides and is intentionally
    conservative. Extend this as additional configuration needs arise.
    """

    cfg = base or DEFAULT_CONFIG

    chroma_dir = os.getenv("CHROMA_DIR")
    chroma_collection = os.getenv("CHROMA_COLLECTION")

    updates = {}
    if chroma_dir:
        updates["chroma_dir"] = cfg.project_root / chroma_dir
    if chroma_collection:
        updates["chroma_collection"] = chroma_collection

    return replace(cfg, **updates)

