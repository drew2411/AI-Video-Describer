"""Evaluation helpers for generated audio descriptions.

This module will wrap the legacy metrics from `core/evaluate_ads.py` and
expose a cleaner API for use in tests and scripts.
"""

from __future__ import annotations

from typing import Any, Dict, List


def evaluate_ads(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """Placeholder evaluation wrapper.

    In the future this function will call into the ported BLEU/ROUGE/METEOR
    utilities. For now it raises NotImplementedError to make the missing
    functionality explicit.
    """

    raise NotImplementedError(
        "Evaluation has not yet been migrated from core/evaluate_ads.py."
    )

