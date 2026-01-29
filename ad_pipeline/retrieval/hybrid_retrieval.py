"""Hybrid visual+text retrieval logic for MAD.

This module is the new home for the `retrieve_hybrid_context`-style
functionality from `core/mad_ad_generator_final.py`.

At this stage it only defines a placeholder API; the actual logic will
be migrated here progressively.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import chromadb


def retrieve_hybrid_context(
    collection: chromadb.Collection,
    shot_emb: np.ndarray,
    shot_info: Dict,
    mad_data: Dict,
    visual_k: int = 3,
    text_k: int = 5,
) -> Dict:
    """Placeholder for hybrid retrieval.

    Parameters mirror the legacy function so that the pipeline can be
    refactored with minimal friction. The implementation should combine
    visual and cross-modal retrieval using ChromaDB, then return a
    dictionary with at least the keys:

    - "combined_context"
    - "combined_distances"
    - "combined_sources"
    """

    # TODO: move the real implementation here from the legacy script.
    return {
        "visual_context": [],
        "visual_distances": [],
        "cross_modal_context": [],
        "cross_modal_distances": [],
        "combined_context": [],
        "combined_distances": [],
        "combined_sources": [],
    }

