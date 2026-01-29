"""Shot detection utilities.

This module is intended to host the shot detection and pooling logic
from `core/mad_ad_generator_final.py`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def detect_shots(
    embeddings: np.ndarray,
    sim_threshold: float,
    min_shot_len: int,
) -> List[Tuple[int, int]]:
    """Very simple shot detection based on cosine similarity.

    This is copied conceptually from the legacy implementation; the
    exact parameters and behavior can be tuned as we continue the
    refactor.
    """

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms

    # Cosine similarity between consecutive frames
    sims = (normed[1:] * normed[:-1]).sum(axis=1)

    shots: List[Tuple[int, int]] = []
    start = 0
    for i, s in enumerate(sims, start=1):
        if s < sim_threshold and (i - start) >= min_shot_len:
            shots.append((start, i))
            start = i

    # Final shot
    if start < len(embeddings) - 1:
        shots.append((start, len(embeddings) - 1))

    return shots

