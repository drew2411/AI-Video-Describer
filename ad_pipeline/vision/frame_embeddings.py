"""Frame embedding utilities.

This module will centralize logic for working with frame-level visual
embeddings, currently stored in H5 files for the MAD dataset.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py


def load_frame_embeddings(movie_id: str, h5_path: Path) -> np.ndarray:
    """Load frame embeddings for a single movie from an H5 file."""

    with h5py.File(h5_path, "r") as f:
        if movie_id not in f:
            raise KeyError(f"Movie {movie_id!r} not found in frame embeddings file {h5_path}")
        return np.asarray(f[movie_id], dtype=np.float32)

