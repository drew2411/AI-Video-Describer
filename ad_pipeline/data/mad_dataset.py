"""MAD dataset loading utilities.

This module will host helpers for loading MAD annotations (JSON/CSV) and
frame embeddings (H5), refactored out of the legacy scripts in `core/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal
import json

import h5py
import numpy as np

from ad_pipeline.config.base_config import DEFAULT_CONFIG


SplitName = Literal["train", "val", "test"]


def load_mad_json(split: SplitName, json_paths: Dict[str, Path]) -> Dict:
    """Load MAD JSON annotations for the given split.

    Parameters
    ----------
    split:
        One of "train", "val", or "test".
    json_paths:
        Mapping from split name to JSON path. During the transition this
        lets us re-use the existing configuration, and it can be folded
        into a richer config object later.
    """

    path = json_paths[split]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_frame_embeddings(movie_id: str, h5_path: Path | None = None) -> np.ndarray:
    """Load frame embeddings for a given movie from an H5 file.

    This is a thin wrapper that will eventually replace the equivalent
    function in `core/mad_ad_generator_final.py`.
    """

    h5_path = h5_path or DEFAULT_CONFIG.h5_frames_path

    with h5py.File(h5_path, "r") as f:
        if movie_id not in f:
            raise KeyError(f"Movie {movie_id!r} not found in frame embeddings file {h5_path}")
        return np.asarray(f[movie_id], dtype=np.float32)

