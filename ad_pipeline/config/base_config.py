"""Base configuration for the MAD audio description pipeline.

This module will centralize default paths, model names, and other settings
that were previously hard-coded in scripts such as `core/mad_ad_generator_final.py`.

During the initial refactor, treat these values as placeholders and gradually
migrate real configuration here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseConfig:
    """Default configuration values for the pipeline.

    These values replace the hard-coded paths that previously lived in
    the legacy scripts (e.g. ``core/mad_ad_generator_final.py``). They
    are intentionally conservative and can be overridden via
    :func:`ad_pipeline.config.env_config.load_config_from_env`.
    """

    # Project-level roots
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    outputs_dir: Path = project_root / "outputs"

    # ChromaDB defaults (equivalent to Config.CHROMA_* in the legacy code)
    chroma_dir: Path = project_root / "chroma_db_mad_joint"
    chroma_collection: str = "mad_joint_clip_L14"

    # MAD dataset root; all other MAD paths are derived from this.
    mad_root: Path = project_root / "MAD"

    # --- Derived MAD paths -------------------------------------------------
    @property
    def mad_features_dir(self) -> Path:
        return self.mad_root / "features"

    @property
    def mad_annotations_dir(self) -> Path:
        return self.mad_root / "annotations"

    @property
    def h5_frames_path(self) -> Path:
        return self.mad_features_dir / "CLIP_L14_frames_features_5fps.h5"

    @property
    def h5_text_path(self) -> Path:
        return self.mad_features_dir / "CLIP_L14_language_tokens_features.h5"

    @property
    def json_train(self) -> Path:
        return self.mad_annotations_dir / "MAD-v1" / "MAD_train.json"

    @property
    def json_val(self) -> Path:
        return self.mad_annotations_dir / "MAD-v1" / "MAD_val.json"

    @property
    def json_test(self) -> Path:
        return self.mad_annotations_dir / "MAD-v1" / "MAD_test.json"

    @property
    def csv_ad(self) -> Path:
        return self.mad_annotations_dir / "MAD-v2" / "mad-v2-ad-unnamed.csv"

    @property
    def csv_subs(self) -> Path:
        return self.mad_annotations_dir / "MAD-v2" / "mad-v2-subs.csv"


DEFAULT_CONFIG = BaseConfig()

