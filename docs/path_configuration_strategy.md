# Path Configuration Strategy

This document describes how to replace hard-coded, Windows-only paths
with a flexible, cross-platform configuration system.

## 1. Current state and issues

The legacy MAD pipeline uses absolute Windows paths baked into the code.
Hard-coded examples include:

- `core/mad_ad_generator_final.py` (class `Config`):
  - `H5_FRAMES = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_frames_features_5fps.h5"`
  - `H5_TEXT = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_language_tokens_features.h5"`
  - `JSON_TRAIN`, `JSON_VAL`, `JSON_TEST` under
    `...\MAD\annotations\MAD-v1\`.
  - `CSV_AD`, `CSV_SUBS` under `...\MAD\annotations\MAD-v2\`.
- `core/chroma_index.py` and `core/chroma_resume_frames.py` define
  similar `H5_FRAMES_PATH`, `H5_TEXT_PATH`, and `ANNOTATIONS_JSON`
  pointing to the same Windows-only locations.

Problems:

- The code does not run on non-Windows machines without editing source
  files.
- Paths are duplicated across multiple files, making changes error-prone
  and inconsistent.
- Dataset location cannot be changed without modifying code.

## 2. Design goals

- **Cross-platform**: use `pathlib.Path` and relative paths from the
  project root instead of OS-specific separators.
- **Single source of truth**: define dataset and index paths in one
  configuration object.
- **Override-friendly**: allow users to change dataset / ChromaDB
  locations via environment variables or config files, not code edits.
- **Backwards-compatible**: introduce the new config layer without
  immediately rewriting all legacy code, to avoid blocking algorithmic
  work.

## 3. Proposed configuration system

1. Use `BaseConfig` in `ad_pipeline.config.base_config` as the central
   configuration object:
   - `project_root`: inferred from the location of the code.
   - `mad_root`: default `project_root / "MAD"`.
   - Derived properties:
     - `h5_frames_path`, `h5_text_path`.
     - `json_train`, `json_val`, `json_test`.
     - `csv_ad`, `csv_subs`.
   - Chroma settings: `chroma_dir`, `chroma_collection`.
2. Use `load_config_from_env` in `env_config.py` to override paths via
   environment variables, for example:
   - `MAD_ROOT=/mnt/data/MAD` (future extension).
   - `CHROMA_DIR=/mnt/data/chroma_db_mad_joint`.
3. New code in `ad_pipeline/` should only depend on `BaseConfig` paths
   (e.g. `DEFAULT_CONFIG.h5_frames_path`) and never embed absolute
   strings.

## 4. Migration plan

**Stage 1  configuration layer (non-breaking)**

- Keep the legacy `Config` class in `core/mad_ad_generator_final.py` and
  related files unchanged for now.
- Implement and validate `BaseConfig` and its MAD path properties (this
  has been done in `base_config.py`).
- Update new helper modules (e.g. `ad_pipeline.data.mad_dataset`) to use
  `DEFAULT_CONFIG` instead of hard-coded paths.

**Stage 2  gradually refactor legacy code**

- In `core/mad_ad_generator_final.py`, replace direct string constants
  with references to `BaseConfig`, for example:
  - `Config.H5_FRAMES` > `cfg.h5_frames_path`.
  - `Config.JSON_TRAIN` > `cfg.json_train`.
- Pass a `BaseConfig` (or derived config) instance through the pipeline
  rather than relying on static class attributes.
- Apply the same pattern in `core/chroma_index.py` and
  `core/chroma_resume_frames.py`.

**Stage 3  fully migrate to `ad_pipeline/`**

- Once `ad_pipeline.pipeline.offline_movie.process_movie` is populated
  with the real logic, source all paths exclusively from `BaseConfig` /
  `load_config_from_env`.
- Keep the old `Config` class only as a compatibility shim (or remove it
  entirely once no callers depend on it).

## 5. Timing decision: fix now or later?

**Option A  fix path configuration now**

Pros:

- Immediately makes the project runnable on different machines
  (including macOS and Linux) without editing code.
- Reduces duplication and risk of inconsistent fixes across multiple
  files.
- Provides a solid foundation for future features (memory, timing,
  improved prompts) without revisiting path logic later.
- Changes are mostly mechanical and low risk if done incrementally.

Cons:

- Requires some up-front refactoring effort before adding new AD
  features.

**Option B  defer until after algorithm improvements**

Pros:

- Lets you prototype new AD algorithms directly in the legacy script
  without touching configuration.

Cons:

- You continue to accumulate technical debt around paths and config.
- Every collaborator must hand-edit paths to run experiments.
- Refactoring later will be more error-prone and may break in-flight
  experiments.

### Recommendation

Adopt a **hybrid version of Option A**:

- **Do the configuration work now**, but in a backwards-compatible way:
  - Keep the legacy `Config` class temporarily.
  - Introduce and stabilize `BaseConfig` + environment overrides.
  - Start porting helpers (data loading, retrieval, index building) to
    use `BaseConfig`.
- Defer full removal of the legacy `Config` until after the new
  `ad_pipeline` pipeline is feature-complete and well tested.

This approach keeps development moving while preventing path-related
technical debt from growing and blocking collaboration or deployment
later.

