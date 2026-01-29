# Migration Guide: Legacy MAD Pipeline → `ad_pipeline/`

This guide explains how the original MAD-based audio description (AD) codebase has been reorganized into a modular `ad_pipeline/` package and a `deprecated/` area for legacy scripts.

## 1. High-Level Changes

- Introduced a new package: `ad_pipeline/` for all current and future AD pipeline code.
- Created a `deprecated/` directory for experimental, legacy, or superseded scripts.
- Moved Jupyter notebooks into a `notebooks/` folder.
- Added `docs/architecture_review_2026-01-29.md` to capture the design rationale.

## 2. New Package Layout (`ad_pipeline/`)

The new package is structured as follows:

- `ad_pipeline/`
  - `config/`
    - `base_config.py` – central configuration (paths, model names, defaults).
    - `env_config.py` – environment / `.env` based overrides.
  - `data/`
    - `mad_dataset.py` – MAD dataset loaders (JSON/CSV/H5).
    - `mad_downloader.py` – downloader / setup helpers (from `MAD_Downloader.py`).
  - `vision/`
    - `frame_embeddings.py` – frame-embedding utilities.
    - `shot_detector.py` – shot detection and pooling.
  - `retrieval/`
    - `chroma_client.py` – ChromaDB client and configuration helpers.
    - `hybrid_retrieval.py` – joint visual+text retrieval logic.
    - `index_builder.py` – index-building utilities (from `core/chroma_index.py`, `core/chroma_resume_frames.py`).
  - `memory/`
    - `entity_memory.py` – placeholder for character/location/object memory.
    - `episodic_memory.py` – placeholder for time-ordered event memory.
  - `generation/`
    - `groq_client.py` – Groq (or other LLM) client initialization.
    - `prompts.py` – prompt templates and shot-type logic.
    - `ad_generator.py` – AD generation functions.
  - `pipeline/`
    - `offline_movie.py` – orchestrates offline movie AD generation.
    - `cli.py` – command-line entry point.
    - `evaluation.py` – wrappers around evaluation utilities.

> Note: Many of these modules start as thin wrappers or placeholders and will be filled in by refactoring logic out of the old scripts.

## 3. Old → New Mapping

### Core scripts

- `core/mad_ad_generator_final.py`
  - Config → `ad_pipeline/config/base_config.py`, `env_config.py`.
  - Data loading → `ad_pipeline/data/mad_dataset.py` and `ad_pipeline/vision/frame_embeddings.py`.
  - Shot detection → `ad_pipeline/vision/shot_detector.py`.
  - Chroma verification & retrieval → `ad_pipeline/retrieval/chroma_client.py`, `hybrid_retrieval.py`.
  - LLM initialization → `ad_pipeline/generation/groq_client.py`.
  - Prompting & shot classification → `ad_pipeline/generation/prompts.py`.
  - AD generation → `ad_pipeline/generation/ad_generator.py`.
  - Main pipeline (`process_movie`) → `ad_pipeline/pipeline/offline_movie.py`.
  - CLI → `ad_pipeline/pipeline/cli.py`.

- `core/chroma_index.py` → `ad_pipeline/retrieval/index_builder.py`.
- `core/chroma_resume_frames.py` → `ad_pipeline/retrieval/index_builder.py` (or a helper next to it).
- `core/evaluate_ads.py` → `ad_pipeline/pipeline/evaluation.py`.
- `MAD_Downloader.py` → `ad_pipeline/data/mad_downloader.py`.

### Notebooks

- `core/data_exploration.ipynb` → `notebooks/data_exploration.ipynb`.
- `core/single_movie_run.ipynb` → `notebooks/single_movie_run.ipynb`.

## 4. Deprecated / Legacy Code

A new `deprecated/` directory contains older or experimental code that is not part of the supported pipeline:

- `deprecated/old/` – the original `old/` directory, containing early prototypes such as:
  - `10142_test.py`
  - `analyser.py`
  - `analyser_moondream.py`
  - `claude_enhanced.py`
  - `query_index.py`
  - `video_analysis.json`
  - `video_processor.py`
- `deprecated/legacy_scripts/` – one-off debug or helper scripts (e.g., `test_chroma.py`).
- `deprecated/legacy_docs/` – outdated notes or TODO lists, if any.

These files remain available for reference but should be treated as read-only historical context.

## 5. How to Run the New Pipeline (Planned)

The goal is to replace the old entry point:

- **Old:** `python core/mad_ad_generator_final.py --movie 10142 --max_shots 20`

With a new, clearer CLI:

- **New (planned):** `python -m ad_pipeline.pipeline.cli --movie 10142 --max_shots 20`

During the transition, both may coexist until the new CLI is fully wired up and tested.

## 6. Status

- Directory structure and placeholder modules for `ad_pipeline/` have been created.
- Legacy scripts have been grouped under `deprecated/`.
- Next steps: progressively migrate logic from legacy scripts into the appropriate `ad_pipeline/` modules and add tests.

