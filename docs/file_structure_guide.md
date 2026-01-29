# File Structure Guide for AI-Video-Describer

This document explains how the repository is organized after the initial
refactor toward the modular `ad_pipeline/` package.

## 1. Top-level layout

- `README.md` – high-level project description and quick-start notes.
- `requirements.txt` – Python dependencies.
- `core/` – original MAD-based pipeline and utilities (still the only
  fully wired pipeline at the moment).
- `ad_pipeline/` – new, modular package where refactored code lives
  (work in progress).
- `deprecated/` – legacy or experimental scripts and notes kept for
  reference but not part of the supported pipeline.
- `docs/` – architecture review and design documentation.
- `notebooks/` – exploratory Jupyter notebooks.

## 2. The `ad_pipeline/` package

The goal of `ad_pipeline/` is to host a clean, testable implementation
of the MAD-based audio description pipeline, separated into focused
modules.

### 2.1 `config/`

- `base_config.py` – defines `BaseConfig` and `DEFAULT_CONFIG`, which
  centralize project paths (project root, MAD data root, ChromaDB
  directory, etc.). This is the main replacement for hard-coded paths in
  the legacy scripts.
- `env_config.py` – provides `load_config_from_env`, which applies
  environment-variable overrides (e.g. `CHROMA_DIR`, `CHROMA_COLLECTION`)
  on top of `DEFAULT_CONFIG`.

### 2.2 `data/`

- `mad_dataset.py` – helpers for loading MAD annotations and frame
  embeddings. Currently thin wrappers around H5/JSON loading; the intent
  is to move `load_mad_json` and related functions here from
  `core/mad_ad_generator_final.py`.
- `mad_downloader.py` – placeholder for dataset download/setup routines
  (logic to be migrated from `MAD_Downloader.py`).

### 2.3 `vision/`

- `frame_embeddings.py` – future home for shared utilities dealing with
  CLIP frame embeddings (loading, normalization, batching, etc.).
- `shot_detector.py` – contains a standalone `detect_shots` function
  based on the legacy cosine-similarity implementation. The intention is
  to keep all shot-boundary logic here, separate from dataset loading
  and generation.

### 2.4 `retrieval/`

- `chroma_client.py` – helpers for creating and configuring a
  `chromadb.PersistentClient` given a path.
- `hybrid_retrieval.py` – placeholder for the main
  `retrieve_hybrid_context` logic that currently lives in
  `core/mad_ad_generator_final.py`.
- `index_builder.py` – placeholder for ChromaDB index-building utilities
  corresponding to `core/chroma_index.py` and
  `core/chroma_resume_frames.py`.

### 2.5 `memory/`

- `entity_memory.py` – defines a simple `Entity` dataclass as the
  starting point for tracking recurring characters, locations, and
  objects across a movie.
- `episodic_memory.py` – defines `EpisodeEvent` and `EpisodicMemory` as
  a minimal event history, to later support long-range context and
  repetition avoidance.

### 2.6 `generation/`

- `groq_client.py` – provides `init_groq_from_env`, a helper for
  constructing a Groq client using the `GROQ_API_KEY` environment
  variable.
- `prompts.py` – placeholder for prompt templates and shot-type
  classification logic used during AD generation.
- `ad_generator.py` – defines a `generate_ad` function with a signature
  compatible with the legacy implementation; the body is currently a
  stub and will be filled in as logic is ported.

### 2.7 `pipeline/`

- `__init__.py` – marks the subpackage; no logic.
- `offline_movie.py` – placeholder `process_movie` function, which will
  eventually absorb the main movie-level pipeline from
  `core/mad_ad_generator_final.py`.
- `cli.py` – planned command-line entry point
  (`python -m ad_pipeline.pipeline.cli`) mirroring the existing legacy
  CLI options while delegating to `offline_movie.process_movie`.
- `evaluation.py` – placeholder wrapper around evaluation metrics to be
  moved from `core/evaluate_ads.py`.

## 3. Legacy vs. new code

- The **legacy pipeline** still lives in `core/mad_ad_generator_final.py`
  and related `core/*` helpers; it remains the single source of truth
  for end-to-end behavior.
- The **new pipeline** under `ad_pipeline/` is partially implemented:
  directory structure and some utilities exist, but most heavy logic is
  still in the legacy files.
- `MIGRATION_GUIDE.md` documents, at a finer level, which functions and
  responsibilities in `core/` are intended to move into which
  `ad_pipeline/` modules.

## 4. How to find things

- **Shot detection logic** – see `core/mad_ad_generator_final.py` today;
  new implementation lives in `ad_pipeline/vision/shot_detector.py`.
- **MAD dataset I/O** – legacy in `core/mad_ad_generator_final.py` and
  `core/chroma_index.py`; new home is `ad_pipeline/data/mad_dataset.py`.
- **ChromaDB index and queries** – legacy in `core/chroma_index.py` and
  `core/mad_ad_generator_final.py`; new home is
  `ad_pipeline/retrieval/`.
- **LLM client and prompts** – legacy in
  `core/mad_ad_generator_final.py`; new home is `ad_pipeline/generation/`.
- **End-to-end pipeline orchestration** – currently only in
  `core/mad_ad_generator_final.py`; planned home is
  `ad_pipeline/pipeline/offline_movie.py` plus `cli.py`.

