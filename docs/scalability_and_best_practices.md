# Scalability and Best Practices Plan

This document outlines how to evolve the AI-Video-Describer codebase
into a scalable, maintainable system while iterating on the AD
algorithms.

## 1. Code organization and modularity

- Keep **legacy** code in `core/` as a stable reference until the new
  pipeline is feature-complete.
- Concentrate all **new** work inside `ad_pipeline/` with clear
  boundaries: `config/`, `data/`, `vision/`, `retrieval/`, `memory/`,
  `generation/`, `pipeline/`.
- For each concern (e.g. retrieval, timing, memory), define a small
  public API (functions or classes) and keep implementation details
  private to that subpackage.
- Avoid monolithic scripts; prefer importable modules with small
  entry-point wrappers (e.g. `ad_pipeline.pipeline.cli`).
- Use simple interfaces (plain functions / dataclasses) instead of
  deeply nested class hierarchies.

## 2. Configuration and dependency injection

- Use `BaseConfig` and `DEFAULT_CONFIG` (in
  `ad_pipeline.config.base_config`) as the single source of truth for
  project paths and default settings.
- Use `load_config_from_env` (in `env_config.py`) to override paths via
  environment variables or `.env` files (e.g. `CHROMA_DIR`,
  `CHROMA_COLLECTION`).
- Pass configuration objects explicitly into functions instead of using
  global state; e.g. `build_chroma_client(cfg: BaseConfig)` rather than
  reading from `os.environ` inside the function.
- Treat external services (ChromaDB, Groq, future VLMs) as injectable
  dependencies: construct clients once in a top-level module or CLI, and
  pass them downward.
- When adding new modules, design functions to accept generic
  interfaces (e.g. an abstract retrieval API) so they can be swapped out
  without changing callers.

## 3. Error handling and logging

- Adopt the standard library `logging` module for structured logging
  with log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- Define a small logging helper in `ad_pipeline/` (e.g.
  `ad_pipeline.logging_utils`) that configures a root logger with a
  consistent format (timestamp, level, component, message).
- Use clear, domain-specific exceptions (e.g. `MissingMADDataError`,
  `ChromaIndexNotFoundError`, `GroqCallError`) instead of generic
  `Exception` where it aids debugging.
- At the pipeline orchestration level, catch expected exceptions and
  emit user-friendly messages while logging full stack traces for
  debugging.
- For external services (Chroma, Groq), add lightweight retry logic
  around transient errors, with backoff and maximum attempts.

## 4. Testing strategy

- Standardize on **pytest** for tests.
- Mirror the `ad_pipeline/` structure under `tests/`, e.g.:
  - `tests/test_data_mad_dataset.py`
  - `tests/test_vision_shot_detector.py`
  - `tests/test_retrieval_hybrid.py`
- Provide small, synthetic fixtures for unit tests (tiny H5 files,
  minimal JSON/CSV snippets) instead of relying on full MAD files.
- Mock external services:
  - ChromaDB: use an in-memory client or a fake implementation.
  - Groq: stub the client so tests never hit the real API.
- Add regression tests for key behaviors:
  - Shot detection on hand-crafted embedding sequences.
  - Hybrid retrieval ranking and filtering.
  - Prompt formatting and basic AD generation constraints.
- Aim for incremental coverage growth; start with critical modules
  (timing, memory, retrieval) and expand over time.

## 5. Documentation standards

- Use clear module-level docstrings to describe responsibilities.
- For public functions and classes, adopt a consistent docstring style
  (Google or NumPy style) and add type hints everywhere.
- Keep high-level design documents in `docs/`:
  - Architecture review (`architecture_review_*.md`).
  - File structure guide (`file_structure_guide.md`).
  - Scalability and path configuration docs (this file and
    `path_configuration_strategy.md`).
- When adding new pipeline features (e.g. memory, timing), document the
  data flow and main abstractions in a short design note before or as
  you implement them.

## 6. Performance and scalability

- Prefer **batching** for expensive operations:
  - Batch Chroma queries where possible.
  - Batch LLM calls for multiple shots when latency budget allows.
- Add simple **caching** layers:
  - Cache frame embeddings and subtitle segments on disk.
  - Cache intermediate retrieval results during experimentation.
- Use `numpy` and vectorized operations instead of per-element Python
  loops in tight spots (e.g. similarity computations).
- For I/O-heavy tasks (reading H5, writing JSON), consider using
  asynchronous I/O or background workers once the basic pipeline is
  stable.
- Introduce profiling (e.g. `cProfile`, line-profiler) to identify
  genuine hotspots before attempting micro-optimizations.

## 7. Deployment and packaging

- Move toward a `pyproject.toml`-based package with a clear name (for
  example, `ai_video_describer`) so the pipeline can be installed as a
  library.
- Define console entry points (e.g. `blv-ad-offline`) that map to
  `ad_pipeline.pipeline.cli:main`.
- Provide a minimal Dockerfile or environment specification capturing
  Python version and required system libraries.
- Keep configuration (paths, API keys, model choices) outside the
  container image via environment variables and/or config files.
- Version the package and maintain a simple changelog once the API
  stabilizes.

