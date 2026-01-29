# Architecture Review – MAD Audio Description Pipeline (2026-01-29)

This document summarizes the earlier review of your inherited MAD-based audio description (AD) system and outlines key strengths, limitations, and recommended directions.

## 1. Current System Overview

- Built around the MAD dataset (Movie Audio Description) with frame embeddings saved in H5.
- Core script: `core/mad_ad_generator_final.py` – end-to-end AD generation for a movie:
  - Load CLIP L/14 embeddings at ~5 FPS.
  - Detect shots via cosine similarity and simple thresholds.
  - Use a joint ChromaDB collection (frame_shot + text_caption) for hybrid retrieval.
  - Use Groq LLM (e.g., `llama-3.3-70b-versatile`) to generate AD per shot.
  - Optionally skip low-confidence shots.
- Supporting scripts: Chroma index builders, evaluation utilities, and exploration notebooks.

The current pipeline is a good research prototype but not yet a modular, real-time system.

## 2. What Works Well

**Pipeline structure**
- Clear offline phases: load embeddings → detect shots → retrieve context → call LLM → write JSON.
- Shot detection based on CLIP similarity is a reasonable baseline.

**Hybrid retrieval with ChromaDB**
- Joint collection with both visual (frame_shot) and textual (text_caption) entries.
- `retrieve_hybrid_context`:
  - Visual similarity → similar shots → align retrieved shots with AD timestamps in MAD.
  - Cross-modal retrieval → text captions directly from visual embeddings.
  - Combines results with weighted distances (visual boosted, cross-modal penalized).

**Context-aware prompting**
- Derives a provisional movie name and synopsis from early subtitles using Groq.
- Distinguishes special shot types (e.g., logos/credits) from normal content.
- Uses local subtitle window around shot time for extra context.

**Evaluation & experimentation**
- `core/evaluate_ads.py` provides BLEU/ROUGE/METEOR style metrics.
- Notebooks (`core/data_exploration.ipynb`, `core/single_movie_run.ipynb`) show prior analyses and prototyping.

These pieces are worth preserving and refactoring into a cleaner package.

## 3. Key Limitations vs Your Goals

**No persistent memory or deduplication**
- Shots are processed independently; no movie-level state.
- No tracking of characters, locations, or objects across time.
- No mechanism to avoid re-describing the same house/person repeatedly.

**No explicit timing or pause-aware planning**
- AD is generated per shot, not per speech-free gap.
- The system does not explicitly find or exploit silent / low-dialogue intervals.
- There is no planner to choose when to speak or how long the description should be for a given pause.

**Offline, non-streaming design**
- Processes full movies by loading all embeddings plus subtitle CSVs from disk.
- ChromaDB is used in batch mode, not incrementally.

**Architecture and engineering debt**
- `mad_ad_generator_final.py` is monolithic, mixing config, IO, retrieval, prompting, and CLI.
- Hard-coded absolute paths in `Config` reduce portability.
- Retrieval over MAD annotations is inefficient (linear scan instead of indexed by time/movie).
- One LLM call per shot with no batching or latency considerations.

**RAG limitations**
- RAG is purely per-shot; lacks long-range plot and character modeling.

## 4. Recommended Modeling Directions

**Backbone models**
- Short term: keep CLIP L/14 as you already have.
- Medium term: consider video-language models such as:
  - Video-LLaVA / Video-LLaMA / Video-ChatGPT-like models.
  - Qwen2-VL, InternVL2, or similar general VLMs with multi-frame support.
  - TimeSformer / ViViT / X-CLIP as stronger video feature extractors.

**AD and long-context research**
- MAD (Soldan et al., CVPR 2022) – base dataset and baselines.
- AutoAD II – focuses on who/when/what in AD; useful for timing logic.
- LoCo-MAD (ACCV 2024) – long-range context model for plot-centric MAD.
- LLM-AD (2024) – LLM-based AD architecture, shows how to integrate LLMs with AD-specific constraints.

**Memory mechanisms**
- Episodic memory: time-ordered events with their ADs, entities, and embeddings.
- Entity memory: characters/locations/objects with names, descriptions, and first/last times seen.
- Use vector stores and symbolic state to:
  - Avoid repetition.
  - Maintain consistent references to recurring entities.

**Context-aware generation**
- Use RAG to retrieve few-shot MAD examples for style and structure, not just raw text.
- Tailor prompts for:
  - First appearance vs repeat appearance of entities.
  - Detail levels (short/medium/long) based on available pause duration.

## 5. Proposed Modular Architecture (High Level)

Introduce a new `ad_pipeline/` package with submodules:

- `config/`: paths, model names, and environment handling.
- `data/`: MAD dataset loaders and download utilities.
- `vision/`: frame embedding utilities and shot detection.
- `retrieval/`: ChromaDB clients and hybrid retrieval logic.
- `memory/`: entity and episodic memory components.
- `generation/`: Groq (or other LLM) clients, prompts, and AD generators.
- `pipeline/`: orchestrators for offline movie processing and, later, streaming.

This separates concerns and makes it easier to extend the system.

## 6. Roadmap Summary

Phase 0 – Clean up & modularize:
- Extract logic from `core/mad_ad_generator_final.py` into the new package.
- Centralize configuration and paths.

Phase 1 – Add memory & deduplication:
- Implement episodic and entity memory.
- Add rules and prompts to avoid redundant descriptions.

Phase 2 – Timing & adaptive detail:
- Detect dialogue-free gaps.
- Add an AD planner that schedules descriptions and sets detail levels.

Phase 3 – Stronger models & fine-tuning:
- Integrate a video-language model and, if possible, fine-tune on MAD/related datasets.

Phase 4 – Real-time / streaming:
- Split into streaming services (frame encoder, ASR, planner, generator).
- Optimize latency and prompt size for live use.

This document should serve as a stable reference while we refactor the codebase toward a modular, real-time-capable AD system.
