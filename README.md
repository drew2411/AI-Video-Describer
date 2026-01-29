# AI-Video-Describer

This project generates movie audio descriptions (AD) for blind and
low-vision (BLV) users, originally using the MAD dataset and a
monolithic script-based pipeline. The codebase is being refactored
into a modular `ad_pipeline/` package.

## Quick Start (Legacy Prototype)

The original prototype entrypoint (may be replaced by the new CLI):

```bash
python -m venv venv
source venv/bin/activate  # or `venv/Scripts/Activate.ps1` on Windows
pip install -r requirements.txt
python core/mad_ad_generator_final.py --movie 10142 --max_shots 20
```

## New Structure (Work in Progress)

- `ad_pipeline/` – main package for the refactored pipeline:
  - `config/` – configuration and environment handling.
  - `data/` – MAD dataset loading helpers.
  - `vision/` – shot detection and visual processing.
  - `retrieval/` – ChromaDB integration and hybrid retrieval.
  - `memory/` – entity and episodic memory (placeholders for now).
  - `generation/` – LLM clients and prompt logic.
  - `pipeline/` – orchestration and (future) CLI entrypoints.
- `deprecated/` – legacy and experimental scripts moved out of the
  main code path.
- `docs/` – documentation, including
  `docs/architecture_review_2026-01-29.md`.
- `notebooks/` – Jupyter notebooks for exploration.

For details on how legacy files map into the new structure, see
`MIGRATION_GUIDE.md`.