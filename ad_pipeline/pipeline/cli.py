"""Command-line entry point for the refactored AD pipeline.

Intended replacement for `core/mad_ad_generator_final.py`'s CLI once the
pipeline has been migrated into :mod:`ad_pipeline.pipeline.offline_movie`.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from .offline_movie import process_movie


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline MAD-based audio description generation",
    )
    parser.add_argument("--movie", required=True, help="MAD movie id (e.g. 10142)")
    parser.add_argument(
        "--max_shots",
        type=int,
        default=None,
        help="Maximum number of shots to process (for debugging)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Run quick verification checks without generating full output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    process_movie(
        movie_id=str(args.movie),
        max_shots=args.max_shots,
        verify_only=args.verify_only,
    )


if __name__ == "__main__":
    main()

