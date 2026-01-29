"""ChromaDB client helpers.

This module will encapsulate creation and configuration of ChromaDB
clients used throughout the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import chromadb


def get_chroma_client(path: Path) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client for the given path."""

    return chromadb.PersistentClient(path=str(path))

