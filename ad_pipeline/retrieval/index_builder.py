"""Index-building utilities for ChromaDB.

This module is intended to absorb the logic from `core/chroma_index.py`
and `core/chroma_resume_frames.py` so that index construction is
reusable and testable.
"""

from __future__ import annotations

from pathlib import Path

import chromadb


def ensure_joint_index(db_path: Path, collection_name: str) -> chromadb.Collection:
    """Placeholder for index-creation logic.

    Implementors should migrate the existing indexing pipeline here and
    return a handle to the Chroma collection.
    """

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)
    return collection

