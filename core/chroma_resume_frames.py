"""
chroma_indexer_frames_autoresume.py

Continues building the joint CLIP-based ChromaDB index by ingesting
remaining visual (frame) embeddings from CLIP_L14_frames_features_5fps.h5.

This version automatically detects which movies have already been indexed
and resumes from the next unprocessed movie, preserving the same schema
used in chroma_indexer.py.
"""

import h5py
import numpy as np
import chromadb
from chromadb.config import Settings
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
H5_FRAMES_PATH = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_frames_features_5fps.h5"
CHROMA_DIR = "./chroma_db_mad_joint"
COLLECTION_NAME = "mad_joint_clip_L14"

# Controls
BATCH_SIZE = 100
SIM_SPACE = "cosine"


# ----------------------------
# Helpers
# ----------------------------
def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def aggregate_shot_embeddings(embeddings: np.ndarray, sim_threshold: float = 0.8, min_len: int = 3):
    """Group consecutive frames into shots based on cosine similarity."""
    emb = normalize_rows(embeddings)
    sims = np.sum(emb[:-1] * emb[1:], axis=1)
    cuts = np.where(sims < sim_threshold)[0]

    shots = [0]
    for pos in cuts:
        if pos + 1 - shots[-1] >= min_len:
            shots.append(pos + 1)
    shots.append(len(embeddings))

    shot_vecs = []
    for i in range(len(shots) - 1):
        s, e = shots[i], shots[i + 1]
        pooled = embeddings[s:e].mean(axis=0)
        pooled /= np.linalg.norm(pooled) + 1e-8
        shot_vecs.append((i, pooled.astype(np.float32), (s, e)))
    return shot_vecs


def create_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": SIM_SPACE}
    )
    return client, coll


def get_already_indexed_movies(coll):
    """Return a set of movie keys that have already been indexed into Chroma."""
    try:
        results = coll.get(where={"type": "frame_shot"}, include=["metadatas", "ids"])
        metas = results.get("metadatas", [])
        movies = {m["movie"] for m in metas if "movie" in m}
        print(f"Found {len(movies)} movies already indexed in Chroma.")
        return movies
    except Exception as e:
        print(f"Warning: could not query existing movie data → {e}")
        return set()


# ----------------------------
# Main continuation builder
# ----------------------------
def continue_frame_indexing():
    print("=== Continuing CLIP frame Chroma indexing (auto-resume mode) ===")

    client, coll = create_collection()

    print(f"Loading frame embeddings from {H5_FRAMES_PATH}")
    with h5py.File(H5_FRAMES_PATH, "r") as f:
        all_movie_keys = list(f.keys())
        total_movies = len(all_movie_keys)

        print(f"Total movies in H5: {total_movies}")

        # Detect already indexed movies
        indexed_movies = get_already_indexed_movies(coll)
        remaining_movies = [m for m in all_movie_keys if m not in indexed_movies]

        if not remaining_movies:
            print("✅ All movies have already been indexed. Nothing to do.")
            return

        print(f"Resuming from movie index {all_movie_keys.index(remaining_movies[0])}")
        print(f"Processing {len(remaining_movies)} remaining movies...")

        for key in tqdm(remaining_movies, desc="Processing remaining movies"):
            arr = np.array(f[key]).astype(np.float32)
            shot_vecs = aggregate_shot_embeddings(arr)

            batch_embs, batch_ids, batch_meta = [], [], []
            for shot_id, vec, (s, e) in shot_vecs:
                batch_embs.append(vec.tolist())
                batch_ids.append(f"{key}_shot_{shot_id}")
                batch_meta.append({
                    "type": "frame_shot",
                    "movie": key,
                    "shot_id": shot_id,
                    "start_frame": int(s),
                    "end_frame": int(e),
                })

                if len(batch_embs) >= BATCH_SIZE:
                    coll.add(embeddings=batch_embs, ids=batch_ids, metadatas=batch_meta)
                    batch_embs, batch_ids, batch_meta = [], [], []

            if batch_embs:
                coll.add(embeddings=batch_embs, ids=batch_ids, metadatas=batch_meta)

    print(f"✅ Finished ingesting remaining {len(remaining_movies)} visual movies into Chroma.")
    print("✅ Auto-resume visual Chroma indexing complete.")


# ----------------------------
# CLI Entry
# ----------------------------
if __name__ == "__main__":
    continue_frame_indexing()
