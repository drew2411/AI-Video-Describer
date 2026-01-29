"""
chroma_indexer.py (auto-resumable)

Builds a joint CLIP-based ChromaDB index combining:
- Visual embeddings (from CLIP_L14_frames_features_5fps.h5)
- Text embeddings (from CLIP_L14_language_tokens_features.h5)

Auto-resumes from the number of text records already in Chroma.
"""

import h5py
import numpy as np
import chromadb
from chromadb.config import Settings
from pathlib import Path
from tqdm import tqdm
import json

# ----------------------------
# Config
# ----------------------------
H5_FRAMES_PATH = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_frames_features_5fps.h5"
H5_TEXT_PATH   = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_language_tokens_features.h5"
ANNOTATIONS_JSON = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v2\mad-v2-annotations.json"

CHROMA_DIR = "./chroma_db_mad_joint"
COLLECTION_NAME = "mad_joint_clip_L14"

# Controls
MAX_MOVIES = 5           # how many movies to process for frames
TEXT_LIMIT = 100000       # how many text embeddings to process this run
BATCH_SIZE = 100
SIM_SPACE = "cosine"


# ----------------------------
# Helpers
# ----------------------------
def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def aggregate_shot_embeddings(embeddings: np.ndarray, sim_threshold: float = 0.8, min_len: int = 3):
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


def load_text_keys(h5_path: str):
    print(f"=== Loading text keys from {h5_path} ===")
    f = h5py.File(h5_path, "r")
    keys = list(f.keys())
    return keys, f


# ----------------------------
# Chroma setup
# ----------------------------
def create_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": SIM_SPACE}
    )
    return client, coll


def get_existing_text_count(coll) -> int:
    """Counts only text_caption entries in Chroma."""
    try:
        res = coll.get(where={"type": "text_caption"})
        return len(res.get("ids", []))
    except Exception:
        return 0


# ----------------------------
# Main builder
# ----------------------------
def build_joint_index():
    print("=== Building joint CLIP frame+text Chroma index ===")

    # Chroma setup
    client, coll = create_collection()

    # --------------------
    # Step 1: Frame embeddings
    # --------------------
    print(f"Loading frame embeddings from {H5_FRAMES_PATH}")
    with h5py.File(H5_FRAMES_PATH, "r") as f:
        movie_keys = list(f.keys())

        for key in tqdm(movie_keys[:MAX_MOVIES], desc="Processing movies"):
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

    # --------------------
    # Step 2: Text embeddings (auto-resume)
    # --------------------
    keys, f = load_text_keys(H5_TEXT_PATH)
    existing_text_count = get_existing_text_count(coll)
    print(f"Found {existing_text_count} existing text entries in Chroma.")

    start_index = existing_text_count
    end_index = min(start_index + TEXT_LIMIT, len(keys))
    total = len(keys)

    print(f"Resuming from {start_index}/{total}, processing {end_index - start_index} entries → up to {end_index}")

    # optional captions
    captions = {}
    if Path(ANNOTATIONS_JSON).exists():
        with open(ANNOTATIONS_JSON, "r", encoding="utf-8") as jf:
            captions = json.load(jf)

    for i in tqdm(range(start_index, end_index), initial=start_index, total=total, desc="Indexing text"):
        text_id = keys[i]
        try:
            emb = np.array(f[text_id]).astype(np.float32)
        except Exception:
            continue

        meta = {"type": "text_caption", "source_id": text_id}
        if text_id in captions:
            meta["caption"] = captions[text_id]

        coll.add(
            embeddings=[emb.mean(axis=0).tolist()],
            ids=[f"text_{text_id}"],
            metadatas=[meta],
            documents=[meta.get("caption", "")]
        )

    print(f"✅ Indexed entries {start_index} → {end_index} ({end_index}/{total})")
    print("✅ Joint CLIP frame+text Chroma index built successfully.")


# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    build_joint_index()
