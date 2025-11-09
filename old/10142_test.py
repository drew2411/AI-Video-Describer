"""
process_movie_10142.py
----------------------
Processes the first 100 frames of movie 10142 from the MAD dataset.

For each frame:
- Loads its visual CLIP embedding from .h5
- Queries the Chroma database (built from MAD_train.json text embeddings)
- Retrieves top-k nearest text embedding IDs
- Maps them to human-readable sentences using MAD_train.json
"""

import h5py
import json
import numpy as np
from chromadb import PersistentClient
from tqdm import tqdm

# ========== CONFIG ==========
MOVIE_ID = "10142"
MAX_FRAMES = 10
TOP_K = 5

# Paths
FRAMES_H5 = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_frames_features_5fps.h5"
TEXT_H5 = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_language_tokens_features.h5"
MAD_JSON = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_train.json"

CHROMA_PATH = "./chroma_db_mad_joint"
COLLECTION_NAME = "mad_joint_clip_L14"
# =============================

# --- Load datasets ---
print("Loading MAD JSONs (train + val + test)...")

mad_data = {}

for split in ["train", "val", "test"]:
    path = fr"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_{split}.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
            mad_data.update(data)
            print(f"  Loaded {len(data)} entries from {split}")
    except FileNotFoundError:
        print(f"  ⚠️  Skipped {split} (file not found)")


print("Loading frame embeddings for movie", MOVIE_ID)
with h5py.File(FRAMES_H5, "r") as f_frames:
    frames = f_frames[MOVIE_ID][:]
print("  Loaded", frames.shape[0], "frames, using first", min(MAX_FRAMES, frames.shape[0]))

# Normalize for cosine similarity consistency
frames = frames[:MAX_FRAMES]
frames = frames / np.linalg.norm(frames, axis=1, keepdims=True)

# --- Connect to Chroma ---
print("Connecting to Chroma collection...")
client = PersistentClient(path=CHROMA_PATH)
coll = client.get_collection(COLLECTION_NAME)

# --- Query each frame ---
for i, frame_vec in enumerate(tqdm(frames, desc="Processing frames")):
    # Query top-k nearest text embeddings
    result = coll.query(
        query_embeddings=[frame_vec.tolist()],
        n_results=TOP_K,
        where={"type": "text_caption"},
    )

    text_ids = [id_.replace("text_", "") for id_ in result["ids"][0]]
    retrieved_sentences = []

    for tid in text_ids:
        if tid in mad_data:
            sent = mad_data[tid]["sentence"]
            retrieved_sentences.append(sent)
        else:
            retrieved_sentences.append("[UNKNOWN ID]")

    # --- Diagnostic: check which JSON file contains the ID ---
    for tid in text_ids:
        found_in = []
        for split in ["train", "val", "test"]:
            path = fr"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_{split}.json"
            with open(path, "r") as f:
                data = json.load(f)
                if tid in data:
                    found_in.append(split)
        if not found_in:
            print(f"[!] ID {tid} not found in ANY of the JSON files.")
        else:
            print(f"[✓] ID {tid} found in {found_in}")


    print(f"\nFrame {i}: Top {TOP_K} retrieved ADs:")
    for j, sent in enumerate(retrieved_sentences):
        print(f"  {j+1}. {sent[:120]}...")
