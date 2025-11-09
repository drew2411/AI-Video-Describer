#!/usr/bin/env python3
"""
query_index.py

Multimodal retrieval test tool for the MAD Chroma index.

Usage examples:
    # Query by a frame embedding in the HDF5 file
    python query_index.py --movie_key "0001_American_Beauty" --frame_index 120 --top_k 5

    # Query by a shot id that was stored in Chroma (if available)
    python query_index.py --shot_id "10142_shot_10" --top_k 5

    # Change collection or chroma dir
    python query_index.py --movie_key 10142 --frame_index 100 --collection "mad_shots_v2" --chroma_dir "./chroma_db_mad"
"""
import argparse
from pathlib import Path
import numpy as np
import h5py
import chromadb
import json
import sys
from typing import Optional

# ----------------------------
# Default config - edit if needed
# ----------------------------
DEFAULT_H5_PATH = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_frames_features_5fps.h5"
DEFAULT_CHROMA_DIR = "./chroma_db_mad_joint"
DEFAULT_COLLECTION = "mad_joint_clip_L14"
DEFAULT_SUBS_CSV = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v2\mad-v2-subs.csv"

# ----------------------------
# Utilities
# ----------------------------
def normalize(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n <= 1e-8:
        return v
    return v / n

def load_frame_embedding(h5_path: str, movie_key: str, frame_index: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if movie_key not in f:
            raise KeyError(f"Movie key '{movie_key}' not found in {h5_path}.")
        arr = np.array(f[movie_key]).astype(np.float32)
        if frame_index < 0 or frame_index >= arr.shape[0]:
            raise IndexError(f"Frame index {frame_index} out of range (0..{arr.shape[0]-1}).")
        emb = arr[frame_index]
    return emb

def load_shot_pooled_embedding(h5_path: str, movie_key: str, shot_start: int, shot_end: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        arr = np.array(f[movie_key]).astype(np.float32)
        seg = arr[shot_start:shot_end]
        if seg.shape[0] == 0:
            raise ValueError("Empty shot interval.")
        pooled = seg.mean(axis=0)
    return pooled

def pretty_print_result(rank: int, id_: str, distance: float, doc: Optional[str], metadata: Optional[dict]):
    print(f"{rank:02d}. id: {id_}   distance: {distance:.4f}")
    if metadata:
        # print limited metadata keys
        md_preview = {k: metadata[k] for k in metadata.keys() if k in ("movie_key","movie_numeric_id","shot_id","start_frame","end_frame","duration","type") and k in metadata}
        if md_preview:
            print("    metadata:", json.dumps(md_preview, ensure_ascii=False))
    if doc:
        print("    document/text:", doc if len(doc) < 400 else doc[:400] + " ...")
    print()

# ----------------------------
# Main retrieval logic
# ----------------------------
def run_query(
    h5_path: str,
    chroma_dir: str,
    collection_name: str,
    movie_key: Optional[str],
    frame_index: Optional[int],
    shot_id: Optional[str],
    top_k: int,
    subs_csv: Optional[str],
):
    # open chroma
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Failed to open collection '{collection_name}' in {chroma_dir}: {e}")
        sys.exit(1)
    print(f"Loaded Chroma collection: '{collection_name}' (path={chroma_dir})")

    # build query embedding
    if shot_id:
        # if user provided shot id, try to fetch its embedding directly from chroma
        try:
            rec = collection.get(ids=[shot_id], include=["embeddings","metadatas","documents"])
            emb = None
            if rec and rec.get("embeddings") and rec["embeddings"][0]:
                emb = np.array(rec["embeddings"][0], dtype=np.float32)
            if emb is None:
                print(f"No embedding found in Chroma for id {shot_id}.")
                sys.exit(1)
            query_emb = normalize(emb)
            print(f"Using embedding pulled from Chroma for id {shot_id}.")
        except Exception as e:
            print(f"Error retrieving shot id embedding from Chroma: {e}")
            sys.exit(1)

    elif movie_key is not None and frame_index is not None:
        try:
            emb = load_frame_embedding(h5_path, movie_key, frame_index)
            query_emb = normalize(emb)
            print(f"Loaded frame embedding: movie='{movie_key}', frame={frame_index}")
        except Exception as e:
            print(f"Error loading frame embedding: {e}")
            sys.exit(1)
    else:
        print("You must provide either --shot_id or both --movie_key and --frame_index.")
        sys.exit(1)

    # do the query - restrict to text entries if possible
    where_clause = None
    # prefer text entries (type or metadata might differ depending on how you indexed)
    # We'll try a small set of likely filters; if it errors, fall back to no filter.
    try_filters = [
        {"type": {"$eq": "text_token"}},
        {"type": {"$eq": "text"}},
        {"category": {"$eq": "text"}},
        {"doc_type": {"$eq": "text"}},
    ]
    last_exc = None
    for f in try_filters:
        try:
            res = collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k,
                # where=f,
                where={"type": {"$eq": "text_caption"}},
                include=["metadatas","documents","distances"]
            )
            # if we got results, accept
            ids = res.get("ids", [[]])[0]
            if ids:
                results = res
                break
        except Exception as e:
            last_exc = e
            results = None
    else:
        # try without filter
        try:
            res = collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k,
                where={"type": {"$eq": "text_caption"}},
                include=["metadatas","documents","distances"]
            )
            results = res
        except Exception as e:
            print(f"Chroma query failed: {e}")
            sys.exit(1)

    ids = results.get("ids", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    print("\n=== Retrieval results ===")
    if not ids:
        print("No results found.")
        return

    for i, (id_, meta, doc, dist) in enumerate(zip(ids, metas, docs, dists), start=1):
        pretty_print_result(i, id_, dist, doc if doc else None, meta if meta else None)

    # Optionally, try to show matching subtitle lines from CSV for convenience
    if subs_csv and movie_key:
        try:
            import pandas as pd
            df = pd.read_csv(subs_csv)
            # attempt to infer numeric movie id from movie_key prefix
            prefix = str(movie_key).split("_")[0]
            try:
                mid = int(prefix)
            except:
                mid = None

            if mid is not None:
                print("\nLooking up subtitle rows for the queried frame/shot in the CSV (if any)...")
                # approximate time of query frame
                # we need a FPS; assume 5 FPS for MAD frames
                FPS = 5.0
                if frame_index is not None:
                    t = frame_index / FPS
                else:
                    # if shot_id provided, try to parse start_time from metadata returned above
                    t = None
                    # look for metadata in first result
                    if metas and metas[0] and metas[0].get("start_frame") is not None:
                        try:
                            start_f = int(metas[0]["start_frame"])
                            t = start_f / FPS
                        except:
                            t = None

                if t is not None:
                    mask = df["movie"] == mid
                    # select subtitles within +-2s of t
                    sel = df[mask & ( (df["start"] <= t + 2) & (df["end"] >= t - 2) )]
                    if not sel.empty:
                        print(f"Subtitle lines near t={t:.2f}s (movie id={mid}):")
                        for _, row in sel.iterrows():
                            print(f"  [{row['start']:.2f}-{row['end']:.2f}] {row['text']}")
                    else:
                        print("No subtitle lines found near that time.")
                else:
                    print("Couldn't compute time for subtitle lookup (no frame_index and no start_frame metadata).")
        except Exception as e:
            print(f"Subtitle CSV lookup failed: {e}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Query Chroma multimodal index (frame->text).")
    p.add_argument("--h5_path", type=str, default=DEFAULT_H5_PATH, help="Path to CLIP frames .h5 file")
    p.add_argument("--chroma_dir", type=str, default=DEFAULT_CHROMA_DIR, help="Chroma DB directory")
    p.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Chroma collection name")
    p.add_argument("--movie_key", type=str, help="HDF5 movie key (e.g., '0001_American_Beauty' or '10142')")
    p.add_argument("--frame_index", type=int, help="Frame index within the HDF5 movie key to query")
    p.add_argument("--shot_id", type=str, help="If set, use an embedding already stored in Chroma (id)")
    p.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors to return")
    p.add_argument("--subs_csv", type=str, default=DEFAULT_SUBS_CSV, help="Subtitles CSV (optional) for showing matching text lines")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_query(
        h5_path=args.h5_path,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        movie_key=args.movie_key,
        frame_index=args.frame_index,
        shot_id=args.shot_id,
        top_k=args.top_k,
        subs_csv=args.subs_csv,
    )
