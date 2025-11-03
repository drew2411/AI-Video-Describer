"""
mad_groq_pipeline.py

.h5 (CLIP L/14) -> shot detection (cosine similarity) -> ChromaDB -> Groq text LLM AD generation (text-only)

Designed for local Windows use. Configure MOVIE_IDENTIFIER and paths below.
"""

import os
import time
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# vector DB
import chromadb
from chromadb.config import Settings

# Groq (text-only)
try:
    from groq import Groq
except Exception as e:
    Groq = None  # will raise later if used

# ----------------------------
# Config (edit these values)
# ----------------------------
class Config:
    # HDF5 file (CLIP L/14)
    H5_PATH = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_frames_features_5fps.h5"

    # Subtitles CSV (used as optional text context)
    SUBS_CSV = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v2\mad-v2-subs.csv"

    # ChromaDB persistence
    CHROMA_DIR = "./chroma_db_mad"
    CHROMA_COLLECTION = "mad_shots_v2"

    # Shot detection params
    SIM_THRESHOLD = 0.80
    MIN_SHOT_LEN = 3
    FPS = 5.0  # embeddings are at 5 FPS

    # Groq
    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    # GROQ API KEY must be set in .env as GROQ_API_KEY

    # Retrieval / context
    CONTEXT_K = 5           # how many neighbor shots to retrieve
    SUB_CONTEXT_WINDOW = 3  # seconds of subtitle history to include before shot start

    # Output
    OUTPUT_DIR = Path("./mad_groq_outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Utilities - H5 & mapping
# ----------------------------
def list_h5_keys(h5_path: str, n: int = 20) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())[:n]
    return keys

def load_embeddings_for_key(h5_path: str, movie_key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if movie_key not in f:
            raise KeyError(f"Key {movie_key} not found in {h5_path}")
        arr = np.array(f[movie_key]).astype(np.float32)
    return arr

def try_map_movie_identifier(h5_path: str, identifier: str) -> Optional[str]:
    """
    Mapping heuristic:
    - If identifier exactly matches an h5 key -> return it.
    - If identifier is numeric (e.g., '10142' or 10142), try to find any h5 key starting with that numeric prefix.
    - Else return None to signal manual mapping needed.
    """
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

    # if exact match
    if str(identifier) in keys:
        return str(identifier)

    # if numeric: try prefix match (handles '0001_American_Beauty' or '10142_Title' or plain '34433')
    try:
        num = str(int(identifier))
    except Exception:
        num = None

    if num:
        # Try exact numeric key
        if num in keys:
            return num
        # Try padded numeric with leading zeros: check both exact and padded
        # Many h5 keys have zero padding like '0001_...'. Try left-zero padded lengths used in keys.
        for k in keys:
            if k.startswith(num + "_") or k.startswith(num):
                return k
        # try matching numeric anywhere at start (strip leading zeros on key)
        for k in keys:
            prefix = k.split("_")[0]
            try:
                if int(prefix) == int(num):
                    return k
            except:
                continue

    # fallback: no mapping found
    return None

# ----------------------------
# Shot detection (cosine similarity)
# ----------------------------
def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def detect_shots_by_similarity(embeddings: np.ndarray, sim_threshold: float, min_shot_len: int) -> List[Tuple[int,int]]:
    """
    Return list of (start_idx, end_idx) (end exclusive).
    """
    if embeddings.shape[0] == 0:
        return []

    emb = normalize_rows(embeddings)
    sims = np.sum(emb[:-1] * emb[1:], axis=1)  # length N-1
    cut_positions = np.where(sims < sim_threshold)[0]

    cuts = [0]
    for pos in cut_positions:
        if pos + 1 - cuts[-1] >= min_shot_len:
            cuts.append(pos + 1)
    if cuts[-1] != embeddings.shape[0]:
        cuts.append(embeddings.shape[0])

    intervals = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]

    # Merge too-short shots into previous if needed
    merged = []
    for s,e in intervals:
        if not merged:
            merged.append((s,e))
            continue
        prev_s, prev_e = merged[-1]
        if (e - s) < min_shot_len and (prev_e - prev_s) >= min_shot_len:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s,e))
    return merged

def aggregate_shot_embedding(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    seg = embeddings[start:end]
    if seg.shape[0] == 0:
        return np.zeros((embeddings.shape[1],), dtype=np.float32)
    pooled = seg.mean(axis=0)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
    return pooled.astype(np.float32)

# ----------------------------
# ChromaDB helper
# ----------------------------
def create_chroma_collection(chroma_dir: str, collection_name: str):
    print("Trying to create chroma client")
    client = chromadb.PersistentClient(
        path=chroma_dir,
    )
    print("Trying to create chroma collection")
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return client, collection

# ----------------------------
# Subtitles loader (for context only)
# ----------------------------
def load_subtitles_csv(subs_csv_path: str) -> pd.DataFrame:
    if not Path(subs_csv_path).exists():
        return pd.DataFrame(columns=["text","start","end","speaker","speech_type","movie"])
    df = pd.read_csv(subs_csv_path)
    # ensure numeric columns
    if 'start' in df.columns:
        df['start'] = df['start'].astype(float)
    if 'end' in df.columns:
        df['end'] = df['end'].astype(float)
    if 'movie' in df.columns:
        df['movie'] = df['movie'].astype(int)
    return df

def get_subtitles_for_shot(subs_df: pd.DataFrame, movie_numeric_id: Optional[int], shot_start_frame: int, shot_end_frame: int, fps: float, window_seconds: float) -> str:
    if subs_df.empty or movie_numeric_id is None:
        return ""
    start_sec = shot_start_frame / fps
    # include subtitles that end before shot start but within window_seconds, or that overlap shot
    mask = (subs_df['movie'] == movie_numeric_id) & (
        ((subs_df['end'] >= start_sec - window_seconds) & (subs_df['end'] <= start_sec + 1)) |
        ((subs_df['start'] >= start_sec) & (subs_df['start'] <= shot_end_frame / fps))
    )
    rows = subs_df[mask]
    return " ".join(rows['text'].astype(str).tolist())[:1000]  # limit length

# ----------------------------
# Groq LLM text call (with retry)
# ----------------------------
def init_groq_client():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment (.env)")
    if Groq is None:
        raise ImportError("Groq SDK not installed or import failed. Install package providing `groq` client.")
    client = Groq(api_key=api_key)
    print("Loaded API key")
    return client

def generate_ad_with_groq(client: Any, model: str, shot_meta: Dict[str,Any], context_text: str, max_tokens: int = 150) -> str:
    """
    client: Groq client
    model: string model id
    shot_meta: dict with keys like start/end/duration/characters etc
    context_text: optional textual context (subtitles + retrieved shots text)
    """
    sys_msg = (
        "You are an expert describer who writes vivid, factual audio descriptions (AD) "
        "for blind and low-vision audiences. Focus on visible actions, key objects, "
        "and emotional expressions, but avoid technical or camera terms."
    )

    user_prompt = (
        f"Duration: {shot_meta['duration']:.2f}s. "
        f"Visual context: {context_text}\n\n"
        "Describe what happens in this shot in one sentence, "
        "capturing the main action, objects, and emotional tone."
    )


    # Retry loop
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system", "content": sys_msg},
                    {"role":"user", "content": user_prompt}
                ],
                temperature=0.6,
                max_tokens=max_tokens
            )
            # extract text
            out = ""
            try:
                out = resp.choices[0].message.content.strip()
            except Exception:
                # fallback if different shape
                out = getattr(resp, "output_text", "") or str(resp)
            return out.strip()
        except Exception as e:
            print(f"[Groq] attempt {attempt+1} failed: {e}")
            time.sleep(2 * (attempt+1))
    return "[AD generation failed]"

# ----------------------------
# Pipeline main
# ----------------------------
def run_single_movie_pipeline(
    movie_identifier: str,
    h5_path: str = Config.H5_PATH,
    max_shots: Optional[int] = 10,
    offset_shot: int = 0
):
    print("Loading H5 keys (sample):", list_h5_keys(h5_path, n=20))
    mapped_key = try_map_movie_identifier(h5_path, movie_identifier)
    if mapped_key is None:
        raise KeyError(f"Could not map identifier '{movie_identifier}' to any HDF5 key. Provide exact key or mapping.")
    print(f"Using HDF5 key: {mapped_key}")

    # load embeddings
    embeddings = load_embeddings_for_key(h5_path, mapped_key)
    n_frames, dim = embeddings.shape
    print(f"Loaded embeddings: frames={n_frames}, dim={dim}")

    # Attempt to infer numeric movie id (from key's numeric prefix) for subtitle lookup
    movie_numeric_id = None
    try:
        prefix = mapped_key.split("_")[0]
        movie_numeric_id = int(prefix)
    except Exception:
        movie_numeric_id = None

    # shot detection
    shots = detect_shots_by_similarity(embeddings, Config.SIM_THRESHOLD, Config.MIN_SHOT_LEN)
    print(f"Detected {len(shots)} shots")

    # build shot-level pooled vectors & metadata
    shot_list = []
    for i, (s,e) in enumerate(shots):
        pooled = aggregate_shot_embedding(embeddings, s, e)
        duration = (e - s) / Config.FPS
        shot_meta = {
            "shot_id": i,
            "start_frame": int(s),
            "end_frame": int(e),
            "start_time_sec": round(s / Config.FPS, 2),
            "end_time_sec": round(e / Config.FPS, 2),
            "duration": float(duration),
            "characters": [],  # stub; later can be filled via NER on CSV if desired
        }
        shot_list.append((pooled, shot_meta))

    # ChromaDB - store shot vectors
    client, collection = create_chroma_collection(Config.CHROMA_DIR, Config.CHROMA_COLLECTION)
    # prepare lists
    embeddings_to_add = [v.tolist() for v, m in shot_list]
    ids_to_add = [f"{mapped_key}_shot_{m['shot_id']}" for v,m in shot_list]
    metadatas_to_add = [{
        "movie_key": mapped_key,
        "movie_numeric_id": int(movie_numeric_id) if movie_numeric_id is not None else None,
        "shot_id": int(m['shot_id']),
        "start_frame": int(m['start_frame']),
        "end_frame": int(m['end_frame']),
        "duration": float(m['duration']),
        "characters": ""
    } for v,m in shot_list]

    # Add to Chroma (safe: chunk add if large)
    print("Storing shot vectors into ChromaDB...")
    # if collection already contains same ids, add may fail; ignore duplicates by trying add in try/except
    try:
        collection.add(embeddings=embeddings_to_add, ids=ids_to_add, metadatas=metadatas_to_add)
    except Exception as e:
        print(f"Warning: collection.add() raised: {e}. Attempting to continue.")

    # load subs csv
    subs_df = load_subtitles_csv(Config.SUBS_CSV)
    print(f"Loaded subtitles rows: {len(subs_df)}")

    # init Groq client
    groq_client = init_groq_client()

    # iterate shots and create ADs for top max_shots
    results = []
    start_idx = offset_shot
    end_idx = min(start_idx + (max_shots or len(shot_list)), len(shot_list))
    print(f"Processing shots {start_idx} → {end_idx - 1} ({end_idx - start_idx} total)")

    for idx in range(start_idx, end_idx):        
        pooled, meta = shot_list[idx]
        print(f"\nProcessing shot {meta['shot_id']} | frames {meta['start_frame']}-{meta['end_frame']} "
            f"({meta['start_time_sec']:.2f}s → {meta['end_time_sec']:.2f}s, {meta['duration']:.2f}s)")
        # retrieve neighbors for context
        try:
            retrieval = collection.query(
                query_embeddings=[pooled.tolist()],
                n_results=Config.CONTEXT_K * 2,  # fetch a bit more
                where={"shot_id": {"$lt": meta["shot_id"]}}  # only earlier shots
            )
            neighbor_meta = retrieval.get("metadatas", [[]])[0]
            # keep only past shots
            neighbor_meta = [m for m in neighbor_meta if m.get("shot_id", 0) < meta["shot_id"]]
            neighbor_meta = neighbor_meta[:Config.CONTEXT_K]
        except Exception as e:
            print(f"Chroma query failed: {e}")
            neighbor_meta = []

        # build short textual context from neighbor metadata (shot ids)
        neighbor_text = ""
        if neighbor_meta:
            try:
                # neighbor_meta is list-of-lists per query
                nm = neighbor_meta[0]
                neighbor_text = "Similar shots: " + ", ".join([f"shot_{m.get('shot_id','?')}" for m in nm[:3]])
            except Exception:
                neighbor_text = ""

        # subtitle context
        subs_context = get_subtitles_for_shot(subs_df, movie_numeric_id, meta['start_frame'], meta['end_frame'], Config.FPS, Config.SUB_CONTEXT_WINDOW)

        # combine context (short)
        context_text = (neighbor_text + " " + subs_context).strip()

        # prepare shot_meta for prompt
        shot_prompt_meta = {
            "duration": meta['duration'],
            "characters": meta.get('characters', []),
            "labels": ""  # placeholder - could add clustering labels later
        }

        # call Groq to generate AD sentence
        ad_text = generate_ad_with_groq(groq_client, Config.GROQ_MODEL, shot_prompt_meta, context_text, max_tokens=120)
        print(f"AD: {ad_text}")

        # append
        results.append({
            "shot_id": meta['shot_id'],
            "start_frame": meta['start_frame'],
            "end_frame": meta['end_frame'],
            "duration": meta['duration'],
            "context": context_text,
            "ad": ad_text
        })

        # also optionally store final AD as a 'document' record in Chroma for future context
        try:
            collection.add(
                embeddings=[pooled.tolist()],
                ids=[f"{mapped_key}_shot_{meta['shot_id']}_ad"],
                metadatas=[{"movie_key": mapped_key, "shot_id": meta['shot_id'], "final_ad": ad_text}],
                documents=[ad_text]
            )
        except Exception:
            # ignore duplicates / minor errors
            pass

    # save results file
    out_file = Config.OUTPUT_DIR / f"{mapped_key}_{OFFSET_SHOT}_{OFFSET_SHOT+MAX_SHOTS_TO_PROCESS}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"movie_key": mapped_key, "movie_numeric_id": movie_numeric_id, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {out_file}")

    return results

# ----------------------------
# CLI-like invocation (edit this)
# ----------------------------
if __name__ == "__main__":
    # Edit this identifier to either:
    # - exact HDF5 key (e.g. "0001_American_Beauty"), OR
    # - numeric movie id (e.g. "10142" or 10142)
    MOVIE_IDENTIFIER = "10142"   # <--- change to the movie you want to run

    # how many shots to process for quick runs
    MAX_SHOTS_TO_PROCESS = 20

    OFFSET_SHOT =0
    # run
    print("Starting MAD -> Groq pipeline")
    results = run_single_movie_pipeline(str(MOVIE_IDENTIFIER), Config.H5_PATH, max_shots=MAX_SHOTS_TO_PROCESS, offset_shot=OFFSET_SHOT)
    print("Done.")