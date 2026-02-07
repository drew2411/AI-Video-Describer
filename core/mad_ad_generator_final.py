"""
mad_ad_generator_final.py

Complete pipeline for generating Audio Descriptions from MAD dataset.
Fixed to work with joint ChromaDB collection (frame_shot + text_caption).

Usage:
    python mad_ad_generator_final.py --movie 10142 --max_shots 20
"""

import os
import json
import h5py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import chromadb
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime



# ----------------------------
# CONFIG
# ----------------------------
class Config:
    # Paths
    H5_FRAMES = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_frames_features_5fps.h5"
    H5_TEXT = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\features\CLIP_L14_language_tokens_features.h5"
    
    JSON_TRAIN = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_train.json"
    JSON_VAL = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_val.json"
    JSON_TEST = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_test.json"
    
    CSV_AD = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v2\mad-v2-ad-unnamed.csv"
    CSV_SUBS = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v2\mad-v2-subs.csv"
    
    # ChromaDB - using your existing joint collection
    CHROMA_DIR = "./chroma_db_mad_joint"
    COLLECTION_NAME = "mad_joint_clip_L14"
    
    # Shot detection
    SIM_THRESHOLD = 0.80
    MIN_SHOT_LEN = 3
    FPS = 5.0
    
    # Retrieval (increased to compensate for filtering)
    TEXT_CONTEXT_K = 5    # how many similar ADs to retrieve
    VISUAL_CONTEXT_K = 3  # how many nearby shots to find
    
    # Groq
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    # Output
    OUTPUT_DIR = Path("./mad_generated_ads")
    OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# SHOT DETECTION
# ----------------------------
def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each row to unit length."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def detect_shots(embeddings: np.ndarray, threshold: float, min_len: int) -> List[Tuple[int, int]]:
    """
    Detect shot boundaries using cosine similarity.
    Returns list of (start_frame, end_frame) tuples.
    """
    if len(embeddings) == 0:
        return []
    
    emb_norm = normalize_rows(embeddings)
    sims = np.sum(emb_norm[:-1] * emb_norm[1:], axis=1)
    cuts = np.where(sims < threshold)[0]
    
    boundaries = [0]
    for pos in cuts:
        if pos + 1 - boundaries[-1] >= min_len:
            boundaries.append(pos + 1)
    boundaries.append(len(embeddings))
    
    shots = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
    
    # Merge shots that are too short
    merged = []
    for s, e in shots:
        if not merged:
            merged.append((s, e))
            continue
        prev_s, prev_e = merged[-1]
        if (e - s) < min_len and len(merged) > 0:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    
    return merged


def pool_shot_embedding(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """Average pool frames in shot and normalize."""
    seg = embeddings[start:end]
    if len(seg) == 0:
        return np.zeros(embeddings.shape[1], dtype=np.float32)
    pooled = seg.mean(axis=0)
    return pooled / (np.linalg.norm(pooled) + 1e-8)


# ----------------------------
# DATA LOADING
# ----------------------------
def load_mad_json(split: str = "train") -> Dict:
    """Load MAD-v1 JSON annotations."""
    path_map = {
        "train": Config.JSON_TRAIN,
        "val": Config.JSON_VAL,
        "test": Config.JSON_TEST
    }
    print(f"Loading {split} JSON...")
    with open(path_map[split], "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries")
    return data


def load_frame_embeddings(movie_id: str) -> np.ndarray:
    """Load frame embeddings for a movie."""
    with h5py.File(Config.H5_FRAMES, "r") as f:
        if movie_id not in f:
            raise KeyError(f"Movie {movie_id} not found in frame embeddings")
        return np.array(f[movie_id]).astype(np.float32)


# ----------------------------
# CHROMA VERIFICATION
# ----------------------------
def verify_chroma_structure():
    """Verify your existing Chroma collection structure."""
    client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
    
    try:
        collection = client.get_collection(Config.COLLECTION_NAME)
    except:
        print(f"❌ Collection '{Config.COLLECTION_NAME}' not found in {Config.CHROMA_DIR}")
        print("\nAvailable collections:")
        for c in client.list_collections():
            print(f"  - {c.name}")
        return None
    
    #Test to see if movie key is a string or not
    sample = collection.get(limit=5, include=["metadatas"])
    for m in sample["metadatas"]:
        print(m.get("movie"), type(m.get("movie")))

    total = collection.count()
    print(f"\n{'='*60}")
    print(f"ChromaDB Collection: {Config.COLLECTION_NAME}")
    print(f"{'='*60}")
    print(f"Total entries: {total:,}")
    
    # Sample entries
    sample = collection.get(limit=10, include=["metadatas"])
    print("\nSample entries:")
    for id_, meta in zip(sample["ids"], sample["metadatas"]):
        entry_type = meta.get('type', 'UNKNOWN')
        movie = meta.get('movie_key', meta.get('movie', 'UNKNOWN'))
        print(f"  {id_:20s} | type={entry_type:15s} | movie={movie}")
    
    # Count by type
    try:
        text_count = len(collection.get(where={"type": "text_caption"}, limit=1000000)["ids"])
        print(f"\nText caption entries: {text_count:,}")
    except Exception as e:
        print(f"\n⚠️  Could not count text entries: {e}")
    
    try:
        frame_count = len(collection.get(where={"type": "frame_shot"}, limit=1000000)["ids"])
        print(f"Frame shot entries: {frame_count:,}")
    except Exception as e:
        print(f"⚠️  Could not count frame entries: {e}")
    
    print(f"{'='*60}\n")
    return collection


# ----------------------------
# HYBRID RETRIEVAL
# ----------------------------
def get_representative_frame_embedding(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """Get embedding from middle frame (more representative for short shots)."""
    mid_frame = (start + end) // 2
    emb = embeddings[mid_frame]
    return emb / (np.linalg.norm(emb) + 1e-8)


def retrieve_hybrid_context(
    collection: chromadb.Collection,
    shot_emb: np.ndarray,
    shot_info: Dict,
    mad_data: Dict,
    visual_k: int = 3,
    text_k: int = 5
) -> Dict:
    """
    Hybrid retrieval strategy:
    1. Visual similarity → find similar shots → get their ADs (intra-modal)
    2. Cross-modal → direct visual-to-text retrieval (inter-modal)
    3. Combine with confidence weighting
    """
    
    results = {
        "visual_context": [],
        "visual_distances": [],
        "cross_modal_context": [],
        "cross_modal_distances": [],
        "combined_context": [],
        "combined_distances": [],
        "combined_sources": []
    }
    
    # ============================================
    # STAGE 1: VISUAL SIMILARITY (Intra-Modal)
    # ============================================
    try:
        visual_retrieval = collection.query(
            query_embeddings=[shot_emb.tolist()],
            n_results=visual_k * 10,  # Over-fetch to allow manual filtering
            where={
                "$and": [
                    {"type": "frame_shot"},
                    {"movie_key": {"$ne": str(shot_info.get("movie_id", shot_info.get("movie", None)))}}
                ]
            },
            include=["metadatas", "distances"]
        )
        
        # Start of debugging for movie id check
        current_movie = str(shot_info.get("movie_id", shot_info.get("movie", None)))
        if visual_retrieval.get("metadatas") and visual_retrieval["metadatas"] and visual_retrieval["metadatas"][0]:
            vis_movies = []
            for m in visual_retrieval["metadatas"][0]:
                vis_movies.append(str(m.get("movie_key", m.get("movie"))))
            print(f"[DEBUG] Visual retrieved movies (current={current_movie}): {vis_movies}")
            if current_movie in vis_movies:
                print(f"[DEBUG] Visual retrieval contains current movie: {current_movie}")
        #end of debugging for movie id check
        
        if visual_retrieval["ids"] and visual_retrieval["ids"][0]:
            for idx, shot_id in enumerate(visual_retrieval["ids"][0]):
                meta = visual_retrieval["metadatas"][0][idx]
                distance = visual_retrieval["distances"][0][idx]
                
                # Extract movie and temporal info
                movie_key = meta.get("movie_key", meta.get("movie"))
                meta_movie = str(meta.get("movie_key", meta.get("movie", meta.get("movie_id", None))))
                if meta_movie == current_movie:
                    continue
                shot_start = meta.get("start_frame", 0) / Config.FPS
                shot_end = meta.get("end_frame", 0) / Config.FPS
                
                if not movie_key:
                    continue
                
                # Find ADs that overlap with this retrieved shot
                for text_id, entry in mad_data.items():
                    if entry["movie"] != str(movie_key):
                        continue
                    
                    ad_start, ad_end = entry["timestamps"]
                    
                    # Check temporal overlap
                    overlap = min(shot_end, ad_end) - max(shot_start, ad_start)
                    if overlap > 0.5:  # At least 0.5s overlap
                        results["visual_context"].append(entry["sentence"])
                        results["visual_distances"].append(distance)
                        
                        if len(results["visual_context"]) >= visual_k:
                            break
                
                if len(results["visual_context"]) >= visual_k:
                    break
    
    except Exception as e:
        print(f"  ⚠️  Visual retrieval failed: {e}")
    
    # ============================================
    # STAGE 2: CROSS-MODAL RETRIEVAL (Inter-Modal)
    # ============================================
    try:
        text_retrieval = collection.query(
            query_embeddings=[shot_emb.tolist()],
            n_results=text_k * 10,
            where={
                "$and": [
                    {"type": "text_caption"},
                    {"movie": {"$ne": str(shot_info.get("movie_id", shot_info.get("movie", None)))}}
                ]
            },
            include=["metadatas", "distances"]
        )

        # Start of debugging for movie id check
        current_movie = str(shot_info.get("movie_id", shot_info.get("movie", None)))
        if text_retrieval.get("metadatas") and text_retrieval["metadatas"] and text_retrieval["metadatas"][0]:
            txt_movies = []
            for m in text_retrieval["metadatas"][0]:
                txt_movies.append(str(m.get("movie", m.get("movie_key"))))
            print(f"[DEBUG] Text retrieved movies (current={current_movie}): {txt_movies}")
            if current_movie in txt_movies:
                print(f"[DEBUG] Text retrieval contains current movie: {current_movie}")
        #end of debugging for movie id check

        if text_retrieval["ids"] and text_retrieval["ids"][0]:
            for idx, text_id in enumerate(text_retrieval["ids"][0]):
                meta = text_retrieval["metadatas"][0][idx]
                meta_movie = str(meta.get("movie", meta.get("movie_key", meta.get("movie_id", None))))
                if meta_movie == current_movie:
                    continue
                clean_id = text_id.replace("text_", "")

                if clean_id in mad_data:
                    sentence = mad_data[clean_id]["sentence"]
                    distance = text_retrieval["distances"][0][idx]
                    
                    # Skip if already in visual context
                    if sentence not in results["visual_context"]:
                        results["cross_modal_context"].append(sentence)
                        results["cross_modal_distances"].append(distance)
                        
                        if len(results["cross_modal_context"]) >= text_k:
                            break
    
    except Exception as e:
        print(f"  ⚠️  Cross-modal retrieval failed: {e}")
    
    # ============================================
    # STAGE 3: COMBINE WITH CONFIDENCE WEIGHTING
    # ============================================
    # Visual similarity is more reliable → boost it (reduce distance)
    # Cross-modal is less reliable → penalize it (increase distance)
    
    visual_weighted = [
        (ad, dist * 0.7, "visual")  # 30% boost for visual similarity
        for ad, dist in zip(results["visual_context"], results["visual_distances"])
    ]
    
    cross_modal_weighted = [
        (ad, dist * 1.1, "cross_modal")  # 10% penalty for cross-modal
        for ad, dist in zip(results["cross_modal_context"], results["cross_modal_distances"])
    ]
    
    # Combine and sort by weighted distance
    all_weighted = visual_weighted + cross_modal_weighted
    all_weighted.sort(key=lambda x: x[1])
    
    # Take top K overall
    max_results = max(text_k, visual_k)
    results["combined_context"] = [ad for ad, _, _ in all_weighted[:max_results]]
    results["combined_distances"] = [dist for _, dist, _ in all_weighted[:max_results]]
    results["combined_sources"] = [src for _, _, src in all_weighted[:max_results]]
    
    return results


def should_generate_ad(distances: List[float], threshold: float = 0.7) -> Tuple[bool, str]:
    """
    Check if retrieval quality is good enough to generate AD.
    Returns (should_generate, reason)
    """
    if not distances:
        return False, "No context retrieved"
    
    best_distance = max(distances)
    
    if best_distance < threshold:
        return False, f"Low confidence (distance={best_distance:.3f} > {threshold})"
    
    return True, "OK"


# ----------------------------
# GROQ AD GENERATION
# ----------------------------
def init_groq():
    """Initialize Groq client."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env file")
    return Groq(api_key=api_key)


def classify_shot_type(context_ads: List[str], subtitle_text: str) -> str:
    """Classify shot type based on context to tailor prompt."""
    if not context_ads:
        return "generic_scene"
    
    context_lower = " ".join(context_ads).lower()
    
    # Logo/credits detection
    logo_keywords = ["logo", "credits", "presents", "production", "pictures", "entertainment", 
                     "company", "universal", "paramount", "warner", "disney", "fox"]
    if any(kw in context_lower for kw in logo_keywords) and len(subtitle_text) < 10:
        return "logo_or_credits"
    
    # Dialogue scene
    if len(subtitle_text) > 50:
        return "dialogue_scene"
    
    # Action scene
    action_keywords = ["runs", "jumps", "fights", "chase", "explosion", "crashes", "shoots"]
    if any(kw in context_lower for kw in action_keywords):
        return "action_scene"
    
    return "generic_scene"


def derive_movie_context(client: Groq, subs_df: pd.DataFrame) -> Dict[str, str]:
    """Derive movie name guess and short synopsis from early subtitles."""
    try:
        print(f"[DEBUG] derive_movie_context: subs_df empty={subs_df is None or subs_df.empty}")
        if subs_df is None or subs_df.empty:
            print("[DEBUG] derive_movie_context: no subtitles available, returning empty context")
            return {"movie_name": "", "movie_synopsis": ""}
        first_subs = subs_df.sort_values(by="start").head(50)["text"].astype(str).tolist()
        seed_text = " \n".join(first_subs)[:2000]
        print(f"[DEBUG] derive_movie_context: seed_text_len={len(seed_text)}")
        if not seed_text.strip():
            print("[DEBUG] derive_movie_context: seed_text is blank, returning empty context")
            return {"movie_name": "", "movie_synopsis": ""}
        system_prompt = (
            "You analyze movie subtitles and infer metadata." 
            " Given a few early subtitle lines, guess the movie name if possible"
            " (or leave blank if uncertain) and write a concise 1-2 sentence synopsis."
        )
        user_prompt = (
            "Early subtitles (may be noisy):\n" + seed_text +
            "\n\nReturn as JSON with keys movie_name, movie_synopsis."
        )
        resp = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        content = resp.choices[0].message.content.strip()
        print(f"[DEBUG] derive_movie_context: raw_response_preview={content[:200]}")
        try:
            parsed = json.loads(content)
            # Debugging for movie context
            print(f"[DEBUG] Movie context: {parsed}")
            return {
                "movie_name": str(parsed.get("movie_name", ""))[:120],
                "movie_synopsis": str(parsed.get("movie_synopsis", ""))[:400],
            }
        except Exception:
            # Fallback: treat whole output as synopsis
            print("[DEBUG] derive_movie_context: JSON parse failed, using fallback synopsis")
            return {"movie_name": "", "movie_synopsis": content[:400]}
    except Exception as e:
        print(f"[DEBUG] derive_movie_context: exception={e}")
        return {"movie_name": "", "movie_synopsis": ""}


def generate_ad(
    client: Groq,
    shot_info: Dict,
    context_ads: List[str],
    context_sources: List[str],
    subtitle_text: str = "",
    movie_name: str = "",
    movie_synopsis: str = "",
    previous_ad: str = ""
) -> str:
    """Generate AD using Groq LLM with improved prompting and continuity."""
    
    shot_type = classify_shot_type(context_ads, subtitle_text)
    
    # Different prompts for different shot types
    if shot_type == "logo_or_credits":
        system_prompt = """You are an expert audio description writer.
Generate ONE sentence describing a production logo or opening credits.
Focus on: company names, visual elements (logos, text, animations).
Be factual and concise. DO NOT invent details not in the context."""
        
        user_prompt = f"""Context from similar logos/credits:
{chr(10).join(f"- {ad[:150]}" for ad in context_ads[:2])}

Generate one sentence describing this logo or credit sequence."""
        
        temperature = 0.3  # Lower for factual content
    
    else:
        system_prompt = """You are an expert audio description writer for blind audiences.
Generate ONE vivid, factual sentence describing the visual content of a movie scene.
Focus on: visible actions, key objects, character appearances, emotional expressions.
CRITICAL: Only describe what is CLEARLY indicated in the context. DO NOT invent specific details.
Maintain narrative continuity with the previous shot if provided.
Before writing, silently decide which provided context items are actually relevant (some may be off-topic). Do NOT output your reasoning; only output the final description."""
        
        # Build context with source indicators
        context_parts = []
        if context_ads:
            context_parts.append("Context from similar scenes:")
            for i, (ad, src) in enumerate(zip(context_ads[:3], context_sources[:3]), 1):
                source_marker = "🎬" if src == "visual" else "📝"
                context_parts.append(f"{i}. {source_marker} {ad[:150]}")
        
        if subtitle_text:
            context_parts.append(f"\nDialogue: {subtitle_text[:200]}")
        if movie_name or movie_synopsis:
            meta_line = "Movie: " + (movie_name or "Unknown")
            if movie_synopsis:
                meta_line += f" | Synopsis: {movie_synopsis}"
            context_parts.append(meta_line)
        
        if previous_ad:
            context_parts.append(f"\nPrevious Shot AD: {previous_ad}")
        
        context_text = "\n".join(context_parts) if context_parts else "No context available."
        
        user_prompt = f"""Scene duration: {shot_info['duration']:.1f} seconds

{context_text}

Instructions:
- The provided context ADs might include irrelevant or misleading items. First, internally choose the 1-2 most relevant items (if any) and ignore the rest.
- Give priority to items marked with 🎬 (visual) over 📝 (cross-modal) if relevance is similar.
- Consider the 'Previous Shot AD' for continuity (e.g. use "He" if the character was just introduced), but focus on the CURRENT shot's action.
- Then generate ONE sentence describing what happens visually, grounded in the chosen context and the dialogue.
- Focus on observable actions and objects, not interpretations.
- Do NOT include your reasoning in the output."""
        
        temperature = 0.5  # Moderate creativity
    
    try:
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  Groq error: {e}")
        return "[Generation failed]"


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def process_movie(
    movie_id: str,
    collection: chromadb.Collection,
    groq_client: Groq,
    mad_data: Dict,
    max_shots: Optional[int] = None,
    start_time: float = 0.0,
    shot_offset: int = 0
) -> List[Dict]:
    """Process one movie: detect shots → retrieve context → generate ADs."""
    
    print(f"\n{'='*60}")
    print(f"Processing Movie: {movie_id}")
    print(f"{'='*60}")
    
    # Load frame embeddings
    embeddings = load_frame_embeddings(movie_id)
    print(f"✓ Loaded {len(embeddings)} frames @ {Config.FPS} FPS ({len(embeddings)/Config.FPS:.1f}s)")
    
    # Detect shots
    shots = detect_shots(embeddings, Config.SIM_THRESHOLD, Config.MIN_SHOT_LEN)
    print(f"✓ Detected {len(shots)} shots")
    
    # Filter shots by start_time
    if start_time > 0:
        shots = [(s, e) for s, e in shots if (e / Config.FPS) >= start_time]
        print(f"  Filtered to {len(shots)} shots after {start_time}s")
    
    # Apply shot offset
    if shot_offset and shot_offset > 0:
        prev = len(shots)
        shots = shots[shot_offset:]
        print(f"  Skipping first {shot_offset} shots ({prev} -> {len(shots)})")
    
    if max_shots:
        shots = shots[:max_shots]
        print(f"  Processing first {max_shots} shots")
    
    # Load subtitles
    try:
        subs_df = pd.read_csv(Config.CSV_SUBS)
        movie_numeric = int(movie_id)
        subs_df = subs_df[subs_df['movie'] == movie_numeric]
        print(f"✓ Loaded {len(subs_df)} subtitle entries")
    except Exception as e:
        print(f"⚠️  Could not load subtitles: {e}")
        subs_df = pd.DataFrame()
    # Derive movie context once per movie
    movie_ctx = derive_movie_context(groq_client, subs_df)
    print(f"[DEBUG] derive_movie_context: final_context={movie_ctx}")
    
    # Process each shot
    results = []
    failed_retrievals = 0
    low_confidence_skips = 0
    last_valid_ad = ""
    
    for shot_idx, (start_frame, end_frame) in enumerate(tqdm(shots, desc="Generating ADs")):
        # Shot metadata
        start_time = start_frame / Config.FPS
        end_time = end_frame / Config.FPS
        duration = end_time - start_time
        
        shot_info = {
            "shot_id": shot_idx,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "duration": float(duration),
            "movie_id": str(movie_id)
        }
        
        # Choose embedding strategy based on shot duration
        if duration < 3.0:
            # Use middle frame for short shots (more representative)
            shot_emb = get_representative_frame_embedding(embeddings, start_frame, end_frame)
        else:
            # Use pooled embedding for longer shots
            shot_emb = pool_shot_embedding(embeddings, start_frame, end_frame)
        
        # === HYBRID RETRIEVAL ===
        retrieval_results = retrieve_hybrid_context(
            collection=collection,
            shot_emb=shot_emb,
            shot_info=shot_info,
            mad_data=mad_data,
            visual_k=Config.VISUAL_CONTEXT_K,
            text_k=Config.TEXT_CONTEXT_K
        )
        
        context_ads = retrieval_results["combined_context"]
        distances = retrieval_results["combined_distances"]
        sources = retrieval_results["combined_sources"]
        
        if not context_ads:
            failed_retrievals += 1
        
        # Get subtitle context (within 3s before shot)
        subtitle_text = ""
        if not subs_df.empty:
            mask = (subs_df['start'] <= end_time) & (subs_df['end'] >= start_time - 3)
            relevant_subs = subs_df[mask]['text'].tolist()
            subtitle_text = " ".join(str(s) for s in relevant_subs)[:300]
        
        # Check if we should generate (confidence filtering)
        should_gen, reason = should_generate_ad(distances, threshold=0.7)
        
        if should_gen:
            # Generate AD with Groq
            generated_ad = generate_ad(
                groq_client, 
                shot_info, 
                context_ads, 
                sources,
                subtitle_text,
                movie_ctx.get("movie_name", ""),
                movie_ctx.get("movie_synopsis", ""),
                previous_ad=last_valid_ad
            )
            
            # Update history if generation was successful and not a failure message
            if generated_ad and not generated_ad.startswith("["):
                 last_valid_ad = generated_ad
        else:
            generated_ad = f"[Skipped: {reason}]"
            low_confidence_skips += 1
        
        # Find ground truth (if exists in same movie)
        ground_truth = None
        ground_truth_id = None
        for text_id, entry in mad_data.items():
            if entry['movie'] != movie_id:
                continue
            gt_start, gt_end = entry['timestamps']
            # Check temporal overlap
            overlap = min(end_time, gt_end) - max(start_time, gt_start)
            if overlap > 0.5 * duration:  # >50% overlap
                ground_truth = entry['sentence']
                ground_truth_id = text_id
                break
        
        results.append({
            **shot_info,
            "generated_ad": generated_ad,
            "ground_truth": ground_truth,
            "ground_truth_id": ground_truth_id,
            "visual_context": retrieval_results["visual_context"][:2],
            "cross_modal_context": retrieval_results["cross_modal_context"][:2],
            "combined_context": context_ads[:3],
            "combined_distances": [float(d) for d in distances[:3]],
            "combined_sources": sources[:3],
            "subtitle_context": subtitle_text
        })
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total shots processed: {len(results)}")
    print(f"  Failed retrievals: {failed_retrievals}")
    print(f"  Low confidence skips: {low_confidence_skips}")
    print(f"  Shots with ground truth: {sum(1 for r in results if r['ground_truth'])}")
    print(f"{'='*60}")
    
    return results


def save_results(movie_id: str, results: List[Dict]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = Config.OUTPUT_DIR / f"{movie_id}_results_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "movie_id": movie_id,
            "total_shots": len(results),
            "config": {
                "sim_threshold": Config.SIM_THRESHOLD,
                "min_shot_len": Config.MIN_SHOT_LEN,
                "text_context_k": Config.TEXT_CONTEXT_K,
                "groq_model": Config.GROQ_MODEL
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved results to: {output_file}")


def print_sample_results(results: List[Dict], n: int = 3):
    """Print sample results for inspection."""
    print(f"\n{'='*60}")
    print(f"SAMPLE RESULTS (first {n} shots)")
    print(f"{'='*60}")
    
    for r in results[:n]:
        print(f"\n📍 Shot {r['shot_id']} | {r['start_time']:.1f}s - {r['end_time']:.1f}s ({r['duration']:.1f}s)")
        print(f"   Generated: {r['generated_ad']}")
        
        if r['ground_truth']:
            print(f"   Ground Truth: {r['ground_truth'][:100]}...")
        else:
            print(f"   Ground Truth: [None found]")
        
        if r['combined_distances']:
            dists = ", ".join(f"{d:.3f}" for d in r['combined_distances'])
            print(f"   Retrieval distances: [{dists}]")
        
        if r['combined_context']:
            print(f"   Top context: {r['combined_context'][0][:80]}...")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Audio Descriptions for MAD movies")
    parser.add_argument("--movie", type=str, default="10142", help="Movie ID to process")
    parser.add_argument("--max_shots", type=int, default=20, help="Max shots to process (None = all)")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time in seconds (skip intro)")
    parser.add_argument("--shot_offset", type=int, default=0, help="Number of detected shots to skip from the start")
    parser.add_argument("--verify_only", action="store_true", help="Only verify ChromaDB structure")
    args = parser.parse_args()
    
    # Verify ChromaDB
    collection = verify_chroma_structure()
    if collection is None:
        return
    
    if args.verify_only:
        return
    
    # Initialize
    print("\n🚀 Initializing pipeline...")
    groq_client = init_groq()
    mad_data = load_mad_json("train")  # Use train split for context
    
    # Apply start_time filter if specified
    if args.start_time > 0:
        print(f"⏩ Skipping intro: starting from {args.start_time}s")
    
    # Process movie
    results = process_movie(
        movie_id=args.movie,
        collection=collection,
        groq_client=groq_client,
        mad_data=mad_data,
        max_shots=args.max_shots,
        start_time=args.start_time,
        shot_offset=args.shot_offset
    )
    
    # Save and display
    save_results(args.movie, results)
    print_sample_results(results, n=3)


if __name__ == "__main__":
    main()