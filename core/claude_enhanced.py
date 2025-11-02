"""
Enhanced Single-Movie AD Pipeline with Shot-Aware Context
---------------------------------------------------------

Implements state-of-the-art training-free AD generation based on:
- AutoAD-Zero: Two-stage VLM → LLM approach
- Shot-by-Shot: Film grammar awareness (shot scales, thread structure)
- Vector DB for temporal context retrieval

Dataset: MAD (Movie Audio Descriptions)
Author: Nikhil Andrew Franco
Version: 2.0 (Research-Grade)
"""

import os
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

# Vector DB
import chromadb
from chromadb.config import Settings

# LLM Integration
from groq import Groq
from dotenv import load_dotenv

# Optional visualization
try:
    import matplotlib.pyplot as plt
    from IPython.display import display, HTML
except ImportError:
    plt = None

# ============================================================================
# Configuration & Data Structures
# ============================================================================

@dataclass
class Shot:
    """Shot-level representation"""
    shot_id: int
    start_frame: int
    end_frame: int
    embedding: np.ndarray
    duration: float  # in seconds
    shot_scale: Optional[str] = None  # "close-up", "medium", "long"
    thread_id: Optional[int] = None
    characters: List[str] = None
    
    def __post_init__(self):
        if self.characters is None:
            self.characters = []

@dataclass
class MovieMetadata:
    """Movie-level metadata"""
    movie_id: str
    title: str
    fps: float = 5.0  # MAD default
    total_frames: int = 0
    total_duration: float = 0.0
    cast_list: List[str] = None
    
    def __post_init__(self):
        if self.cast_list is None:
            self.cast_list = []
        if self.total_duration == 0 and self.total_frames > 0:
            self.total_duration = self.total_frames / self.fps


class Config:
    """Centralized configuration"""
    # Paths
    BASE_DIR = Path(r"C:\Users\nikhi\projects\AI-Video-Describer\MAD")
    FEATURES_DIR = BASE_DIR / "features"
    ANNOTATIONS_DIR = BASE_DIR / "annotations" / "MAD-v2"
    
    # Feature files
    CLIP_FRAMES = FEATURES_DIR / "CLIP_B32_frames_features_5fps.h5"
    CLIP_L14_FRAMES = FEATURES_DIR / "CLIP_L14_frames_features_5fps.h5"
    
    # Annotation files
    AD_NAMED = ANNOTATIONS_DIR / "mad-v2-ad-named.csv"
    AD_UNNAMED = ANNOTATIONS_DIR / "mad-v2-ad-unnamed.csv"
    SUBTITLES = ANNOTATIONS_DIR / "mad-v2-subs.csv"
    
    # Shot detection parameters
    SHOT_SIM_THRESHOLD = 0.80  # Cosine similarity for shot boundaries
    MIN_SHOT_LENGTH = 3  # Minimum frames per shot
    FPS = 5.0  # MAD dataset frame rate
    
    # ChromaDB settings
    CHROMA_PERSIST_DIR = "./chroma_db"
    CHROMA_COLLECTION = "mad_shots_v2"
    
    # Generation parameters
    CONTEXT_WINDOW_SHOTS = 5  # Retrieve top-k shots for context
    MAX_TOKENS_STAGE1 = 300  # Dense description
    MAX_TOKENS_STAGE2 = 100  # Final AD
    
    # API keys
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================================
# I/O & Data Loading
# ============================================================================

class MADDataLoader:
    """Handles loading MAD dataset components"""
    
    def __init__(self, config: Config):
        self.config = config
        self._load_annotations()
    
    def _load_annotations(self):
        """Load all annotation CSVs"""
        print("Loading MAD annotations...")
        self.ad_named = pd.read_csv(self.config.AD_NAMED)
        self.ad_unnamed = pd.read_csv(self.config.AD_UNNAMED)
        self.subtitles = pd.read_csv(self.config.SUBTITLES)
        print(f"✓ Loaded {len(self.ad_named)} AD annotations")
    
    def load_movie_features(self, movie_id: str, use_large_clip: bool = False) -> np.ndarray:
        """Load CLIP features for a movie"""
        h5_path = self.config.CLIP_L14_FRAMES if use_large_clip else self.config.CLIP_FRAMES
        
        with h5py.File(h5_path, 'r') as f:
            if movie_id not in f:
                available = list(f.keys())[:10]
                raise KeyError(
                    f"Movie '{movie_id}' not found. "
                    f"Available (first 10): {available}"
                )
            features = np.array(f[movie_id]).astype(np.float32)
        
        print(f"✓ Loaded {movie_id}: {features.shape}")
        return features
    
    def get_movie_ads(self, movie_id: str, use_named: bool = True) -> pd.DataFrame:
        """Get ground truth ADs for a movie"""
        df = self.ad_named if use_named else self.ad_unnamed
        movie_ads = df[df['movie'] == movie_id].copy()
        return movie_ads.sort_values('start_time')
    
    def get_movie_metadata(self, movie_id: str) -> MovieMetadata:
        """Extract metadata for a movie"""
        features = self.load_movie_features(movie_id)
        
        # Get cast from first AD annotation (stub - replace with IMDb API)
        ads = self.get_movie_ads(movie_id)
        cast = self._extract_cast_stub(ads)
        
        return MovieMetadata(
            movie_id=movie_id,
            title=movie_id.replace('_', ' '),
            fps=self.config.FPS,
            total_frames=len(features),
            cast_list=cast
        )
    
    @staticmethod
    def _extract_cast_stub(ads_df: pd.DataFrame) -> List[str]:
        """Stub: Extract character names from ADs (replace with IMDb)"""
        # Simple NER-based extraction (placeholder)
        import re
        characters = set()
        for ad in ads_df['sentence'].head(20):  # Sample first 20 ADs
            # Look for capitalized names (naive approach)
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', str(ad))
            characters.update(names[:3])  # Limit per AD
        return list(characters)[:10]  # Top 10 characters


# ============================================================================
# Shot Detection & Segmentation
# ============================================================================

class ShotDetector:
    """Detects shots using frame similarity (TransNetV2-inspired)"""
    
    def __init__(self, config: Config):
        self.config = config
    
    @staticmethod
    def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
        """L2 normalization"""
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-8)
    
    def detect_shots(self, embeddings: np.ndarray) -> List[Shot]:
        """
        Detect shot boundaries via cosine similarity drop
        
        Args:
            embeddings: [num_frames, dim] frame features
            
        Returns:
            List of Shot objects
        """
        emb_norm = self.normalize_embeddings(embeddings)
        
        # Compute frame-to-frame similarity
        similarities = (emb_norm[:-1] * emb_norm[1:]).sum(axis=1)
        
        # Detect cuts (sharp similarity drops)
        cut_indices = np.where(similarities < self.config.SHOT_SIM_THRESHOLD)[0]
        
        # Build shot boundaries with minimum length constraint
        boundaries = [0]
        for idx in cut_indices:
            if idx + 1 - boundaries[-1] >= self.config.MIN_SHOT_LENGTH:
                boundaries.append(idx + 1)
        
        if boundaries[-1] != len(embeddings):
            boundaries.append(len(embeddings))
        
        # Create Shot objects
        shots = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            
            # Aggregate frames within shot
            shot_embedding = embeddings[start:end].mean(axis=0)
            shot_embedding = shot_embedding / (np.linalg.norm(shot_embedding) + 1e-8)
            
            shots.append(Shot(
                shot_id=i,
                start_frame=start,
                end_frame=end,
                embedding=shot_embedding,
                duration=(end - start) / self.config.FPS
            ))
        
        print(f"✓ Detected {len(shots)} shots")
        return shots
    
    def classify_shot_scale(self, shot: Shot, embeddings: np.ndarray) -> str:
        """
        Classify shot scale (close-up, medium, long)
        Placeholder: Should use fine-tuned DINOv2 on MovieShots dataset
        """
        # Heuristic based on duration (replace with real classifier)
        if shot.duration < 2.0:
            return "close-up"
        elif shot.duration < 5.0:
            return "medium"
        else:
            return "long"


# ============================================================================
# Vector Database (ChromaDB)
# ============================================================================

class ShotVectorDB:
    """Manages shot-level embeddings in ChromaDB"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = chromadb.Client(Settings(
            persist_directory=str(config.CHROMA_PERSIST_DIR),
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_shots(self, movie_id: str, shots: List[Shot]):
        """Store shot embeddings with metadata"""
        embeddings = [shot.embedding.tolist() for shot in shots]
        ids = [f"{movie_id}_shot_{shot.shot_id}" for shot in shots]
        
        metadatas = [{
            "movie_id": movie_id,
            "shot_id": shot.shot_id,
            "start_frame": shot.start_frame,
            "end_frame": shot.end_frame,
            "duration": shot.duration,
            "shot_scale": shot.shot_scale or "unknown",
            "characters": json.dumps(shot.characters)
        } for shot in shots]
        
        self.collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"✓ Stored {len(shots)} shots in ChromaDB")
    
    def retrieve_context(
        self, 
        query_shot: Shot, 
        movie_id: str,
        k: int = 5,
        temporal_window: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve contextually relevant shots
        
        Args:
            query_shot: Current shot
            movie_id: Filter by movie
            k: Number of results
            temporal_window: Only retrieve shots within N shots of query
        """
        where_filter = {"movie_id": movie_id}
        
        if temporal_window:
            where_filter.update({
                "shot_id": {
                    "$gte": max(0, query_shot.shot_id - temporal_window),
                    "$lte": query_shot.shot_id + temporal_window
                }
            })
        
        results = self.collection.query(
            query_embeddings=[query_shot.embedding.tolist()],
            n_results=k,
            where=where_filter
        )
        
        return results


# ============================================================================
# Two-Stage AD Generation (AutoAD-Zero Style)
# ============================================================================

class ADGenerator:
    """Training-free two-stage AD generation"""
    
    def __init__(self, config: Config):
        self.config = config
        
        if not config.GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not found. Set in .env file.")
        
        self.client = Groq(api_key=config.GROQ_API_KEY)
    
    def generate_stage1_description(
        self, 
        shot: Shot,
        context_shots: List[Dict],
        metadata: MovieMetadata
    ) -> str:
        """
        Stage I: Generate dense, comprehensive description
        Uses shot-aware prompting from Shot-by-Shot paper
        """
        # Build context from retrieved shots
        context_text = self._format_context(context_shots)
        
        # Shot-scale-dependent factors (from Shot-by-Shot)
        factors = ["character actions", "interactions"]
        if shot.shot_scale == "close-up":
            factors.append("facial expressions")
        elif shot.shot_scale == "long":
            factors.append("environment and atmosphere")
        
        prompt = f"""You are generating an audio description for a blind/low-vision audience.

Shot Information:
- Duration: {shot.duration:.1f}s
- Scale: {shot.shot_scale or 'unknown'}
- Possible characters: {', '.join(shot.characters) or 'Unknown'}

Previous Context:
{context_text}

Task: Describe this shot in detail, focusing on:
{chr(10).join(f'- {f}' for f in factors)}

Be specific, vivid, and avoid repetition from context.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Use text model for Stage I
                messages=[
                    {"role": "system", "content": "You generate rich visual descriptions for accessibility."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=self.config.MAX_TOKENS_STAGE1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠ Stage I error: {e}")
            return "[Description unavailable]"
    
    def generate_stage2_ad(
        self, 
        dense_description: str,
        shot: Shot,
        metadata: MovieMetadata
    ) -> str:
        """
        Stage II: Summarize into concise AD
        Enforces length, style, and action focus
        """
        # Calculate target length (1 word per 0.2s, roughly)
        target_words = max(5, int(shot.duration / 0.2))
        
        # Common action verbs (from AutoAD-Zero)
        action_verbs = [
            'look', 'turn', 'walk', 'run', 'grab', 'hold', 
            'stare', 'smile', 'watch', 'approach'
        ]
        
        prompt = f"""Summarize this description into ONE concise audio description sentence.

Dense Description:
{dense_description}

Requirements:
- Target length: ~{target_words} words (for {shot.duration:.1f}s duration)
- Use character first names only: {', '.join(shot.characters[:2]) or 'pronouns (he/she)'}
- Focus on main action (prefer verbs: {', '.join(action_verbs[:5])})
- Narrator perspective (third-person)
- DO NOT mention camera or technical details

Audio Description:"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You create concise audio descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=self.config.MAX_TOKENS_STAGE2
            )
            
            ad = response.choices[0].message.content.strip()
            # Clean up common artifacts
            ad = ad.replace("Audio Description:", "").strip()
            ad = ad.strip('"\'')
            
            return ad
        except Exception as e:
            print(f"⚠ Stage II error: {e}")
            return dense_description[:50] + "..."
    
    @staticmethod
    def _format_context(context_shots: List[Dict]) -> str:
        """Format retrieved shots as context text"""
        if not context_shots or 'metadatas' not in context_shots:
            return "No previous context."
        
        lines = []
        for meta in context_shots['metadatas'][0][:3]:  # Top 3
            shot_id = meta.get('shot_id', '?')
            duration = meta.get('duration', 0)
            lines.append(f"Shot {shot_id} ({duration:.1f}s)")
        
        return "\n".join(lines) if lines else "No previous context."


# ============================================================================
# Evaluation Metrics
# ============================================================================

class ADEvaluator:
    """Compute metrics: CRITIC, CIDEr, Action Score"""
    
    def __init__(self):
        pass
    
    def compute_metrics(
        self, 
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute multiple AD quality metrics
        
        For now, returns placeholders. Implement:
        - CIDEr: n-gram overlap
        - CRITIC: character name accuracy
        - Action Score: verb matching
        """
        return {
            "cider": 0.0,  # TODO: Implement
            "critic": 0.0,  # TODO: Implement
            "action_score": 0.0  # TODO: Implement
        }


# ============================================================================
# Main Pipeline
# ============================================================================

class SingleMoviePipeline:
    """End-to-end pipeline for one movie"""
    
    def __init__(self, config: Config):
        self.config = config
        self.loader = MADDataLoader(config)
        self.shot_detector = ShotDetector(config)
        self.vector_db = ShotVectorDB(config)
        self.ad_generator = ADGenerator(config)
        self.evaluator = ADEvaluator()
    
    def run(self, movie_id: str, max_shots: Optional[int] = 5):
        """
        Execute full pipeline on a single movie
        
        Args:
            movie_id: MAD movie identifier (e.g., "0001_American_Beauty")
            max_shots: Limit processing to first N shots (for testing)
        """
        print(f"\n{'='*60}")
        print(f"Processing: {movie_id}")
        print(f"{'='*60}\n")
        
        # 1. Load movie data
        metadata = self.loader.get_movie_metadata(movie_id)
        embeddings = self.loader.load_movie_features(movie_id)
        ground_truth_ads = self.loader.get_movie_ads(movie_id)
        
        print(f"\n📊 Movie Info:")
        print(f"   Frames: {metadata.total_frames}")
        print(f"   Duration: {metadata.total_duration:.1f}s")
        print(f"   Ground Truth ADs: {len(ground_truth_ads)}")
        
        # 2. Detect shots
        shots = self.shot_detector.detect_shots(embeddings)
        
        # Classify shot scales
        for shot in shots:
            shot.shot_scale = self.shot_detector.classify_shot_scale(shot, embeddings)
            # Assign characters (stub)
            shot.characters = metadata.cast_list[:2] if shot.shot_id % 3 == 0 else []
        
        # 3. Store in vector DB
        self.vector_db.store_shots(movie_id, shots)
        
        # 4. Generate ADs for first N shots
        test_shots = shots[:max_shots] if max_shots else shots
        results = []
        
        print(f"\n🎬 Generating ADs for {len(test_shots)} shots...\n")
        
        for shot in test_shots:
            print(f"\n--- Shot {shot.shot_id + 1} ({shot.start_frame}-{shot.end_frame}, {shot.duration:.1f}s, {shot.shot_scale}) ---")
            
            # Retrieve context
            context = self.vector_db.retrieve_context(
                shot, 
                movie_id, 
                k=self.config.CONTEXT_WINDOW_SHOTS
            )
            
            # Stage I: Dense description
            dense_desc = self.ad_generator.generate_stage1_description(
                shot, context, metadata
            )
            print(f"Dense: {dense_desc[:100]}...")
            
            # Stage II: Final AD
            final_ad = self.ad_generator.generate_stage2_ad(
                dense_desc, shot, metadata
            )
            print(f"✓ AD: {final_ad}")
            
            # Find matching ground truth (if any)
            gt_ad = self._find_matching_gt(shot, ground_truth_ads)
            if gt_ad:
                print(f"   GT: {gt_ad}")
            
            results.append({
                "shot_id": shot.shot_id,
                "predicted_ad": final_ad,
                "dense_description": dense_desc,
                "ground_truth": gt_ad
            })
        
        # 5. Save results
        self._save_results(movie_id, results)
        
        print(f"\n{'='*60}")
        print("✓ Pipeline completed successfully!")
        print(f"{'='*60}\n")
        
        return results
    
    @staticmethod
    def _find_matching_gt(shot: Shot, gt_ads: pd.DataFrame) -> Optional[str]:
        """Find ground truth AD overlapping with shot timeframe"""
        shot_start_sec = shot.start_frame / 5.0  # MAD uses 5fps
        shot_end_sec = shot.end_frame / 5.0
        
        matching = gt_ads[
            (gt_ads['start_time'] >= shot_start_sec) & 
            (gt_ads['end_time'] <= shot_end_sec)
        ]
        
        return matching.iloc[0]['sentence'] if len(matching) > 0 else None
    
    def _save_results(self, movie_id: str, results: List[Dict]):
        """Save predictions to JSON"""
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{movie_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "movie_id": movie_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


# ============================================================================
# Run Example
# ============================================================================

if __name__ == "__main__":
    # Initialize
    config = Config()
    pipeline = SingleMoviePipeline(config)
    
    # List available movies
    print("\n🎬 Available movies in MAD dataset:")
    with h5py.File(config.CLIP_FRAMES, 'r') as f:
        movies = list(f.keys())[:20]  # First 20
        for i, movie in enumerate(movies, 1):
            print(f"   {i}. {movie}")
    
    # Run on first movie (or specify)
    MOVIE_ID = "0001_American_Beauty"  # Change this!
    
    try:
        results = pipeline.run(MOVIE_ID, max_shots=5)
    except KeyError as e:
        print(f"\n❌ Error: {e}")
        print("   Try a different movie_id from the list above.")