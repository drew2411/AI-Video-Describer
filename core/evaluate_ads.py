"""
evaluate_ads.py

Evaluate generated Audio Descriptions against ground truth using Cosine Similarity.

Usage:
    python evaluate_ads.py --results ./mad_generated_ads/10142_results.json
    python evaluate_ads.py --results ./mad_generated_ads/10142_results.json --detailed
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

# Sentence Transformers (for Cosine Similarity)
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("⚠️  sentence-transformers not installed. Install: pip install sentence-transformers")
    SentenceTransformer = None


class ADEvaluator:
    """Evaluator for Audio Descriptions using Cosine Similarity."""
    
    def __init__(self):
        # Initialize Sentence Transformer model
        if SentenceTransformer:
            print("⏳ Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Model loaded.")
        else:
            self.st_model = None
            print("❌ SentenceTransformer not available. Cosine similarity will be 0.")

    def evaluate_single(self, generated: str, reference: str) -> Dict:
        """Evaluate single generated AD against reference."""
        
        metrics = {}
        
        # Cosine Similarity (Sentence Transformers)
        if self.st_model:
            embeddings = self.st_model.encode([generated, reference], convert_to_tensor=True)
            cosine_sim = util.cos_sim(embeddings[0], embeddings[1])
            metrics['cosine_similarity'] = float(cosine_sim.item())
        else:
             metrics['cosine_similarity'] = 0.0
        
        return metrics
    
    def evaluate_file(self, results_path: str, detailed: bool = False) -> Dict:
        """Evaluate all results in a file."""
        
        print(f"\n{'='*70}")
        print(f"EVALUATING: {results_path}")
        print(f"{'='*70}")
        
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        movie_id = data.get('movie_id', 'unknown')
        total_shots = data.get('total_shots', len(data['results']))
        
        # Separate shots with/without ground truth
        with_gt = []
        
        for result in data['results']:
            if result.get('ground_truth') and result.get('generated_ad') != "[Generation failed]":
                with_gt.append(result)
        
        if not with_gt:
            print("\n⚠️  No shots with ground truth found - cannot compute metrics!")
            return {}
        
        # Compute metrics
        all_scores = []
        for result in with_gt:
            score = self.evaluate_single(result['generated_ad'], result['ground_truth'])
            score['shot_id'] = result['shot_id']
            score['start_time'] = result['start_time']
            all_scores.append(score)
        
        # Aggregate
        metric_names = ['cosine_similarity']
        avg_scores = {
            metric: np.mean([s[metric] for s in all_scores])
            for metric in metric_names
        }
        std_scores = {
            metric: np.std([s[metric] for s in all_scores])
            for metric in metric_names
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EVALUATION METRICS (n={len(with_gt)})")
        print(f"{'='*70}")
        
        print(f"\n📊 Cosine Similarity: {avg_scores['cosine_similarity']:.4f} ± {std_scores['cosine_similarity']:.4f}")
        
        # Detailed per-shot analysis
        if detailed and len(all_scores) > 0:
            print(f"\n{'='*70}")
            print(f"DETAILED PER-SHOT ANALYSIS")
            print(f"{'='*70}")
            
            # Sort by Cosine Similarity
            sorted_scores = sorted(all_scores, key=lambda x: x['cosine_similarity'], reverse=True)
            
            print(f"\n🏆 Top 5 shots:")
            for i, score in enumerate(sorted_scores[:5], 1):
                shot_id = score['shot_id']
                # find result by shot_id specifically to be safe, though index might match if preserved
                result = next((r for r in with_gt if r['shot_id'] == shot_id), None)
                if result:
                    print(f"\n  {i}. Shot {shot_id} @ {score['start_time']:.1f}s | Cosine={score['cosine_similarity']:.3f}")
                    print(f"     Generated: {result['generated_ad']}")
                    print(f"     Reference: {result['ground_truth']}")
            
            print(f"\n❌ Bottom 5 shots:")
            for i, score in enumerate(sorted_scores[-5:], 1):
                shot_id = score['shot_id']
                result = next((r for r in with_gt if r['shot_id'] == shot_id), None)
                if result:
                    print(f"\n  {i}. Shot {shot_id} @ {score['start_time']:.1f}s | Cosine={score['cosine_similarity']:.3f}")
                    print(f"     Generated: {result['generated_ad']}")
                    print(f"     Reference: {result['ground_truth']}")
        
        return {
            'movie_id': movie_id,
            'total_shots': total_shots,
            'evaluated_shots': len(with_gt),
            'avg_scores': avg_scores,
            'std_scores': std_scores,
            'all_scores': all_scores if detailed else None
        }
    
    def compare_multiple(self, results_paths: List[str]):
        """Compare results across multiple movies."""
        
        all_results = []
        for path in results_paths:
            result = self.evaluate_file(path, detailed=False)
            if result:
                all_results.append(result)
        
        if len(all_results) < 2:
            return
        
        print(f"\n{'='*70}")
        print(f"CROSS-MOVIE COMPARISON")
        print(f"{'='*70}")
        
        metric = 'cosine_similarity'
        values = [r['avg_scores'][metric] for r in all_results if metric in r['avg_scores']]
        
        if values:
            print(f"\n{metric}:")
            for r in all_results:
                if metric in r['avg_scores']:
                    print(f"  {r['movie_id']:10s}: {r['avg_scores'][metric]:.4f}")
            print(f"  {'AVERAGE':10s}: {np.mean(values):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated Audio Descriptions")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-shot analysis")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare multiple result files")
    args = parser.parse_args()
    
    evaluator = ADEvaluator()
    
    if args.compare:
        evaluator.compare_multiple(args.compare)
    else:
        evaluator.evaluate_file(args.results, detailed=args.detailed)


if __name__ == "__main__":
    main()