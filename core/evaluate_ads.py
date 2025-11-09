"""
evaluate_ads.py

Evaluate generated Audio Descriptions against ground truth.
Supports BLEU, METEOR, ROUGE, and semantic similarity metrics.

Usage:
    python evaluate_ads.py --results ./mad_generated_ads/10142_results.json
    python evaluate_ads.py --results ./mad_generated_ads/10142_results.json --detailed
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# NLP metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    try:
        nltk.data.find('wordnet')
    except:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
except ImportError:
    print("⚠️  NLTK not installed. Install: pip install nltk")
    sentence_bleu = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("⚠️  rouge-score not installed. Install: pip install rouge-score")
    rouge_scorer = None


class ADEvaluator:
    """Evaluator for Audio Descriptions."""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if sentence_bleu else None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) if rouge_scorer else None
    
    def evaluate_single(self, generated: str, reference: str) -> Dict:
        """Evaluate single generated AD against reference."""
        
        metrics = {}
        
        # Tokenize
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        # Length metrics
        metrics['gen_length'] = len(gen_tokens)
        metrics['ref_length'] = len(ref_tokens)
        metrics['length_ratio'] = len(gen_tokens) / max(len(ref_tokens), 1)
        
        # BLEU scores
        if sentence_bleu:
            metrics['bleu1'] = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
            metrics['bleu2'] = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
            metrics['bleu4'] = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
        
        # METEOR
        if sentence_bleu:  # meteor_score requires same nltk install
            try:
                metrics['meteor'] = meteor_score([ref_tokens], gen_tokens)
            except Exception as e:
                metrics['meteor'] = 0.0
        
        # ROUGE
        if self.rouge_scorer:
            rouge_scores = self.rouge_scorer.score(reference, generated)
            metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        # Token overlap (simple)
        gen_set = set(gen_tokens)
        ref_set = set(ref_tokens)
        if ref_set:
            metrics['token_precision'] = len(gen_set & ref_set) / len(gen_set) if gen_set else 0
            metrics['token_recall'] = len(gen_set & ref_set) / len(ref_set)
            metrics['token_f1'] = 2 * metrics['token_precision'] * metrics['token_recall'] / (metrics['token_precision'] + metrics['token_recall'] + 1e-9)
        
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
        without_gt = []
        failed_gen = []
        
        for result in data['results']:
            if result['generated_ad'] == "[Generation failed]":
                failed_gen.append(result)
            elif result['ground_truth']:
                with_gt.append(result)
            else:
                without_gt.append(result)
        
        print(f"\nMovie ID: {movie_id}")
        print(f"Total shots: {total_shots}")
        print(f"  With ground truth: {len(with_gt)}")
        print(f"  Without ground truth: {len(without_gt)}")
        print(f"  Failed generation: {len(failed_gen)}")
        
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
        metric_names = [k for k in all_scores[0].keys() if k not in ['shot_id', 'start_time']]
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
        
        print("\n📊 Language Similarity:")
        for metric in ['bleu1', 'bleu2', 'bleu4', 'meteor']:
            if metric in avg_scores:
                print(f"  {metric:15s}: {avg_scores[metric]:.4f} ± {std_scores[metric]:.4f}")
        
        print("\n📊 ROUGE Scores:")
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            if metric in avg_scores:
                print(f"  {metric:15s}: {avg_scores[metric]:.4f} ± {std_scores[metric]:.4f}")
        
        print("\n📊 Token Overlap:")
        for metric in ['token_precision', 'token_recall', 'token_f1']:
            if metric in avg_scores:
                print(f"  {metric:15s}: {avg_scores[metric]:.4f} ± {std_scores[metric]:.4f}")
        
        print("\n📊 Length Statistics:")
        for metric in ['gen_length', 'ref_length', 'length_ratio']:
            if metric in avg_scores:
                print(f"  {metric:15s}: {avg_scores[metric]:.2f} ± {std_scores[metric]:.2f}")
        
        # Detailed per-shot analysis
        if detailed and len(all_scores) > 0:
            print(f"\n{'='*70}")
            print(f"DETAILED PER-SHOT ANALYSIS")
            print(f"{'='*70}")
            
            # Sort by BLEU-4 (or first available metric)
            sort_key = 'bleu4' if 'bleu4' in all_scores[0] else metric_names[0]
            sorted_scores = sorted(all_scores, key=lambda x: x[sort_key], reverse=True)
            
            print(f"\n🏆 Top 5 shots (by {sort_key}):")
            for i, score in enumerate(sorted_scores[:5], 1):
                shot_id = score['shot_id']
                result = with_gt[shot_id] if shot_id < len(with_gt) else None
                if result:
                    print(f"\n  {i}. Shot {shot_id} @ {score['start_time']:.1f}s | {sort_key}={score[sort_key]:.3f}")
                    print(f"     Generated: {result['generated_ad']}")
                    print(f"     Reference: {result['ground_truth'][:100]}...")
            
            print(f"\n❌ Bottom 5 shots (by {sort_key}):")
            for i, score in enumerate(sorted_scores[-5:], 1):
                shot_id = score['shot_id']
                result = with_gt[shot_id] if shot_id < len(with_gt) else None
                if result:
                    print(f"\n  {i}. Shot {shot_id} @ {score['start_time']:.1f}s | {sort_key}={score[sort_key]:.3f}")
                    print(f"     Generated: {result['generated_ad']}")
                    print(f"     Reference: {result['ground_truth'][:100]}...")
        
        # Retrieval quality analysis
        print(f"\n{'='*70}")
        print(f"RETRIEVAL QUALITY ANALYSIS")
        print(f"{'='*70}")
        
        retrieval_distances = []
        for result in data['results']:
            if result.get('context_distances'):
                retrieval_distances.extend(result['context_distances'])
        
        if retrieval_distances:
            print(f"  Total retrievals: {len(retrieval_distances)}")
            print(f"  Mean distance: {np.mean(retrieval_distances):.4f}")
            print(f"  Median distance: {np.median(retrieval_distances):.4f}")
            print(f"  Min distance: {np.min(retrieval_distances):.4f}")
            print(f"  Max distance: {np.max(retrieval_distances):.4f}")
        
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
        
        # Aggregate across movies
        all_metric_names = set()
        for r in all_results:
            all_metric_names.update(r['avg_scores'].keys())
        
        for metric in sorted(all_metric_names):
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