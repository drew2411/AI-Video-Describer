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

# BERTScore (optional)
try:
    from bert_score import score as bertscore_score
except ImportError:
    print("⚠️  BERTScore not installed. Install: pip install bert-score")
    bertscore_score = None


class ADEvaluator:
    """Evaluator for Audio Descriptions."""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if sentence_bleu else None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) if rouge_scorer else None
        self.idf_stats = None  # for CIDEr
        self.num_refs = 0

    # ----------------------------
    # CIDEr (simplified implementation)
    # ----------------------------
    def _ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)] if n <= len(tokens) else []

    def _build_idf(self, references: List[List[str]]):
        """Build document frequency (DF) and IDF for 1..4-grams over reference corpus."""
        df = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}
        num_docs = len(references)
        for ref in references:
            ref_tokens = ref
            for n in (1, 2, 3, 4):
                seen = set(self._ngrams(ref_tokens, n))
                for ng in seen:
                    df[n][ng] += 1
        # idf: log((N+1)/(df+1))
        idf = {n: {} for n in (1, 2, 3, 4)}
        for n in (1, 2, 3, 4):
            for ng, cnt in df[n].items():
                idf[n][ng] = np.log((num_docs + 1.0) / (cnt + 1.0))
        self.idf_stats = idf
        self.num_refs = num_docs

    def _cider(self, gen_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute a simple CIDEr-D-like score using TF-IDF cosine over 1..4-grams."""
        if not self.idf_stats or self.num_refs == 0:
            return 0.0
        sims = []
        for n in (1, 2, 3, 4):
            # TF
            g_ngrams = self._ngrams(gen_tokens, n)
            r_ngrams = self._ngrams(ref_tokens, n)
            if len(g_ngrams) == 0 or len(r_ngrams) == 0:
                sims.append(0.0)
                continue
            g_tf = defaultdict(float)
            r_tf = defaultdict(float)
            for ng in g_ngrams:
                g_tf[ng] += 1.0
            for ng in r_ngrams:
                r_tf[ng] += 1.0
            # L1 normalize TF
            g_len = sum(g_tf.values())
            r_len = sum(r_tf.values())
            for k in list(g_tf.keys()):
                g_tf[k] /= max(g_len, 1e-9)
            for k in list(r_tf.keys()):
                r_tf[k] /= max(r_len, 1e-9)
            # Apply IDF weights
            def weighted(vec):
                out = {}
                for k, v in vec.items():
                    idf = self.idf_stats[n].get(k, np.log((self.num_refs + 1.0) / 1.0))
                    out[k] = v * idf
                return out
            g_w = weighted(g_tf)
            r_w = weighted(r_tf)
            # Cosine similarity
            common = set(g_w.keys()) & set(r_w.keys())
            dot = sum(g_w[k] * r_w[k] for k in common)
            g_norm = np.sqrt(sum(v*v for v in g_w.values()))
            r_norm = np.sqrt(sum(v*v for v in r_w.values()))
            sim = dot / (g_norm * r_norm + 1e-9)
            sims.append(sim)
        # Average and scale similar to CIDEr (×10 for readability)
        return float(np.mean(sims) * 10.0)
    
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
        
        # BERTScore (F1)
        if bertscore_score:
            try:
                P, R, F1 = bertscore_score([generated], [reference], lang='en', rescale_with_baseline=True)
                metrics['bertscore_f1'] = float(F1[0].item())
            except Exception:
                metrics['bertscore_f1'] = 0.0
        
        # CIDEr (simplified)
        try:
            metrics['cider'] = self._cider(gen_tokens, ref_tokens)
        except Exception:
            metrics['cider'] = 0.0
        
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
        
        # Build IDF corpus for CIDEr using references
        references_corpus = [r['ground_truth'].lower().split() for r in with_gt]
        if references_corpus:
            self._build_idf(references_corpus)

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
        for metric in ['bleu1', 'bleu2', 'bleu4', 'meteor', 'bertscore_f1', 'cider']:
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