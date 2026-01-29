# to run
# Quick suggestions with estimated offsets:
# python core/find_good_movies.py --top_n 30 --min_ads 50 --window_sec 600
# With exact shot offsets:
# python core/find_good_movies.py --top_n 30 --min_ads 50 --window_sec 600 --compute_offsets
# This loads embeddings and runs your detect_shots to map recommend_start_time to a shot_offset.
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from mad_ad_generator_final import (
        Config,
        load_mad_json,
        detect_shots,
        load_frame_embeddings,
    )
except Exception:
    # Fallback simple Config if direct import fails
    class Config:  # type: ignore
        JSON_TRAIN = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_train.json"
        JSON_VAL = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_val.json"
        JSON_TEST = r"C:\Users\nikhi\projects\AI-Video-Describer\MAD\annotations\MAD-v1\MAD_test.json"

    def load_mad_json(split: str = "train") -> Dict:
        path_map = {
            "train": Config.JSON_TRAIN,
            "val": Config.JSON_VAL,
            "test": Config.JSON_TEST,
        }
        with open(path_map[split], "r", encoding="utf-8") as f:
            return json.load(f)

    def detect_shots(embeddings, threshold, min_len):
        # Minimal fallback if import fails; not used for exact offsets without full module
        return []

    def load_frame_embeddings(movie_id: str):
        raise RuntimeError("Exact shot offsets not available without mad_ad_generator_final imports")


def gather_entries(splits: List[str]) -> Dict[str, Dict]:
    combined: Dict[str, Dict] = {}
    for sp in splits:
        data = load_mad_json(sp)
        combined.update(data)
    return combined


def per_movie_stats(entries: Dict[str, Dict], window_sec: float) -> List[Dict]:
    by_movie: Dict[str, List[Tuple[float, float, str]]] = {}
    for _, e in entries.items():
        mv = str(e.get("movie"))
        ts = e.get("timestamps", [0.0, 0.0])
        sent = e.get("sentence", "")
        if mv not in by_movie:
            by_movie[mv] = []
        by_movie[mv].append((float(ts[0]), float(ts[1]), sent))

    results = []
    for mv, items in by_movie.items():
        items.sort(key=lambda x: x[0])
        starts = [s for s, _, _ in items]
        ends = [e for _, e, _ in items]
        n = len(items)
        min_t = starts[0] if n > 0 else 0.0
        max_t = max(ends) if n > 0 else 0.0
        best_count = 0
        best_start = 0.0
        i = 0
        for j in range(n):
            t_start = starts[j]
            t_end = t_start + window_sec
            while i < n and starts[i] < t_start:
                i += 1
            k = j
            while k < n and starts[k] <= t_end:
                k += 1
            count = k - j
            if count > best_count:
                best_count = count
                best_start = t_start
        density = best_count / window_sec if window_sec > 0 else 0.0
        results.append({
            "movie": mv,
            "total_ads": n,
            "first_ts": float(min_t),
            "last_ts": float(max_t),
            "span_sec": float(max(0.0, max_t - min_t)),
            "best_window_sec": float(window_sec),
            "best_window_start": float(best_start),
            "best_window_count": int(best_count),
            "best_window_density_per_sec": float(density),
            "recommend_start_time": float(best_start),
        })
    results.sort(key=lambda x: (x["total_ads"], x["best_window_count"], x["best_window_density_per_sec"]), reverse=True)
    return results


def _compute_shot_offset_exact(movie_id: str, start_time_sec: float) -> int:
    try:
        emb = load_frame_embeddings(movie_id)
        shots = detect_shots(emb, Config.SIM_THRESHOLD, Config.MIN_SHOT_LEN)
        # Find first shot whose end_time >= start_time_sec
        for idx, (s, e) in enumerate(shots):
            end_t = e / Config.FPS
            if end_t >= start_time_sec:
                return idx
        return max(0, len(shots) - 1)
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser(description="Find movies with many ground-truth ADs and suggest start times")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"], help="Which MAD splits to use")
    parser.add_argument("--min_ads", type=int, default=30, help="Minimum total ADs to include")
    parser.add_argument("--window_sec", type=float, default=300.0, help="Window size in seconds for density search")
    parser.add_argument("--top_n", type=int, default=50, help="How many movies to list")
    parser.add_argument("--output", type=str, default="core/good_movies_suggestions.json", help="Path to save JSON suggestions")
    parser.add_argument("--compute_offsets", action="store_true", help="Compute exact shot offsets using embeddings and shot detection for top movies")
    parser.add_argument("--estimate_offset_sec", type=float, default=3.0, help="Average seconds per shot to estimate shot_offset when exact computation not possible")
    args = parser.parse_args()

    print("Loading MAD entries...")
    entries = gather_entries(args.splits)
    print(f"Loaded {len(entries)} entries from splits: {args.splits}")

    stats = per_movie_stats(entries, args.window_sec)
    filtered = [r for r in stats if r["total_ads"] >= args.min_ads]

    top = filtered[: args.top_n]

    # Attach shot offsets
    enriched = []
    for r in top:
        movie_id = str(r["movie"])  # may be name-like; exact offset may fail if embeddings key differs
        start_time = float(r["recommend_start_time"]) if r.get("recommend_start_time") is not None else 0.0
        exact_offset = _compute_shot_offset_exact(movie_id, start_time) if args.compute_offsets else -1
        # Fallback estimate
        est_offset = int(round(start_time / max(args.estimate_offset_sec, 1e-6)))
        r_out = dict(r)
        r_out["shot_offset"] = exact_offset if exact_offset >= 0 else None
        r_out["estimated_shot_offset"] = est_offset
        enriched.append(r_out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "splits": args.splits,
            "min_ads": args.min_ads,
            "window_sec": args.window_sec,
            "top_n": args.top_n,
            "movies": enriched,
        }, f, indent=2)

    print(f"Saved suggestions to: {out_path}")
    print("Top suggestions:")
    for r in enriched[:10]:
        extra = (
            f" | shot_offset={r['shot_offset']}"
            if r.get('shot_offset') is not None
            else f" | est_shot_offset≈{r['estimated_shot_offset']}"
        )
        st = r.get('recommend_start_time', 0.0)
        print(
            f"  movie={r['movie']} | total_ads={r['total_ads']} | best_window_count={r['best_window_count']} "
            f"| start_time≈{st:.1f}s{extra}"
        )


if __name__ == "__main__":
    main()
