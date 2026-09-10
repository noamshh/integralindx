#!/usr/bin/env python
"""Profile webapp startup times to identify bottlenecks.

This script mimics the actual startup flow in src/web/main.py to measure
the time taken by each component loading step.

Usage:
    python scripts/profile_startup.py
    python scripts/profile_startup.py --iterations 5
    python scripts/profile_startup.py --embedder ii-cl-19m_prod
"""

import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List

from src.utils.paths import get_paths
from src.database.integral_db import IntegralDatabase
from src.search import embedding_cache
from src.models.egen.contrastive_embedder import CLEmbedder
from src.search.similarity_engine import IntegrandGroupSearch

logging.basicConfig(level=logging.WARNING)  # suppress info logs for clean output


class Timer:
    """context manager for timing code blocks"""
    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def profile_startup(embedder_name: str) -> Dict[str, float]:
    """profile webapp startup sequence

    Args:
        embedder_name: name of embedder to load

    Returns:
        dict mapping component names to elapsed times (seconds)
    """
    timings = {}

    # step 1: load database
    with Timer("database_load") as t:
        paths = get_paths()
        db_path = Path(paths['data']['integral_db'])
        database = IntegralDatabase(db_path, fallback_to_jsonl=False)
        _ = database.count_groups(exclude_curated=False)
    timings["database_load"] = t.elapsed

    # step 2: check cache exists and get metadata
    with Timer("cache_info") as t:
        cache_exists = embedding_cache.cache_exists(embedder_name)
        if not cache_exists:
            raise FileNotFoundError(f"embedding cache not found: {embedder_name}")
        cache_info = embedding_cache.get_cache_info(embedder_name)
    timings["cache_info"] = t.elapsed

    # step 3: load model checkpoint
    with Timer("model_load") as t:
        checkpoint_path = Path(cache_info.get('checkpoint_path'))
        config_path_str = cache_info.get('config_path')
        config_path = Path(config_path_str) if config_path_str else None
        embedder = CLEmbedder.load(checkpoint_path, config_path=config_path)
    timings["model_load"] = t.elapsed

    # step 4: load search engine from cache (FAISS mmap + idx→hash dict)
    with Timer("search_engine_load") as t:
        search_engine = IntegrandGroupSearch.load_cache(embedder_name, embedder, database)
    timings["search_engine_load"] = t.elapsed

    return timings


def format_results(timings_list: List[Dict[str, float]]) -> None:
    """print formatted timing results

    Args:
        timings_list: list of timing dicts from multiple iterations
    """
    # average across iterations
    avg_timings = {}
    for key in timings_list[0].keys():
        avg_timings[key] = sum(t[key] for t in timings_list) / len(timings_list)

    total_time = sum(avg_timings.values())

    print("\n" + "="*60)
    print("WEBAPP STARTUP PROFILING RESULTS")
    print("="*60)
    print(f"Iterations: {len(timings_list)}")
    print(f"Total startup time: {total_time:.3f}s\n")

    # sort by time (longest first)
    sorted_timings = sorted(avg_timings.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Component':<30} {'Time (s)':<12} {'% of Total':<12}")
    print("-"*60)

    for name, elapsed in sorted_timings:
        percentage = (elapsed / total_time) * 100
        print(f"{name:<30} {elapsed:>8.3f}s    {percentage:>6.1f}%")

    print("="*60)
    print(f"{'TOTAL':<30} {total_time:>8.3f}s    {100.0:>6.1f}%")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Profile webapp startup performance")
    parser.add_argument(
        '--embedder',
        default='ii-cl-19m_prod',
        help='embedder name to profile (default: ii-cl-19m_prod)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='number of iterations to average (default: 3)'
    )
    args = parser.parse_args()

    print(f"\nProfiling webapp startup with embedder: {args.embedder}")
    print(f"Running {args.iterations} iteration(s)...\n")

    timings_list = []
    for i in range(args.iterations):
        print(f"Iteration {i+1}/{args.iterations}...", end=" ", flush=True)
        timings = profile_startup(args.embedder)
        timings_list.append(timings)
        total = sum(timings.values())
        print(f"done ({total:.3f}s)")

    format_results(timings_list)

    # provide recommendations
    avg_total = sum(sum(t.values()) for t in timings_list) / len(timings_list)
    print("RECOMMENDATIONS:")
    if avg_total < 3.0:
        print("✅ Startup time is excellent (<3s). Auto-stop mode is viable.")
    elif avg_total < 5.0:
        print("⚠️  Startup time is acceptable (3-5s). Consider optimizations or keep-alive.")
    else:
        print("❌ Startup time is slow (>5s). Recommend optimizations or always-on mode.")
    print()


if __name__ == '__main__':
    main()
