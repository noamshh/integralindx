"""
Benchmark script for E-Gen performance testing

Measures the time to generate equivalents for a subset of synthetic seeds
to estimate the total time needed for full training data generation.
"""

import sys
import time
from pathlib import Path
from statistics import mean, stdev, median
from src.egraph.egen_wrapper import generate_equivalents, EGenConfig
from src.utils.prefix_notation import sympy_to_prefix
from src.utils.paths import get_paths
import sympy as sp

def main():
    # configure paths
    paths = get_paths()
    project_root = Path(paths['project_root'])
    binary_path = project_root / "bin" / "egen.exe"  # Windows

    if not binary_path.exists():
        # try Unix name
        binary_path = project_root / "bin" / "egen"

    if not binary_path.exists():
        print(f"ERROR: E-Gen binary not found at {binary_path}")
        print("Please build the binary first.")
        sys.exit(1)

    # load synthetic seeds
    seeds_file = Path(paths['data']['ml_seeds']) / "synthetic_seeds.txt"
    if not seeds_file.exists():
        print(f"ERROR: Seeds file not found at {seeds_file}")
        sys.exit(1)

    print(f"Loading seeds from: {seeds_file}")
    seeds = []
    with open(seeds_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                seeds.append(line)

    print(f"Loaded {len(seeds)} seeds")
    print()

    # limit to first N seeds for benchmarking
    n_benchmark = 50
    seeds = seeds[:n_benchmark]
    print(f"Benchmarking first {n_benchmark} seeds")
    print("=" * 70)

    # convert to prefix notation
    print("Converting seeds to prefix notation...")
    prefix_seeds = []
    failed_parsing = []

    for i, seed in enumerate(seeds):
        try:
            expr = sp.sympify(seed)
            prefix = sympy_to_prefix(expr)
            prefix_seeds.append((seed, prefix))
        except Exception as e:
            failed_parsing.append((i, seed, str(e)))

    print(f"Successfully converted {len(prefix_seeds)}/{len(seeds)} seeds")
    if failed_parsing:
        print(f"Failed to parse {len(failed_parsing)} seeds:")
        for idx, seed, err in failed_parsing[:5]:
            print(f"  {idx}. {seed}: {err}")
        if len(failed_parsing) > 5:
            print(f"  ... and {len(failed_parsing) - 5} more")
    print()

    # configuration for generation
    config = EGenConfig(
        binary_path=binary_path,
        n_equiv=100,  # target: 100 equivalents per seed
        token_limit=12,
        time_limit=120  # 2 minute timeout per seed
    )

    print(f"E-Gen Configuration:")
    print(f"  n_equiv: {config.n_equiv}")
    print(f"  token_limit: {config.token_limit}")
    print(f"  time_limit: {config.time_limit}s")
    print("=" * 70)
    print()

    # benchmark each seed
    times = []
    equiv_counts = []
    failed_generation = []

    print("Starting benchmark...")
    print()

    for i, (original, prefix) in enumerate(prefix_seeds, 1):
        print(f"[{i}/{len(prefix_seeds)}] {original[:50]}...")

        start = time.time()
        try:
            equivalents = generate_equivalents(prefix, config)
            elapsed = time.time() - start

            num_equiv = len(equivalents)
            times.append(elapsed)
            equiv_counts.append(num_equiv)

            print(f"  ✓ {num_equiv} equivalents in {elapsed:.2f}s ({num_equiv/elapsed:.1f} equiv/s)")

        except Exception as e:
            elapsed = time.time() - start
            failed_generation.append((i, original, str(e)))
            print(f"  ✗ FAILED after {elapsed:.2f}s: {e}")

        print()

    # compute statistics
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()

    if times:
        print(f"Successful generations: {len(times)}/{len(prefix_seeds)}")
        print()
        print("Time per seed:")
        print(f"  Mean:   {mean(times):.2f}s")
        print(f"  Median: {median(times):.2f}s")
        print(f"  StDev:  {stdev(times):.2f}s" if len(times) > 1 else "  StDev:  N/A")
        print(f"  Min:    {min(times):.2f}s")
        print(f"  Max:    {max(times):.2f}s")
        print()

        print("Equivalents per seed:")
        print(f"  Mean:   {mean(equiv_counts):.1f}")
        print(f"  Median: {median(equiv_counts):.1f}")
        print(f"  Min:    {min(equiv_counts)}")
        print(f"  Max:    {max(equiv_counts)}")
        print()

        # extrapolate to full dataset
        avg_time = mean(times)
        total_seeds = 5000  # target dataset size

        print(f"Extrapolation for {total_seeds} seeds:")
        print(f"  Sequential time:  {avg_time * total_seeds / 3600:.1f} hours")
        print(f"  Parallel (4 cores): {avg_time * total_seeds / 3600 / 4:.1f} hours")
        print(f"  Parallel (8 cores): {avg_time * total_seeds / 3600 / 8:.1f} hours")
        print(f"  Parallel (16 cores): {avg_time * total_seeds / 3600 / 16:.1f} hours")
        print()

    if failed_generation:
        print(f"Failed generations: {len(failed_generation)}")
        for idx, seed, err in failed_generation[:5]:
            print(f"  {idx}. {seed[:40]}... : {err}")
        if len(failed_generation) > 5:
            print(f"  ... and {len(failed_generation) - 5} more")
        print()

    # success rate
    success_rate = len(times) / len(prefix_seeds) * 100 if prefix_seeds else 0
    print(f"Overall success rate: {success_rate:.1f}%")
    print()

    if success_rate >= 90:
        print("✓ Benchmark passed (≥90% success rate)")
        return 0
    else:
        print("✗ Benchmark below target (<90% success rate)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
