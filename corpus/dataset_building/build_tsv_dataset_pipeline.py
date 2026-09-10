"""
Stage 2: TSV Dataset Construction Pipeline

Constructs train/val/test TSV files from pre-generated equivalence files.
Implements seed-level splitting (not line-level) to prevent data leakage
as described in the E-Gen paper. https://arxiv.org/html/2501.14951v2

Key Design Principles:
1. Seed-level splitting: Each seed's equivalents stay within one split
2. No data leakage: Test equivalents never seen during training
3. Intra-split negatives: Negative samples drawn from same split
4. Ordered pairs: Generate all (query, positive) pairs within cluster

Data Flow:
    equivalents/*.txt (seed equiv files)
        ↓ seed-level split (80/10/10)
    train_seeds  / val_seeds  / test_seeds
        ↓ ordered pair generation + negative sampling
    train.tsv / val.tsv / test.tsv

Usage:
    # full construction
    python -m corpus.dataset_building.build_tsv_dataset_pipeline

    # test mode (uses equivalents_test/)
    python -m corpus.dataset_building.build_tsv_dataset_pipeline \\
        dataset/tsv_construction=test

    # custom output version
    python -m corpus.dataset_building.build_tsv_dataset_pipeline \\
        dataset.tsv_construction.output.version=v2.0

Output:
    data/ml_training/v1.0/
        ├── train.tsv
        ├── val.tsv
        ├── test.tsv
        └── metadata.json
"""

import json
import logging
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import sympy as sp
from tqdm import tqdm

from corpus.pipeline_utils import LOGGING_FORMAT, TSVPipelineConfig, load_tsv_pipeline_config
from src.utils.prefix_notation import sympy_to_prefix

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def load_equivalence_file(filepath: Path) -> Tuple[str, List[str]]:
    with open(filepath) as f:
        lines = f.readlines()
    # first line is comment with seed
    seed_line = lines[0].strip()
    if seed_line.startswith('# SEED: '):
        seed_original = seed_line[8:]  # remove "# SEED: " prefix
    else:
        seed_original = filepath.stem

    # convert seed to prefix notation
    try:
        seed_prefix = sympy_to_prefix(sp.sympify(seed_original))
    except Exception as e:
        logger.warning(f"failed to convert seed to prefix: {seed_original}, error: {e}")
        seed_prefix = None

    # remaining lines are equivalents (already in prefix notation)
    equivalents = [line.strip() for line in lines[1:] if line.strip()]

    # include seed as first equivalent (if conversion succeeded)
    if seed_prefix:
        equivalents = [seed_prefix] + equivalents

    return seed_original, equivalents


def load_all_equivalences(equivalents_dir: Path) -> Dict[str, List[str]]:
    if not equivalents_dir.exists():
        raise FileNotFoundError(f"equivalents directory not found: {equivalents_dir}")
    equivalence_files = list(equivalents_dir.glob("*.txt"))
    if not equivalence_files:
        raise ValueError(f"no equivalence files found in {equivalents_dir}")
    logger.info(f"loading {len(equivalence_files)} equivalence files from {equivalents_dir}")
    all_equivalences = {}
    for filepath in tqdm(equivalence_files, desc="Loading equivalence files", unit="file"):
        seed, equivalents = load_equivalence_file(filepath)
        # use hash (filename stem) as key for deduplication
        all_equivalences[filepath.stem] = equivalents
    logger.info(f"loaded {len(all_equivalences)} unique seed equivalence sets")
    return all_equivalences


def split_seeds(
    seed_hashes: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    split seed hashes into train/val/test sets

    This is SEED-LEVEL splitting, not line-level splitting.
    Each seed's equivalents stay within one split to prevent data leakage.

    Args:
        seed_hashes: list of seed hashes
        train_ratio: fraction for training
        val_ratio: fraction for validation
        test_ratio: fraction for testing
        random_seed: random seed for reproducibility

    Returns:
        (train_hashes, val_hashes, test_hashes) tuple
    """
    # validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    # shuffle seeds
    random.seed(random_seed)
    seed_hashes = seed_hashes.copy()
    random.shuffle(seed_hashes)

    # calculate split indices
    n = len(seed_hashes)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_hashes = seed_hashes[:train_end]
    val_hashes = seed_hashes[train_end:val_end]
    test_hashes = seed_hashes[val_end:]

    logger.info(f"split seeds: train={len(train_hashes)}, val={len(val_hashes)}, test={len(test_hashes)}")
    return train_hashes, val_hashes, test_hashes


def generate_tsv_lines_for_split(
    split_equivalences: Dict[str, List[str]],
    n_exprs_per_line: int,
    random_seed: int,
    split_name: str,
    output_file,
    large_cluster_threshold: int = 100,
    max_positives_per_query: int = 50
) -> int:
    """
    generate TSV lines for a single split (train/val/test) with stream writing

    Implements:
    1. Ordered pair generation:
       - Small clusters (N < threshold): all N×(N-1) pairs
       - Large clusters (N ≥ threshold): sample max_positives_per_query per query
    2. Intra-split negatives: negative samples from different clusters within same split
    3. Stream writing: writes directly to file to avoid memory accumulation

    Args:
        split_equivalences: dict mapping seed hash to equivalents (within this split)
        n_exprs_per_line: number of expressions per line (query + positive + negatives)
        random_seed: random seed for negative sampling
        split_name: name of split (for logging)
        output_file: file handle to write TSV lines to
        large_cluster_threshold: use sampling for clusters >= this size
        max_positives_per_query: max positives to sample per query in large clusters

    Returns:
        total number of TSV lines written
    """
    if n_exprs_per_line < 3:
        raise ValueError(f"n_exprs_per_line must be at least 3, got {n_exprs_per_line}")

    line_count = 0
    random.seed(random_seed)

    all_seed_hashes = list(split_equivalences.keys())
    k_negatives = n_exprs_per_line - 2  # e.g., 3 negatives for 5-lets

    logger.info(f"generating {split_name} TSV lines with {k_negatives} negatives per pair...")
    logger.info(f"smart sampling: threshold={large_cluster_threshold}, max_positives={max_positives_per_query}")

    # progress bar over seeds
    pbar = tqdm(
        split_equivalences.items(),
        desc=f"Generating {split_name} pairs",
        unit="seed",
        total=len(split_equivalences)
    )

    for seed_hash, equivs in pbar:
        # all expressions in this cluster
        all_exprs = equivs

        if len(all_exprs) < 2:
            logger.warning(f"seed {seed_hash} has only {len(all_exprs)} equivalents, skipping")
            continue

        # smart sampling: all pairs for small clusters, sampled for large clusters
        is_large_cluster = len(all_exprs) >= large_cluster_threshold

        for query in all_exprs:
            # determine positives for this query
            if is_large_cluster:
                # sample K positives (avoid combinatorial explosion)
                possible_positives = [e for e in all_exprs if e != query]
                if len(possible_positives) > max_positives_per_query:
                    positives = random.sample(possible_positives, max_positives_per_query)
                else:
                    positives = possible_positives
            else:
                # generate all pairs (small cluster)
                positives = [e for e in all_exprs if e != query]

            # for each (query, positive) pair, sample negatives and write line
            for positive in positives:
                # sample K negatives from different clusters (within this split)
                negatives = []
                for _ in range(k_negatives):
                    # choose random cluster (not current one)
                    if len(all_seed_hashes) < 2:
                        break

                    neg_seed_hash = random.choice([h for h in all_seed_hashes if h != seed_hash])
                    neg_pool = split_equivalences[neg_seed_hash]
                    if not neg_pool:
                        continue
                    negative = random.choice(neg_pool)
                    negatives.append(negative)

                # write line if we have enough negatives
                if len(negatives) == k_negatives:
                    tsv_line = '\t'.join([query, positive] + negatives)
                    output_file.write(tsv_line + '\n')
                    line_count += 1

        # update progress bar with current stats
        pbar.set_postfix({'lines': line_count, 'avg': f'{line_count/(pbar.n+1):.0f}'})

    pbar.close()

    logger.info(f"generated {line_count:,} {split_name} TSV lines from {len(split_equivalences)} seeds")
    if split_equivalences:
        logger.info(f"average lines per seed: {line_count / len(split_equivalences):.1f}")

    return line_count


def log_tsv_stats(line_count: int, path: Path):
    """log statistics for completed TSV file"""
    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"completed {path.name}: {line_count:,} lines ({file_size_mb:.1f} MB)")


def save_metadata(config: TSVPipelineConfig, stats: Dict, output_dir: Path):
    """save metadata about dataset construction"""
    metadata = {
        'version': config.output_version,
        'stage': 'tsv_construction',
        'created': datetime.now().isoformat(),
        'config': {
            'equivalents_dir': str(config.equivalents_dir),
            'n_exprs_per_line': config.n_exprs_per_line,
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio,
            'random_seed': config.random_seed,
            'large_cluster_threshold': config.large_cluster_threshold,
            'max_positives_per_query': config.max_positives_per_query,
        },
        'statistics': stats
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"saved metadata to {metadata_path}")


@hydra.main(version_base=None, config_path="../../config", config_name="tsv_construction_config")
def main(cfg: DictConfig):
    """main pipeline execution"""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    config = load_tsv_pipeline_config(cfg)

    # setup logging
    logging.getLogger().setLevel(config.log_level)
    if config.log_file:
        # ensure log directory exists
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(config.log_file)
        fh.setLevel(config.log_level)
        fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
        logging.getLogger().addHandler(fh)

    # ========== STEP 1: Load equivalence files ==========
    logger.info("=" * 60)
    logger.info("STEP 1: Loading equivalence files")
    logger.info("=" * 60)

    all_equivalences = load_all_equivalences(config.equivalents_dir)

    # ========== STEP 2: Seed-level splitting ==========
    logger.info("=" * 60)
    logger.info("STEP 2: Seed-level splitting (prevents data leakage)")
    logger.info("=" * 60)

    seed_hashes = list(all_equivalences.keys())
    train_hashes, val_hashes, test_hashes = split_seeds(
        seed_hashes,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed
    )

    # partition equivalences by split
    train_equivalences = {h: all_equivalences[h] for h in train_hashes}
    val_equivalences = {h: all_equivalences[h] for h in val_hashes}
    test_equivalences = {h: all_equivalences[h] for h in test_hashes}

    # ========== STEP 3: Generate and write TSV files (stream writing) ==========
    logger.info("=" * 60)
    logger.info("STEP 3: Generating and writing TSV files (stream mode)")
    logger.info("=" * 60)

    output_dir = config.output_base_dir / config.output_version
    output_dir.mkdir(parents=True, exist_ok=True)

    # generate train split
    logger.info("\n--- Training split ---")
    train_path = output_dir / 'train.tsv'
    with open(train_path, 'w') as train_file:
        train_lines = generate_tsv_lines_for_split(
            train_equivalences,
            config.n_exprs_per_line,
            config.random_seed,
            "train",
            train_file,
            config.large_cluster_threshold,
            config.max_positives_per_query
        )
    log_tsv_stats(train_lines, train_path)

    # generate val split
    logger.info("\n--- Validation split ---")
    val_path = output_dir / 'val.tsv'
    with open(val_path, 'w') as val_file:
        val_lines = generate_tsv_lines_for_split(
            val_equivalences,
            config.n_exprs_per_line,
            config.random_seed + 1,  # different seed for val
            "val",
            val_file,
            config.large_cluster_threshold,
            config.max_positives_per_query
        )
    log_tsv_stats(val_lines, val_path)

    # generate test split
    logger.info("\n--- Test split ---")
    test_path = output_dir / 'test.tsv'
    with open(test_path, 'w') as test_file:
        test_lines = generate_tsv_lines_for_split(
            test_equivalences,
            config.n_exprs_per_line,
            config.random_seed + 2,  # different seed for test
            "test",
            test_file,
            config.large_cluster_threshold,
            config.max_positives_per_query
        )
    log_tsv_stats(test_lines, test_path)

    # ========== STEP 5: Save metadata ==========
    logger.info("=" * 60)
    logger.info("STEP 5: Saving metadata")
    logger.info("=" * 60)

    stats = {
        'total_seed_files': len(all_equivalences),
        'train_seeds': len(train_hashes),
        'val_seeds': len(val_hashes),
        'test_seeds': len(test_hashes),
        'train_lines': train_lines,
        'val_lines': val_lines,
        'test_lines': test_lines,
        'total_lines': train_lines + val_lines + test_lines,
    }

    save_metadata(config, stats, output_dir)

    # ========== PIPELINE COMPLETE ==========
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Train: {train_lines:,} lines from {len(train_hashes)} seeds")
    logger.info(f"Val: {val_lines:,} lines from {len(val_hashes)} seeds")
    logger.info(f"Test: {test_lines:,} lines from {len(test_hashes)} seeds")
    logger.info(f"Total: {train_lines + val_lines + test_lines:,} lines")


if __name__ == '__main__':
    main()
