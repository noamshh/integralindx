import logging
import random
from datetime import datetime
from pathlib import Path

import sympy as sp

from src.database.integral_db import IntegralDatabase
from src.utils.paths import get_paths
from src.utils.prefix_notation import sympy_to_prefix
from corpus.pipeline_utils import load_seeds, LOGGING_FORMAT

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def count_tokens(sympy_expr: str) -> int:
    """count tokens in sympy expression using prefix notation
    Args:
        sympy_expr: sympy expression as string
    Returns:
        number of tokens in prefix notation"""
    try:
        expr = sp.sympify(sympy_expr)
        prefix_str = sympy_to_prefix(expr)
        tokens = prefix_str.split()
        return len(tokens)
    except Exception as e:
        logger.warning(f"failed to count tokens for {sympy_expr}: {e}")
        return 999  # return large number to exclude from filtering


def filter_seeds_by_token_limit(seeds: list[str], token_limit: int) -> list[str]:
    """filter seeds to only those with token count <= token_limit
    Args:
        seeds: list of sympy expression strings
        token_limit: maximum token count
    Returns:
        filtered list of seeds"""
    filtered = []
    excluded = []
    for seed in seeds:
        token_count = count_tokens(seed)
        if token_count <= token_limit:
            filtered.append(seed)
        else:
            excluded.append((seed, token_count))

    logger.info(f"filtered {len(filtered)}/{len(seeds)} seeds (token_limit={token_limit})")
    if excluded:
        logger.info(f"excluded {len(excluded)} seeds exceeding token limit")
        # log a few examples
        for seed, token_count in excluded[:5]:
            logger.debug(f"  excluded: {seed} ({token_count} tokens)")

    return filtered


def get_database_seeds_by_token_limit(db: IntegralDatabase, token_limit: int) -> list[str]:
    """get all integrand expressions from database with token count <= token_limit
    Args:
        db: database instance
        token_limit: maximum token count
    Returns:
        list of integrand expressions"""
    # get all groups (no limit)
    all_groups = []
    offset = 0
    batch_size = 5000
    while True:
        batch = db.get_all_groups(limit=batch_size, offset=offset, exclude_curated=True)
        if not batch:
            break
        all_groups.extend(batch)
        offset += batch_size
        logger.info(f"loaded {len(all_groups)} groups from database...")

    logger.info(f"database has {len(all_groups)} integrand groups")

    # filter by token limit
    integrands = []
    for group in all_groups:
        if group.integrand_canonical:
            integrands.append(group.integrand_canonical)

    filtered_integrands = filter_seeds_by_token_limit(integrands, token_limit)
    logger.info(f"filtered {len(filtered_integrands)} integrands from database (token_limit={token_limit})")

    return filtered_integrands


def write_frozen_seeds(synthetic: list[str], database: list[str], output_path: Path, token_limit: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# IntegralIndx Training Seeds\n")
        f.write(f"# Created: {datetime.now().isoformat()}\n")
        f.write(f"# Total seeds: {len(synthetic) + len(database)}\n")
        f.write(f"# Synthetic seeds: {len(synthetic)}\n")
        f.write(f"# Database seeds: {len(database)}\n")
        f.write(f"# Token limit: {token_limit}\n")
        f.write("#\n")
        f.write("# Selection criteria:\n")
        f.write("#   - All synthetic seeds with <= token_limit tokens\n")
        f.write("#   - ALL database integrands with <= token_limit tokens\n")
        f.write("#\n")
        f.write("# Format: One sympy expression per line\n")
        f.write("# Lines starting with '#' are comments and will be ignored\n")
        f.write("#\n")
        f.write("# DO NOT MODIFY THIS FILE\n")
        f.write("# This file is frozen to ensure reproducible training data generation\n")
        f.write("#\n")
        f.write("# ========== SYNTHETIC SEEDS ==========\n")
        for seed in synthetic:
            f.write(f"{seed}\n")
        f.write("#\n")
        f.write("# ========== DATABASE SEEDS ==========\n")
        for seed in database:
            f.write(f"{seed}\n")

    logger.info(f"wrote {len(synthetic) + len(database)} frozen seeds to {output_path}")


def main():
    paths = get_paths()
    project_root = Path(paths['project_root'])
    SYNTHETIC_SEEDS_PATH = project_root / "data/ml_seeds/synthetic_seeds.txt"
    INTEGRAL_DB_PATH = project_root / paths['data']['integral_db']
    FROZEN_SEEDS_PATH = project_root / "data/ml_seeds/frozen_seeds.txt"
    TOKEN_LIMIT = 8

    logger.info("=" * 60)
    logger.info("Freezing ML Training Seeds")
    logger.info("=" * 60)
    logger.info(f"token limit: {TOKEN_LIMIT}")

    # load and filter synthetic seeds
    logger.info("\nProcessing synthetic seeds...")
    all_synthetic_seeds = load_seeds(SYNTHETIC_SEEDS_PATH, logger)
    synthetic_seeds = filter_seeds_by_token_limit(all_synthetic_seeds, TOKEN_LIMIT)

    # get and filter database seeds
    logger.info("\nProcessing database seeds...")
    db = IntegralDatabase(INTEGRAL_DB_PATH)
    database_seeds = get_database_seeds_by_token_limit(db, TOKEN_LIMIT)

    # write frozen seeds
    write_frozen_seeds(
        synthetic_seeds,
        database_seeds,
        FROZEN_SEEDS_PATH,
        TOKEN_LIMIT
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total seeds: {len(synthetic_seeds) + len(database_seeds)}")
    logger.info(f"  Synthetic: {len(synthetic_seeds)} (filtered from {len(all_synthetic_seeds)})")
    logger.info(f"  Database: {len(database_seeds)}")
    logger.info(f"Token limit: {TOKEN_LIMIT}")
    logger.info(f"Output: {FROZEN_SEEDS_PATH}")


if __name__ == '__main__':
    main()
