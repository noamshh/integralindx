#!/usr/bin/env python3
"""Validate deployment readiness for IntegralIndx.

Checks that all required files exist and estimates deployment size.

Usage:
    python scripts/prepare_deployment.py --embedder ii-cl-19m_v1.0
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


def get_dir_size(path: Path) -> int:
    """Recursively calculate directory size in bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except Exception as e:
        logger.warning(f"error calculating size for {path}: {e}")
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def check_deployment_requirements(embedder_name: str) -> Tuple[bool, List[str], Dict[str, int]]:
    """Check that all required files exist for deployment.

    Args:
        embedder_name: name of embedder (e.g., 'ii-cl-19m_v1.0')

    Returns:
        (all_checks_passed, error_messages, file_sizes)
    """
    errors = []
    sizes = {}

    # check database
    db_path = Path('data/integral.db')
    if not db_path.exists():
        errors.append(f"❌ Database not found: {db_path}")
    else:
        sizes['database'] = db_path.stat().st_size
        logger.info(f"✓ Database found: {format_size(sizes['database'])}")

    # check embedder cache
    cache_dir = Path('data/embeddings') / embedder_name
    if not cache_dir.exists():
        errors.append(f"❌ Embedder cache not found: {cache_dir}")
        errors.append(f"   Run: python scripts/build_embedding_cache.py --embedder {embedder_name}")
    else:
        index_path = cache_dir / 'index.faiss'
        groups_path = cache_dir / 'groups.pkl'
        metadata_path = cache_dir / 'metadata.json'

        if not index_path.exists():
            errors.append(f"❌ FAISS index not found: {index_path}")
        if not groups_path.exists():
            errors.append(f"❌ Groups file not found: {groups_path}")
        if not metadata_path.exists():
            errors.append(f"❌ Metadata not found: {metadata_path}")

        if index_path.exists() and groups_path.exists() and metadata_path.exists():
            sizes['faiss_cache'] = get_dir_size(cache_dir)
            logger.info(f"✓ FAISS cache found: {format_size(sizes['faiss_cache'])}")

    # check model checkpoint (if E-Gen model)
    if embedder_name not in ['tfidf', 'sentence_bert']:
        # try to find checkpoint in cache metadata
        metadata_path = cache_dir / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            checkpoint_path_str = metadata.get('checkpoint_path')
            if checkpoint_path_str:
                checkpoint_path = Path(checkpoint_path_str)
                if not checkpoint_path.exists():
                    errors.append(f"❌ Model checkpoint not found: {checkpoint_path}")
                else:
                    sizes['model_checkpoint'] = checkpoint_path.stat().st_size
                    logger.info(f"✓ Model checkpoint found: {format_size(sizes['model_checkpoint'])}")

    # check source code
    src_dir = Path('src')
    if not src_dir.exists():
        errors.append(f"❌ Source directory not found: {src_dir}")
    else:
        sizes['source_code'] = get_dir_size(src_dir)
        logger.info(f"✓ Source code found: {format_size(sizes['source_code'])}")

    # check latex2sympy2 custom package
    latex_pkg = Path('latex2sympy2_custom')
    if not latex_pkg.exists():
        errors.append(f"❌ Custom latex2sympy2 package not found: {latex_pkg}")
    else:
        sizes['latex2sympy2'] = get_dir_size(latex_pkg)
        logger.info(f"✓ Custom latex2sympy2 package found: {format_size(sizes['latex2sympy2'])}")

    # check config
    config_dir = Path('config')
    if not config_dir.exists():
        errors.append(f"❌ Config directory not found: {config_dir}")
    else:
        sizes['config'] = get_dir_size(config_dir)
        logger.info(f"✓ Config found: {format_size(sizes['config'])}")

    # check Dockerfile
    if not Path('Dockerfile').exists():
        errors.append(f"❌ Dockerfile not found")
    else:
        logger.info(f"✓ Dockerfile found")

    # check fly.toml
    if not Path('fly.toml').exists():
        errors.append(f"❌ fly.toml not found")
    else:
        logger.info(f"✓ fly.toml found")

    # check requirements-prod.txt
    if not Path('requirements-prod.txt').exists():
        errors.append(f"❌ requirements-prod.txt not found")
    else:
        logger.info(f"✓ requirements-prod.txt found")

    all_passed = len(errors) == 0
    return all_passed, errors, sizes


def estimate_docker_image_size(deployment_sizes: Dict[str, int]) -> int:
    """Estimate final Docker image size.

    Base image: ~150 MB (python:3.11-slim)
    ML dependencies: ~1.2 GB (PyTorch, transformers, FAISS)
    Application: deployment_sizes

    Returns:
        estimated size in bytes
    """
    base_image_mb = 150
    ml_deps_mb = 1200

    total_mb = base_image_mb + ml_deps_mb

    for size_bytes in deployment_sizes.values():
        total_mb += size_bytes / (1024 ** 2)

    return int(total_mb * 1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Validate deployment readiness for IntegralIndx"
    )
    parser.add_argument(
        '--embedder',
        type=str,
        default='ii-cl-19m_v1.0',
        help="Embedder name (must match FAISS cache name)"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("INTEGRALINDX DEPLOYMENT READINESS CHECK")
    logger.info("="*80)
    logger.info(f"Embedder: {args.embedder}")
    logger.info("")

    # run checks
    all_passed, errors, sizes = check_deployment_requirements(args.embedder)

    logger.info("")
    logger.info("="*80)

    if all_passed:
        logger.info("✓ ALL CHECKS PASSED")
        logger.info("="*80)

        # calculate total deployment size
        total_size = sum(sizes.values())
        logger.info("")
        logger.info("DEPLOYMENT SIZE SUMMARY:")
        logger.info("-" * 40)
        for name, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {name:20s}: {format_size(size):>12s}")
        logger.info("-" * 40)
        logger.info(f"  {'Total assets':20s}: {format_size(total_size):>12s}")
        logger.info("")

        # estimate Docker image size
        docker_size = estimate_docker_image_size(sizes)
        logger.info(f"Estimated Docker image size: {format_size(docker_size)}")
        logger.info("")
        logger.info("READY FOR DEPLOYMENT!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Test locally:  docker build -t integralindx:local .")
        logger.info("  2. Deploy:        flyctl deploy")
        logger.info("")

    else:
        logger.error("✗ DEPLOYMENT CHECKS FAILED")
        logger.info("="*80)
        logger.info("")
        logger.info("ERRORS:")
        for error in errors:
            logger.error(f"  {error}")
        logger.info("")
        logger.info("Fix these issues before deploying.")
        exit(1)


if __name__ == '__main__':
    main()
