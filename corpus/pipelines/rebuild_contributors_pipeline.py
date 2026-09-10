import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Set

from src.utils.paths import get_paths
from corpus.pipeline_utils import LOGGING_FORMAT

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def load_contributing_ids(ids_file: Path) -> Set[str]:
    """Load contributing source IDs from file"""
    contributing_ids = set()
    if not ids_file.exists():
        logger.error(f"Contributing IDs file not found: {ids_file}")
        return contributing_ids

    logger.info(f"Loading contributing source IDs from: {ids_file}")
    with open(ids_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                contributing_ids.add(line)

    logger.info(f"Loaded {len(contributing_ids):,} contributing source IDs")
    return contributing_ids


def filter_normalized_to_contributors(normalized_file: Path, contributing_ids: Set[str],
                                     output_file: Path) -> int:
    """Filter normalized formulas to only include contributors"""
    if not normalized_file.exists():
        logger.error(f"Normalized formulas file not found: {normalized_file}")
        return 0

    logger.info(f"Filtering normalized formulas from: {normalized_file}")
    logger.info(f"Output: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_count = 0

    with open(normalized_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                source_id = data.get('source_id')
                if source_id in contributing_ids:
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    filtered_count += 1

            except Exception as e:
                if line_num % 10000 == 0:
                    logger.warning(f"Error reading normalized formula line {line_num}: {e}")
                continue

            if line_num % 50000 == 0:
                logger.info(f"Processed {line_num:,} normalized formulas, kept {filtered_count:,}...")

    logger.info(f"Filtered to {filtered_count:,} contributing formulas")
    return filtered_count


def run_pipeline_step(description: str, command: list, cwd: str = None) -> bool:
    """Run a pipeline step with error handling"""
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} FAILED")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False


def main():
    import argparse

    paths = get_paths()

    parser = argparse.ArgumentParser(
        description="Rebuild database from contributing sources with ML-optimized variable normalization"
    )
    parser.add_argument('--limit', type=int,
                       help='Process only first N contributing formulas (for testing)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging in all pipeline steps')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last successful step (skips already completed steps)')
    parser.add_argument('--skip-filter', action='store_true',
                       help='Skip filter step (use existing filtered data)')
    parser.add_argument('--skip-normalize', action='store_true',
                       help='Skip normalization step (use existing normalized data)')
    parser.add_argument('--skip-canonicalize', action='store_true',
                       help='Skip canonicalization step (use existing canonical data)')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # setup paths
    contributing_ids_file = Path("data/contributing_source_ids.txt")
    normalized_all_file = Path(paths['files']['normalized_accepted'])

    # output directories for each stage
    contributors_dir = Path("data/contributors_rebuild")
    filtered_dir = contributors_dir / "filtered"
    var_normalized_dir = contributors_dir / "var_normalized"
    canonicalized_dir = contributors_dir / "canonicalized"
    grouped_dir = contributors_dir / "grouped"

    # create output directories
    for dir_path in [contributors_dir, filtered_dir, var_normalized_dir, canonicalized_dir, grouped_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("REBUILD CONTRIBUTORS PIPELINE - ML-OPTIMIZED")
    logger.info("="*80)
    logger.info(f"Contributing IDs file: {contributing_ids_file}")
    logger.info(f"Output directory: {contributors_dir}")
    if args.limit:
        logger.info(f"TESTING MODE: Processing first {args.limit} formulas only")
    logger.info("="*80)

    # Step 0: Load contributing IDs
    contributing_ids = load_contributing_ids(contributing_ids_file)
    if not contributing_ids:
        logger.error("No contributing IDs loaded. Exiting.")
        return 1

    # Step 1: Filter normalized formulas to contributors
    contributors_normalized = contributors_dir / "normalized_contributors.jsonl"
    if not args.resume or not contributors_normalized.exists():
        logger.info("\n=== STEP 1: Filter to Contributing Sources ===")
        filtered_count = filter_normalized_to_contributors(
            normalized_all_file,
            contributing_ids,
            contributors_normalized
        )
        if filtered_count == 0:
            logger.error("No contributors filtered. Exiting.")
            return 1
    else:
        logger.info(f"\n=== STEP 1: SKIPPED (using existing {contributors_normalized}) ===")

    # Step 2: Run filter pipeline (LaTeX rejection + symbol validation, auto_canonicalize=False)
    if not args.skip_filter:
        filter_cmd = [
            sys.executable, '-m', 'corpus.pipelines.filter_pipeline',
            '--input', str(contributors_normalized),
            '--output-dir', str(filtered_dir),
            '--progress'
        ]
        if args.limit:
            filter_cmd.extend(['--limit', str(args.limit)])
        if args.debug:
            filter_cmd.append('--debug')

        if not run_pipeline_step("Filter Pipeline (LaTeX rejection + symbol validation)", filter_cmd):
            logger.error("Filter pipeline failed. Exiting.")
            return 1
    else:
        logger.info("\n=== STEP 2: SKIPPED (--skip-filter) ===")

    # Step 3: Run variable normalization (x, a, b, c)
    if not args.skip_normalize:
        normalize_cmd = [
            sys.executable, '-m', 'corpus.pipelines.variable_normalization_pipeline',
            '--input', str(filtered_dir / 'integrals_all.jsonl'),
            '--output-dir', str(var_normalized_dir)
        ]
        if args.debug:
            normalize_cmd.append('--debug')

        if not run_pipeline_step("Variable Normalization (x, a, b, c)", normalize_cmd):
            logger.error("Variable normalization failed. Exiting.")
            return 1
    else:
        logger.info("\n=== STEP 3: SKIPPED (--skip-normalize) ===")

    # Step 4: Run canonicalization
    if not args.skip_canonicalize:
        canonicalize_cmd = [
            sys.executable, '-m', 'corpus.pipelines.canonicalize_pipeline',
            '--input', str(var_normalized_dir / 'integrals_all.jsonl'),
            '--output-dir', str(canonicalized_dir)
        ]
        if args.debug:
            canonicalize_cmd.append('--debug')

        if not run_pipeline_step("Canonicalization", canonicalize_cmd):
            logger.error("Canonicalization failed. Exiting.")
            return 1
    else:
        logger.info("\n=== STEP 4: SKIPPED (--skip-canonicalize) ===")

    # Step 5: Run integrand grouping
    grouping_cmd = [
        sys.executable, '-m', 'corpus.pipelines.integrand_group_pipeline',
        '--input', str(canonicalized_dir / 'integrals_all.jsonl'),
        '--output-dir', str(grouped_dir),
        '--write-groups'
    ]

    if not run_pipeline_step("Integrand Grouping & Deduplication", grouping_cmd):
        logger.error("Integrand grouping failed. Exiting.")
        return 1

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {contributors_dir}")
    logger.info(f"  - Filtered integrals: {filtered_dir / 'integrals_all.jsonl'}")
    logger.info(f"  - Variable normalized: {var_normalized_dir / 'integrals_all.jsonl'}")
    logger.info(f"  - Canonicalized: {canonicalized_dir / 'integrals_all.jsonl'}")
    logger.info(f"  - Integrand groups: {grouped_dir / 'integrand_groups.jsonl'}")
    logger.info(f"  - All grouped integrals: {grouped_dir / 'integrals_all.jsonl'}")
    logger.info(f"\nNext step: Migrate to database")
    logger.info(f"  {sys.executable} -m corpus.scripts.migrate_to_database \\")
    logger.info(f"    --input-dir {grouped_dir} \\")
    logger.info(f"    --rebuild --preserve-curation")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
