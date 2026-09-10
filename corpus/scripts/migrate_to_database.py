"""
Usage:
    python -m corpus.scripts.migrate_to_database --rebuild --preserve-curation
    python -m corpus.scripts.migrate_to_database --rebuild  # without curation preservation
    python -m corpus.scripts.migrate_to_database --stats    # show database statistics
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.database.integral_db import IntegralDatabase
from src.utils.paths import get_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_rebuild(db: IntegralDatabase, preserve_curation: bool = True, input_dir: Path = None):
    """
    rebuild database from JSONL files
    Args:
        db: IntegralDatabase instance
        preserve_curation: if True, preserve curation log across rebuild
        input_dir: optional directory containing grouped JSONL files (defaults to contributors_rebuild/grouped/)
    """
    logger.info("starting database rebuild...")
    # export curation log if requested
    curation_log = []
    if preserve_curation:
        logger.info("exporting curation log...")
        curation_log = db.export_curation_log()
        logger.info(f"exported {len(curation_log)} curation log entries")
    # get source paths
    if input_dir:
        groups_path = input_dir / 'integrand_groups.jsonl'
        integrals_path = input_dir / 'integrals_all.jsonl'
        logger.info(f"using custom input directory: {input_dir}")
    else:
        paths = get_paths()
        groups_path = Path(paths['files']['integrand_groups'])
        integrals_path = Path(paths['files']['integrals_all_grouped'])
        logger.info(f"using default paths from paths.yaml")
    if not groups_path.exists():
        logger.error(f"groups file not found: {groups_path}")
        logger.error("run grouping pipeline first: corpus.pipelines.integrand_group_pipeline")
        return
    if not integrals_path.exists():
        logger.error(f"integrals file not found: {integrals_path}")
        logger.error("run grouping pipeline first: corpus.pipelines.integrand_group_pipeline")
        return
    # rebuild from JSONL
    start_time = datetime.now()
    db.rebuild_from_jsonl(groups_path, integrals_path)
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"database rebuild complete in {elapsed:.2f}s")
    # restore curation log
    if preserve_curation and curation_log:
        logger.info("restoring curation log...")
        db.import_curation_log(curation_log)
        logger.info(f"restored {len(curation_log)} curation log entries")
    # show statistics
    show_statistics(db)


def show_statistics(db: IntegralDatabase):
    logger.info("\n" + "=" * 60)
    logger.info("DATABASE STATISTICS")
    logger.info("=" * 60)
    total_groups = db.count_groups(exclude_curated=False)
    total_instances = db.count_instances(exclude_curated=False)
    curated_groups = db.count_groups(exclude_curated=True)
    curated_instances = db.count_instances(exclude_curated=True)
    removed_count = len(db.get_removed_instance_ids())
    logger.info(f"Total integrand groups:     {total_groups:,}")
    logger.info(f"Total integral instances:   {total_instances:,}")
    logger.info(f"Removed instances:          {removed_count:,}")
    logger.info(f"Active groups (curated):    {curated_groups:,}")
    logger.info(f"Active instances (curated): {curated_instances:,}")
    logger.info("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Migrate JSONL data to SQLite database")
    parser.add_argument('--rebuild', action='store_true', help='rebuild database from JSONL files')
    parser.add_argument('--preserve-curation', action='store_true', default=True,
                        help='preserve curation log across rebuilds (default: True)')
    parser.add_argument('--no-preserve-curation', dest='preserve_curation', action='store_false',
                        help='do not preserve curation log')
    parser.add_argument('--stats', action='store_true', help='show database statistics')
    parser.add_argument('--input-dir', type=str,
                        help='input directory containing grouped JSONL files (defaults to data/grouped/)')
    args = parser.parse_args()
    paths = get_paths()
    db_path = Path(paths['data']['integral_db'])
    logger.info(f"database path: {db_path}")
    if args.rebuild:
        # for rebuild, create database if it doesn't exist
        db = IntegralDatabase(db_path, fallback_to_jsonl=True)
    else:
        # for stats, require existing database
        db = IntegralDatabase(db_path, fallback_to_jsonl=False)
    if args.rebuild:
        input_dir = Path(args.input_dir) if args.input_dir else None
        migrate_rebuild(db, preserve_curation=args.preserve_curation, input_dir=input_dir)
    elif args.stats:
        if not db_path.exists():
            logger.error(f"database not found: {db_path}")
            logger.error("run --rebuild first")
            return
        show_statistics(db)
    else:
        logger.error("no action specified")
        logger.error("use --rebuild or --stats")
        parser.print_help()


if __name__ == '__main__':
    main()
