#!/usr/bin/env python3
"""
backfill author information to existing integral datasets

reads integral JSONL files, extracts author info from original blobs,
and writes updated records with author_name and author_link fields
"""
import json
import argparse
import logging
from pathlib import Path
from typing import List
import shutil

from corpus.formula_models import IntegralFormula
from corpus.scripts.blob_parser import load_raw_formula_blob_paths, extract_author_from_formula, extract_author_from_blob
from src.utils.paths import get_paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# cache for loaded blobs (blob_path -> parsed JSON)
_blob_cache = {}

def load_blob_cached(blob_path: str) -> dict:
    """load blob JSON with caching"""
    if blob_path not in _blob_cache:
        try:
            with open(blob_path, 'r', encoding='utf-8') as f:
                _blob_cache[blob_path] = json.load(f)
        except Exception:
            _blob_cache[blob_path] = {}
    return _blob_cache[blob_path]

def extract_author_cached(blob_path: str, item_id: int, origin: str) -> tuple:
    """extract author with blob caching"""
    try:
        data = load_blob_cached(blob_path)
        items = data.get('items', [])
        if not items:
            return None, None

        is_answer = 'answer' in origin

        for item in items:
            if is_answer:
                if item.get('answer_id') == item_id:
                    owner = item.get('owner', {})
                    return owner.get('display_name'), owner.get('link')
            else:
                if item.get('question_id') == item_id:
                    owner = item.get('owner', {})
                    return owner.get('display_name'), owner.get('link')

        return None, None
    except Exception:
        return None, None

def backfill_integral_file(input_path: Path, output_path: Path, blob_map: dict, dry_run: bool = False) -> dict:
    """
    backfill author info for a single integral JSONL file

    returns:
        stats dict with counts
    """
    stats = {
        'total': 0,
        'with_author': 0,
        'without_author': 0,
        'errors': 0
    }

    logger.info(f"Processing: {input_path}")

    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return stats

    # create backup
    if not dry_run and output_path.exists():
        backup_path = output_path.with_suffix('.jsonl.bak')
        shutil.copy2(output_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    updated_integrals = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                integral = IntegralFormula.from_dict(data)
                stats['total'] += 1

                # progress logging
                if stats['total'] % 5000 == 0:
                    logger.info(f"Progress: {stats['total']:,} integrals, {stats['with_author']:,} with authors ({stats['with_author']/max(stats['total'],1)*100:.1f}%)")

                # extract author info
                source_id = integral.source_id

                # determine item_id and origin from integral fields
                item_id = None
                origin = ''

                # check if it's an answer or question
                if integral.mse_answer_id:
                    item_id = integral.mse_answer_id
                    origin = 'answer_body'
                elif integral.mse_question_id:
                    item_id = integral.mse_question_id
                    origin = 'question_body'

                author_name, author_link = None, None

                if source_id and item_id and origin and source_id in blob_map:
                    blob_paths = blob_map[source_id]

                    # prioritize answer_blob for answer-origin formulas
                    is_answer = 'answer' in origin
                    blob_path = None

                    if is_answer and blob_paths.get('answer_blob'):
                        blob_path = blob_paths['answer_blob']
                    elif blob_paths.get('raw_blob'):
                        blob_path = blob_paths['raw_blob']

                    if blob_path:
                        author_name, author_link = extract_author_cached(blob_path, item_id, origin)

                # update integral with author info
                if author_name or author_link:
                    stats['with_author'] += 1
                else:
                    stats['without_author'] += 1

                # create updated integral
                updated_data = data.copy()
                updated_data['author_name'] = author_name
                updated_data['author_link'] = author_link

                updated_integrals.append(updated_data)

            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                stats['errors'] += 1
                continue

    # write updated integrals
    if not dry_run:
        with open(output_path, 'w', encoding='utf-8') as f:
            for integral_data in updated_integrals:
                f.write(json.dumps(integral_data, ensure_ascii=False) + '\n')
        logger.info(f"Wrote {len(updated_integrals)} integrals to {output_path}")
    else:
        logger.info(f"[DRY RUN] Would write {len(updated_integrals)} integrals to {output_path}")

    return stats

def main():
    parser = argparse.ArgumentParser(description="Backfill author information to integral datasets")
    parser.add_argument('--input', '-i', type=str, help='Input JSONL file (if not specified, processes standard datasets)')
    parser.add_argument('--output', '-o', type=str, help='Output JSONL file (defaults to overwriting input)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without writing files')
    parser.add_argument('--curated-only', action='store_true', help='Process only curated dataset')
    args = parser.parse_args()

    paths = get_paths()

    # load blob path mapping
    logger.info("Loading blob paths...")
    blob_map = load_raw_formula_blob_paths()
    logger.info(f"Loaded {len(blob_map):,} blob paths")

    if args.input:
        # process single file
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path

        stats = backfill_integral_file(input_path, output_path, blob_map, dry_run=args.dry_run)

        logger.info(f"\n=== BACKFILL SUMMARY ===")
        logger.info(f"Total integrals: {stats['total']:,}")
        logger.info(f"With author: {stats['with_author']:,} ({stats['with_author']/max(stats['total'],1)*100:.1f}%)")
        logger.info(f"Without author: {stats['without_author']:,}")
        logger.info(f"Errors: {stats['errors']:,}")

    else:
        # process standard datasets
        datasets = []

        if args.curated_only:
            # curated only
            datasets = [
                paths['data']['curated']['root'] / 'integrals_all.jsonl',
                paths['data']['curated']['root'] / 'integrals_definite.jsonl',
                paths['data']['curated']['root'] / 'integrals_indefinite.jsonl',
            ]
        else:
            # all datasets
            data_root = paths['data']['root']
            datasets = [
                data_root / 'filtered' / 'integrals_all.jsonl',
                data_root / 'filtered' / 'integrals_definite.jsonl',
                data_root / 'filtered' / 'integrals_indefinite.jsonl',
                data_root / 'cleaned' / 'integrals_all.jsonl',
                data_root / 'cleaned' / 'integrals_definite.jsonl',
                data_root / 'cleaned' / 'integrals_indefinite.jsonl',
                data_root / 'grouped' / 'integrals_all.jsonl',
                data_root / 'grouped' / 'integrals_definite.jsonl',
                data_root / 'grouped' / 'integrals_indefinite.jsonl',
                data_root / 'normalized_vars' / 'integrals_all.jsonl',
                data_root / 'normalized_vars' / 'integrals_definite.jsonl',
                data_root / 'normalized_vars' / 'integrals_indefinite.jsonl',
                data_root / 'normalized_vars_fixed' / 'integrals_all.jsonl',
                data_root / 'normalized_vars_fixed' / 'integrals_definite.jsonl',
                data_root / 'normalized_vars_fixed' / 'integrals_indefinite.jsonl',
                data_root / 'curated' / 'integrals_all.jsonl',
                data_root / 'curated' / 'integrals_definite.jsonl',
                data_root / 'curated' / 'integrals_indefinite.jsonl',
            ]

        total_stats = {
            'total': 0,
            'with_author': 0,
            'without_author': 0,
            'errors': 0,
            'files_processed': 0
        }

        for dataset_path in datasets:
            if not dataset_path.exists():
                logger.info(f"Skipping (not found): {dataset_path}")
                continue

            stats = backfill_integral_file(dataset_path, dataset_path, blob_map, dry_run=args.dry_run)

            total_stats['total'] += stats['total']
            total_stats['with_author'] += stats['with_author']
            total_stats['without_author'] += stats['without_author']
            total_stats['errors'] += stats['errors']
            total_stats['files_processed'] += 1

        logger.info(f"\n=== TOTAL BACKFILL SUMMARY ===")
        logger.info(f"Files processed: {total_stats['files_processed']}")
        logger.info(f"Total integrals: {total_stats['total']:,}")
        logger.info(f"With author: {total_stats['with_author']:,} ({total_stats['with_author']/max(total_stats['total'],1)*100:.1f}%)")
        logger.info(f"Without author: {total_stats['without_author']:,}")
        logger.info(f"Errors: {total_stats['errors']:,}")

if __name__ == '__main__':
    main()
