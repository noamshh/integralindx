#!/usr/bin/env python3
"""
scrape answer blobs from MSE API for one-time backfill

reads filtered integral dataset, extracts unique answer IDs,
fetches answer data from MSE API, and saves answer blobs + updates raw formulas
"""
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict
import requests
import yaml

from src.utils.paths import get_paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_api_key() -> str:
    """load MSE API key from config"""
    paths = get_paths()
    api_keys_path = paths['config']['api_keys']

    with open(api_keys_path, 'r') as f:
        api_keys = yaml.safe_load(f)
        return api_keys['mse']['api_key']

def collect_answer_ids(filtered_path: Path) -> Dict[int, int]:
    """
    collect unique answer IDs from filtered dataset

    returns:
        dict mapping answer_id to question_id
    """
    answer_map = {}  # answer_id -> question_id

    logger.info(f"Collecting answer IDs from {filtered_path}")

    with open(filtered_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                mse_answer_id = data.get('mse_answer_id')
                mse_question_id = data.get('mse_question_id')

                if mse_answer_id and mse_question_id:
                    answer_map[mse_answer_id] = mse_question_id
            except Exception as e:
                logger.warning(f"Error parsing line: {e}")
                continue

    logger.info(f"Collected {len(answer_map):,} unique answer IDs")
    return answer_map

def fetch_answer_from_api(question_id: int, answer_id: int, api_key: str, session: requests.Session) -> dict:
    """
    fetch single answer from MSE API

    returns:
        answer JSON data or None
    """
    # use answers endpoint filtered by ID
    url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
    params = {
        'site': 'math.stackexchange.com',
        'filter': 'withbody',
        'key': api_key
    }

    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if 'error_message' in data:
            logger.error(f"API error for answer {answer_id}: {data['error_message']}")
            return None

        items = data.get('items', [])
        if items:
            return items[0]  # return first (should be only) answer

        return None
    except Exception as e:
        logger.error(f"Failed to fetch answer {answer_id}: {e}")
        return None

def save_answer_blob(answer_data: dict, answer_blobs_dir: Path) -> str:
    """
    save answer blob to disk

    returns:
        path to saved blob file
    """
    answer_id = answer_data.get('answer_id')
    if not answer_id:
        return None

    blob_filename = f"answer_{answer_id}.json"
    blob_path = answer_blobs_dir / blob_filename

    with open(blob_path, 'w', encoding='utf-8') as f:
        json.dump({'items': [answer_data]}, f, indent=2, ensure_ascii=False)

    return str(blob_path)

def update_all_raw_formulas(answer_blob_map: Dict[int, str], raw_formulas_path: Path) -> int:
    """
    update raw formulas JSONL to add answer_blob paths in ONE pass

    args:
        answer_blob_map: dict mapping answer_id to blob_path
        raw_formulas_path: path to mse.jsonl

    returns:
        count of formulas updated
    """
    logger.info("Updating raw formulas with answer blob paths...")

    formulas = []
    updated_count = 0

    with open(raw_formulas_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            if line_num % 50000 == 0:
                logger.info(f"Processing line {line_num:,}...")

            try:
                data = json.loads(line)
                provenance = data.get('provenance', {})
                item_id = provenance.get('item_id')
                origin = provenance.get('origin', '')

                # check if this formula is from an answer we fetched
                if 'answer' in origin and item_id in answer_blob_map:
                    # add answer_blob to blob_paths
                    if 'blob_paths' not in data:
                        data['blob_paths'] = {}
                    data['blob_paths']['answer_blob'] = answer_blob_map[item_id]
                    updated_count += 1

                formulas.append(data)
            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {e}")
                formulas.append(json.loads(line))
                continue

    # write back
    logger.info(f"Writing updated formulas to {raw_formulas_path}...")
    with open(raw_formulas_path, 'w', encoding='utf-8') as f:
        for formula in formulas:
            f.write(json.dumps(formula, ensure_ascii=False) + '\n')

    logger.info(f"Updated {updated_count:,} formulas with answer blob paths")
    return updated_count

def main():
    parser = argparse.ArgumentParser(description="Scrape answer blobs for backfill")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without API calls')
    parser.add_argument('--limit', type=int, help='Limit number of answers to fetch (for testing)')
    args = parser.parse_args()

    paths = get_paths()

    # setup paths
    filtered_path = paths['data']['filtered']['root'] / 'integrals_all.jsonl'
    answer_blobs_dir = paths['data']['raw']['root'] / 'answer_blobs'
    raw_formulas_path = paths['data']['raw']['formulas'] / 'mse.jsonl'

    answer_blobs_dir.mkdir(parents=True, exist_ok=True)

    # load API key
    api_key = load_api_key()
    logger.info("Loaded MSE API key")

    # collect answer IDs
    answer_map = collect_answer_ids(filtered_path)

    if args.limit:
        # limit for testing
        answer_map = dict(list(answer_map.items())[:args.limit])
        logger.info(f"Limited to {len(answer_map)} answers for testing")

    # scrape answers (fetch only, don't update raw formulas yet)
    session = requests.Session()
    stats = {
        'total': len(answer_map),
        'fetched': 0,
        'failed': 0,
        'formulas_updated': 0
    }

    # map of answer_id -> blob_path for batch update
    answer_blob_paths = {}

    for i, (answer_id, question_id) in enumerate(answer_map.items(), 1):
        if args.dry_run:
            logger.info(f"[DRY RUN] Would fetch answer {answer_id} (question {question_id})")
            continue

        # rate limiting
        if i > 1:
            time.sleep(0.1)  # 10 requests/sec max

        # progress
        if i % 100 == 0:
            logger.info(f"Progress: {i:,}/{stats['total']:,} ({i/stats['total']*100:.1f}%) | "
                       f"Fetched: {stats['fetched']:,} | Failed: {stats['failed']:,}")

        # fetch answer
        answer_data = fetch_answer_from_api(question_id, answer_id, api_key, session)

        if answer_data:
            # save blob
            blob_path = save_answer_blob(answer_data, answer_blobs_dir)
            if blob_path:
                answer_blob_paths[answer_id] = blob_path
                stats['fetched'] += 1
            else:
                stats['failed'] += 1
        else:
            stats['failed'] += 1

    # update raw formulas in ONE pass
    if not args.dry_run and answer_blob_paths:
        stats['formulas_updated'] = update_all_raw_formulas(answer_blob_paths, raw_formulas_path)

    logger.info(f"\n=== ANSWER BLOB SCRAPING SUMMARY ===")
    logger.info(f"Total answers: {stats['total']:,}")
    logger.info(f"Fetched: {stats['fetched']:,}")
    logger.info(f"Failed: {stats['failed']:,}")
    logger.info(f"Formulas updated: {stats['formulas_updated']:,}")

if __name__ == '__main__':
    main()
