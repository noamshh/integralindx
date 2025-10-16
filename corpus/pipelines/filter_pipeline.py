import json
import argparse
import logging
from pathlib import Path

from corpus.pipeline_utils import LOGGING_FORMAT, load_integrals_from_output_dir, write_jsonl
from corpus.filters.sympy_integral_filter import SymPyIntegralFilter
from src.utils.provenance import now_iso

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def load_processed_normalized_ids(output_dir: Path) -> tuple[set, int]:
    all_output = output_dir / "integrals_all.jsonl"
    processed_ids = set()
    integral_count = 0
    if not all_output.exists():
        return processed_ids, 0
    logger.info("loading existing processed formulas...")
    with open(all_output, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            integral_count += 1
            try:
                data = json.loads(line)
                normalized_source_id = data.get('normalized_source_id')
                if normalized_source_id:
                    processed_ids.add(normalized_source_id)
            except Exception as e:
                if line_num % 1000 == 0:
                    logger.warning(f"error reading processed ID from line {line_num}: {e}")
                continue
    logger.info(f"loaded {len(processed_ids):,} processed formula IDs from {integral_count:,} existing integrals")
    return processed_ids, integral_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Input normalized .jsonl')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory')
    parser.add_argument('--limit', type=int, default=None, help='Process only N formulas')
    parser.add_argument('--force-reprocess', action='store_true', help='Reprocess all formulas')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--progress', action='store_true', help='Show progress')
    parser.add_argument('--start', type=int, default=0, help='Start processing from line N')
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        logger.error(f"input file not found: {input_path}")
        return 1
    logger.info(f"starting integral filtering from: {input_path}")
    # load already processed normalized IDs for deduplication and count existing integrals
    processed_ids = set()
    existing_integral_count = 0
    if not args.force_reprocess:
        processed_ids, existing_integral_count = load_processed_normalized_ids(output_dir)
        if processed_ids:
            logger.info(f"found {len(processed_ids):,} already processed formulas generating {existing_integral_count:,} integrals")
            logger.info("(use --force-reprocess to ignore)")
        else:
            logger.info("no previously processed formulas found")
    else:
        logger.info("force reprocessing enabled - ignoring already processed formulas")
    
    if args.limit:
        logger.info(f"processing limit: {args.limit} formulas")

    if args.start > 0:
        logger.info(f"starting from line: {args.start}")

    all_output = output_dir / "integrals_all.jsonl"
    logger.info(f"writing results to: {all_output}")
    filter = SymPyIntegralFilter(debug=args.debug, progress=args.progress)
    stats = filter.filter_formulas(
        input_file=str(input_path),
        output_file=str(all_output),
        processed_ids=processed_ids,
        force_reprocess=args.force_reprocess,
        limit=args.limit,
        existing_integral_count=existing_integral_count,
        start_line=args.start
    )

    integrals = load_integrals_from_output_dir(output_dir)
    definite_integrals = [i for i in integrals if i.integral_type == "definite"]
    indefinite_integrals = [i for i in integrals if i.integral_type == "indefinite"]
    definite_output = output_dir / "integrals_definite.jsonl"
    indefinite_output = output_dir / "integrals_indefinite.jsonl"
    stats_output = output_dir / "filter_stats.json"
    write_jsonl(str(definite_output), definite_integrals)
    write_jsonl(str(indefinite_output), indefinite_integrals)
    final_stats = {
        'input_file': str(input_path),
        'total_formulas': stats['total_formulas'],
        'skipped_already_processed': stats['skipped_already_processed'],
        'skipped_no_integral': stats['skipped_no_integral'],
        'integrals_found': stats['integrals_found'],
        'processing_errors': stats['processing_errors'],
        'definite_integrals': len(definite_integrals),
        'indefinite_integrals': len(indefinite_integrals),
        'acceptance_rate': stats['integrals_found'] / max(stats['total_formulas'], 1) if stats['total_formulas'] > 0 else 0,
        'limit_applied': args.limit,
        'force_reprocess': args.force_reprocess,
        'generated_at': now_iso(),
        'output_files': {
            'all_integrals': str(all_output),
            'definite_integrals': str(definite_output),
            'indefinite_integrals': str(indefinite_output)
        }
    }
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

    logger.info('='*80)
    logger.info(f"total formulas processed: {final_stats['total_formulas']:,}")
    logger.info(f"already processed (skipped): {final_stats['skipped_already_processed']:,}")
    logger.info(f"no integral pattern (skipped): {final_stats['skipped_no_integral']:,}")
    logger.info(f"integrals found: {final_stats['integrals_found']:,}")
    logger.info(f"acceptance rate: {final_stats['acceptance_rate']:.1%}")
    logger.info(f"  - definite integrals: {final_stats['definite_integrals']:,}")
    logger.info(f"  - indefinite integrals: {final_stats['indefinite_integrals']:,}")
    logger.info(f"processing errors: {final_stats['processing_errors']:,}")
    logger.info(f"\noutput files:")
    logger.info(f"  all integrals:        {all_output}")
    logger.info(f"  definite integrals:   {definite_output}")
    logger.info(f"  indefinite integrals: {indefinite_output}")
    logger.info(f"  statistics:           {stats_output}")
    return 0


if __name__ == "__main__":
    exit(main())