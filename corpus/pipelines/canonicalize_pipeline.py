import json
import argparse
import logging
from pathlib import Path
from typing import Optional
from dataclasses import replace

from corpus.formula_models import IntegralFormula
from corpus.pipeline_utils import LOGGING_FORMAT
from src.utils.integrand_canonicalization import canonicalize_integrand, get_integrand_family
from src.utils.provenance import now_iso

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def canonicalize_normalized_integral(integral: IntegralFormula, debug: bool = False) -> Optional[IntegralFormula]:
    try:
        canonical, hash_val = canonicalize_integrand(integral.sympy_integrand)
        if canonical is None or hash_val is None:
            if debug:
                logger.debug(f"canonicalization failed for {integral.id}")
            return None
        family = get_integrand_family(canonical)
        canonicalized = replace(integral, integrand_canonical=canonical, integrand_hash=hash_val, integrand_family=family)
        return canonicalized
    except Exception as e:
        if debug:
            logger.warning(f"exception canonicalizing {integral.id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Input jsonl file with variable-normalized integrals')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug')
    parser.add_argument('--limit', type=int, help='Limit number of integrals to process')
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_output = output_dir / "integrals_all.jsonl"
    definite_output = output_dir / "integrals_definite.jsonl"
    indefinite_output = output_dir / "integrals_indefinite.jsonl"
    stats_output = output_dir / "canonicalization_stats.json"

    logger.info("=== canonicalization pipeline ===")
    logger.info(f"input: {input_path}")
    logger.info(f"output directory: {output_dir}")
    processed_ids = set()
    resuming = all_output.exists() and all_output.stat().st_size > 0
    if resuming:
        logger.info(f"resuming: loading already processed IDs from {all_output}...")
        with open(all_output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_ids.add(data['id'])
                    except:
                        pass
        logger.info(f"found {len(processed_ids):,} already processed integrals")
    logger.info(f"loading integrals from: {input_path}")
    integrals = []
    load_errors = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                integral = IntegralFormula.from_dict(data)
                integrals.append(integral)
            except Exception as e:
                load_errors += 1
                if args.debug:
                    logger.warning(f"error reading integral line {line_num}: {e}")
                continue

    logger.info(f"loaded {len(integrals):,} integrals ({load_errors} parse errors)")

    if args.limit:
        integrals = integrals[:args.limit]
        logger.info(f"limited to {len(integrals):,} integrals for testing")

    output_mode = 'a' if resuming else 'w'
    all_file = open(all_output, output_mode, encoding='utf-8')
    definite_file = open(definite_output, output_mode, encoding='utf-8')
    indefinite_file = open(indefinite_output, output_mode, encoding='utf-8')

    stats = {
        'total_input': len(integrals),
        'already_processed': 0,
        'successfully_canonicalized': 0,
        'canonicalization_failed': 0,
        'load_errors': load_errors
    }

    for i, integral in enumerate(integrals):
        if integral.id in processed_ids:
            stats['already_processed'] += 1
            continue
        try:
            canonicalized = canonicalize_normalized_integral(integral, args.debug)
            if canonicalized is None:
                stats['canonicalization_failed'] += 1
                if args.debug:
                    logger.debug(f"failed to canonicalize {integral.id}")
                continue
            all_file.write(canonicalized.to_json() + '\n')
            all_file.flush()
            if canonicalized.integral_type == 'definite':
                definite_file.write(canonicalized.to_json() + '\n')
                definite_file.flush()
            else:
                indefinite_file.write(canonicalized.to_json() + '\n')
                indefinite_file.flush()
            stats['successfully_canonicalized'] += 1
        except Exception as e:
            stats['canonicalization_failed'] += 1
            if args.debug:
                logger.warning(f"unexpected error processing {integral.id}: {e}")

        if (i + 1) % 1000 == 0:
            progress_pct = 100 * (i + 1) / len(integrals)
            logger.info(f"progress: {i + 1:,}/{len(integrals):,} ({progress_pct:.1f}%) | "
                       f"canonicalized: {stats['successfully_canonicalized']:,} | "
                       f"failed: {stats['canonicalization_failed']}")

    all_file.close()
    definite_file.close()
    indefinite_file.close()

    final_stats = {
        'input_file': str(input_path),
        'total_input': stats['total_input'],
        'already_processed': stats['already_processed'],
        'successfully_canonicalized': stats['successfully_canonicalized'],
        'canonicalization_failed': stats['canonicalization_failed'],
        'load_errors': stats['load_errors'],
        'generated_at': now_iso(),
        'output_files': {
            'all_integrals': str(all_output),
            'definite_integrals': str(definite_output),
            'indefinite_integrals': str(indefinite_output)
        }
    }

    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

    logger.info("\n=== CANONICALIZATION SUMMARY ===")
    logger.info(f"input integrals: {stats['total_input']:,}")
    logger.info(f"already processed (skipped): {stats['already_processed']:,}")
    logger.info(f"successfully canonicalized: {stats['successfully_canonicalized']:,}")
    logger.info(f"canonicalization failed: {stats['canonicalization_failed']:,}")
    logger.info(f"load errors: {stats['load_errors']}")
    logger.info(f"\noutput files:")
    logger.info(f"  all: {all_output}")
    logger.info(f"  sefinite: {definite_output}")
    logger.info(f"  indefinite: {indefinite_output}")
    logger.info(f"  stats: {stats_output}")
    return 0


if __name__ == "__main__":
    exit(main())
