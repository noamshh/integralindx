import json
import logging
from pathlib import Path
import argparse
from typing import Dict, Optional
from dataclasses import replace
from sympy import  sympify

from corpus.formula_models import IntegralFormula
from corpus.pipeline_utils import load_integrals, LOGGING_FORMAT
from src.utils.paths import get_paths
from src.utils.provenance import now_iso
from src.utils.timeout import with_timeout, TimeoutError
from src.utils.variable_normalization import extract_variables_from_expression, create_variable_mapping, apply_variable_substitution

NORMALIZATION_TIMEOUT_SECONDS = 20

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def normalize_expression_variables(expr_str: str, var_mapping: Dict[str, str], debug: bool = False) -> Optional[str]:
    """Normalize variable names in a sympy expression string. Returns None if normalization fails or times out."""
    if not var_mapping: return expr_str
    try:
        logger.debug(f"sympify vars: {expr_str[:100]}")
        expr = with_timeout(sympify, NORMALIZATION_TIMEOUT_SECONDS)(expr_str)
        # apply variable substitution )
        logger.debug(f"subs vars: {var_mapping}")
        def do_subs():
            return apply_variable_substitution(expr, var_mapping)
        normalized_expr = with_timeout(do_subs, NORMALIZATION_TIMEOUT_SECONDS)()
        logger.debug(f"subs completed")
        return str(normalized_expr)
    except TimeoutError:
        logger.warning(f"TIMEOUT normalizing expression vars '{expr_str[:100]}'")
        return None
    except Exception as e:
        if debug:
            logger.warning(f"failed to normalize expression '{expr_str}': {e}")
        return None


def normalize_integral_variables(integral: IntegralFormula, debug: bool = False) -> Optional[IntegralFormula]:
    # reject expressions with Sum or nested Integral (causes hang)
    if 'Sum(' in integral.sympy_integrand or 'Integral(' in integral.sympy_integrand:
        if debug:
            logger.debug(f"Rejecting integral {integral.id}: contains Sum/Integral (causes hang)")
        return None
    # original integrand
    normalized_integrand = integral.sympy_integrand
    # extract variables from integrand
    integrand_vars = extract_variables_from_expression(normalized_integrand)
    # create and apply mappings
    var_mapping = create_variable_mapping(integral.sympy_variable, integrand_vars)
    if var_mapping:
        normalized_integrand = normalize_expression_variables(normalized_integrand, var_mapping, debug)
        if normalized_integrand is None:
            return None
    normalized_lower = integral.sympy_lower_bound
    normalized_upper = integral.sympy_upper_bound
    if integral.sympy_lower_bound and var_mapping:
        normalized_lower = normalize_expression_variables(integral.sympy_lower_bound, var_mapping, debug)
        if normalized_lower is None:
            return None
    if integral.sympy_upper_bound and var_mapping:
        normalized_upper = normalize_expression_variables(integral.sympy_upper_bound, var_mapping, debug)
        if normalized_upper is None:
            return None
    # set canonical fields to None (will be populated later)
    integrand_canonical = None
    integrand_hash = None
    normalized_integral = replace(
        integral,
        sympy_integrand=normalized_integrand,
        sympy_variable='x',
        sympy_lower_bound=normalized_lower,
        sympy_upper_bound=normalized_upper,
        integrand_canonical=integrand_canonical,
        integrand_hash=integrand_hash
    )
    return normalized_integral



def main():
    paths = get_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Input jsonl file with integral formulas')
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
    stats_output = output_dir / "normalization_stats.json"

    logger.info("=== VARIABLE NORMALIZATION PIPELINE ===")
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
        logger.info(f"found {len(processed_ids):,} already processed integrals, will skip them")

    integrals= load_integrals(input_path, logger)
    if not integrals:
        logger.error("no integrals loaded. Exiting.")
        return 1

    if args.limit:
        integrals = integrals[:args.limit]
        logger.info(f"limited to {len(integrals):,} integrals for testing")

    logger.info(f"total integrals to process: {len(integrals):,}")

    output_mode = 'a' if resuming else 'w'
    all_file = open(all_output, output_mode, encoding='utf-8')
    definite_file = open(definite_output, output_mode, encoding='utf-8')
    indefinite_file = open(indefinite_output, output_mode, encoding='utf-8')

    normalization_stats = {
        'total_processed': 0,
        'variables_normalized': 0,
        'normalization_failures_rejected': 0,
        'skipped_already_processed': 0,
    }

    for i, integral in enumerate(integrals):
        if integral.id in processed_ids:
            normalization_stats['skipped_already_processed'] += 1
            continue
        try:
            if args.debug:
                logger.debug(f"processing integral {i}: {integral.id}")
            original_vars = extract_variables_from_expression(integral.sympy_integrand)
            normalized_integral = normalize_integral_variables(integral, args.debug)
            if normalized_integral is None:
                normalization_stats['normalization_failures_rejected'] += 1
                if args.debug:
                    logger.debug(f"rejected integral {integral.id}: normalization failed")
                continue
            normalized_vars = extract_variables_from_expression(normalized_integral.sympy_integrand)
            all_file.write(normalized_integral.to_json() + '\n')
            all_file.flush()
            if normalized_integral.integral_type == 'definite':
                definite_file.write(normalized_integral.to_json() + '\n')
                definite_file.flush()
            else:
                indefinite_file.write(normalized_integral.to_json() + '\n')
                indefinite_file.flush()
            normalization_stats['total_processed'] += 1
            if original_vars != normalized_vars:
                normalization_stats['variables_normalized'] += 1
                if args.debug and normalization_stats['variables_normalized'] <= 10:
                    logger.debug(f"normalized variables: {original_vars} → {normalized_vars}")

        except Exception as e:
            normalization_stats['normalization_failures_rejected'] += 1
            if args.debug:
                logger.warning(f"unexpected error processing integral {integral.id}: {e}")
        if (i + 1) % 100 == 0:
            progress_pct = 100 * (i + 1) / len(integrals)
            logger.info(f"progress: {i + 1:,}/{len(integrals):,} ({progress_pct:.1f}%) | "
                       f"processed: {normalization_stats['total_processed']:,} | "
                       f"skipped: {normalization_stats['skipped_already_processed']:,} | "
                       f"rejected: {normalization_stats['normalization_failures_rejected']}")
    all_file.close()
    definite_file.close()
    indefinite_file.close()
    logger.info("files written and closed")
    final_stats = {
        'input_file': str(input_path),
        'input_integrals': len(integrals),
        'normalization_stats': normalization_stats,
        'generated_at': now_iso(),
        'output_files': {
            'all_integrals': str(all_output),
            'definite_integrals': str(definite_output),
            'indefinite_integrals': str(indefinite_output)
        }
    }
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

    logger.info("\n=== VARIABLE NORMALIZATION SUMMARY ===")
    logger.info(f"input integrals: {len(integrals):,}")
    logger.info(f"load parse errors: {normalization_stats['load_parse_errors']}")
    logger.info(f"skipped (already processed): {normalization_stats['skipped_already_processed']:,}")
    logger.info(f"successfully processed: {normalization_stats['total_processed']:,}")
    logger.info(f"rejected (normalization failures): {normalization_stats['normalization_failures_rejected']:,}")
    logger.info(f"variables normalized to (x, a, b, c): {normalization_stats['variables_normalized']:,}")
    logger.info(f"output files:")
    logger.info(f"  - all: {all_output}")
    logger.info(f"  - definite: {definite_output}")
    logger.info(f"  - indefinite: {indefinite_output}")
    logger.info(f"  - stats: {stats_output}")
    return 0


if __name__ == "__main__":
    exit(main())