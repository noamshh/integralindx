import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
from dataclasses import replace
from collections import defaultdict

from corpus.formula_models import IntegralFormula, IntegrandGroup
from corpus.pipeline_utils import load_integrals, write_jsonl, LOGGING_FORMAT
from src.utils.provenance import now_iso
from src.utils.latex_patterns import INEQUALITY_PATTERNS
from src.utils.integrand_canonicalization import get_integrand_family
from src.utils.symbol_validation import validate_parsed_symbols

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def contains_inequality(raw_latex: str) -> bool:
    if not raw_latex:
        return False
    for pattern in INEQUALITY_PATTERNS:
        if pattern.search(raw_latex):
            return True
    return False

def clean_inequality_equivalent_forms(integrals: List[IntegralFormula]) -> List[IntegralFormula]:
    cleaned_integrals = []
    inequality_count = 0
    cleared_count = 0
    for integral in integrals:
        if contains_inequality(integral.raw_latex):
            inequality_count += 1
            if integral.equivalent_forms:
                cleaned_integral = replace(integral, equivalent_forms=[])
                cleaned_integrals.append(cleaned_integral)
                cleared_count += 1
                logger.debug(f"cleared {len(integral.equivalent_forms)} equivalent forms from inequality: {integral.id}")
            else:
                cleaned_integrals.append(integral)
        else:
            cleaned_integrals.append(integral)
    logger.info(f"found {inequality_count:,} integrals with inequalities in raw_latex")
    logger.info(f"cleared equivalent forms from {cleared_count:,} integrals")
    return cleaned_integrals


def filter_by_vocab_compatibility(integrals: List[IntegralFormula]) -> List[IntegralFormula]:
    filtered = []
    rejected_by_reason = defaultdict(int)
    rejected_examples = defaultdict(list)
    for integral in integrals:
        integrand_to_check = integral.integrand_canonical if integral.integrand_canonical else integral.sympy_integrand
        is_valid, reason, _ = validate_parsed_symbols(integrand_to_check, integral.sympy_variable, debug=False)
        if is_valid:
            filtered.append(integral)
        else:
            rejected_by_reason[reason] += 1
            if len(rejected_examples[reason]) < 3:
                rejected_examples[reason].append({
                    'id': integral.id,
                    'integrand_original': integral.sympy_integrand,
                    'integrand_canonical': integrand_to_check,
                    'normalized_latex': integral.normalized_latex[:100]
                })
    rejected_count = len(integrals) - len(filtered)
    logger.info(f"vocab compatibility filter: rejected {rejected_count:,} integrals")
    if rejected_count > 0:
        logger.info(f"rejection breakdown:")
        for reason, count in sorted(rejected_by_reason.items(), key=lambda x: -x[1]):
            logger.info(f"- {reason}: {count:,} integrals")
            if rejected_examples[reason]:
                logger.debug(f"examples:")
                for ex in rejected_examples[reason]:
                    logger.debug(f"  {ex['id']}: {ex['integrand_canonical']}")
    logger.info(f"kept {len(filtered):,} vocab-compatible integrals")
    return filtered


def populate_integrand_families(integrals: List[IntegralFormula]) -> List[IntegralFormula]:
    populated_count = 0
    updated_integrals = []
    for integral in integrals:
        if integral.integrand_canonical and not integral.integrand_family:
            family = get_integrand_family(integral.integrand_canonical)
            updated_integral = replace(integral, integrand_family=family)
            updated_integrals.append(updated_integral)
            populated_count += 1
        else:
            updated_integrals.append(integral)
    logger.info(f"populated integrand_family for {populated_count:,} integrals")
    return updated_integrals


def create_integrand_groups(integrals: List[IntegralFormula]) -> Tuple[List[IntegrandGroup], List[IntegralFormula]]:
    """Group integrals by integrand_hash and create IntegrandGroup objects.
    Returns: (integrand_groups, integrals_with_no_hash)"""
    # group by integrand_hash
    hash_to_integrals = defaultdict(list)
    integrals_with_no_hash = []
    for integral in integrals:
        if integral.integrand_hash:
            hash_to_integrals[integral.integrand_hash].append(integral)
        else:
            # keep integrals without hash (canonicalization timed out)
            integrals_with_no_hash.append(integral)
    logger.info(f"found {len(hash_to_integrals)} unique integrand hashes")
    logger.info(f"found {len(integrals_with_no_hash)} integrals without hash (canonicalization timeout)")
    # create IntegrandGroup objects
    integrand_groups = []
    for integrand_hash, group_integrals in hash_to_integrals.items():
        # get canonical form from first integral
        integrand_canonical = group_integrals[0].integrand_canonical or "unknown"
        integrand_family = group_integrals[0].integrand_family
        indefinite_instances = [i.id for i in group_integrals if i.integral_type == "indefinite"]
        definite_instances = [i.id for i in group_integrals if i.integral_type == "definite"]
        unique_mse_questions = {i.mse_question_id for i in group_integrals if i.mse_question_id}
        latex_variants = list(set(i.normalized_latex for i in group_integrals))
        group = IntegrandGroup(
            id=IntegrandGroup.make_id(integrand_hash),
            integrand_canonical=integrand_canonical,
            integrand_hash=integrand_hash,
            indefinite_instances=indefinite_instances,
            definite_instances=definite_instances,
            unique_mse_questions=unique_mse_questions,
            latex_variants=latex_variants,
            integrand_family=integrand_family
        )
        integrand_groups.append(group)
    logger.info(f"created {len(integrand_groups)} integrand groups")
    return integrand_groups, integrals_with_no_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Input jsonl file')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without writing files')
    parser.add_argument('--debug', action='store_true', help='Enable debug')
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Integrand Group Pipeline ===")
    logger.info(f"input: {input_path}")
    logger.info(f"output directory: {output_dir}")
    logger.info(f"dry run: {args.dry_run}")
    integrals = load_integrals(input_path, logger)
    if not integrals:
        logger.error("no integrals loaded, exiting")
        return 1
    original_count = len(integrals)
    logger.info(f"starting with {original_count:,} integrals")
    logger.info("\n--- filtering by vocab compatibility ---")
    integrals = filter_by_vocab_compatibility(integrals)
    logger.info("\n--- cleaning inequality equivalent forms ---")
    integrals = clean_inequality_equivalent_forms(integrals)
    logger.info("\n--- populating integrand families ---")
    integrals = populate_integrand_families(integrals)
    logger.info("\n--- creating integrand groups ---")
    integrand_groups, _ = create_integrand_groups(integrals)
    definite_integrals = [i for i in integrals if i.integral_type == "definite"]
    indefinite_integrals = [i for i in integrals if i.integral_type == "indefinite"]
    all_output = output_dir / 'integrals_all.jsonl'
    definite_output = output_dir / 'integrals_definite.jsonl'
    indefinite_output = output_dir / 'integrals_indefinite.jsonl'
    groups_output = output_dir / 'integrand_groups.jsonl'
    stats_output = output_dir / 'integrand_group_stats.json'

    if not args.dry_run:
        logger.info(f"\n--- writing cleaned results ---")
        write_jsonl(str(all_output), integrals)
        write_jsonl(str(definite_output), definite_integrals)
        write_jsonl(str(indefinite_output), indefinite_integrals)
        if integrand_groups:
            write_jsonl(str(groups_output), integrand_groups)
            logger.info(f"integrand groups: {groups_output}")
        final_stats = {
            'input_file': str(input_path),
            'original_count': original_count,
            'final_count': len(integrals),
            'definite_integrals': len(definite_integrals),
            'indefinite_integrals': len(indefinite_integrals),
            'removed_count': original_count - len(integrals),
            'integrand_groups_count': len(integrand_groups),
            'processing_applied': {
                'filter_vocab': True,
                'clean_inequalities': True,
                'populate_families': True,
                'create_groups': True
            },
            'generated_at': now_iso(),
            'output_files': {
                'all_integrals': str(all_output),
                'definite_integrals': str(definite_output),
                'indefinite_integrals': str(indefinite_output),
                'integrand_groups': str(groups_output) if integrand_groups else None
            }
        }

        with open(stats_output, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"all integrals: {all_output}")
        logger.info(f"definite integrals: {definite_output}")
        logger.info(f"indefinite integrals: {indefinite_output}")
        logger.info(f"statistics: {stats_output}")
    else:
        logger.info("\n--- dry run completed (no files written) ---")

    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"original integrals: {original_count:,}")
    logger.info(f"final integrals: {len(integrals):,}")
    logger.info(f"  - definite: {len(definite_integrals):,}")
    logger.info(f"  - indefinite: {len(indefinite_integrals):,}")
    logger.info(f"removed: {original_count - len(integrals):,}")
    return 0


if __name__ == "__main__":
    exit(main())