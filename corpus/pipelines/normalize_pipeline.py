import argparse
import json
import time
import logging
from pathlib import Path
from typing import Optional

from corpus.normalizers.mse_normalizer import MSENormalizer
from corpus.formula_models import RawFormula, NormalizedFormula
from corpus.pipeline_utils import append_jsonl, LOGGING_FORMAT
from src.utils.paths import get_paths

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

def normalize(input_file: Path, output_dir: Path, max_lines: Optional[int] = None, debug: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "normalized_accepted.jsonl"
    rejected_path = output_dir / "normalized_rejected.jsonl"
    stats_path = output_dir / "normalization_stats.json"
    if accepted_path.exists():
        accepted_path.unlink()
    if rejected_path.exists():
        rejected_path.unlink()
    logger.info(f"Processing: {input_file}")
    logger.info(f"Output dir: {output_dir}")
    start_time = time.time()
    total_processed = 0
    total_accepted = 0
    total_rejected = 0
    rejection_reasons = {}
    equivalent_forms_stats = {"has_equivalents": 0, "no_equivalents": 0}
    normalizer = MSENormalizer()

    with input_file.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_lines and total_processed >= max_lines:
                logger.info(f"Reached max_lines limit ({max_lines}), stopping")
                break
            total_processed += 1
            if total_processed % 500 == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                acceptance_rate = total_accepted / total_processed if total_processed > 0 else 0
                logger.info(f"  Progress: {total_processed:,} processed | "
                      f"{total_accepted:,} accepted ({acceptance_rate:.1%}) | "
                      f"{total_rejected:,} rejected | "
                      f"{rate:.1f}/sec")
            line = line.strip()
            if not line:
                continue
            try:
                raw_data = json.loads(line)
                raw_formula = RawFormula.from_dict(raw_data)
            except Exception as e:
                logger.info(f"error parsing line {line_num}: {e}")
                continue
            if not raw_formula.raw_latex:
                total_rejected += 1
                rejected_formula = NormalizedFormula.rejected(raw_formula.id, "", "empty_latex")
                append_jsonl(str(rejected_path), rejected_formula)
                continue
            try:
                normalized_list = normalizer.normalize_latex(raw_formula.raw_latex, raw_formula.id)
                for normalized in normalized_list:
                    if normalized.accepted:
                        normalized.provenance = raw_formula.provenance
                        normalized.compute_checksum("leading_expression")
                        append_jsonl(str(accepted_path), normalized)
                        total_accepted += 1
                        if normalized.equivalent_forms:
                            equivalent_forms_stats["has_equivalents"] += 1
                            if debug:
                                logger.debug(f"  [EQUIVALENTS] Leading: {normalized.leading_expression}")
                                for i, equiv in enumerate(normalized.equivalent_forms):
                                    logger.debug(f"    Equiv {i + 1}: {equiv}")
                        else:
                            equivalent_forms_stats["no_equivalents"] += 1
                    else:
                        normalized.provenance = raw_formula.provenance
                        append_jsonl(str(rejected_path), normalized)
                        total_rejected += 1
                        reason = normalized.rejection_reason or "unknown"
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        if debug and total_rejected <= 5:
                            logger.debug(f"  [REJECTED] {reason}: {raw_formula.raw_latex[:100]}...")

            except Exception as e:
                logger.info(f"error processing line {line_num}: {e}")
                rejected_formula = NormalizedFormula.rejected(raw_formula.id, raw_formula.raw_latex, f"processing_error: {str(e)}")
                append_jsonl(str(rejected_path), rejected_formula)
                total_rejected += 1

    elapsed = time.time() - start_time
    acceptance_rate = total_accepted / total_processed if total_processed > 0 else 0
    stats = {
        "input_file": str(input_file),
        "total_processed": total_processed,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "acceptance_rate": acceptance_rate,
        "processing_time_seconds": elapsed,
        "processing_rate_per_second": total_processed / elapsed if elapsed > 0 else 0,
        "rejection_reasons": dict(sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)),
        "equivalent_forms_stats": equivalent_forms_stats,
        "output_files": {
            "accepted": str(accepted_path),
            "rejected": str(rejected_path)
        }
    }
    with stats_path.open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info("\n=== NORMALIZATION SUMMARY ===")
    logger.info(f"total processed: {total_processed:,}")
    logger.info(f"total accepted:  {total_accepted:,} ({acceptance_rate:.1%})")
    logger.info(f"total rejected:  {total_rejected:,} ({100 - acceptance_rate * 100:.1%})")
    logger.info(f"processing time: {elapsed:.1f}s ({total_processed / elapsed:.1f}/sec)")
    logger.info(f"\nequivalent forms:")
    logger.info(f"  with equivalents: {equivalent_forms_stats['has_equivalents']:,}")
    logger.info(f"  no equivalents:   {equivalent_forms_stats['no_equivalents']:,}")
    logger.info(f"\ntop rejection reasons:")
    for reason, count in list(rejection_reasons.items())[:5]:
        logger.info(f"  {reason}: {count:,}")
    logger.info(f"\noutput files:")
    logger.info(f"  accepted: {accepted_path}")
    logger.info(f"  rejected: {rejected_path}")
    logger.info(f"  stats:    {stats_path}")
    return stats


def main():
    paths = get_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=paths['data']['raw']['formulas'] / "mse.jsonl",
                        help="Input jsonl file with RawFormula data")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--max", type=int, default=100, help="Maximum number of lines to process (default: 100)")
    parser.add_argument("--debug", action="store_true", help="Enable debug")
    args = parser.parse_args()
    normalize(Path(args.input), Path(args.output_dir), max_lines=args.max, debug=args.debug)


if __name__ == "__main__":
    exit(main())
