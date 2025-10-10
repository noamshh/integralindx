import logging
from pathlib import Path
from typing import List
from corpus.formula_models import IntegralFormula
import json


LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def load_integrals(path: Path, logger: logging.Logger) -> List[IntegralFormula]:
    """Load integral formulas from input JSONL file"""
    integrals = []
    if not path.exists():
        logger.error(f"File not found: {path}")
        return integrals
    logger.info(f"Loading integrals from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                integral = IntegralFormula.from_dict(data)
                integrals.append(integral)
            except Exception as e:
                logger.warning(f"Error reading integral line {line_num}: {e}")
                continue
    logger.info(f"Loaded {len(integrals):}")
    return integrals


def load_integrals_from_output_dir(output_dir: Path, logger: logging.Logger) -> List[IntegralFormula]:
    all_output = output_dir / "integrals_all.jsonl"
    return load_integrals(all_output, logger)


def write_jsonl(path: str, items: List) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.to_json() + "\n")

def append_jsonl(path: str, item) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(item.to_json() + "\n")
