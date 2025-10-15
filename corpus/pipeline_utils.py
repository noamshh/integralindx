import logging
import json
from pathlib import Path
from typing import List
from omegaconf import DictConfig
from dataclasses import dataclass

from src.utils.paths import get_paths
from corpus.formula_models import IntegralFormula


LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def load_integrals(path: Path, logger: logging.Logger) -> List[IntegralFormula]:
    """Load integral formulas from input jsonl file"""
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



@dataclass
class EGenPipelineConfig:
    """configuration for equivalence generation pipeline"""
    frozen_seeds_path: Path
    egraph_binary_path: Path
    egraph_n_equivalents: int
    egraph_token_limit: int
    egraph_max_token_limit: int
    egraph_time_limit: int
    output_equivalents_dir: Path
    output_metadata_file: str
    output_failed_seeds_file: str
    batch_size: int
    n_workers: int | None
    log_level: str
    log_file: Path | None
    progress_interval: int


def load_egen_pipeline_config(cfg: DictConfig) -> EGenPipelineConfig:
    paths = get_paths()
    project_root = Path(paths['project_root'])
    ds = cfg.dataset.egen_generation
    return EGenPipelineConfig(
        frozen_seeds_path=project_root / ds.data.frozen_seeds,
        egraph_binary_path=project_root / ds.egraph.binary_path,
        egraph_n_equivalents=ds.egraph.n_equivalents,
        egraph_token_limit=ds.egraph.token_limit,
        egraph_max_token_limit=ds.egraph.max_token_limit,
        egraph_time_limit=ds.egraph.time_limit,
        output_equivalents_dir=project_root / ds.output.equivalents_dir,
        output_metadata_file=ds.output.metadata_file,
        output_failed_seeds_file=ds.output.failed_seeds_file,
        batch_size=ds.processing.batch_size,
        n_workers=ds.processing.get('n_workers', None),
        log_level=ds.logging.level,
        log_file=project_root / ds.logging.file if ds.logging.file else None,
        progress_interval=ds.logging.progress_interval
    )


@dataclass
class TSVPipelineConfig:
    """configuration for TSV construction pipeline"""
    equivalents_dir: Path
    n_exprs_per_line: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    large_cluster_threshold: int
    max_positives_per_query: int
    output_base_dir: Path
    output_version: str
    log_level: str
    log_file: Path | None
    progress_interval: int


def load_tsv_pipeline_config(cfg: DictConfig) -> TSVPipelineConfig:
    paths = get_paths()
    project_root = Path(paths['project_root'])
    ds = cfg.dataset.tsv_construction

    return TSVPipelineConfig(
        equivalents_dir=project_root / ds.data.equivalents_dir,
        n_exprs_per_line=ds.data.n_exprs_per_line,
        train_ratio=ds.splits.train,
        val_ratio=ds.splits.val,
        test_ratio=ds.splits.test,
        random_seed=ds.splits.random_seed,
        large_cluster_threshold=ds.sampling.large_cluster_threshold,
        max_positives_per_query=ds.sampling.max_positives_per_query,
        output_base_dir=project_root / ds.output.base_dir,
        output_version=ds.output.version,
        log_level=ds.logging.level,
        log_file=project_root / ds.logging.file if ds.logging.file else None,
        progress_interval=ds.logging.progress_interval
    )


def load_seeds(path: Path, logger: logging.Logger) -> List[str]:
    """load seeds from file, skipping comments and blank lines"""
    if not path.exists():
        raise FileNotFoundError(f"seeds file not found: {path}")
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            seeds.append(line)
    logger.info(f"loaded {len(seeds)} seeds from {path}")
    return seeds
