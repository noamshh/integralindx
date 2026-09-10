import json
import logging
import hashlib
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import sympy as sp

from src.egraph.egen_wrapper import EGenConfig, generate_batch
from src.utils.prefix_notation import sympy_to_prefix
from corpus.pipeline_utils import LOGGING_FORMAT, EGenPipelineConfig, load_egen_pipeline_config, load_seeds

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def compute_seed_hash(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:8]

def convert_to_prefix(expr_str: str) -> str | None:
    try:
        expr = sp.sympify(expr_str)
        prefix = sympy_to_prefix(expr)
        return prefix
    except Exception as e:
        logger.warning(f"failed to convert expression to prefix: {expr_str}\nerror: {e}")
        return None


def is_seed_already_processed(seed_prefix: str, output_dir: Path, min_equivalents: int = 1) -> bool:
    """
    check if a seed has already been processed (valid output file exists)

    Args:
        seed_prefix: seed in prefix notation
        output_dir: directory containing equivalence files
        min_equivalents: minimum number of equivalents required for valid file

    Returns:
        True if seed has valid output file, False otherwise
    """
    seed_hash = compute_seed_hash(seed_prefix)
    filepath = output_dir / f"{seed_hash}.txt"

    # file must exist
    if not filepath.exists():
        return False

    # file must be non-empty and have expected format
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # must have header + at least min_equivalents lines
        if len(lines) < 1 + min_equivalents:
            return False

        # first line must be seed header
        if not lines[0].startswith("# SEED:"):
            return False

        # remaining lines should be non-empty equivalents
        equivalents = [line.strip() for line in lines[1:] if line.strip()]
        if len(equivalents) < min_equivalents:
            return False

        return True
    except Exception as e:
        logger.warning(f"error checking file {filepath}: {e}")
        return False


def filter_unprocessed_seeds(
    seed_mapping: Dict[str, str],
    output_dir: Path,
    min_equivalents: int = 1
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    separate already-processed from unprocessed seeds

    Args:
        seed_mapping: dict mapping original seed to prefix notation
        output_dir: directory containing equivalence files
        min_equivalents: minimum equivalents required for valid file

    Returns:
        tuple of (unprocessed_mapping, skipped_mapping)
    """
    unprocessed = {}
    skipped = {}

    for original, prefix in seed_mapping.items():
        if is_seed_already_processed(prefix, output_dir, min_equivalents):
            skipped[original] = prefix
        else:
            unprocessed[original] = prefix

    return unprocessed, skipped


def process_single_batch(
    batch_data: tuple[int, List[str], List[str], EGenPipelineConfig, Path]
) -> Dict:
    """
    worker function to process a single batch of seeds

    Args:
        batch_data: tuple of (batch_num, batch_original, batch_prefix, config, output_dir)

    Returns:
        dict with 'batch_num', 'success_count', 'failure_count', 'failures', 'saved_files'
    """
    batch_num, batch_original, batch_prefix, config, output_dir = batch_data

    # configure E-Gen
    egraph_config = EGenConfig(
        binary_path=config.egraph_binary_path,
        n_equiv=config.egraph_n_equivalents,
        token_limit=config.egraph_token_limit,
        max_token_limit=config.egraph_max_token_limit,
        time_limit=config.egraph_time_limit
    )

    logger.info(f"[batch {batch_num}] processing {len(batch_prefix)} seeds...")

    success_count = 0
    failure_count = 0
    failures = []
    saved_files = []

    try:
        # generate equivalents for batch
        batch_results = generate_batch(
            exprs=batch_prefix,
            config=egraph_config,
            fail_on_error=False
        )

        # save results
        for original_seed, prefix_seed in zip(batch_original, batch_prefix):
            if prefix_seed in batch_results:
                filepath = save_equivalence_file(
                    seed_original=original_seed,
                    seed_prefix=prefix_seed,
                    equivalents=batch_results[prefix_seed],
                    output_dir=output_dir
                )
                saved_files.append(str(filepath))
                success_count += 1
            else:
                # record E-Gen generation failure
                error_msg = "E-Gen failed to generate equivalents"
                failures.append({
                    'original': original_seed,
                    'prefix': prefix_seed,
                    'type': 'egen_generation_failed',
                    'error': error_msg
                })
                failure_count += 1

        logger.info(f"[batch {batch_num}] complete: {success_count}/{len(batch_prefix)} succeeded")

    except Exception as e:
        logger.error(f"[batch {batch_num}] failed: {e}")
        # record failures for all seeds in failed batch
        for original_seed, prefix_seed in zip(batch_original, batch_prefix):
            error_msg = f"batch processing error: {str(e)}"
            failures.append({
                'original': original_seed,
                'prefix': prefix_seed,
                'type': 'egen_generation_failed',
                'error': error_msg
            })
        failure_count = len(batch_prefix)

    return {
        'batch_num': batch_num,
        'success_count': success_count,
        'failure_count': failure_count,
        'failures': failures,
        'saved_files': saved_files
    }


def generate_and_save_parallel(
    seed_mapping: Dict[str, str],
    config: EGenPipelineConfig,
    output_dir: Path,
    failures_data: Dict,
    failure_tracking_path: Path,
    batch_size: int = 50,
    limit: int | None = None
) -> Dict[str, int]:
    """
    generate equivalent expressions for seed expressions using E-Gen (parallel batch processing)

    Args:
        seed_mapping: dict mapping original seed to prefix notation
        config: pipeline configuration
        output_dir: directory to save equivalence files
        failures_data: failure tracking dict (modified in-place)
        failure_tracking_path: path to save failure tracking file
        batch_size: number of seeds per batch
        limit: limit number of seeds to process (for testing)

    Returns:
        statistics dict with success/failure counts
    """
    import os
    import concurrent.futures
    from tqdm import tqdm

    # limit seeds if requested
    if limit:
        limited_mapping = dict(list(seed_mapping.items())[:limit])
        logger.info(f"limiting to {limit} seeds for testing")
    else:
        limited_mapping = seed_mapping

    # check if binary exists
    if not config.egraph_binary_path.exists():
        raise FileNotFoundError(
            f"E-Gen binary not found: {config.egraph_binary_path}\n"
            f"Build with: cd egen_integralindx && cargo build --release\n"
            f"Install with: cp target/release/egen ../bin/egen"
        )

    # determine number of workers
    n_workers = config.n_workers
    if n_workers is None:
        n_workers = os.cpu_count() or 1
        logger.info(f"auto-detected {n_workers} CPU cores")
    else:
        logger.info(f"using {n_workers} worker processes")

    logger.info(f"generating equivalences for {len(limited_mapping)} seeds (parallel batch processing)...")
    logger.info(f"E-Gen config: n_equiv={config.egraph_n_equivalents}, "
                f"token_limit={config.egraph_token_limit}, "
                f"max_token_limit={config.egraph_max_token_limit}, "
                f"time_limit={config.egraph_time_limit}")
    logger.info(f"batch size: {batch_size}, workers: {n_workers}")

    # convert to lists for batching
    original_seeds = list(limited_mapping.keys())
    prefix_seeds = list(limited_mapping.values())

    # prepare batches
    total_seeds = len(prefix_seeds)
    batches = []
    for batch_start in range(0, total_seeds, batch_size):
        batch_end = min(batch_start + batch_size, total_seeds)
        batch_num = (batch_start // batch_size) + 1

        batch_prefix = prefix_seeds[batch_start:batch_end]
        batch_original = original_seeds[batch_start:batch_end]

        batch_data = (batch_num, batch_original, batch_prefix, config, output_dir)
        batches.append(batch_data)

    total_batches = len(batches)
    logger.info(f"prepared {total_batches} batches")

    # process batches in parallel
    success_count = 0
    failure_count = 0
    saved_files = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # submit all batches
        future_to_batch = {executor.submit(process_single_batch, batch): batch[0] for batch in batches}

        # collect results with progress bar
        with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    result = future.result()

                    # update counters
                    success_count += result['success_count']
                    failure_count += result['failure_count']
                    saved_files.extend(result['saved_files'])

                    # record failures
                    for failure in result['failures']:
                        record_failure(
                            failures_data,
                            failure['original'],
                            failure['prefix'],
                            failure['type'],
                            failure['error']
                        )

                    # save failure tracking after each batch completes
                    save_failure_tracking(failure_tracking_path, failures_data)

                    pbar.update(1)
                    pbar.set_postfix({
                        'success': success_count,
                        'failed': failure_count
                    })

                except Exception as e:
                    logger.error(f"batch {batch_num} raised exception: {e}")
                    pbar.update(1)

    logger.info(f"parallel processing complete: {success_count} succeeded, {failure_count} failed")

    return {
        'total_seeds': total_seeds,
        'success_count': success_count,
        'failure_count': failure_count,
        'saved_files': len(saved_files)
    }


def generate_and_save_batched(
    seed_mapping: Dict[str, str],
    config: EGenPipelineConfig,
    output_dir: Path,
    failures_data: Dict,
    failure_tracking_path: Path,
    batch_size: int = 50,
    progress_interval: int = 10,
    limit: int | None = None
) -> Dict[str, int]:
    """
    generate equivalent expressions for seed expressions using E-Gen (batch processing with incremental saving)

    Args:
        seed_mapping: dict mapping original seed to prefix notation
        config: pipeline configuration
        output_dir: directory to save equivalence files
        failures_data: failure tracking dict (modified in-place)
        failure_tracking_path: path to save failure tracking file
        batch_size: number of seeds to process per E-Gen call
        progress_interval: log progress every N seeds
        limit: limit number of seeds to process (for testing)

    Returns:
        statistics dict with success/failure counts
    """
    # limit seeds if requested
    if limit:
        limited_mapping = dict(list(seed_mapping.items())[:limit])
        logger.info(f"limiting to {limit} seeds for testing")
    else:
        limited_mapping = seed_mapping

    # configure E-Gen
    egraph_config = EGenConfig(
        binary_path=config.egraph_binary_path,
        n_equiv=config.egraph_n_equivalents,
        token_limit=config.egraph_token_limit,
        max_token_limit=config.egraph_max_token_limit,
        time_limit=config.egraph_time_limit
    )

    # check if binary exists
    if not egraph_config.binary_path.exists():
        raise FileNotFoundError(
            f"E-Gen binary not found: {egraph_config.binary_path}\n"
            f"Build with: cd egen_integralindx && cargo build --release\n"
            f"Install with: cp target/release/egen ../bin/egen"
        )

    logger.info(f"generating equivalences for {len(limited_mapping)} seeds (batch processing)...")
    logger.info(f"E-Gen config: n_equiv={egraph_config.n_equiv}, "
                f"token_limit={egraph_config.token_limit}, "
                f"max_token_limit={egraph_config.max_token_limit}, "
                f"time_limit={egraph_config.time_limit}")
    logger.info(f"batch size: {batch_size}")

    # convert to lists for batching
    original_seeds = list(limited_mapping.keys())
    prefix_seeds = list(limited_mapping.values())

    # process in batches
    total_seeds = len(prefix_seeds)
    success_count = 0
    failure_count = 0
    saved_files = []

    for batch_start in range(0, total_seeds, batch_size):
        batch_end = min(batch_start + batch_size, total_seeds)
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_seeds + batch_size - 1) // batch_size

        batch_prefix = prefix_seeds[batch_start:batch_end]
        batch_original = original_seeds[batch_start:batch_end]

        logger.info(f"processing batch {batch_num}/{total_batches} ({len(batch_prefix)} seeds)...")

        try:
            # generate equivalents for batch
            batch_results = generate_batch(
                exprs=batch_prefix,
                config=egraph_config,
                fail_on_error=False
            )

            # save results immediately
            for original_seed, prefix_seed in zip(batch_original, batch_prefix):
                if prefix_seed in batch_results:
                    filepath = save_equivalence_file(
                        seed_original=original_seed,
                        seed_prefix=prefix_seed,
                        equivalents=batch_results[prefix_seed],
                        output_dir=output_dir
                    )
                    saved_files.append(filepath)
                    success_count += 1
                else:
                    # record E-Gen generation failure
                    error_msg = "E-Gen failed to generate equivalents"
                    record_failure(failures_data, original_seed, prefix_seed, "egen_generation_failed", error_msg)
                    failure_count += 1

            # save failure tracking after each batch
            save_failure_tracking(failure_tracking_path, failures_data)

            logger.info(f"batch {batch_num}/{total_batches} complete: {len(batch_results)}/{len(batch_prefix)} succeeded")

        except Exception as e:
            logger.error(f"batch {batch_num}/{total_batches} failed: {e}")
            # record failures for all seeds in failed batch
            for original_seed, prefix_seed in zip(batch_original, batch_prefix):
                error_msg = f"batch processing error: {str(e)}"
                record_failure(failures_data, original_seed, prefix_seed, "egen_generation_failed", error_msg)
            failure_count += len(batch_prefix)
            # save failure tracking after batch error
            save_failure_tracking(failure_tracking_path, failures_data)
            continue

    logger.info(f"batch processing complete: {success_count} succeeded, {failure_count} failed")

    return {
        'total_seeds': total_seeds,
        'success_count': success_count,
        'failure_count': failure_count,
        'saved_files': len(saved_files)
    }


def save_equivalence_file(
    seed_original: str,
    seed_prefix: str,
    equivalents: List[str],
    output_dir: Path
) -> Path:
    """
    save equivalents to hash-named file

    Format:
        Line 1: # SEED: <original seed expression>
        Line 2-N: <equivalent expressions in prefix notation>

    Args:
        seed_original: original seed expression (for documentation)
        seed_prefix: seed in prefix notation (for hashing)
        equivalents: list of equivalent expressions
        output_dir: directory to save file

    Returns:
        path to saved file
    """
    # compute hash based on seed prefix
    seed_hash = compute_seed_hash(seed_prefix)
    filepath = output_dir / f"{seed_hash}.txt"

    # write file
    with open(filepath, 'w') as f:
        # first line: original seed for human inspection
        f.write(f"# SEED: {seed_original}\n")
        # remaining lines: equivalents
        for equiv in equivalents:
            f.write(f"{equiv}\n")

    return filepath


def save_metadata(
    config: EGenPipelineConfig,
    stats: Dict,
    output_dir: Path,
    seed_mapping: Dict[str, str]
):
    """
    save generation metadata

    Args:
        config: pipeline configuration
        stats: generation statistics
        output_dir: output directory
        seed_mapping: dict mapping hash to original seed
    """
    metadata = {
        'version': '1.0',
        'stage': 'equivalence_generation',
        'created': datetime.now().isoformat(),
        'config': {
            'frozen_seeds_path': str(config.frozen_seeds_path),
            'egraph_n_equivalents': config.egraph_n_equivalents,
            'egraph_token_limit': config.egraph_token_limit,
            'egraph_max_token_limit': config.egraph_max_token_limit,
            'egraph_time_limit': config.egraph_time_limit,
        },
        'statistics': stats,
        'seed_mapping': seed_mapping  # hash -> original seed
    }

    metadata_path = output_dir / config.output_metadata_file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"saved metadata to {metadata_path}")


# ========== failure tracking functions ==========

def load_failure_tracking(path: Path) -> Dict:
    """
    load failure tracking data from JSON file

    Args:
        path: path to failure tracking file

    Returns:
        dict with 'version' and 'failures' keys
    """
    if not path.exists():
        return {
            'version': '1.0',
            'failures': {}
        }

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"loaded {len(data.get('failures', {}))} failed seeds from {path}")
        return data
    except Exception as e:
        logger.warning(f"failed to load failure tracking file {path}: {e}")
        return {
            'version': '1.0',
            'failures': {}
        }


def save_failure_tracking(path: Path, failures_data: Dict):
    """
    save failure tracking data to JSON file

    Args:
        path: path to failure tracking file
        failures_data: dict with 'version' and 'failures' keys
    """
    try:
        with open(path, 'w') as f:
            json.dump(failures_data, f, indent=2)
    except Exception as e:
        logger.error(f"failed to save failure tracking file {path}: {e}")


def record_failure(
    failures_data: Dict,
    seed_original: str,
    seed_prefix: str | None,
    failure_type: str,
    error_msg: str = ""
):
    """
    record a seed failure

    Args:
        failures_data: failure tracking dict (modified in-place)
        seed_original: original seed expression
        seed_prefix: seed in prefix notation (None if prefix conversion failed)
        failure_type: "prefix_conversion_failed" or "egen_generation_failed"
        error_msg: error message
    """
    # use hash if we have prefix, otherwise use original seed as key
    if seed_prefix:
        key = compute_seed_hash(seed_prefix)
    else:
        key = hashlib.sha256(seed_original.encode()).hexdigest()[:8]

    failures = failures_data['failures']

    if key in failures:
        # increment attempt counter
        failures[key]['attempts'] += 1
        failures[key]['last_attempt'] = datetime.now().isoformat()
    else:
        # new failure
        failures[key] = {
            'seed_original': seed_original,
            'seed_prefix': seed_prefix,
            'failure_type': failure_type,
            'error_message': error_msg,
            'timestamp': datetime.now().isoformat(),
            'attempts': 1
        }


def filter_failed_seeds_prefix(
    seeds: List[str],
    failures_data: Dict
) -> tuple[List[str], List[str]]:
    """
    filter out seeds that previously failed prefix conversion

    Args:
        seeds: list of original seed expressions
        failures_data: failure tracking dict

    Returns:
        tuple of (non_failed_seeds, failed_seeds)
    """
    failures = failures_data['failures']
    non_failed = []
    failed = []

    for seed in seeds:
        # check if this seed failed before (use original seed as key for prefix failures)
        key = hashlib.sha256(seed.encode()).hexdigest()[:8]
        if key in failures and failures[key]['failure_type'] == 'prefix_conversion_failed':
            failed.append(seed)
        else:
            non_failed.append(seed)

    return non_failed, failed


def filter_failed_seeds_egen(
    seed_mapping: Dict[str, str],
    failures_data: Dict
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    filter out seeds that previously failed E-Gen generation

    Args:
        seed_mapping: dict mapping original seed to prefix notation
        failures_data: failure tracking dict

    Returns:
        tuple of (non_failed_mapping, failed_mapping)
    """
    failures = failures_data['failures']
    non_failed = {}
    failed = {}

    for original, prefix in seed_mapping.items():
        # check if this seed failed E-Gen before (use prefix hash as key)
        key = compute_seed_hash(prefix)
        if key in failures and failures[key]['failure_type'] == 'egen_generation_failed':
            failed[original] = prefix
        else:
            non_failed[original] = prefix

    return non_failed, failed


@hydra.main(version_base=None, config_path="../../config", config_name="egen_generation_config")
def main(cfg: DictConfig):
    """main pipeline execution"""
    limit = cfg.get('limit', None)

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    config = load_egen_pipeline_config(cfg)

    # setup logging
    logging.getLogger().setLevel(config.log_level)
    if config.log_file:
        fh = logging.FileHandler(config.log_file, mode='a')
        fh.setLevel(config.log_level)
        fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
        logging.getLogger().addHandler(fh)

    # create output directory
    config.output_equivalents_dir.mkdir(parents=True, exist_ok=True)

    # ========== STEP 0: Load failure tracking ==========
    logger.info("=" * 60)
    logger.info("STEP 0: Loading failure tracking")
    logger.info("=" * 60)

    failure_tracking_path = config.output_equivalents_dir / config.output_failed_seeds_file
    failures_data = load_failure_tracking(failure_tracking_path)
    if len(failures_data['failures']) > 0:
        logger.info(f"found {len(failures_data['failures'])} previously failed seeds (will skip)")
    else:
        logger.info("no previously failed seeds found")

    # ========== STEP 1: Load frozen seeds ==========
    logger.info("=" * 60)
    logger.info("STEP 1: Loading frozen seeds")
    logger.info("=" * 60)

    frozen_seeds = load_seeds(config.frozen_seeds_path, logger)

    # ========== STEP 2: Convert to prefix notation ==========
    logger.info("=" * 60)
    logger.info("STEP 2: Converting to prefix notation")
    logger.info("=" * 60)

    # filter out seeds that previously failed prefix conversion
    non_failed_seeds, prefix_failed_seeds = filter_failed_seeds_prefix(frozen_seeds, failures_data)
    if len(prefix_failed_seeds) > 0:
        logger.info(f"skipping {len(prefix_failed_seeds)} seeds that previously failed prefix conversion")

    seed_mapping = {}  # original -> prefix
    prefix_seeds = []
    failed_conversions = 0

    for seed in non_failed_seeds:
        prefix = convert_to_prefix(seed)
        if prefix:
            seed_mapping[seed] = prefix
            prefix_seeds.append(prefix)
        else:
            # record new prefix conversion failure
            error_msg = f"failed to convert to prefix notation"
            record_failure(failures_data, seed, None, "prefix_conversion_failed", error_msg)
            failed_conversions += 1

    logger.info(f"converted {len(prefix_seeds)}/{len(non_failed_seeds)} seeds to prefix notation")
    if failed_conversions > 0:
        logger.warning(f"{failed_conversions} seeds failed conversion (recorded in failure tracking)")
    if len(prefix_failed_seeds) > 0:
        logger.info(f"total skipped due to previous prefix failures: {len(prefix_failed_seeds)}")

    # ========== STEP 3: Filter already-processed seeds ==========
    logger.info("=" * 60)
    logger.info("STEP 3: Checking for already-processed seeds")
    logger.info("=" * 60)

    # filter out seeds that previously failed E-Gen generation
    non_egen_failed_mapping, egen_failed_mapping = filter_failed_seeds_egen(seed_mapping, failures_data)
    if len(egen_failed_mapping) > 0:
        logger.info(f"skipping {len(egen_failed_mapping)} seeds that previously failed E-Gen generation")

    unprocessed_mapping, skipped_mapping = filter_unprocessed_seeds(
        non_egen_failed_mapping,
        config.output_equivalents_dir,
        min_equivalents=1
    )

    logger.info(f"found {len(skipped_mapping)} already-processed seeds (skipping)")
    logger.info(f"need to process {len(unprocessed_mapping)} seeds")
    if len(egen_failed_mapping) > 0:
        logger.info(f"total skipped due to previous E-Gen failures: {len(egen_failed_mapping)}")

    # extract unprocessed prefix seeds
    unprocessed_prefix_seeds = list(unprocessed_mapping.values())

    # ========== STEP 4: Generate and save equivalences ==========
    logger.info("=" * 60)
    logger.info("STEP 4: Generating and saving equivalences with E-Gen")
    logger.info("=" * 60)

    if len(unprocessed_mapping) == 0:
        logger.info("no seeds to process - all seeds already have valid output files")
        gen_stats = {
            'total_seeds': 0,
            'success_count': 0,
            'failure_count': 0,
            'saved_files': 0
        }
    else:
        gen_stats = generate_and_save_parallel(
            seed_mapping=unprocessed_mapping,
            config=config,
            output_dir=config.output_equivalents_dir,
            failures_data=failures_data,
            failure_tracking_path=failure_tracking_path,
            batch_size=config.batch_size,
            limit=limit
        )

    # save failure tracking after generation (final save for prefix conversion failures)
    save_failure_tracking(failure_tracking_path, failures_data)

    # ========== STEP 5: Build complete hash-to-seed mapping ==========
    logger.info("=" * 60)
    logger.info("STEP 5: Building hash-to-seed mapping")
    logger.info("=" * 60)

    hash_to_seed = {}  # hash -> original seed

    # add newly processed seeds
    for original_seed, prefix_seed in unprocessed_mapping.items():
        seed_hash = compute_seed_hash(prefix_seed)
        # check if file exists (to handle failed generations)
        filepath = config.output_equivalents_dir / f"{seed_hash}.txt"
        if filepath.exists():
            hash_to_seed[seed_hash] = original_seed

    # add skipped seeds to hash mapping (for complete metadata)
    for original_seed, prefix_seed in skipped_mapping.items():
        seed_hash = compute_seed_hash(prefix_seed)
        hash_to_seed[seed_hash] = original_seed

    logger.info(f"newly saved files: {gen_stats['saved_files']}")
    logger.info(f"total files in output directory: {len(hash_to_seed)}")

    # ========== STEP 6: Save metadata ==========
    logger.info("=" * 60)
    logger.info("STEP 6: Saving metadata")
    logger.info("=" * 60)

    stats = {
        'total_frozen_seeds': len(frozen_seeds),
        'skipped_previous_prefix_failures': len(prefix_failed_seeds),
        'prefix_converted': len(prefix_seeds),
        'new_prefix_conversion_failures': failed_conversions,
        'skipped_previous_egen_failures': len(egen_failed_mapping),
        'already_processed_seeds': len(skipped_mapping),
        'seeds_to_process': len(unprocessed_mapping),
        'generation_success': gen_stats['success_count'],
        'new_egen_generation_failures': gen_stats['failure_count'],
        'newly_saved_files': gen_stats['saved_files'],
        'total_files': len(hash_to_seed),
        'total_failed_seeds_tracked': len(failures_data['failures']),
    }

    save_metadata(config, stats, config.output_equivalents_dir, hash_to_seed)

    # ========== PIPELINE COMPLETE ==========
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {config.output_equivalents_dir}")
    logger.info(f"")
    logger.info(f"Skip statistics:")
    logger.info(f"  - Already processed (success): {len(skipped_mapping)}")
    logger.info(f"  - Previous prefix conversion failures: {len(prefix_failed_seeds)}")
    logger.info(f"  - Previous E-Gen generation failures: {len(egen_failed_mapping)}")
    logger.info(f"")
    logger.info(f"Generation statistics:")
    logger.info(f"  - New generation success: {gen_stats['success_count']}")
    logger.info(f"  - New prefix conversion failures: {failed_conversions}")
    logger.info(f"  - New E-Gen generation failures: {gen_stats['failure_count']}")
    logger.info(f"")
    logger.info(f"File statistics:")
    logger.info(f"  - Newly saved files: {gen_stats['saved_files']}")
    logger.info(f"  - Total success files: {len(hash_to_seed)}")
    logger.info(f"  - Total failed seeds tracked: {len(failures_data['failures'])}")
    logger.info(f"  - Failure tracking file: {failure_tracking_path}")


if __name__ == '__main__':
    main()
