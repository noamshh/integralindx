import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class EGenConfig:
    binary_path: Path                  # path to egen binary (required)
    n_equiv: int = 20                  # number of equivalent expressions to generate
    token_limit: int = 12              # max token limit for saturation
    time_limit: int = 300              # time limit in seconds


def _run_egen(expr: str, config: EGenConfig) -> subprocess.CompletedProcess:
    """
    Run the E-Gen Rust binary on a single expression
    Args:
        expr: prefix notation expression (space-separated)
        config: e-graph configuration
    Returns:
        subprocess result
    Raises:
        FileNotFoundError: if binary not found or invalid
        subprocess.TimeoutExpired: if execution times out
        subprocess.CalledProcessError: if binary fails
    """
    binary_path = config.binary_path
    if not binary_path.exists():
        raise FileNotFoundError(f"E-Gen binary not found at: {binary_path}")
    # build command
    # E-Gen CLI takes -n <n_equiv> -l <token_limit> -t <time_limit> <expr>
    cmd = [
        str(binary_path),
        '-n', str(config.n_equiv),
        '-l', str(config.token_limit),
        '-t', str(config.time_limit),
        expr
    ]
    logger.debug(f"running E-Gen: {' '.join(cmd)}")
    timeout = config.time_limit + 60
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True
    )
    return result


def _parse_output(output: str) -> List[str]:
    """
    Parse E-Gen binary output into list of equivalent expressions
    output format (from Rust binary):
    - first line: original expression
    - subsequent lines: equivalent expressions
    - empty lines separate sections
    Args:
        output: stdout from E-Gen binary
    Returns:
        list of equivalent expressions (excluding original)
    """
    lines = output.strip().split('\n')
    equivalents = []
    skip_first = True  # first line is the original expression
    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        if skip_first:
            skip_first = False
            continue
        # clean up parentheses if needed (Rust outputs s-expressions)
        # example: "(+ x 1)" -> we might want "add x 1"
        # but for now, keep as-is since Rust uses s-expr format
        equivalents.append(line)

    return equivalents


def generate_equivalents(expr: str, config: EGenConfig) -> List[str]:
    """
    Generate equivalent expressions for a single expression
    Args:
        expr: expression in prefix notation (space-separated)
        config: e-graph configuration
    Returns:
        list of equivalent expressions in same format
    Raises:
        FileNotFoundError: if E-Gen binary not found
        subprocess.TimeoutExpired: if generation times out
        RuntimeError: if generation fails
    """
    try:
        result = _run_egen(expr, config)
        equivalents = _parse_output(result.stdout)
        logger.info(
            f"generated {len(equivalents)} equivalents for: {expr[:50]}..."
        )
        return equivalents
    except subprocess.TimeoutExpired as e:
        logger.error(f"E-Gen timed out after {config.time_limit}s for: {expr}")
        raise RuntimeError(f"E-Gen timeout: {e}")
    except subprocess.CalledProcessError as e:
        logger.error(f"E-Gen failed for: {expr}\nstderr: {e.stderr}")
        raise RuntimeError(f"E-Gen error: {e.stderr}")
    except Exception as e:
        logger.error(f"unexpected error generating equivalents: {e}")
        raise


def generate_batch(exprs: List[str], config: EGenConfig, fail_on_error: bool = False) -> Dict[str, List[str]]:
    """
    Generate equivalent expressions for multiple expressions
    Args:
        exprs: list of expressions in prefix notation
        config: e-graph configuration with binary path
        fail_on_error: if True, raise on first error; if False, skip failed expressions

    Returns:
        dict mapping original expression to list of equivalents
        failed expressions are excluded from results (unless fail_on_error=True)
    """
    results = {}
    failed = []
    for i, expr in enumerate(exprs):
        try:
            equivalents = generate_equivalents(expr, config)
            results[expr] = equivalents
            if (i + 1) % 10 == 0:
                logger.info(f"processed {i + 1}/{len(exprs)} expressions")
        except Exception as e:
            logger.warning(f"failed to process expression: {expr}\nerror: {e}")
            failed.append((expr, str(e)))
            if fail_on_error:
                raise
    if failed:
        logger.warning(f"failed to generate equivalents for {len(failed)}/{len(exprs)} expressions")
    logger.info(f"batch complete: {len(results)}/{len(exprs)} successful, {len(failed)} failed")
    return results
