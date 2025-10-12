import subprocess
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from src.utils.sexpression import prefix_to_sexp
from src.models.egen.vocab import SEXP_TO_PREFIX

logger = logging.getLogger(__name__)


def _convert_integer_token(token: str) -> List[str]:
    """convert E-Gen integer output to vocabulary format.

    E-Gen outputs integers like: 0, 1, 2, ..., -1, -2, ...
    Vocabulary expects: 0-9 (single digits), INT+ d1 d2 ... (positive), INT- d1 d2 ... (negative)

    Args:
        token: single token from E-Gen output

    Returns:
        list of tokens in vocabulary format"""
    # check if token is an integer
    try:
        val = int(token)
    except ValueError:
        # not an integer, return as-is
        return [token]

    # single digit non-negative (0-9)
    if 0 <= val <= 9:
        return [token]

    # multi-digit positive
    if val > 9:
        digits = list(str(val))
        return ["INT+"] + digits

    # negative
    if val < 0:
        digits = list(str(abs(val)))
        return ["INT-"] + digits

    return [token]

@dataclass
class EGenConfig:
    binary_path: Path
    n_equiv: int = 20
    token_limit: int = 12
    max_token_limit: int = 30
    time_limit: int = 300
    optimized: bool = True


def generate_batch(exprs: List[str], config: EGenConfig, fail_on_error: bool = False) -> Dict[str, List[str]]:
    if not exprs:
        return {}
    logger.info(f"input expressions: {exprs}")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
        input_path = Path(input_file.name)
        for expr in exprs:
            sexp = prefix_to_sexp(expr)
            input_file.write(sexp + '\n')
    output_path = input_path.with_suffix('.out.txt')
    logger.info(f"input file: {input_path}, output file: {output_path}")
    try:
        cmd = [str(config.binary_path)]
        if config.optimized:
            cmd.append('-f')
        cmd.extend([
            '-i', str(input_path),
            '-o', str(output_path),
            '-n', str(config.n_equiv),
            '-l', str(config.token_limit),
            '-m', str(config.max_token_limit),
            '-t', str(config.time_limit),
        ])
        logger.info(f"running E-Gen on {len(exprs)} expressions...")
        logger.info(f"command: {' '.join(cmd)}")
        timeout = config.time_limit * len(exprs) + 60
        logger.info(f"timeout: {timeout}s")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        logger.info(f"E-Gen completed with return code: {result.returncode}")
        if not output_path.exists():
            logger.error(f"E-Gen did not create output file: {output_path}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            if fail_on_error:
                raise RuntimeError("E-Gen did not create output file")
            return {}
        with open(output_path) as f:
            output_content = f.read()
        logger.debug(f"output file size: {len(output_content)} chars")

        lines = output_content.strip().split('\n')
        groups = []
        current_group = []
        for line in lines:
            line = line.strip()
            if not line:  # blank line separates groups
                if current_group:
                    groups.append(current_group)
                    current_group = []
            else:
                current_group.append(line)
        if current_group:
            groups.append(current_group)
        results = {}
        for i, (expr, group) in enumerate(zip(exprs, groups)):
            if len(group) <= 1:
                logger.warning(f"no equivalents generated for: {expr}")
                continue
            equiv_lines = group[1:]  # skip seed echo
            group_equivalents = []
            for equiv_line in equiv_lines:
                tokens = equiv_line.split()
                # convert S-expression operators to prefix notation
                converted_tokens = []
                for token in tokens:
                    # first try operator conversion (S-exp -> prefix)
                    prefix_token = SEXP_TO_PREFIX.get(token, token)
                    # then convert integers to vocab format
                    int_tokens = _convert_integer_token(prefix_token)
                    converted_tokens.extend(int_tokens)
                group_equivalents.append(' '.join(converted_tokens))
            results[expr] = group_equivalents
        logger.info(f"batch complete: generated equivalents for {len(results)}/{len(exprs)} expressions")
        return results

    except subprocess.TimeoutExpired as e:
        logger.error(f"E-Gen timed out after {timeout}s")
        if fail_on_error:
            raise RuntimeError(f"E-Gen timeout: {e}")
        return {}
    except subprocess.CalledProcessError as e:
        logger.error(f"E-Gen failed:\n{e.stderr}")
        if fail_on_error:
            raise RuntimeError(f"E-Gen error: {e.stderr}")
        return {}
    except Exception as e:
        logger.error(f"unexpected error in batch generation: {e}")
        if fail_on_error:
            raise
        return {}
    finally:
        # cleanup temp files
        try:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
