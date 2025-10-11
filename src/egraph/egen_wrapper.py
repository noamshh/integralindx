import subprocess
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from src.utils.sexpression import prefix_to_sexp
from src.models.egen.vocab import SEXP_TO_PREFIX

logger = logging.getLogger(__name__)

@dataclass
class EGenConfig:
    binary_path: Path
    n_equiv: int = 20
    token_limit: int = 12
    time_limit: int = 300
    optimized: bool = True


def generate_batch(exprs: List[str], config: EGenConfig, fail_on_error: bool = False) -> Dict[str, List[str]]:
    """
    Generate equivalent expressions for multiple expressions
    Args:
        exprs: list of expressions in prefix notation
        config: e-graph configuration
        fail_on_error: if True, raise on first error; if False, skip failed expressions
    Returns:
        dict mapping original expression to list of equivalents
        failed expressions are excluded from results (unless fail_on_error=True)
    """

    if not exprs:
        return {}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
        input_path = Path(input_file.name)
        for expr in exprs:
            sexp = prefix_to_sexp(expr)
            input_file.write(sexp + '\n')
    output_path = input_path.with_suffix('.out.txt')
    try:
        cmd = [str(config.binary_path)]
        if config.optimized:
            cmd.append('-f')
        cmd.extend([
            '-i', str(input_path),
            '-o', str(output_path),
            '-n', str(config.n_equiv),
            '-l', str(config.token_limit),
            '-t', str(config.time_limit),
        ])
        logger.info(f"running E-Gen on {len(exprs)} expressions...")
        logger.debug(f"command: {' '.join(cmd)}")
        timeout = config.time_limit * len(exprs) + 60
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
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
                converted_tokens = [SEXP_TO_PREFIX.get(token, token) for token in tokens]
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
