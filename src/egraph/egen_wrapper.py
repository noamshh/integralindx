import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from src.utils.sexpression import prefix_to_sexp

logger = logging.getLogger(__name__)

@dataclass
class EGenConfig:
    binary_path: Path
    n_equiv: int = 20
    token_limit: int = 12
    time_limit: int = 300
    optimized: bool = True

def _parse_file_output(output_content: str) -> List[str]:
    lines = output_content.strip().split('\n')
    equivalents = []
    SYMBOL_TO_WORD_MAP = {
        '+': 'add',
        '-': 'sub',
        '*': 'mul',
        '/': 'div',
    }
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if tokens and tokens[0] in SYMBOL_TO_WORD_MAP:
            tokens[0] = SYMBOL_TO_WORD_MAP[tokens[0]]
        converted = ' '.join(tokens)
        equivalents.append(converted)
    return equivalents


def generate_batch(exprs: List[str], config: EGenConfig, fail_on_error: bool = False) -> Dict[str, List[str]]:
    """
    Generate equivalent expressions for multiple expressions using file I/O

    More efficient than single-expression mode - writes all seeds to input file,
    runs E-Gen once, then parses output file.
    Args:
        exprs: list of expressions in prefix notation (e.g., ["add x 1", "mul x 2"])
        config: e-graph configuration with binary path
        fail_on_error: if True, raise on first error; if False, skip failed expressions
    Returns:
        dict mapping original expression to list of equivalents
        failed expressions are excluded from results (unless fail_on_error=True)
    """
    import tempfile

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

        # parse output file - format is:
        # original1
        # equiv1_1
        # equiv1_2
        # <blank>
        # original2
        # equiv2_1
        # ...
        all_equivalents = _parse_file_output(output_content)
        results = {}
        current_group = []
        for equiv in all_equivalents:
            current_group.append(equiv)

        # simple grouping: assume output is in same order as input
        # first expr in each group is the original (in our format), rest are equivalents
        # split by finding where we have exactly n_equiv + 1 expressions
        if all_equivalents:
            equiv_idx = 0
            for expr in exprs:
                group_equivalents = []
                if equiv_idx < len(all_equivalents):
                    equiv_idx += 1
                while equiv_idx < len(all_equivalents) and len(group_equivalents) < config.n_equiv:
                    group_equivalents.append(all_equivalents[equiv_idx])
                    equiv_idx += 1
                if group_equivalents:
                    results[expr] = group_equivalents
                else:
                    logger.warning(f"no equivalents generated for: {expr}")
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
