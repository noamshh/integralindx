import logging
from typing import Optional, Set, Tuple

from sympy import Expr

from src.utils.safe_math import safe_sympify
from src.utils.symbol_validation import validate_parsed_symbols
from src.utils.variable_normalization import apply_variable_substitution, create_variable_mapping

logger = logging.getLogger(__name__)

CANONICAL_PARAMS = {'a', 'b', 'c'}


def _integration_var(var_names: Set[str]) -> Optional[str]:
    candidates = sorted(var_names - CANONICAL_PARAMS)
    return candidates[-1] if candidates else None


def normalize_sympy_expression(expr: Expr) -> Optional[str]:
    try:
        var_names = {str(s) for s in expr.free_symbols}
        integration_var = _integration_var(var_names) if var_names else None
        if integration_var is None:
            return str(expr)
        var_mapping = create_variable_mapping(integration_var, var_names)
        if not var_mapping:
            return str(expr)
        return str(apply_variable_substitution(expr, var_mapping))
    except Exception as e:
        logger.warning(f"failed to normalize expression '{expr}': {e}")
        return None


def parse_query_to_sympy_integrand(query: str) -> Tuple[Optional[str], str]:
    query = query.strip()
    if not query:
        return None, 'empty query provided'
    try:
        parsed = safe_sympify(query)
        var_names = {str(s) for s in parsed.free_symbols}
        if var_names:
            multi_letter = {v for v in var_names if len(v) > 1}
            if multi_letter:
                return None, f'multi-letter variables not allowed: {multi_letter}'
            integration_var = _integration_var(var_names) or 'x'
            is_valid, reason, _ = validate_parsed_symbols(str(parsed), integration_var, debug=False)
            if not is_valid:
                return None, f'invalid expression: {reason}'
        normalized = normalize_sympy_expression(parsed)
        if normalized is None:
            return None, f'failed to normalize expression: {query}'
        return normalized, 'normalized sympy expression'
    except Exception as e:
        return None, f'parsing error: {str(e)}'
