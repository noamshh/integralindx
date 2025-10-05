import logging
from typing import Optional, Tuple
from sympy import sympify
from corpus.pipelines.variable_normalization_pipeline import (
    extract_variables_from_expression,
    create_variable_mapping,
    normalize_expression_variables
)
from src.utils.timeout import with_timeout, TimeoutError

logger = logging.getLogger(__name__)


def normalize_sympy_expression(sympy_expr: str) -> Optional[str]:
    """
    Normalize a SymPy expression string with variable normalization.
    Uses same normalization as pipeline to ensure query matches database.
    Args:
        sympy_expr: SymPy expression as string
    Returns:
        Normalized SymPy expression string (variables normalized to x,s,t,u...), or None if parsing fails
    Examples:
        normalize_sympy_expression("y**2") -> "x**2"
        normalize_sympy_expression("a*sin(t)") -> "s*sin(x)"
    """
    try:
        # parse and re-stringify to normalize
        parsed = sympify(sympy_expr)
        stringified = str(parsed)
        # extract variables from expression
        integrand_vars = extract_variables_from_expression(stringified)
        if not integrand_vars:
            return stringified
        # create variable mapping (last var → x, rest → s,t,u,v...)
        # assume last alphabetically is the integration variable
        sorted_vars = sorted(integrand_vars)
        integration_var = sorted_vars[-1] if sorted_vars else None
        var_mapping = create_variable_mapping(integration_var, integrand_vars)
        # apply variable normalization
        if var_mapping:
            normalized = normalize_expression_variables(stringified, var_mapping, debug=False)
            return normalized if normalized else stringified
        return stringified
    except Exception as e:
        logger.warning(f"failed to normalize SymPy expression '{sympy_expr}': {e}")
        return None


def parse_query_to_sympy_integrand(query: str) -> Tuple[Optional[str], str]:
    query = query.strip()
    if not query:
        return None, 'empty query provided'
    try:
        normalized = with_timeout(normalize_sympy_expression, timeout_seconds=2)(query)
        if normalized is None:
            return None, f'failed to parse SymPy expression: {query}'
        return normalized, 'normalized SymPy expression'
    except TimeoutError as e:
        return None, f'parsing timed out - expression too complex: {str(e)}'
    except Exception as e:
        return None, f'parsing error: {str(e)}'
