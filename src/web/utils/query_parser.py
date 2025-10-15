import logging
from typing import Optional, Tuple
from sympy import sympify
from corpus.pipelines.variable_normalization_pipeline import (
    extract_variables_from_expression,
    create_variable_mapping,
    normalize_expression_variables
)
from src.utils.timeout import with_timeout, TimeoutError
from src.utils.symbol_validation import validate_parsed_symbols

logger = logging.getLogger(__name__)


def normalize_sympy_expression(sympy_expr: str) -> Optional[str]:
    """
    Applies variable normalization to sympy expression
    Args:
        sympy_expr: sympy expression as string
    Returns:
        Normalized sympy expression string (integration var -> x, params -> a,b,c), or None if parsing fails
    """
    try:
        parsed = sympify(sympy_expr)
        stringified = str(parsed)
        integrand_vars = extract_variables_from_expression(stringified)
        if not integrand_vars:
            return stringified
        # assume last alphabetically is the integration variable
        sorted_vars = sorted(integrand_vars)
        integration_var = sorted_vars[-1] if sorted_vars else None
        var_mapping = create_variable_mapping(integration_var, integrand_vars)
        if var_mapping:
            normalized = normalize_expression_variables(stringified, var_mapping, debug=False)
            return normalized if normalized else stringified
        return stringified
    except Exception as e:
        logger.warning(f"failed to normalize sympy expression '{sympy_expr}': {e}")
        return None


def parse_query_to_sympy_integrand(query: str) -> Tuple[Optional[str], str]:
    query = query.strip()
    if not query:
        return None, 'empty query provided'
    try:
        parsed = sympify(query)
        stringified = str(parsed)
        integrand_vars = extract_variables_from_expression(stringified)
        if integrand_vars:
            sorted_vars = sorted(integrand_vars)
            integration_var = sorted_vars[-1] if sorted_vars else 'x'
            is_valid, reason, params = validate_parsed_symbols(stringified, integration_var, debug=False)
            if not is_valid:
                return None, f'invalid expression: {reason}'

        normalized = with_timeout(normalize_sympy_expression, timeout_seconds=2)(query)
        if normalized is None:
            return None, f'failed to parse sympy expression: {query}'
        return normalized, 'normalized sympy expression'
    except TimeoutError as e:
        return None, f'parsing timed out - expression too complex: {str(e)}'
    except Exception as e:
        return None, f'parsing error: {str(e)}'
