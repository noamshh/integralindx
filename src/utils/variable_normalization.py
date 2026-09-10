import logging
import re
from typing import Dict, Set
from sympy import symbols, simplify, Function
from sympy.core.function import UndefinedFunction

from src.utils.safe_math import safe_sympify

logger = logging.getLogger(__name__)

# canonical parameter alphabet for ML tokenization
# max 3 parameters normalized to a, b, c (deterministic ordering)
# this ensures fixed vocabulary for transformer model
CANONICAL_PARAMS = ['a', 'b', 'c']


def extract_variables_from_expression(expr_str: str) -> Set[str]:
    """
    Extract all variable/parameter names from a SymPy expression.
    Args:
        expr_str: SymPy expression as string

    Returns:
        Set of variable names (excluding function names)
    """
    try:
        expr = safe_sympify(expr_str)
        # get free symbols (variables/parameters)
        symbols_in_expr = expr.free_symbols
        return {str(s) for s in symbols_in_expr}
    except Exception:
        # fallback: extract using regex for common variable patterns
        var_pattern = r'\b[a-zA-Z][a-zA-Z0-9_]*\b'
        return set(re.findall(var_pattern, expr_str))


def create_variable_mapping(integration_var: str, integrand_vars: Set[str]) -> Dict[str, str]:
    """
    Create mapping from original variable names to canonical names.

    Maps integration variable to 'x' and up to 3 parameters to 'a', 'b', 'c'.
    Uses sorted order for deterministic mapping.

    Args:
        integration_var: The integration variable (e.g., 't', 'u', 'alpha')
        integrand_vars: All variables found in the integrand

    Returns:
        Dictionary mapping original variables to normalized names
        Max 3 params → a, b, c (sorted alphabetically for determinism)

    Examples:
        >>> create_variable_mapping('t', {'t', 'alpha', 'beta'})
        {'t': 'x', 'alpha': 'a', 'beta': 'b'}

        >>> create_variable_mapping('x', {'x', 'r', 's', 't'})
        {'x': 'x', 'r': 'a', 's': 'b', 't': 'c'}
    """
    mapping = {}

    # integration variable always maps to 'x'
    if integration_var and integration_var in integrand_vars:
        mapping[integration_var] = 'x'
        integrand_vars = integrand_vars - {integration_var}

    # map remaining variables to canonical parameter names (a, b, c)
    # sort for deterministic mapping
    remaining_vars = sorted(integrand_vars)

    for i, var in enumerate(remaining_vars):
        if i < len(CANONICAL_PARAMS):  # max 3 params
            mapping[var] = CANONICAL_PARAMS[i]
        else:
            # this shouldn't happen if symbol validation is applied first
            # but provide fallback just in case
            logger.warning(f"More than {len(CANONICAL_PARAMS)} parameters found: {remaining_vars}")
            mapping[var] = f'param{i+1}'  # fallback naming

    return mapping


def apply_variable_substitution(expr, var_mapping: Dict[str, str]):
    """
    Apply variable substitution to a SymPy expression, handling name collisions.
    Args:
        expr: SymPy expression object
        var_mapping: Dictionary mapping old variable names to new names

    Returns:
        SymPy expression with substituted variables
    """
    if not var_mapping:
        return expr
    # Match the expression's own symbols by name: Symbol('t') and
    # Symbol('t', real=True) are distinct, and subs() silently misses otherwise.
    by_name = {str(s): s for s in expr.free_symbols}
    old_names = set(var_mapping.keys())
    new_names = set(var_mapping.values())
    has_collision = bool(old_names & new_names)
    if has_collision:
        # use two-phase substitution with temporary names to avoid collision
        temp_mapping = {}
        temp_to_final = {}
        for i, (old_var, new_var) in enumerate(var_mapping.items()):
            if old_var not in by_name:
                continue
            temp_name = f'__temp_{i}'
            temp_mapping[by_name[old_var]] = symbols(temp_name)
            temp_to_final[symbols(temp_name)] = symbols(new_var)
        temp_expr = expr.subs(temp_mapping)
        return temp_expr.subs(temp_to_final)
    else:
        # no collision, direct substitution is safe
        subs_mapping = {}
        for old_var, new_var in var_mapping.items():
            if old_var in by_name:
                subs_mapping[by_name[old_var]] = symbols(new_var)
        return expr.subs(subs_mapping)


def normalize_parameters(expr_str: str, integration_var: str = 'x') -> str:
    """
    Normalize parameter names in an expression to canonical form.
    This ensures that expressions with different parameter names but same structure
    get mapped to the same canonical form:
    - x^n/(1+x) → x^s/(1+x)
    - x^q/(1+x) → x^s/(1+x)
    - x^alpha/(1+x) → x^s/(1+x)

    Args:
        expr_str: SymPy expression string (e.g., "x**n/(x + 1)")
        integration_var: The integration variable (default: 'x')

    Returns:
        Normalized expression string with canonical parameter names
    """
    try:
        expr = safe_sympify(expr_str)
        # extract all symbols
        all_symbols = expr.free_symbols
        # separate integration variable from parameters
        integrand_vars = {str(s) for s in all_symbols}
        # create variable mapping
        var_mapping = create_variable_mapping(integration_var, integrand_vars)
        if not var_mapping:
            # no parameters to normalize
            return str(simplify(expr))
        # apply variable substitution (handles name collisions)
        normalized_expr = apply_variable_substitution(expr, var_mapping)
        canonical_expr = simplify(normalized_expr)
        return str(canonical_expr)
    except Exception as e:
        # fallback: return original expression
        return expr_str
