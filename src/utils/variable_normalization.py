import re
from typing import Dict, Set
from sympy import sympify, symbols, simplify, Function
from sympy.core.function import UndefinedFunction

# canonical parameter alphabet
# skip common bound names (a,b,c,d,e,i), function names (f,g,h), and confusing letters (o,l)
CANONICAL_PARAMS = ['s', 't', 'u', 'v', 'w', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 'y', 'z']


def extract_variables_from_expression(expr_str: str) -> Set[str]:
    """
    Extract all variable/parameter names from a SymPy expression.
    Args:
        expr_str: SymPy expression as string

    Returns:
        Set of variable names (excluding function names)
    """
    try:
        expr = sympify(expr_str)
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
    Args:
        integration_var: The integration variable (e.g., 't', 'u', 'theta')
        integrand_vars: All variables found in the integrand

    Returns:
        Dictionary mapping original variables to normalized names
    """
    mapping = {}
    # integration variable always maps to 'x'
    if integration_var and integration_var in integrand_vars:
        mapping[integration_var] = 'x'
        integrand_vars = integrand_vars - {integration_var}
    # map remaining variables to canonical parameter names
    remaining_vars = sorted(integrand_vars)  # sort for deterministic mapping
    for i, var in enumerate(remaining_vars):
        if i < len(CANONICAL_PARAMS):
            mapping[var] = CANONICAL_PARAMS[i]
        else:
            # fallback for too many variables
            mapping[var] = f'p{i - len(CANONICAL_PARAMS) + 1}'
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
    # check for name collisions where a new name is also an old name
    # example: r→s, s→t creates collision because 's' is both target and source
    old_names = set(var_mapping.keys())
    new_names = set(var_mapping.values())
    has_collision = bool(old_names & new_names)
    if has_collision:
        # use two-phase substitution with temporary names to avoid collision
        temp_mapping = {}
        temp_to_final = {}
        for i, (old_var, new_var) in enumerate(var_mapping.items()):
            temp_name = f'__temp_{i}'
            temp_mapping[symbols(old_var)] = symbols(temp_name)
            temp_to_final[symbols(temp_name)] = symbols(new_var)
        temp_expr = expr.subs(temp_mapping)
        return temp_expr.subs(temp_to_final)
    else:
        # no collision, direct substitution is safe
        subs_mapping = {}
        for old_var, new_var in var_mapping.items():
            subs_mapping[symbols(old_var)] = symbols(new_var)
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
        expr = sympify(expr_str)
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
