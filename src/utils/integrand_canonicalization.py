import logging
from typing import Tuple
from sympy import sympify, simplify

from src.utils.provenance import sha256_hex
from src.utils.timeout import with_timeout, TimeoutError

logger = logging.getLogger(__name__)


def _simplify_with_timeout(expr, timeout_seconds=20):
    try:
        return with_timeout(simplify, timeout_seconds)(expr)
    except TimeoutError:
        logger.warning(f"Simplify timed out after {timeout_seconds}s, using unsimplified form")
        return expr


def canonicalize_integrand(sympy_integrand_str: str) -> Tuple[str, str]:
    """
    Convert integrand to canonical form and compute hash.
    Args:
        sympy_integrand_str: String representation of integrand from SymPy
    Returns:
        Tuple of (canonical_form, hash)
        - canonical_form: Simplified SymPy string representation (or unsimplified if timeout)
        - hash: 12-character SHA256 hash
    """
    try:
        expr = sympify(sympy_integrand_str)
        canonical_expr = _simplify_with_timeout(expr, timeout_seconds=20)
        canonical = str(canonical_expr)
        integrand_hash = sha256_hex(canonical)[:12]
        return canonical, integrand_hash
    except Exception:
        canonical = sympy_integrand_str.strip()
        integrand_hash = sha256_hex(canonical)[:12]
        return canonical, integrand_hash


def get_integrand_family(integrand_canonical: str) -> str:
    canonical_lower = integrand_canonical.lower()
    functions_found = set()
    for func in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']:
        if func + '(' in canonical_lower:
            functions_found.add(func)
    for func in ['asin', 'acos', 'atan', 'asec', 'acsc', 'acot']:
        if func + '(' in canonical_lower:
            functions_found.add(func)
    for func in ['sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth']:
        if func + '(' in canonical_lower:
            functions_found.add(func)
    for func in ['asinh', 'acosh', 'atanh', 'asech', 'acsch', 'acoth']:
        if func + '(' in canonical_lower:
            functions_found.add(func)
    if 'exp(' in canonical_lower:
        functions_found.add('exp')
    if 'log(' in canonical_lower or 'ln(' in canonical_lower:
        functions_found.add('log')
    if 'sqrt(' in canonical_lower:
        functions_found.add('sqrt')
    if 'abs(' in canonical_lower:
        functions_found.add('abs')
    if 'polylog(' in canonical_lower or 'li(' in canonical_lower:
        functions_found.add('polylog')
    if 'zeta(' in canonical_lower:
        functions_found.add('zeta')
    if 're(' in canonical_lower:
        functions_found.add('re')
    if 'im(' in canonical_lower:
        functions_found.add('im')
    if '**' in canonical_lower or '^' in canonical_lower:
        functions_found.add('pow')
    if '/' in canonical_lower:
        functions_found.add('div')
    if not functions_found:
        return "elementary"
    return ",".join(sorted(functions_found))