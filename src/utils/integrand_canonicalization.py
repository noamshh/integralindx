"""
Utilities for canonical integrand normalization and hashing.
Used for deduplication of mathematical integrands across different expressions.
"""
import logging
from typing import Tuple
from sympy import sympify, simplify

from src.utils.provenance import sha256_hex
from src.utils.timeout import with_timeout, TimeoutError

logger = logging.getLogger(__name__)


def _simplify_with_timeout(expr, timeout_seconds=20):
    """Simplify expression with timeout, fallback to unsimplified if timeout"""
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
        - hash: 12-character SHA256 hash for fast grouping

    Examples:
        canonicalize_integrand("sin(2*x)") -> ("sin(2*x)", "abc123def456")
        canonicalize_integrand("x**2") -> ("x**2", "def456abc123")
    """
    try:
        # Parse the integrand
        expr = sympify(sympy_integrand_str)

        # Try to simplify with timeout, fallback to unsimplified if timeout
        canonical_expr = _simplify_with_timeout(expr, timeout_seconds=20)
        canonical = str(canonical_expr)

        # Compute deterministic hash
        integrand_hash = sha256_hex(canonical)[:12]

        return canonical, integrand_hash

    except Exception:
        # Fallback for parsing errors - use original string
        canonical = sympy_integrand_str.strip()
        integrand_hash = sha256_hex(canonical)[:12]
        return canonical, integrand_hash


def get_integrand_family(integrand_canonical: str) -> str:
    """
    List all mathematical functions present in the integrand.

    Args:
        integrand_canonical: Canonical form of integrand

    Returns:
        Comma-separated list of function names found (e.g., "sin,exp,sqrt")

    Examples:
        get_integrand_family("sin(2*x)") -> "sin"
        get_integrand_family("x**2") -> "pow"
        get_integrand_family("exp(x)*log(x)") -> "exp,log"
        get_integrand_family("sqrt(x)*tan(x)") -> "sqrt,tan"
    """
    canonical_lower = integrand_canonical.lower()
    functions_found = set()

    # Trigonometric
    for func in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']:
        if func + '(' in canonical_lower:
            functions_found.add(func)

    # Inverse trigonometric
    for func in ['asin', 'acos', 'atan', 'asec', 'acsc', 'acot']:
        if func + '(' in canonical_lower:
            functions_found.add(func)

    # Hyperbolic
    for func in ['sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth']:
        if func + '(' in canonical_lower:
            functions_found.add(func)

    # Inverse hyperbolic
    for func in ['asinh', 'acosh', 'atanh', 'asech', 'acsch', 'acoth']:
        if func + '(' in canonical_lower:
            functions_found.add(func)

    # Exponential and logarithmic
    if 'exp(' in canonical_lower:
        functions_found.add('exp')
    if 'log(' in canonical_lower or 'ln(' in canonical_lower:
        functions_found.add('log')

    # Special functions
    if 'sqrt(' in canonical_lower:
        functions_found.add('sqrt')
    if 'polylog(' in canonical_lower or 'li(' in canonical_lower:
        functions_found.add('polylog')
    if 'gamma(' in canonical_lower:
        functions_found.add('gamma')
    if 'erf(' in canonical_lower:
        functions_found.add('erf')
    if 'abs(' in canonical_lower:
        functions_found.add('abs')
    if 'unk_func(' in canonical_lower:  # check both UNK_FUNC and unk_func
        functions_found.add('unk_func')

    # Structural features
    if '**' in canonical_lower or '^' in canonical_lower:
        functions_found.add('pow')
    if '/' in canonical_lower:
        functions_found.add('div')

    if not functions_found:
        return "elementary"

    return ",".join(sorted(functions_found))