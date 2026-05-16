"""
Safe SymPy parsing that prevents code injection.

sympify() uses eval() internally and executes arbitrary Python when given
malicious input. This module replaces it with parse_expr() using:
  - A character whitelist that blocks quotes, brackets, and shell metacharacters
    before any SymPy parsing occurs
  - A tiny global_dict containing only Symbol/Integer/Float/Rational plus an
    empty __builtins__, so __import__/exec/open/eval are unreachable. The
    Symbol constructor must be in scope because parse_expr's auto_symbol
    transformation materializes unknown identifiers as Symbol('name'); the
    numeric constructors are needed by auto_number for literal parsing.
  - A local_dict that maps known names to specific SymPy objects (e.g. 'a',
    'b', 'c' with real=True assumptions). Unknown letters fall through to
    auto_symbol and become plain Symbols.

Semantic restrictions (which symbols/functions are *meaningful* in this app)
are enforced downstream by validate_parsed_symbols in symbol_validation.py,
not here. This module's job is to parse safely; that module's job is to
decide what's allowed.
"""
import re
import logging
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

logger = logging.getLogger(__name__)

# Characters valid in SymPy expression notation:
#   letters (function names, variable names), digits, whitespace,
#   arithmetic operators (+, -, *, /, **), power caret (^),
#   parentheses, comma (multi-arg functions), decimal point, underscore.
# Explicitly excluded: quotes ' ", backticks `, brackets [ ],
#   curly braces { }, semicolons ;, angle brackets < >, at signs @, etc.
_SAFE_CHARS = re.compile(r'^[a-zA-Z0-9\s\+\-\*\/\^\(\)\,\.\_]+$')

_TRANSFORMATIONS = standard_transformations + (convert_xor,)

# Allowed SymPy symbols and functions — mirrors the allowed set in
# symbol_validation.py (COMMON_PARAMS) and vocab.py (FUNCTION_REGISTRY).
SAFE_SYMBOLS_DICT: dict = {
    # integration variable and parameters
    'x': sp.Symbol('x'),
    'a': sp.Symbol('a', real=True),
    'b': sp.Symbol('b', real=True),
    'c': sp.Symbol('c', real=True),
    'r': sp.Symbol('r', real=True),
    's': sp.Symbol('s', real=True),
    't': sp.Symbol('t', real=True),
    'u': sp.Symbol('u', real=True),
    'z': sp.Symbol('z'),
    'n': sp.Symbol('n'),
    'm': sp.Symbol('m'),
    'alpha': sp.Symbol('alpha', real=True),
    'beta':  sp.Symbol('beta',  real=True),
    # mathematical constants
    'pi': sp.pi,
    'E':  sp.E,
    # basic operators (parse_expr resolves these automatically, but being
    # explicit prevents them from falling through to global builtins)
    'Add': sp.Add,
    'Mul': sp.Mul,
    'Pow': sp.Pow,
    # basic unary functions
    'exp':  sp.exp,
    'log':  sp.log,
    'Abs':  sp.Abs,
    'abs':  sp.Abs,
    'sqrt': sp.sqrt,
    # trigonometric
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'cot': sp.cot, 'sec': sp.sec, 'csc': sp.csc,
    # inverse trigonometric
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'acot': sp.acot, 'asec': sp.asec, 'acsc': sp.acsc,
    # hyperbolic
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
    'coth': sp.coth, 'sech': sp.sech, 'csch': sp.csch,
    # inverse hyperbolic
    'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
    'acoth': sp.acoth, 'asech': sp.asech, 'acsch': sp.acsch,
    # special functions
    'polylog': sp.polylog,
    'zeta':    sp.zeta,
}

# Minimal global context passed to parse_expr. __builtins__ is forced to an
# empty dict so __import__/eval/exec/open are unreachable. The four SymPy
# constructors are required for parse_expr's transformations:
#   - Symbol     : auto_symbol wraps unknown identifiers as Symbol('name')
#   - Integer/Float/Rational : auto_number wraps numeric literals
# These constructors are inert — they build symbolic objects, they do not
# evaluate Python code, so exposing them does not weaken the sandbox.
_SAFE_GLOBALS: dict = {
    '__builtins__': {},
    'Symbol':   sp.Symbol,
    'Integer':  sp.Integer,
    'Float':    sp.Float,
    'Rational': sp.Rational,
}


def safe_sympify(expr: str) -> sp.Expr:
    """
    Parse a mathematical expression string safely.

    Raises:
        TypeError:  if expr is not a string
        ValueError: if expr contains disallowed characters or is too long
        Exception:  any SymPy parse error for malformed expressions
    """
    if not isinstance(expr, str):
        raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
    expr = expr.strip()
    if len(expr) > 500:
        raise ValueError("Expression too long (max 500 characters)")
    if not expr:
        raise ValueError("Empty expression")
    if not _SAFE_CHARS.match(expr):
        logger.warning(f"safe_sympify: rejected expression with disallowed chars: {expr[:120]!r}")
        raise ValueError("Expression contains disallowed characters")
    return parse_expr(
        expr,
        global_dict=_SAFE_GLOBALS,
        local_dict=SAFE_SYMBOLS_DICT,
        transformations=_TRANSFORMATIONS,
        evaluate=True,
    )
