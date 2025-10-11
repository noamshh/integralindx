"""Based on reference/hongbozheng-transformer/vocab.py with extensions (polylog, zeta)."""
import sympy as sp
from collections import OrderedDict
from typing import Dict


CONSTANTS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "pi", "e", "INT+", "INT-"]
VARIABLES = OrderedDict({
    'x': sp.Symbol('x', real=None, nonzero=None, positive=None),
    'a': sp.Symbol('a', real=True),
    'b': sp.Symbol('b', real=True),
    'c': sp.Symbol('c', real=True),
})

FUNCTION_REGISTRY = [
    # basic binary operators
    {'sympy': sp.Add, 'prefix': 'add', 'sexp': '+', 'arity': 2},
    {'sympy': sp.Mul, 'prefix': 'mul', 'sexp': '*', 'arity': 2},
    {'sympy': sp.Pow, 'prefix': 'pow', 'sexp': 'pow', 'arity': 2},
    {'sympy': None, 'prefix': 'sub', 'sexp': '-', 'arity': 2},  # derived from Add
    {'sympy': None, 'prefix': 'div', 'sexp': '/', 'arity': 2},  # derived from Mul/Pow
    {'sympy': None, 'prefix': 'rac', 'sexp': 'rac', 'arity': 2},  # rational constructor
    # unary power operators
    {'sympy': None, 'prefix': 'inv', 'sexp': 'inv', 'arity': 1},  # 1/x
    {'sympy': None, 'prefix': 'pow2', 'sexp': 'pow2', 'arity': 1},  # x^2
    {'sympy': None, 'prefix': 'pow3', 'sexp': 'pow3', 'arity': 1},  # x^3
    {'sympy': None, 'prefix': 'pow4', 'sexp': 'pow4', 'arity': 1},  # x^4
    {'sympy': None, 'prefix': 'pow5', 'sexp': 'pow5', 'arity': 1},  # x^5
    {'sympy': None, 'prefix': 'sqrt', 'sexp': 'sqrt', 'arity': 1},  # derived from Pow(x, 1/2)
    # basic unary functions
    {'sympy': sp.exp, 'prefix': 'exp', 'sexp': 'exp', 'arity': 1},
    {'sympy': sp.log, 'prefix': 'log', 'sexp': 'log', 'arity': 1},
    {'sympy': sp.Abs, 'prefix': 'abs', 'sexp': 'abs', 'arity': 1},
    # trigonometric
    {'sympy': sp.sin, 'prefix': 'sin', 'sexp': 'sin', 'arity': 1},
    {'sympy': sp.cos, 'prefix': 'cos', 'sexp': 'cos', 'arity': 1},
    {'sympy': sp.tan, 'prefix': 'tan', 'sexp': 'tan', 'arity': 1},
    {'sympy': sp.cot, 'prefix': 'cot', 'sexp': 'cot', 'arity': 1},
    {'sympy': sp.sec, 'prefix': 'sec', 'sexp': 'sec', 'arity': 1},
    {'sympy': sp.csc, 'prefix': 'csc', 'sexp': 'csc', 'arity': 1},
    # inverse trigonometric
    {'sympy': sp.asin, 'prefix': 'asin', 'sexp': 'asin', 'arity': 1},
    {'sympy': sp.acos, 'prefix': 'acos', 'sexp': 'acos', 'arity': 1},
    {'sympy': sp.atan, 'prefix': 'atan', 'sexp': 'atan', 'arity': 1},
    {'sympy': sp.acot, 'prefix': 'acot', 'sexp': 'acot', 'arity': 1},
    {'sympy': sp.asec, 'prefix': 'asec', 'sexp': 'asec', 'arity': 1},
    {'sympy': sp.acsc, 'prefix': 'acsc', 'sexp': 'acsc', 'arity': 1},
    # hyperbolic
    {'sympy': sp.sinh, 'prefix': 'sinh', 'sexp': 'sinh', 'arity': 1},
    {'sympy': sp.cosh, 'prefix': 'cosh', 'sexp': 'cosh', 'arity': 1},
    {'sympy': sp.tanh, 'prefix': 'tanh', 'sexp': 'tanh', 'arity': 1},
    {'sympy': sp.coth, 'prefix': 'coth', 'sexp': 'coth', 'arity': 1},
    {'sympy': sp.sech, 'prefix': 'sech', 'sexp': 'sech', 'arity': 1},
    {'sympy': sp.csch, 'prefix': 'csch', 'sexp': 'csch', 'arity': 1},
    # inverse hyperbolic
    {'sympy': sp.asinh, 'prefix': 'asinh', 'sexp': 'asinh', 'arity': 1},
    {'sympy': sp.acosh, 'prefix': 'acosh', 'sexp': 'acosh', 'arity': 1},
    {'sympy': sp.atanh, 'prefix': 'atanh', 'sexp': 'atanh', 'arity': 1},
    {'sympy': sp.acoth, 'prefix': 'acoth', 'sexp': 'acoth', 'arity': 1},
    {'sympy': sp.asech, 'prefix': 'asech', 'sexp': 'asech', 'arity': 1},
    {'sympy': sp.acsch, 'prefix': 'acsch', 'sexp': 'acsch', 'arity': 1},
    # special functions
    {'sympy': sp.polylog, 'prefix': 'Li', 'sexp': 'Li', 'arity': 2},
    {'sympy': sp.zeta, 'prefix': 'zeta', 'sexp': 'zeta', 'arity': 1},
]

SYMPY_TO_PREFIX = {entry['sympy']: entry['prefix'] for entry in FUNCTION_REGISTRY if entry['sympy'] is not None}
PREFIX_TO_SEXP = {entry['prefix']: entry['sexp'] for entry in FUNCTION_REGISTRY}
SEXP_TO_PREFIX = {entry['sexp']: entry['prefix'] for entry in FUNCTION_REGISTRY}
PREFIX_ARITY = {entry['prefix']: entry['arity'] for entry in FUNCTION_REGISTRY}
SEXP_ARITY = {entry['sexp']: entry['arity'] for entry in FUNCTION_REGISTRY}

OPERATORS: Dict[str, int] = PREFIX_ARITY.copy()

SPECIAL_TOKENS = {
    "PAD": 0,
    "SOE": 1,
    "EOE": 2,
}

def get_all_tokens():
    tokens = set()
    tokens.update(CONSTANTS)
    tokens.update(VARIABLES.keys())
    tokens.update(OPERATORS.keys())
    tokens.update(SPECIAL_TOKENS.keys())
    return sorted(tokens)

def get_vocab_size():
    return len(get_all_tokens())
