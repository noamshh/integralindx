"""
Based on reference/hongbozheng-transformer/vocab.py with extensions for
integral-specific functions (polylog, zeta, re, im).
"""
import sympy as sp
from collections import OrderedDict


CONSTANTS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "pi", "e",
    "INT+", "INT-",  # multi-digit integer
]

VARIABLES = OrderedDict({
    'x': sp.Symbol('x', real=None, nonzero=None, positive=None),
    'a': sp.Symbol('a', real=True),
    'b': sp.Symbol('b', real=True),
    'c': sp.Symbol('c', real=True),
})

OPERATORS = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "rac": 2,
    "inv": 1,
    "pow2": 1,
    "pow3": 1,
    "pow4": 1,
    "pow5": 1,
    "sqrt": 1,
    "exp": 1,
    "log": 1,
    "abs": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "cot": 1,
    "sec": 1,
    "csc": 1,
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "acot": 1,
    "asec": 1,
    "acsc": 1,
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "coth": 1,
    "sech": 1,
    "csch": 1,
    "asinh": 1,
    "acosh": 1,
    "atanh": 1,
    "acoth": 1,
    "asech": 1,
    "acsch": 1,
    "Li": 2,     # polylog(order, arg)
    "zeta": 1,   # Riemann zeta function
}

SYMPY_TO_PREFIX = {
    sp.Add: "add",
    sp.Mul: "mul",
    sp.Pow: "pow",
    sp.exp: "exp",
    sp.log: "log",
    sp.Abs: "abs",
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.cot: "cot",
    sp.sec: "sec",
    sp.csc: "csc",
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.acot: "acot",
    sp.asec: "asec",
    sp.acsc: "acsc",
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.coth: "coth",
    sp.sech: "sech",
    sp.csch: "csch",
    sp.asinh: "asinh",
    sp.acosh: "acosh",
    sp.atanh: "atanh",
    sp.acoth: "acoth",
    sp.asech: "asech",
    sp.acsch: "acsch",
    sp.polylog: "Li",
    sp.zeta: "zeta",
}

SPECIAL_TOKENS = {
    "PAD": 0,   # padding token
    "SOE": 1,   # start of expression
    "EOE": 2,   # end of expression
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

__all__ = [
    'CONSTANTS',
    'VARIABLES',
    'OPERATORS',
    'SYMPY_TO_PREFIX',
    'SPECIAL_TOKENS',
    'get_all_tokens',
    'get_vocab_size',
]
