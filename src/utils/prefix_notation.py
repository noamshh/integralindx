"""
SymPy to Prefix (Polish) notation converter for E-Gen transformer.

Converts SymPy expressions to space-separated prefix notation suitable for
transformer input. This is the format used by the E-Gen paper and reference implementation.

Example:
    x**2 + sin(x)  →  "add pow x 2 sin x"
    sin(x)*cos(x)  →  "mul sin x cos x"
    polylog(2, x)  →  "Li 2 x"
    f(x)           →  "UNK f x"
"""

import sympy as sp
from sympy import Expr, Number, Symbol, Integer, Rational, Float, Add, Mul, Pow, Function
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# Operator vocabulary matching E-Gen reference implementation
# See: reference/hongbozheng-transformer/vocab.py
SYMPY_TO_PREFIX = {
    sp.Add: "add",
    sp.Mul: "mul",
    sp.Pow: "pow",
    sp.exp: "exp",
    sp.log: "ln",
    sp.Abs: "abs",
    sp.sign: "sign",
    # Trigonometric
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.cot: "cot",
    sp.sec: "sec",
    sp.csc: "csc",
    # Inverse Trig
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.acot: "acot",
    sp.asec: "asec",
    sp.acsc: "acsc",
    # Hyperbolic
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.coth: "coth",
    sp.sech: "sech",
    sp.csch: "csch",
    # Inverse Hyperbolic
    sp.asinh: "asinh",
    sp.acosh: "acosh",
    sp.atanh: "atanh",
    sp.acoth: "acoth",
    sp.asech: "asech",
    sp.acsch: "acsch",
    # Polylogarithm
    sp.polylog: "Li",  # Li(s, z) - polylogarithm of order s
}


def _handle_number(num: Number) -> List[str]:
    """
    Convert SymPy number to prefix tokens.

    Integers: INT+ followed by digits, or direct constants 0-9
    Rationals: div numerator denominator
    Floats: approximated as rationals or integers

    Args:
        num: SymPy Number object

    Returns:
        List of prefix tokens
    """
    if isinstance(num, Integer):
        val = int(num)
        if 0 <= val <= 9:
            return [str(val)]
        elif val > 0:
            # positive multi-digit: INT+ followed by digits
            digits = list(str(val))
            return ["INT+"] + digits
        else:
            # negative: INT- followed by digits (of absolute value)
            digits = list(str(abs(val)))
            return ["INT-"] + digits

    elif isinstance(num, Rational):
        # Rational number: div numerator denominator
        numer_tokens = _handle_number(num.p)
        denom_tokens = _handle_number(num.q)
        return ["div"] + numer_tokens + denom_tokens

    elif isinstance(num, Float):
        # approximate float as rational
        rational = sp.nsimplify(num)
        if isinstance(rational, Rational):
            return _handle_number(rational)
        else:
            # fallback: round to integer
            return _handle_number(Integer(round(float(num))))
    else:
        # fallback for other number types
        return [str(num)]


def sympy_to_prefix(expr: Expr) -> str:
    """
    Convert SymPy expression to prefix (Polish) notation.

    Args:
        expr: SymPy expression

    Returns:
        Space-separated prefix notation string

    Examples:
        >>> sympy_to_prefix(sp.sympify("x**2"))
        "pow x 2"
        >>> sympy_to_prefix(sp.sympify("sin(x) + cos(x)"))
        "add sin x cos x"
        >>> sympy_to_prefix(sp.sympify("x**2 * sin(x)"))
        "mul pow x 2 sin x"
        >>> sympy_to_prefix(sp.polylog(2, sp.Symbol('x')))
        "Li 2 x"
    """
    tokens = _sympy_to_prefix_tokens(expr)
    return " ".join(tokens)


def _sympy_to_prefix_tokens(expr: Expr) -> List[str]:
    """
    Convert SymPy expression to list of prefix tokens.

    Args:
        expr: SymPy expression

    Returns:
        List of prefix tokens
    """
    # Symbol (variable)
    if isinstance(expr, Symbol):
        return [str(expr)]

    # Number (constant)
    if isinstance(expr, Number):
        return _handle_number(expr)

    # Special constants
    if expr == sp.pi:
        return ["pi"]
    if expr == sp.E:
        return ["e"]

    # Get expression type
    expr_type = type(expr)

    # Check if it's a known function/operator
    if expr_type in SYMPY_TO_PREFIX:
        op_name = SYMPY_TO_PREFIX[expr_type]

        # Special handling for different operators
        if expr_type == Add:
            # Add: collect all addends
            # For n-ary add, convert to binary: add a (add b c)
            args = list(expr.args)
            if len(args) == 0:
                return ["0"]
            elif len(args) == 1:
                return _sympy_to_prefix_tokens(args[0])
            elif len(args) == 2:
                return [op_name] + _sympy_to_prefix_tokens(args[0]) + _sympy_to_prefix_tokens(args[1])
            else:
                # n-ary: right-associate add a (add b (add c d))
                result = _sympy_to_prefix_tokens(args[-1])
                for arg in reversed(args[:-1]):
                    result = [op_name] + _sympy_to_prefix_tokens(arg) + result
                return result

        elif expr_type == Mul:
            # Mul: collect all multiplicands
            # Check for special cases like -1*x (negation)
            if len(expr.args) == 2 and expr.args[0] == -1:
                # Negation: sub 0 x
                return ["sub", "0"] + _sympy_to_prefix_tokens(expr.args[1])

            # n-ary multiplication
            args = list(expr.args)
            if len(args) == 0:
                return ["1"]
            elif len(args) == 1:
                return _sympy_to_prefix_tokens(args[0])
            elif len(args) == 2:
                return [op_name] + _sympy_to_prefix_tokens(args[0]) + _sympy_to_prefix_tokens(args[1])
            else:
                # n-ary: right-associate mul a (mul b (mul c d))
                result = _sympy_to_prefix_tokens(args[-1])
                for arg in reversed(args[:-1]):
                    result = [op_name] + _sympy_to_prefix_tokens(arg) + result
                return result

        elif expr_type == Pow:
            # Power: pow base exponent
            base, exp = expr.args

            # Special cases for common powers
            if exp == 2:
                return ["pow2"] + _sympy_to_prefix_tokens(base)
            elif exp == 3:
                return ["pow3"] + _sympy_to_prefix_tokens(base)
            elif exp == 4:
                return ["pow4"] + _sympy_to_prefix_tokens(base)
            elif exp == 5:
                return ["pow5"] + _sympy_to_prefix_tokens(base)
            elif exp == sp.Rational(1, 2):
                return ["sqrt"] + _sympy_to_prefix_tokens(base)
            elif exp == -1:
                return ["inv"] + _sympy_to_prefix_tokens(base)
            else:
                # General power: pow base exponent
                return [op_name] + _sympy_to_prefix_tokens(base) + _sympy_to_prefix_tokens(exp)

        elif expr_type == sp.polylog:
            # Polylogarithm: Li order argument
            # polylog(s, z) → Li s z
            order, arg = expr.args
            return [op_name] + _sympy_to_prefix_tokens(order) + _sympy_to_prefix_tokens(arg)

        else:
            # Function application (unary or binary)
            tokens = [op_name]
            for arg in expr.args:
                tokens.extend(_sympy_to_prefix_tokens(arg))
            return tokens

    # Check for function by func attribute
    if hasattr(expr, 'func'):
        func = expr.func

        # Known function
        if func in SYMPY_TO_PREFIX:
            op_name = SYMPY_TO_PREFIX[func]
            tokens = [op_name]
            for arg in expr.args:
                tokens.extend(_sympy_to_prefix_tokens(arg))
            return tokens

        # Unknown function - special UNK token
        # f(x) → UNK f x
        # g(x, y) → UNK g x y
        # UNK_FUNC(x) → UNK x (special case from variable normalization)
        if isinstance(expr, sp.core.function.AppliedUndef):
            func_name = expr.func.__name__

            # Special case: UNK_FUNC from variable normalization
            if func_name == "UNK_FUNC":
                # UNK_FUNC(x) → UNK x (simplified, no function name)
                tokens = ["UNK"]
                for arg in expr.args:
                    tokens.extend(_sympy_to_prefix_tokens(arg))
                logger.debug(f"UNK_FUNC detected with {len(expr.args)} args")
            else:
                # Regular unknown function: f(x) → UNK f x
                tokens = ["UNK", func_name]
                for arg in expr.args:
                    tokens.extend(_sympy_to_prefix_tokens(arg))
                logger.debug(f"Unknown function detected: {func_name} with {len(expr.args)} args")
            return tokens

    # Fallback: try to stringify
    logger.warning(f"Unknown SymPy expression type: {type(expr)} for {expr}")
    return [str(expr)]


def prefix_to_sympy(prefix_str: str) -> Expr:
    """
    Convert prefix notation to SymPy expression.

    Uses the reference implementation's conversion logic.
    See: reference/hongbozheng-transformer/convert.py

    Args:
        prefix_str: Space-separated prefix notation

    Returns:
        SymPy expression
    """
    # TODO: implement prefix → SymPy conversion
    # For now, import from reference if needed
    raise NotImplementedError("prefix_to_sympy not yet implemented - use reference/hongbozheng-transformer/convert.py")


# Example usage and testing
if __name__ == "__main__":
    test_cases = [
        "x**2",
        "sin(x)",
        "sin(x)*cos(x)",
        "x**2 + sin(x)",
        "log(x + 1)",
        "1/(x + 1)",
        "sqrt(x)",
        "x**(-1)",
        "2*x + 3",
        "-x",
        "polylog(2, x)",
        "polylog(3, 1/x)",
    ]

    print("SymPy → Prefix Notation Conversion Tests:")
    print("=" * 60)

    for expr_str in test_cases:
        try:
            expr = sp.sympify(expr_str)
            prefix = sympy_to_prefix(expr)
            print(f"SymPy: {expr_str:25} → Prefix: {prefix}")
        except Exception as e:
            print(f"ERROR: {expr_str:25} → {e}")

    # Test unknown function
    print("\nUnknown Function Tests:")
    print("=" * 60)
    f = sp.Function('f')
    g = sp.Function('g')
    test_unknown = [
        f(sp.Symbol('x')),
        g(sp.Symbol('x'), sp.Symbol('y')),
        f(sp.sin(sp.Symbol('x'))),
    ]
    for expr in test_unknown:
        prefix = sympy_to_prefix(expr)
        print(f"SymPy: {str(expr):25} → Prefix: {prefix}")
