#  based on hongbozheng/transformer/convert.py
import sympy as sp
from sympy import Expr, Number, Symbol, Integer, Rational, Float, Add, Mul, Pow
from typing import List
import logging

from src.models.egen.vocab import SYMPY_TO_PREFIX, PREFIX_ARITY

logger = logging.getLogger(__name__)

def _handle_number(num: Number) -> List[str]:
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
        # num.p and num.q return Python ints, so wrap in Integer() for SymPy type
        numer_tokens = _handle_number(Integer(num.p))
        denom_tokens = _handle_number(Integer(num.q))
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
    tokens = _sympy_to_prefix_tokens(expr)
    return " ".join(tokens)


def _sympy_to_prefix_tokens(expr: Expr) -> List[str]:
    if isinstance(expr, Symbol):
        return [str(expr)]
    if isinstance(expr, Number):
        return _handle_number(expr)
    if expr == sp.pi:
        return ["pi"]
    if expr == sp.E:
        return ["e"]
    expr_type = type(expr)
    if expr_type in SYMPY_TO_PREFIX:
        op_name = SYMPY_TO_PREFIX[expr_type]
        if expr_type == Add:
            # Add: collect all addends
            # for n-ary add, convert to binary: add a (add b c)
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
            # check for special cases like -1*x (negation)
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
            base, exp = expr.args
            # special case: sqrt(x) = x^(1/2)
            if exp == sp.Rational(1, 2):
                return ["sqrt"] + _sympy_to_prefix_tokens(base)
            # all other powers use standard pow notation
            else:
                return [op_name] + _sympy_to_prefix_tokens(base) + _sympy_to_prefix_tokens(exp)
        elif expr_type == sp.polylog:
            # polylog(order, arg) → Li order arg
            order, arg = expr.args
            return [op_name] + _sympy_to_prefix_tokens(order) + _sympy_to_prefix_tokens(arg)
        else:
            # generic unary/n-ary function
            tokens = [op_name]
            for arg in expr.args:
                tokens.extend(_sympy_to_prefix_tokens(arg))
            return tokens
    # check for function by func attribute
    if hasattr(expr, 'func'):
        func = expr.func
        # known function
        if func in SYMPY_TO_PREFIX:
            op_name = SYMPY_TO_PREFIX[func]
            tokens = [op_name]
            for arg in expr.args:
                tokens.extend(_sympy_to_prefix_tokens(arg))
            return tokens
    # fallback: unknown expression type
    logger.warning(f"Unknown SymPy expression type: {type(expr)} for {expr}")
    return [str(expr)]


def _write_infix(token: str, args: List[str]) -> str:
    if token == 'add':
        return f'({args[0]})+({args[1]})'
    elif token == 'sub':
        return f'({args[0]})-({args[1]})'
    elif token == 'mul':
        return f'({args[0]})*({args[1]})'
    elif token == 'div':
        return f'({args[0]})/({args[1]})'
    elif token == 'pow':
        return f'({args[0]})**({args[1]})'
    elif token == 'inv':
        return f'1/({args[0]})'
    elif token == 'pow2':
        return f'({args[0]})**2'
    elif token == 'pow3':
        return f'({args[0]})**3'
    elif token == 'pow4':
        return f'({args[0]})**4'
    elif token == 'pow5':
        return f'({args[0]})**5'
    elif token == 'sqrt':
        return f'sqrt({args[0]})'
    elif token == 'abs':
        return f'Abs({args[0]})'
    elif token in ['exp', 'log', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                   'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
                   'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
                   'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']:
        return f'{token}({args[0]})'
    elif token == 'Li':
        return f'polylog({args[0]}, {args[1]})'
    elif token == 'zeta':
        return f'zeta({args[0]})'
    elif token.startswith('INT'):
        return token
    else:
        return token


def _parse_int(tokens: List[str]) -> tuple[int, int]:
    if not tokens or tokens[0] not in ['INT+', 'INT-']:
        raise ValueError(f"expected INT+ or INT- token, got: {tokens[0] if tokens else 'empty'}")

    sign = 1 if tokens[0] == 'INT+' else -1
    val = 0
    i = 1
    for token in tokens[1:]:
        if token.isdigit():
            val = val * 10 + int(token)
            i += 1
        else:
            break
    return sign * val, i


def _prefix_to_infix(tokens: List[str]) -> tuple[str, List[str]]:
    if not tokens:
        raise ValueError("empty token list in prefix_to_infix")
    op = tokens[0]
    CONSTANTS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'pi', 'e'}
    VARIABLES = {'x', 'a', 'b', 'c', 's', 't', 'u', 'v', 'w', 'y', 'z'}
    if op in PREFIX_ARITY:
        # operator: recursively parse arguments
        arity = PREFIX_ARITY[op]
        args = []
        remaining = tokens[1:]
        for _ in range(arity):
            arg_infix, remaining = _prefix_to_infix(remaining)
            args.append(arg_infix)
        return _write_infix(op, args), remaining
    elif op in CONSTANTS or op in VARIABLES:
        # constant or variable: return as-is
        return op, tokens[1:]
    elif op in ['INT+', 'INT-']:
        # multi-digit integer
        val, pos = _parse_int(tokens)
        return str(val), tokens[pos:]
    else:
        # unknown token: assume it's a variable or constant
        logger.warning(f"unknown token in prefix notation: {op}")
        return op, tokens[1:]


def prefix_to_sympy(prefix_str: str) -> Expr:
    tokens = prefix_str.strip().split()
    if not tokens:
        raise ValueError("empty prefix expression")
    infix_str, remaining = _prefix_to_infix(tokens)
    if remaining:
        raise ValueError(f"incomplete parse of prefix expression: {prefix_str}\nremaining tokens: {remaining}")
    local_dict = {
        'x': sp.Symbol('x'),
        'a': sp.Symbol('a'),
        'b': sp.Symbol('b'),
        'c': sp.Symbol('c'),
        's': sp.Symbol('s'),
        't': sp.Symbol('t'),
        'u': sp.Symbol('u'),
        'v': sp.Symbol('v'),
        'w': sp.Symbol('w'),
        'y': sp.Symbol('y'),
        'z': sp.Symbol('z'),
    }
    expr = sp.parsing.sympy_parser.parse_expr(
        f'({infix_str})',
        local_dict=local_dict,
        evaluate=True
    )
    return expr
