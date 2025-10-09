import logging
from typing import List, Set

logger = logging.getLogger(__name__)

# frozen global vocabulary reference: src/models/egen/vocab.py
# all tokens must be whitelisted in the global vocabulary
VALID_OPERATORS = {
    "add", "sub", "mul", "div", "pow", "inv", "sqrt",
    "pow2", "pow3", "pow4", "pow5",
    "exp", "ln", "abs", "sign",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
    # special functions (IntegralIndx extensions)
    "Li", "zeta", "re", "im",
}

VALID_CONSTANTS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "pi", "e", "INT+", "INT-"
}

VALID_VARIABLES = {
    "x", "a", "b", "c",
}

SPECIAL_TOKENS = {"PAD", "SOE", "EOE"}


def validate_prefix_expr(expr: str) -> bool:
    """
    validate that expression is valid prefix notation
    checks:
    - all tokens are known (operators, constants, variables)
    - no empty strings
    - basic structure is reasonable
    Args:
        expr: space-separated prefix notation string
    Returns:
        True if valid, False otherwise
    """
    if not expr or not expr.strip():
        logger.debug("empty expression")
        return False
    tokens = expr.strip().split()
    if not tokens:
        logger.debug("no tokens in expression")
        return False
    all_valid = (
        VALID_OPERATORS | VALID_CONSTANTS | VALID_VARIABLES | SPECIAL_TOKENS
    )
    for token in tokens:
        if token not in all_valid:
            logger.debug(f"invalid token: {token}")
            return False
    # basic structural check: should have at least one token
    if len(tokens) == 0:
        return False
    return True


def preprocess_for_egraph(expr: str) -> str:
    """
    Preprocess expression before sending to E-Gen binary
    the Rust E-Gen expects s-expression format: (op arg1 arg2)
    our prefix notation is: op arg1 arg2
    Args:
        expr: prefix notation expression
    Returns:
        s-expression format for Rust binary
    """
    # map our prefix operators to Rust's s-expression operators
    PREFIX_TO_SEXP = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "pow": "pow",
        "sqrt": "sqrt",
        "exp": "exp",
        "ln": "ln",
        "abs": "abs",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "cot": "cot",
        "sec": "sec",
        "csc": "csc",
    }

    tokens = expr.strip().split()
    if not tokens:
        return expr
    # simple conversion: recursively build s-expression
    def convert_tokens(idx: int) -> tuple[str, int]:
        """convert tokens starting at idx, return (s-expr, next_idx)"""
        if idx >= len(tokens):
            return "", idx
        token = tokens[idx]
        # if it's a constant or variable, return as-is
        if token in VALID_CONSTANTS or token in VALID_VARIABLES:
            return token, idx + 1
        # if it's an operator, recursively convert args
        if token in PREFIX_TO_SEXP:
            op = PREFIX_TO_SEXP[token]
            # binary operators need 2 args
            if token in {"add", "sub", "mul", "div", "pow"}:
                arg1, next_idx = convert_tokens(idx + 1)
                arg2, next_idx = convert_tokens(next_idx)
                return f"({op} {arg1} {arg2})", next_idx
            # unary operators need 1 arg
            else:
                arg, next_idx = convert_tokens(idx + 1)
                return f"({op} {arg})", next_idx
        # fallback: return as-is
        return token, idx + 1
    result, _ = convert_tokens(0)
    return result


def postprocess_egraph_output(exprs: List[str]) -> List[str]:
    """
    Postprocess E-Gen binary output
    converts s-expressions back to prefix notation
    cleans up formatting
    Args:
        exprs: list of s-expressions from Rust binary
    Returns:
        list of prefix notation expressions
    """
    # map Rust s-expression operators back to our prefix notation
    SEXP_TO_PREFIX = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "pow": "pow",
        "sqrt": "sqrt",
        "exp": "exp",
        "ln": "ln",
        "abs": "abs",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
    }
    def convert_sexp(sexp: str) -> str:
        """convert single s-expression to prefix notation"""
        sexp = sexp.strip()
        # remove outer parentheses
        if sexp.startswith('(') and sexp.endswith(')'):
            sexp = sexp[1:-1].strip()
        # split by whitespace
        tokens = sexp.split()
        if not tokens:
            return sexp
        # first token should be operator
        op = tokens[0]
        if op in SEXP_TO_PREFIX:
            prefix_op = SEXP_TO_PREFIX[op]
            # recursively convert remaining args
            args = tokens[1:]
            return f"{prefix_op} {' '.join(args)}"
        # fallback: return as-is
        return sexp
    result = []
    for expr in exprs:
        try:
            converted = convert_sexp(expr)
            result.append(converted)
        except Exception as e:
            logger.warning(f"failed to convert s-expression: {expr}\nerror: {e}")
            # keep original if conversion fails
            result.append(expr)
    return result


def deduplicate_equivalents(exprs: List[str]) -> List[str]:
    """Remove duplicate equivalent expressions (preserves order)"""
    seen: Set[str] = set()
    result = []
    for expr in exprs:
        normalized = expr.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    if len(result) < len(exprs):
        logger.debug(f"removed {len(exprs) - len(result)} duplicates")
    return result
