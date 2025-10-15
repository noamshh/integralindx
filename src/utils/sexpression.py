from typing import List

from src.models.egen.vocab import PREFIX_TO_SEXP, PREFIX_ARITY


def prefix_to_sexp(expr: str) -> str:
    tokens = expr.split()
    result, _ = _prefix_to_sexp_recursive(tokens, 0)
    return result


def _prefix_to_sexp_recursive(tokens: List[str], idx: int) -> tuple[str, int]:
    if idx >= len(tokens):
        raise ValueError("unexpected end of expression")
    token = tokens[idx]
    if token in PREFIX_ARITY:
        arity = PREFIX_ARITY[token]
        symbol = PREFIX_TO_SEXP.get(token, token)
        args = []
        next_idx = idx + 1
        for _ in range(arity):
            arg, next_idx = _prefix_to_sexp_recursive(tokens, next_idx)
            args.append(arg)
        return f"({symbol} {' '.join(args)})", next_idx
    elif token in ['INT+', 'INT-']:
        sign = 1 if token == 'INT+' else -1
        value = 0
        next_idx = idx + 1
        while next_idx < len(tokens) and tokens[next_idx].isdigit():
            value = value * 10 + int(tokens[next_idx])
            next_idx += 1
        return str(sign * value), next_idx
    else:
        return token, idx + 1
