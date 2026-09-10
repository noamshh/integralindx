import time
from pathlib import Path

import pytest

from src.utils.safe_math import ExpressionTooComplexError, safe_sympify
from src.web.utils.query_parser import parse_query_to_sympy_integrand

SRC = Path(__file__).resolve().parents[1] / "src"

# Cheap to type, ruinous to evaluate: "9**9**8" is seven characters and runs for
# minutes. parse_expr computes while parsing and cannot be interrupted.
BLOWUPS = [
    "9**9**5",
    "9**9**7",
    "9**9**8",
    "9**9**9**9",
    "2**999999999",
    "x**(9**9**8)",
    "(2**500)**500",
    "((2**500)**500)**500",
]

LEGITIMATE = [
    "x**2",
    "sin(x)/x",
    "x**n/(x+1)",
    "sqrt(x)",
    "x**(1/2)",
    "x**(-1/2)",
    "x**(2/3)",
    "1/(x**2+1)",
    "exp(-a*x**2)",
    "log(x)*atan(x)",
    "a*x**2+b*x+c",
]


@pytest.mark.parametrize("expr", BLOWUPS)
def test_blowups_rejected_immediately(expr):
    start = time.perf_counter()
    with pytest.raises(ValueError):
        safe_sympify(expr)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize("expr", LEGITIMATE)
def test_legitimate_expressions_still_parse(expr):
    assert safe_sympify(expr) is not None


@pytest.mark.parametrize("expr", ["x;import os", "__import__('os')", "eval('1')", "x['a']"])
def test_disallowed_characters_rejected(expr):
    with pytest.raises(Exception):
        safe_sympify(expr)


def test_too_complex_is_a_value_error():
    with pytest.raises(ExpressionTooComplexError):
        safe_sympify("9**9**9")


@pytest.mark.parametrize("expr", BLOWUPS)
def test_query_path_rejects_blowups(expr):
    normalized, _ = parse_query_to_sympy_integrand(expr)
    assert normalized is None


@pytest.mark.parametrize(
    "query,expected",
    [
        ("t**2/(t+1)", "x**2/(x + 1)"),
        ("u**2/(u+1)", "x**2/(x + 1)"),
        ("x**2/(x+1)", "x**2/(x + 1)"),
        ("sin(u)/u", "sin(x)/x"),
        ("t**n/(t+1)", "x**a/(x + 1)"),
    ],
)
def test_integration_variable_is_canonicalised(query, expected):
    # safe_sympify carries assumptions (Symbol('t', real=True)); substituting with
    # a bare Symbol('t') silently misses and leaves the query uncanonicalised,
    # which would stop it matching anything in the index.
    normalized, _ = parse_query_to_sympy_integrand(query)
    assert normalized == expected


# Modules a POST /search can reach. Training and corpus-building code is
# excluded: it runs offline on trusted input, never on a request.
REQUEST_PATH = [
    SRC / "web",
    SRC / "search",
    SRC / "utils" / "variable_normalization.py",
    SRC / "utils" / "symbol_validation.py",
    SRC / "models" / "egen" / "contrastive_embedder.py",
]


def test_request_path_has_no_raw_sympify():
    files = []
    for target in REQUEST_PATH:
        files.extend(target.rglob("*.py") if target.is_dir() else [target])

    offenders = []
    for path in files:
        if "__pycache__" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "safe_sympify" in stripped:
                continue
            if "sympify(" in stripped:
                offenders.append(f"{path.relative_to(SRC)}:{lineno}")
    assert not offenders, f"raw sympify() reachable from a request: {offenders}"
