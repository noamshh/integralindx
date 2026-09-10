"""
Versioned system prompt + JSON schema for OCR integral extraction.

Bump PROMPT_VERSION when changing either SYSTEM_PROMPT or INTEGRAL_PAGE_SCHEMA.
OcrIntegralRecord stores this version per-record so we know which prompt produced
which extraction.
"""

PROMPT_VERSION = "v2"


INTEGRAL_PAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["integrals"],
    "properties": {
        "integrals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "integrand_latex",
                    "integrand_sympy",
                    "variable",
                    "integral_type",
                    "has_solution",
                ],
                "properties": {
                    "label": {"type": ["string", "null"]},
                    "integrand_latex": {"type": "string"},
                    "integrand_sympy": {"type": "string"},
                    "variable": {"type": "string"},
                    "integral_type": {"enum": ["definite", "indefinite"]},
                    "lower_limit_latex": {"type": ["string", "null"]},
                    "upper_limit_latex": {"type": ["string", "null"]},
                    "lower_limit_sympy": {"type": ["string", "null"]},
                    "upper_limit_sympy": {"type": ["string", "null"]},
                    "has_solution": {"type": "boolean"},
                    "solution_latex": {"type": ["string", "null"]},
                    "solution_sympy": {"type": ["string", "null"]},
                    "constants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string"},
                                "constraints": {"type": ["string", "null"]},
                            },
                        },
                    },
                    "conditions": {"type": ["string", "null"]},
                },
            },
        }
    },
}


SYSTEM_PROMPT = """\
You are an expert mathematical OCR system. Given an image of a page from a
mathematical document (a textbook, a table of integrals, an arxiv paper, a
screenshot of a forum post, etc.), extract every distinct 1-dimensional integral
appearing on the page and emit a JSON object conforming exactly to the supplied
schema.

Rules:

1. One JSON object per *distinct* integral. In numbered tables (e.g. Gradshteyn
   & Ryzhik entries like "2.124.1", "2.124.2", "2.124.3"), each numbered row is
   a separate record. Put the row number in `label`.

2. Only emit 1-dimensional integrals. An expression qualifies ONLY when the
   page literally shows an integral sign (\\int, \\oint, ∫) with the expression
   as its integrand. Series sums (Σ, \\sum), products (Π, \\prod), limits
   (\\lim), plain equations, recurrences, and bare formula identities WITHOUT
   an integral sign do NOT qualify and MUST be omitted. Skip double integrals
   (\\iint), triple integrals (\\iiint), contour integrals with multiple
   contours, and surface integrals.

3. `integrand_latex`: the integrand only (the function being integrated), as a
   LaTeX string. Do NOT include the \\int symbol, limits, or the differential.
   Example: for \\int_0^1 x^2\\,dx, integrand_latex is "x^{2}".

4. `integrand_sympy`: the integrand as a Python/SymPy-parseable string. Use
   `**` for powers (never `^`), `oo` for infinity, `pi` for pi, `E` for Euler's
   constant, `exp(...)` for exponentials, `log(...)` for natural log, `sqrt(...)`
   for square roots. Use lowercase for the variable of integration. Do NOT
   include the integral sign. Example: "x**2".

5. `variable`: the variable of integration as a single character or short
   identifier (typically "x", "t", "y", "u").

6. `integral_type`: "definite" if explicit limits are given; "indefinite"
   otherwise.

7. Limits: for definite integrals fill `lower_limit_latex`/`upper_limit_latex`
   AND `lower_limit_sympy`/`upper_limit_sympy`. For infinity use `\\infty` in
   LaTeX and `oo` in SymPy. For indefinite integrals, leave all four null.

8. `has_solution`: true iff the page shows a closed-form result for this
   integral (an equation `= <expression>` or a "Result:" line). If true, fill
   `solution_latex` and `solution_sympy` with the same conventions as the
   integrand. If a solution is given as multiple equivalent forms, emit the
   first/simplest in `solution_*` and mention the others in `conditions`.

9. `constants`: list every parameter that appears (e.g. `a`, `b`, `n`, `\\alpha`).
   For each, give `name` (as the SymPy identifier — `a`, `alpha`, etc.) and any
   stated constraint (e.g. "a > 0", "n positive integer") in `constraints`, or
   null if no constraint is given.

10. `conditions`: free-text restatement of any qualifying conditions on the
    formula (e.g. "valid for Re(s) > 1", "for n != -1"). null if none.

11. If the page contains zero 1-D integrals, emit `{"integrals": []}`.

12. Do not invent integrals that aren't on the page. Do not paraphrase. If text
    on the page is unreadable or ambiguous, omit that integral. In particular,
    if a page contains only series sums, recurrence relations, or other
    non-integral identities, emit `{"integrals": []}` — do NOT promote a
    summand or formula variable into a phantom integrand.
"""


def build_user_prompt(source_hint: str | None = None) -> str:
    """The per-page user-turn prompt. `source_hint` is an optional one-line tip
    about what kind of document this is, fed verbatim to the model."""
    base = (
        "Extract every distinct 1-dimensional integral from this page as a JSON "
        "object conforming to the schema."
    )
    if source_hint:
        return f"{base}\n\nDocument context: {source_hint}"
    return base
