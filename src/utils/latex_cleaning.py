import re
from typing import List
from src.utils.latex_patterns import (
    INVISIBLE_CHARS, ENV_PATTERN, TEX_ENV_PATTERN, DISPLAY_DOLLAR_PATTERN,
    INLINE_DOLLAR_PATTERN, TAG_PATTERN, REQUIRE_PATTERN, DIFFERENTIAL_PATTERN,
    RE_MULTISPACE, RE_SPACE_BEFORE_BRACE, RE_SPACE_AFTER_BRACE,
    RE_BACKSLASH_SPACE, RE_TEXT, APPROX_PATTERNS, INTEGRAL_CANONICALIZATION_PATTERNS,
    DIFFERENTIAL_PATTERNS
)
from src.utils.latex_parsing import (
    _unwrap_size_wrappers, _canonicalize_unbraced_frac_commands,
    _fix_unbraced_fractions, _fix_unbraced_scripts, _remove_limits_commands,
    _match_brace, _canonicalize_over_to_frac
)


def _sanitize_invisible(s: str) -> str:
    """Remove invisible Unicode characters."""
    if not s:
        return s
    for ch in INVISIBLE_CHARS:
        if ch in s:
            s = s.replace(ch, '')
    s = s.replace('\u00A0', ' ')
    return s


def _strip_color_commands(s: str) -> str:
    """Remove \\color{...} commands."""
    if not s:
        return s
    result = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith(r'\color', i):
            j = i + len(r'\color')
            # Skip whitespace
            while j < n and s[j].isspace():
                j += 1
            # If followed by braces, consume the color argument
            if j < n and s[j] == '{':
                brace_end = _match_brace(s, j)
                if brace_end != -1:
                    # Skip the entire \color{...} construct
                    i = brace_end + 1
                    continue
            # If no braces, just skip the \color command
            i = j
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def _clean_alignment_markers(s: str) -> str:
    """Remove LaTeX alignment markers that interfere with relation detection."""
    if not s:
        return s
    # Remove alignment markers from start/end
    s = re.sub(r'^\s*&\s*', '', s)
    s = re.sub(r'\s*&\s*$', '', s)
    # Remove &= patterns (alignment markers)
    s = re.sub(r'&\s*=', '=', s)
    s = re.sub(r'&\s*([<>≤≥≠])', r'\1', s)
    # Replace internal alignment markers with spaces
    s = re.sub(r'\s*&\s*', ' ', s)
    return s


def _normalize_text_inside(match: re.Match) -> str:
    r"""Normalize whitespace inside \\text{...} commands."""
    inner = match.group(1)
    has_lead = bool(inner) and inner[0].isspace()
    has_trail = bool(inner) and inner[-1].isspace()
    collapsed = RE_MULTISPACE.sub(' ', inner).strip()
    if has_lead:
        collapsed = ' ' + collapsed
    if has_trail:
        collapsed = collapsed + ' '
    return r'\text{' + collapsed + '}'


def _strip_math_environments(s: str) -> str:
    """
    Remove display wrappers, environments, size wrappers, tags, and fix some spacing.
    Keep iterating until stable. Use safe size-wrapper unwrapping to avoid nested-brace bugs.
    """
    if not s:
        return s
    s = _sanitize_invisible(s)
    s = re.sub(r'%\s*\n\s*', ' ', s)
    s = re.sub(r'%(?=\s*[{}\\])', '', s)
    prev = None
    iter_i = 0
    s = _unwrap_size_wrappers(s)
    while prev != s:
        prev = s
        iter_i += 1
        s = ENV_PATTERN.sub(r'\2', s)
        s = TEX_ENV_PATTERN.sub(r'\2', s)
        s = DISPLAY_DOLLAR_PATTERN.sub(r'\1', s)
        s = INLINE_DOLLAR_PATTERN.sub(r'\1', s)
        s = _unwrap_size_wrappers(s)
        s = _strip_color_commands(s)
        s = RE_MULTISPACE.sub(' ', s)
    s = TAG_PATTERN.sub('', s)
    s = _clean_alignment_markers(s)
    s = _canonicalize_over_to_frac(s)
    s = _canonicalize_unbraced_frac_commands(s)
    s = REQUIRE_PATTERN.sub('', s)
    s = DIFFERENTIAL_PATTERN.sub(r'd\1', s)
    s = _normalize_differentials(s)
    s = _remove_limits_commands(s)
    s = _fix_unbraced_fractions(s)
    s = _fix_unbraced_scripts(s)
    s = s.replace('\u00A0', ' ')
    s = RE_MULTISPACE.sub(' ', s).strip()
    s = DIFFERENTIAL_PATTERN.sub(r'd\1', s)
    s = _normalize_differentials(s)
    s = _normalize_special_functions(s)
    s = _canonicalize_integral_bounds(s)
    s = _eliminate_remaining_artifacts(s)
    return s


def _normalize_differentials(s: str) -> str:
    """normalize all differential variations to simple dx, dy, etc."""
    if not s:
        return s
    
    # apply comprehensive differential patterns in order
    for pattern in DIFFERENTIAL_PATTERNS:
        s = re.sub(pattern, r'd\1', s)
    
    return s


def _normalize_logarithms(s: str) -> str:
    r"""Normalize all logarithms to \log format with proper bracing."""
    if not s:
        return s
    # Convert \ln to \log - use negative lookahead instead of word boundary
    s = re.sub(r'\\ln(?![a-zA-Z])', r'\\log', s)
    # Handle powers: \log^n2 -> \log(2)^n
    s = re.sub(r'\\log\^(\{[^}]+\}|\d+)\s*([0-9]+)', r'\\log(\2)^{\1}', s)  # \log^32 -> \log(2)^{3}
    s = re.sub(r'\\log\^(\{[^}]+\}|\d+)\s*([a-zA-Z])\b', r'\\log(\2)^{\1}', s)  # \log^2x -> \log(x)^{2}
    # Handle unbraced arguments: \log2 -> \log(2), \log x -> \log(x)
    s = re.sub(r'\\log\s*([0-9]+)', r'\\log(\1)', s)  # \log2 -> \log(2)
    s = re.sub(r'\\log\s*([a-zA-Z])\b', r'\\log(\1)', s)  # \log x -> \log(x)
    return s


def _normalize_special_functions(s: str) -> str:
    """Normalize special function notations."""
    if not s:
        return s
    s = _normalize_logarithms(s)
    trig_funcs = ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']
    hyp_funcs = ['sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth']
    s = _normalize_function_arguments(s, trig_funcs)
    s = _normalize_function_arguments(s, hyp_funcs)
    s = _normalize_inverse_notation(s)
    s = _normalize_function_powers(s)
    s = _standardize_function_representations(s)
    s = _normalize_function_evaluations(s)

    return s

def _normalize_function_arguments(s: str, function_names: List[str]) -> str:
    """Normalize function arguments by adding parentheses to unbraced arguments."""
    if not s:
        return s

    for func in function_names:
        # Use negative lookahead to prevent partial matches (e.g., \sin in \sinh)
        s = re.sub(rf'\\{func}(?![a-zA-Z])\s*([0-9]+[a-zA-Z])', rf'\\{func}(\1)', s)
        s = re.sub(rf'\\{func}(?![a-zA-Z])\s*([a-zA-Z])\b', rf'\\{func}(\1)', s)
    return s


def _normalize_inverse_notation(s: str) -> str:
    r"""Convert \sin^{-1} to \arcsin, etc."""
    if not s:
        return s

    inverse_map = {
        'sin': 'arcsin', 'cos': 'arccos', 'tan': 'arctan',
        'sec': 'arcsec', 'csc': 'arccsc', 'cot': 'arccot',
        'sinh': 'arcsinh', 'cosh': 'arccosh', 'tanh': 'arctanh'
    }

    for func, arc_func in inverse_map.items():
        # Handle various spacing and brace patterns
        # \sin^{-1} -> \arcsin (with optional whitespace)
        s = re.sub(rf'\\{func}(?![a-zA-Z])\s*\^\s*\{{\s*-1\s*\}}', rf'\\{arc_func}', s)
        # \sin^-1 -> \arcsin (with optional whitespace)
        s = re.sub(rf'\\{func}(?![a-zA-Z])\s*\^\s*-1\b', rf'\\{arc_func}', s)
        # \sin^(-1) -> \arcsin
        s = re.sub(rf'\\{func}(?![a-zA-Z])\s*\^\s*\(\s*-1\s*\)', rf'\\{arc_func}', s)

    return s


def _normalize_function_powers(s: str) -> str:
    r"""Convert \sin^2 x to \sin(x)^2, etc.

    FIXED: Avoid transforming nested functions like \ln^3(\sin(x)) incorrectly.
    Only transform when parentheses contain simple expressions (no LaTeX commands).

    Examples:
    - \sin^2 x -> \sin(x)^2 ✓
    - \sin^2(x + y) -> \sin(x + y)^2 ✓
    - \ln^3(\sin(x)) -> \ln^3(\sin(x)) ✓ (unchanged, prevents incorrect transformation)
    - \log^2(\frac{x}{2}) -> \log^2(\frac{x}{2}) ✓ (unchanged, contains LaTeX)
    """
    if not s:
        return s
    func_names = ['sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'log', 'ln']
    for func in func_names:
        # \sin^2 x -> \sin(x)^2 (single variable)
        s = re.sub(rf'\\{func}\^(\{{[^}}]+\}}|\d+)\s*([a-zA-Z])\b', rf'\\{func}(\2)^{{\1}}', s)

        # \sin^2(content) -> \sin(content)^2
        # BUT ONLY if content contains NO backslashes (LaTeX commands)
        # This prevents: \ln^3(\sin(x)) -> \ln(\sin(x)^3) [WRONG]
        # Allows: \sin^2(x + y) -> \sin(x + y)^2 [CORRECT]
        s = re.sub(rf'\\{func}\^(\{{[^}}]+\}}|\d+)(\([^\\)]*\))', rf'\\{func}\2^{{\1}}', s)
    return s


def _clean_side(s: str) -> str:
    """Clean and normalize one side of an equation."""
    if s is None:
        return ""
    s = _clean_alignment_markers(s)
    s = RE_MULTISPACE.sub(' ', s).strip()
    s = re.sub(r'\\\s+(?=[{}()])', '', s)
    s = RE_SPACE_BEFORE_BRACE.sub('{', s)
    s = RE_SPACE_AFTER_BRACE.sub('}', s)
    s = re.sub(r'\s*\(\s*', '(', s)
    s = re.sub(r'\s*\)\s*', ')', s)
    s = RE_BACKSLASH_SPACE.sub(r'\\\1', s)
    s = DIFFERENTIAL_PATTERN.sub(r'd\1', s)
    s = _normalize_differentials(s)
    s = RE_TEXT.sub(_normalize_text_inside, s)
    s = s.rstrip()
    if s and s[-1] in '.;,:' and (len(s) == 1 or s[-2] != '\\'):
        s = s[:-1].rstrip()
    return s



def _standardize_function_representations(s: str) -> str:
    """Consolidate all function representation variants."""
    if not s:
        return s

    function_standardizations = {
        'sech': '\\operatorname{sech}',
        'csch': '\\operatorname{csch}',
        'coth': '\\operatorname{coth}',
        'arccot': '\\operatorname{arccot}',
        'arcsec': '\\operatorname{arcsec}',
        'arccsc': '\\operatorname{arccsc}',
        'arctanh': '\\operatorname{arctanh}',
        'artanh': '\\operatorname{arctanh}',
        'arcsinh': '\\operatorname{arcsinh}',
        'arsinh': '\\operatorname{arcsinh}',
        'arccosh': '\\operatorname{arccosh}',
        'arcosh': '\\operatorname{arccosh}',
        'Li': '\\operatorname{Li}',
        'Si': '\\operatorname{Si}',
        'Ci': '\\operatorname{Ci}',
        'Ei': '\\operatorname{Ei}',
        'erf': '\\operatorname{erf}',
        'erfc': '\\operatorname{erfc}',
        'erfi': '\\operatorname{erfi}',
        'Re': '\\operatorname{Re}',
        'Im': '\\operatorname{Im}',
        'sgn': '\\operatorname{sgn}',
        'sinc': '\\operatorname{sinc}',
    }

    for func, standard in function_standardizations.items():
        escaped_standard = standard.replace('\\', r'\\')
        s = re.sub(rf'\\text\s*\{{\s*{re.escape(func)}\s*\}}', escaped_standard, s)

        if not func.startswith('\\'):
            s = re.sub(rf'\\{re.escape(func)}(?![a-zA-Z])', escaped_standard, s)

    operatorname_fixes = {
        'artanh': 'arctanh',
        'arsinh': 'arcsinh',
        'arcosh': 'arccosh',
        'arsech': 'arcsech',
        'arcsch': 'arccsch',
        'arcoth': 'arccoth',
    }
    for wrong_variant, correct_variant in operatorname_fixes.items():
        s = re.sub(rf'\\operatorname\{{\s*{re.escape(wrong_variant)}\s*\}}',
                   rf'\\operatorname{{{correct_variant}}}', s)
    return s


def _normalize_function_evaluations(s: str) -> str:
    """
    Brace-aware normalization of function-evaluation notation.
    Only convert braced single-argument forms to parentheses for a whitelist of
    true "function" commands (log, sin, Gamma, zeta, etc.). Avoid touching
    structural commands like \\frac, \\binom, \\sqrt, \\left, \\underbrace, etc.
    """
    if not s:
        return s
    # Whitelist of commands that we treat as mathematical functions and that
    # are safe to convert from \cmd{arg} -> \cmd(arg).
    WHITELIST = {
        'log', 'ln', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
        'arcsin', 'arccos', 'arctan', 'arcsec', 'arccsc', 'arccot',
        'sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth',
        'arcsinh', 'arccosh', 'arctanh',
        'exp', 'Gamma', 'gamma', 'zeta', 'psi', 'beta', 'Li',
        'erf', 'erfc', 'Ei', 'Si', 'Ci', 'eta', 'erfi', 'Re', 'Im',
        'sinc'
    }
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '\\' and i + 1 < n:
            # capture command name (letters only) or single-char commands
            j = i + 1
            if s[j].isalpha():
                while j < n and s[j].isalpha():
                    j += 1
                cmd_name = s[i + 1:j]  # without backslash
                cmd_token = s[i:j]     # with backslash
            else:
                # single-character command, copy verbatim
                out.append(s[i])
                i += 1
                continue
            # skip optional whitespace
            k = j
            while k < n and s[k].isspace():
                k += 1
            # Only act when followed by a braced group
            if k < n and s[k] == '{':
                end = _match_brace(s, k)
                if end != -1:
                    inner = s[k + 1:end]  # content inside { ... }
                    # if cmd is in whitelist -> convert braced arg to parentheses
                    # BUT do not convert when inner is empty
                    if cmd_name in WHITELIST and inner.strip():
                        # If inner already starts and ends with parentheses, keep them
                        inner_stripped = inner.strip()
                        if inner_stripped.startswith('(') and inner_stripped.endswith(')'):
                            out.append(cmd_token + inner_stripped)
                        else:
                            out.append(cmd_token + '(' + inner + ')')
                        i = end + 1
                        continue
                    # Special handling for \operatorname{...} -> keep as-is
                    if cmd_name == 'operatorname':
                        out.append(cmd_token)
                        out.append(s[k:end + 1])  # keep whole {\dots}
                        i = end + 1
                        continue
                    # For commands NOT in whitelist (structural), keep original braced group
                    out.append(cmd_token)
                    out.append(s[k:end + 1])
                    i = end + 1
                    continue
            # not followed by braced group — copy the command token only
            out.append(cmd_token)
            i = j
            continue
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)

def _canonicalize_integral_bounds(s: str) -> str:
    """
    Canonicalize integral bounds to consistent \int_{lower}^{upper} format.
    
    Handles complex bounds with nested braces like \int_0^{\frac{\pi}{2}}.
    Uses patterns from latex_patterns.py for consistency.
    """
    if not s or '\\int' not in s:
        return s
    
    # Apply canonicalization patterns in order (most specific first)
    for pattern, replacement in INTEGRAL_CANONICALIZATION_PATTERNS:
        s = re.sub(pattern, replacement, s)
    
    return s

def _eliminate_remaining_artifacts(s: str) -> str:
    """Remove/convert remaining normalization artifacts."""
    if not s:
        return s
    # Remove all size commands - sort by length descending to avoid partial matches
    size_commands = [
        'bigg', 'Bigg', 'biggl', 'biggr', 'Biggl', 'Biggr',
        'big', 'Big', 'bigl', 'bigr', 'Bigl', 'Bigr',
        'small', 'large', 'Large', 'LARGE', 'huge', 'Huge',
        'tiny', 'scriptsize', 'footnotesize', 'normalsize'
    ]
    # Sort by length descending to ensure longer patterns match first
    size_commands.sort(key=len, reverse=True)
    for size in size_commands:
        # Use word boundary to prevent partial matches (e.g., \big from \bigg)
        s = re.sub(rf'\\{re.escape(size)}(?![a-zA-Z])', '', s)
    
    # Remove phantom commands that interfere with parsing
    phantom_commands = ['vphantom', 'hphantom', 'phantom']
    for phantom in phantom_commands:
        s = re.sub(rf'\\{phantom}\s*\{{[^}}]*\}}', '', s)
    
    # Convert ONLY short function-like \text{} to \operatorname{}
    function_like_text = [
        'Li', 'Si', 'Ci', 'Ei', 'sgn', 'erf', 'erfc', 'erfi',
        'sech', 'csch', 'coth', 'arccot', 'arctanh', 'arcsinh',
        'arccosh', 'sinc', 'agm', 'Re', 'Im'
    ]
    for func in function_like_text:
        s = re.sub(rf'\\text\s*\{{\s*{re.escape(func)}\s*\}}', rf'\\operatorname{{{func}}}', s)
    # NOTE: do NOT strip \quad/\qquad here — keep them so split_main_from_conditions can detect spacing-based conditions.
    # Remove \mathrm{} wrappers
    s = re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', s)
    # Normalize \cdot spacing but do not accidentally match prefixes like \cdots
    s = re.sub(r'\\cdot(?![a-zA-Z])', r' \\cdot ', s)
    # Clean up multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s