import re
REL_TOKENS = [
    r'\\stackrel\{[^}]*\}\{=\}',
    r'\\stackrel\?=',
    r'\\coloneqq(?![a-zA-Z])',
    r'\\coloneq(?![a-zA-Z])',
    r'\\eqqcolon(?![a-zA-Z])',
    r'\\eqcolon(?![a-zA-Z])',
    r'\\triangleq(?![a-zA-Z])',
    r'\\equiv(?![a-zA-Z])',
    r'\\defeq(?![a-zA-Z])',
    r'\\doteq(?![a-zA-Z])',
    r'\\approx(?![a-zA-Z])',
    r'≈',
    r'='
]
APPROX_TOKENS = [r'\\approx', r'â‰ˆ', r'\\sim', r'\\cong', r'\\approxeq']
INEQUALITY_TOKENS = [
    r'\\leq(?![a-zA-Z])', r'\\le(?![a-zA-Z])', r'\\geq(?![a-zA-Z])', r'\\ge(?![a-zA-Z])', 
    r'\\neq(?![a-zA-Z])', r'\\ne(?![a-zA-Z])',
    r'\\ll(?![a-zA-Z])', r'\\gg(?![a-zA-Z])', r'\\prec(?![a-zA-Z])', r'\\succ(?![a-zA-Z])', 
    r'\\preceq(?![a-zA-Z])', r'\\succeq(?![a-zA-Z])',
    r'\\subset(?![a-zA-Z])', r'\\supset(?![a-zA-Z])', r'\\subseteq(?![a-zA-Z])', r'\\supseteq(?![a-zA-Z])',
    r'<', r'>', r'≤', r'≥', r'≠', r'≪', r'≫'
]
DISPLAY_MATH_PATTERNS = [
    re.compile(r'\$\$(.+?)\$\$', re.DOTALL),  # $$...$$
    re.compile(r'\\\[(.+?)\\\]', re.DOTALL),  # \[...\]
    re.compile(r'\\begin\{align\*?\}(.+?)\\end\{align\*?\}', re.DOTALL),  # align environments
    re.compile(r'\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}', re.DOTALL),  # equation environments
]
INLINE_DOLLAR_PATTERN = re.compile(r'\$(.+?)\$', re.DOTALL)
DISPLAY_DOLLAR_PATTERN = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
INVISIBLE_CHARS = [
    '\u200b',  # zero width space
    '\u200c',  # zero width non-joiner
    '\u200d',  # zero width joiner
    '\uFEFF',  # BOM
]
SIZE_COMMANDS = [
    "tiny", "scriptsize", "footnotesize", "small", "normalsize",
    "large", "Large", "LARGE", "huge", "Huge"
]
CONNECTIVE_PATTERNS = [
    r'\s*\\qquad\s*\\text\s*\{\s*and\s*\}\s*\\qquad\s*',
    r'\s*\\quad\s*\\text\s*\{\s*and\s*\}\s*\\quad\s*',
    r'\s*\\text\s*\{\s*and\s*\}\s*',
    r'\s*\band\b\s*',
    r'\s*\\text\s*\{\s*or\s*\}\s*',
]
REPETITIVE_SYMBOL_PATTERNS = [
    r'(\\gt;|&gt;|\\&gt;|>){4,}',
    r'(\\lt;|&lt;|\\&lt;|<){4,}',
    r'[=]{3,}',
    r'[.]{4,}',
    r'[-]{5,}',
    r'[*]{3,}',
]
PIECEWISE_INDICATORS = [
            r'\\text\s*\{\s*if\s',
            r'\\text\s*\{\s*for\s',
            r'\\text\s*\{\s*when\s',
            r'\\text\s*\{\s*otherwise',
            r'&\s*\\text\s*\{.*(?:if|for|when|otherwise)',
            r'\\\\.*\\text\s*\{.*(?:if|for|when|otherwise)',
]

# 1D Integral detection patterns with capture groups for bounds
INTEGRAL_1D_PATTERNS = [
    # Definite integrals with bounds - lower first, then upper
    r'\\int_\{([^}]+)\}\^\{([^}]+)\}',   # \int_{a}^{b} -> groups: (a, b)
    r'\\int_([^{^}]+)\^([^{^}]+)',        # \int_a^b -> groups: (a, b)
    r'\\int_\{([^}]+)\}\^([^{^}]+)',      # \int_{a}^b -> groups: (a, b)
    r'\\int_([^{^}]+)\^\{([^}]+)\}',      # \int_a^{b} -> groups: (a, b)
    
    # Definite integrals with bounds - upper first, then lower (note: swapped in groups)
    r'\\int\^\{([^}]+)\}_\{([^}]+)\}',   # \int^{b}_{a} -> groups: (b, a) - need to swap
    r'\\int\^([^{_}]+)_([^{_}]+)',        # \int^b_a -> groups: (b, a) - need to swap
    r'\\int\^\{([^}]+)\}_([^{_}]+)',      # \int^{b}_a -> groups: (b, a) - need to swap
    r'\\int\^([^{_}]+)_\{([^}]+)\}',      # \int^b_{a} -> groups: (b, a) - need to swap
    
    # Definite integrals with spaces before bounds
    r'\\int\s+_\{([^}]+)\}\^\{([^}]+)\}', # \int _{a}^{b} -> groups: (a, b)
    r'\\int\s+_([^{^}]+)\^([^{^}]+)',     # \int _a^b -> groups: (a, b)
    r'\\int\s+\^\{([^}]+)\}_\{([^}]+)\}', # \int ^{b}_{a} -> groups: (b, a) - need to swap
    r'\\int\s+\^([^{_}]+)_([^{_}]+)',     # \int ^b_a -> groups: (b, a) - need to swap
    
    # Indefinite integrals (no bounds) - only after checking all definite patterns
    r'\\int(?!\s*[_{^}])',            # \int not followed by bounds (with optional space)
]

# Integral canonicalization patterns - convert to standard \int_{lower}^{upper} format
INTEGRAL_CANONICALIZATION_PATTERNS = [
    # Mixed formats: \int_a^{b}, \int_{a}^b -> \int_{a}^{b} 
    # Fixed: Use more restrictive patterns to avoid capturing fractions
    (r'\\int_([^{^}\s\\]+)\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\1}^{\2}'),      # \int_a^{b} - exclude backslash
    (r'\\int_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\^([^{^}\s\\]+)', r'\\int_{\1}^{\2}'),      # \int_{a}^b - exclude backslash
    (r'\\int_([^{^}\s\\]+)\^([^{^}\s\\]+)', r'\\int_{\1}^{\2}'),                       # \int_a^b simple - exclude backslash
    
    # Reverse order: \int^{b}_{a} -> \int_{a}^{b}
    (r'\\int\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\2}^{\1}'),  # nested braces
    (r'\\int\^([^{_}\s\\]+)_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\2}^{\1}'),  # \int^b_{a} - exclude backslash
    (r'\\int\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}_([^{_}\s\\]+)', r'\\int_{\2}^{\1}'),  # \int^{b}_a - exclude backslash  
    (r'\\int\^([^{_}\s\\]+)_([^{_}\s\\]+)', r'\\int_{\2}^{\1}'),                   # \int^b_a simple - exclude backslash
    
    # Spaced formats: \int _a^b -> \int_{a}^{b}
    (r'\\int\s+_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\1}^{\2}'),  # nested braces
    (r'\\int\s+_([^{^}\s\\]+)\^([^{^}\s\\]+)', r'\\int_{\1}^{\2}'),                # \int _a^b simple - exclude backslash
    (r'\\int\s+_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\^([^{^}\s\\]+)', r'\\int_{\1}^{\2}'),  # \int _{a}^b - exclude backslash
    (r'\\int\s+_([^{^}\s\\]+)\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\1}^{\2}'),  # \int _a^{b} - exclude backslash
    
    # Spaced reverse order: \int ^{b}_{a} -> \int_{a}^{b}
    (r'\\int\s+\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\2}^{\1}'),  # nested braces
    (r'\\int\s+\^([^{_}\s\\]+)_([^{_}\s\\]+)', r'\\int_{\2}^{\1}'),                # \int ^b_a simple - exclude backslash
    (r'\\int\s+\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}_([^{_}\s\\]+)', r'\\int_{\2}^{\1}'),  # \int ^{b}_a - exclude backslash
    (r'\\int\s+\^([^{_}\s\\]+)_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\2}^{\1}'),  # \int ^b_{a} - exclude backslash
    
    # Single bounds (subscript only) - ensure braces  
    (r'\\int_([^{^}\s\\]+)(?!\^)', r'\\int_{\1}'),                                 # \int_a -> \int_{a} (no superscript) - exclude backslash
]

# Differential patterns - normalize formatted differentials to simple d+variable
DIFFERENTIAL_PATTERNS = [
    r'\\(?:textrm|mathrm|text)\s*\{\s*d\s*\}\s*([a-zA-Z]+)',       # \textrm{d}x, \mathrm{d}x, \text{d}x → dx
    r'\\(?:textrm|mathrm|text)\s*\{\s*d([a-zA-Z]+)\s*\}',          # \textrm{dx}, \mathrm{dx}, \text{dx} → dx
    r'\\,d([a-zA-Z]+)',                                             # \,dx (with thin space) → dx
]

# Special integral bounds patterns
MATHBB_R_PATTERNS = [
    r'\\mathbb\{R\}',      # \mathbb{R}
    r'\\mathbb R',         # \mathbb R (without braces)
    r'mathbb\{R\}',        # mathbb{R} (missing backslash)
    r'mathbb R',           # mathbb R (missing backslash, no braces)
]

# Combined pattern for detecting \mathbb{R} in integral bounds
MATHBB_R_BOUND_PATTERN = r'\\int_\{?(?:' + '|'.join(MATHBB_R_PATTERNS) + r')\}?'

# More refined patterns for checking bound content
MATHBB_R_BOUND_CONTENT_PATTERNS = [
    r'\\mathbb\{R\}',      # \mathbb{R}
    r'\\mathbb R',         # \mathbb R (without braces)  
    r'mathbb\{R\}',        # mathbb{R} (missing backslash)
    r'mathbb R',           # mathbb R (missing backslash, no braces)
]


# Multi-dimensional integrals to exclude
INTEGRAL_MULTIDIM_EXCLUDE = [
    r'\\iint',      # Double integrals
    r'\\iiint',     # Triple integrals  
    r'\\oiint',     # Surface integrals
    r'\\oiiint',    # Volume integrals
]

# Variable assignment patterns (I = \int, J = \int, etc.) - only I and J
INTEGRAL_VARIABLE_PATTERNS = [
    r'[IJ]\s*=\s*\\int',                     # I = \int, J = \int
    r'[IJ]_\{[^}]+\}\s*=\s*\\int',          # I_{n} = \int, J_{n} = \int
    r'[IJ]_[a-zA-Z0-9]\s*=\s*\\int',        # I_n = \int, J_n = \int
    r'[IJ]\([^)]*\)\s*=\s*\\int',           # I(x) = \int, J(x) = \int
]
MATRIX_PATTERNS = [
        r'\\begin\{pmatrix\}',
        r'\\begin\{bmatrix\}',
        r'\\begin\{vmatrix\}',
        r'\\begin\{matrix\}',
]
CONDITION_MARKERS = [
        r'\\text\s*\{\s*for\s*\}',
        r'\\text\s*\{\s*when\s*\}',
        r'\\text\s*\{\s*if\s*\}',
        r'\\text\s*\{\s*where\s*\}',
        r'\\text\s*\{\s*provided\s*\}',
        r'\\text\s*\{\s*given\s*\}',
        r'\\text\s*\{\s*with\s*\}',
]
IMPLICATION_PATTERNS = [
        r'\\mathop\\implies',
        r'\\Rightarrow',
        r'\\Leftarrow',
        r'\\Leftrightarrow',
        r'\\iff',
]
ARG_TAKING_COMMANDS = {
    'frac', 'tfrac', 'dfrac', 'binom', 'choose', 'sqrt',
    'text', 'operatorname', 'underbrace', 'overline', 'stackrel',
    'overset', 'left', 'right', 'vphantom', 'hphantom', 'phantom',
    'raisebox', 'color', 'mathrm', 'mathbf', 'mathbb', 'mathcal',
}
REL_TOKENS = sorted(REL_TOKENS, key=len, reverse=True)
INEQUALITY_TOKENS = sorted(INEQUALITY_TOKENS, key=len, reverse=True)
ALL_RELATION_TOKENS = REL_TOKENS + INEQUALITY_TOKENS
ALL_RELATION_TOKENS = sorted(ALL_RELATION_TOKENS, key=len, reverse=True)
REL_PATTERNS = [re.compile(tok) for tok in REL_TOKENS]
APPROX_PATTERNS = [re.compile(tok) for tok in APPROX_TOKENS]
INEQUALITY_PATTERNS = [re.compile(tok) for tok in INEQUALITY_TOKENS]
ALL_RELATION_PATTERNS = [re.compile(tok) for tok in ALL_RELATION_TOKENS]
ENV_PATTERN = re.compile(
    r'\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?)\}(.*?)\\end\{\1\}',
    re.DOTALL
)
# DISPLAY_PATTERN = re.compile(r'(?:\$\$|\\\[)(.*?)(?:\$\$|\\\])', re.DOTALL)
# INLINE_PATTERN = re.compile(r'\$([^$]*?)\$', re.DOTALL)
RE_MULTISPACE = re.compile(r'\s+')
RE_SPACE_BEFORE_BRACE = re.compile(r'\s*{\s*')
RE_SPACE_AFTER_BRACE = re.compile(r'\s*}\s*')
RE_BACKSLASH_SPACE = re.compile(r'\\\s+([A-Za-z])')
RE_TEXT = re.compile(r'\\text\s*{([^}]*)}')
WS_TOKENS = re.compile(r'^(?:\s|\\;|\\,|\\quad|\\qquad|\\vspace\{[^}]*\})*$')
TAG_PATTERN = re.compile(r'\\tag\{[^}]*\}|\\tag[^\s\\]*')
SIZE_WRAPPER_PATTERN = re.compile(r'\{\\(?:tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\s*(.*?)\}', re.DOTALL)
SIZE_COMMAND_PATTERN = re.compile(r'\\(?:tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\s*')
REQUIRE_PATTERN = re.compile(r'\\require\{[^}]*\}')
DIFFERENTIAL_PATTERN = re.compile(r'\\d([xyztuvsw])\b')
TEX_ENV_PATTERN = re.compile(r'\\(eqalign|align|eqalignno|leqalignno)\{(.*)\}', re.DOTALL)
ALIGNMENT_MARKER_PATTERN = re.compile(r'&.*=')
REPETITIVE_SYMBOL_COMPILED = [re.compile(p) for p in REPETITIVE_SYMBOL_PATTERNS]
CONNECTIVE_COMPILED = [re.compile(p, re.IGNORECASE) for p in CONNECTIVE_PATTERNS]