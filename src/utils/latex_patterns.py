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

INTEGRAL_CANONICALIZATION_PATTERNS = [
    (r'\\int_([^{^}\s\\]+)\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\1}^{\2}'),      # \int_a^{b}
    (r'\\int_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\^([^{^}\s\\]+)', r'\\int_{\1}^{\2}'),      # \int_{a}^b
    (r'\\int_([^{^}\s\\]+)\^([^{^}\s\\]+)', r'\\int_{\1}^{\2}'),                       # \int_a^b
    (r'\\int\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\2}^{\1}'),
    (r'\\int\^([^{_}\s\\]+)_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\2}^{\1}'),  # \int^b_{a}
    (r'\\int\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}_([^{_}\s\\]+)', r'\\int_{\2}^{\1}'),  # \int^{b}_a
    (r'\\int\^([^{_}\s\\]+)_([^{_}\s\\]+)', r'\\int_{\2}^{\1}'),                   # \int^b_a
    (r'\\int\s+_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\1}^{\2}'),
    (r'\\int\s+_([^{^}\s\\]+)\^([^{^}\s\\]+)', r'\\int_{\1}^{\2}'),                # \int _a^b
    (r'\\int\s+_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\^([^{^}\s\\]+)', r'\\int_{\1}^{\2}'),  # \int _{a}^b
    (r'\\int\s+_([^{^}\s\\]+)\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\1}^{\2}'),  # \int _a^{b}
    (r'\\int\s+\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\2}^{\1}'),
    (r'\\int\s+\^([^{_}\s\\]+)_([^{_}\s\\]+)', r'\\int_{\2}^{\1}'),                # \int ^b_a
    (r'\\int\s+\^\{([^}]*(?:\{[^}]*\}[^}]*)*)\}_([^{_}\s\\]+)', r'\\int_{\2}^{\1}'),  # \int ^{b}_a
    (r'\\int\s+\^([^{_}\s\\]+)_\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', r'\\int_{\2}^{\1}'),  # \int ^b_{a}
    (r'\\int_([^{^}\s\\]+)(?!\^)', r'\\int_{\1}'),                                 # \int_a -> \int_{a}
]

DIFFERENTIAL_PATTERNS = [
    r'\\(?:textrm|mathrm|text)\s*\{\s*d\s*\}\s*([a-zA-Z]+)',       # \textrm{d}x, \mathrm{d}x, \text{d}x → dx
    r'\\(?:textrm|mathrm|text)\s*\{\s*d([a-zA-Z]+)\s*\}',          # \textrm{dx}, \mathrm{dx}, \text{dx} → dx
    r'\\,d([a-zA-Z]+)',                                             # \,dx (with thin space) → dx
]
INTEGRAL_MULTIDIM_EXCLUDE = [
    r'\\iint',
    r'\\iiint',
    r'\\oiint',
    r'\\oiiint',
]
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