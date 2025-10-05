import re
from typing import Tuple, List, Optional
from src.utils.latex_patterns import (
    SIZE_COMMANDS, CONNECTIVE_COMPILED, REPETITIVE_SYMBOL_COMPILED,
    PIECEWISE_INDICATORS, MATRIX_PATTERNS, CONDITION_MARKERS, IMPLICATION_PATTERNS,
    ARG_TAKING_COMMANDS
)


def _match_brace(s: str, start_idx: int) -> int:
    """Return index of matching '}' given that s[start_idx] == '{'. -1 if not found."""
    n = len(s)
    depth = 0
    i = start_idx
    while i < n:
        c = s[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def _is_escaped(s: str, pos: int) -> bool:
    """
    Return True if character at s[pos] is escaped (i.e. preceded by an odd number of backslashes).
    pos is index of the character (e.g. comma). If pos == 0 -> False.
    """
    if pos <= 0:
        return False
    # count consecutive backslashes immediately before pos
    i = pos - 1
    count = 0
    while i >= 0 and s[i] == '\\':
        count += 1
        i -= 1
    return (count % 2) == 1


def _remove_limits_commands(s: str) -> str:
    """Remove \\limits commands in e.g. \\int_\\limits{...}"""
    if not s:
        return s
    s = re.sub(r'\\(int|sum|prod|oint|iint|iiint|bigcup|bigcap|bigoplus|bigotimes)_\\limits\s*\{', r'\\\1_{', s)
    s = re.sub(r'\\(int|sum|prod|oint|iint|iiint|bigcup|bigcap|bigoplus|bigotimes)_\\limits(?=\s|$|\^|_)', r'\\\1_', s)
    s = re.sub(r'\\(int|sum|prod|oint|iint|iiint|bigcup|bigcap|bigoplus|bigotimes)\\limits(?=\s|$|\^|_)', r'\\\1', s)
    s = re.sub(r'\\limits(?=\s|$|\^|_)', '', s)
    return s


def _unwrap_size_wrappers(s: str) -> str:
    """Replace occurrences of {\\small ... } with the inner text, but:
    - don't unwrap if they're arguments to commands like \\stackrel
    - keep braces for superscripts/subscripts: ^{\\small ...} -> ^{...}"""
    if not s:
        return s
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '{' and i + 1 < n and s[i + 1] == '\\':
            preceding_context = s[max(0, i - 20):i]
            if re.search(r'\\(?:stackrel|overset|frac|sqrt|text|vphantom|hphantom|phantom|smash|raisebox)\s*$',
                         preceding_context):
                end = _match_brace(s, i)
                if end != -1:
                    j = i + 2
                    while j < n and s[j].isalpha():
                        j += 1
                    cmd = s[i + 2:j]
                    if cmd in SIZE_COMMANDS:
                        inner_start = j
                        while inner_start < end and s[inner_start].isspace():
                            inner_start += 1
                        out.append('{' + s[inner_start:end] + '}')
                        i = end + 1
                        continue
                out.append(s[i])
                i += 1
                continue
            is_script = preceding_context.endswith('^') or preceding_context.endswith('_')
            j = i + 2
            while j < n and s[j].isalpha():
                j += 1
            cmd = s[i + 2:j]
            if cmd in SIZE_COMMANDS:
                end = _match_brace(s, i)
                if end != -1:
                    inner_start = j
                    while inner_start < end and s[inner_start].isspace():
                        inner_start += 1
                    content = s[inner_start:end]
                    if is_script:
                        out.append('{' + content + '}')
                    else:
                        out.append(content)
                    i = end + 1
                    continue
        out.append(s[i])
        i += 1
    return ''.join(out)


def _consume_command_with_arg(s: str, idx: int) -> Tuple[int, str]:
    """
    If s[idx] is at a backslash, try to consume a LaTeX command optionally followed
    by an optional [...] argument and a braced group {..}. Return (new_idx, token_string).
    Conservative: only treat the following {..} as the command's argument if the
    command is known to accept braced arguments (in _ARG_TAKING_COMMANDS).
    """
    n = len(s)
    if idx >= n or s[idx] != '\\':
        return idx, ''
    j = idx + 1
    # command name: sequence of letters (common case) OR a single non-letter char
    if j < n and s[j].isalpha():
        while j < n and s[j].isalpha():
            j += 1
        # allow trailing '*' (e.g. \section*)
        if j < n and s[j] == '*':
            j += 1
    else:
        j += 1
    cmd = s[idx:j]
    cmd_name = cmd[1:] if cmd.startswith('\\') else cmd
    k = j
    while k < n and s[k].isspace():
        k += 1
    if k < n and s[k] == '[':
        depth = 0
        m = k
        while m < n:
            if s[m] == '[':
                depth += 1
            elif s[m] == ']':
                depth -= 1
                if depth == 0:
                    m += 1
                    break
            m += 1
        if m <= n:
            k = m
        while k < n and s[k].isspace():
            k += 1
    if k < n and s[k] == '{' and cmd_name in ARG_TAKING_COMMANDS:
        end = _match_brace(s, k)
        if end != -1:
            return end + 1, s[idx:end + 1]
    return k, cmd


def _consume_denominator_token(s: str, idx: int) -> Tuple[int, str]:
    """
    Consume a "reasonable" denominator token starting at s[idx].
    Returns (new_index, token_string) where new_index is immediately after the consumed token.
    """
    n = len(s)
    i = idx
    while i < n and s[i].isspace():
        i += 1
    if i >= n:
        return i, ''
    # 1) braced group { ... }
    if s[i] == '{':
        end = _match_brace(s, i)
        if end != -1:
            return end + 1, s[i:end + 1]
        # malformed: return rest
        return n, s[i:]
    # 2) \left ... \right (best-effort) or plain parenthesis ( ... )
    if s.startswith(r'\left', i):
        # find the next \right and include any following closing paren
        right_pos = s.find(r'\right', i + 5)
        if right_pos != -1:
            j = right_pos + len(r'\right')
            # include any immediate closing ) ] } characters
            while j < n and s[j] in ')]} ':
                j += 1
            return j, s[i:j]
    if s[i] == '(':
        depth = 0
        j = i
        while j < n:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
                if depth == 0:
                    return j + 1, s[i:j + 1]
            j += 1
        return n, s[i:]  # malformed
    # 3) LaTeX command starting with backslash: consume robustly
    if s[i] == '\\':
        new_idx, token = _consume_command_with_arg(s, i)
        return new_idx, token or s[i:new_idx]
    # 4) Single-digit short form (common in \frac3 4 idiom)
    if s[i].isdigit():
        return i + 1, s[i]
    # 5) Identifier / number / short expression: read until delimiter
    delimiters = {',', ';', ':', '+', '-', '*', '/', '^', ')', ']', '}', '%', ' ', '.', '\\', '{', '}'}
    j = i
    while j < n and not s[j].isspace() and s[j] not in delimiters:
        if (j > i and s[j - 1].isalpha() and s[j].isdigit()):
            break
        if (j > i and s[j - 1].isdigit() and s[j].isalpha()):
            break
        if j + 1 < n and s[j] in 'dD' and s[j + 1] in 'xyztuv':
            break
        if s[j:j + 2] in ('dx', 'dy', 'dz', 'dt', 'du', 'dv'):
            break
        j += 1
    token = s[i:j]
    token = token.rstrip('.,;:')
    if not token:
        return j, ''
    return j, token


def _fix_unbraced_fractions_once(s: str) -> Tuple[str, bool]:
    """
    Do a single left-to-right pass over s and convert occurrences like
    \\frac{a}4  -> \\frac{a}{4}. Return (new_string, changed_flag).
    """
    n = len(s)
    i = 0
    out = []
    last_appended = 0
    changed = False
    while i < n:
        if s.startswith(r'\frac{', i) or s.startswith(r'\tfrac{', i) or s.startswith(r'\dfrac{', i):
            # Get the command length
            if s.startswith(r'\tfrac{', i):
                cmd_len = len(r'\tfrac')
            elif s.startswith(r'\dfrac{', i):
                cmd_len = len(r'\dfrac')
            else:
                cmd_len = len(r'\frac')
            # append any text since last_appended
            if last_appended < i:
                out.append(s[last_appended:i])
            # numerator start and end
            num_start = i + cmd_len
            if num_start >= n or s[num_start] != '{':
                # malformed, append the \frac token and advance
                out.append(s[i:i + cmd_len])
                i = num_start
                last_appended = i
                continue
            num_end = _match_brace(s, num_start)
            if num_end == -1:
                # malformed numerator: append rest and break
                out.append(s[i:])
                last_appended = n
                i = n
                break
            numerator = s[num_start:num_end + 1]  # includes braces
            # get denominator token (starting immediately after numerator)
            den_start = num_end + 1
            den_idx, denom_token = _consume_denominator_token(s, den_start)
            if not denom_token:
                # nothing looks like a denominator -> keep original \frac{...}
                out.append(s[i:i + cmd_len] + numerator)
                i = den_start
                last_appended = i
                continue
            # If denom already braced, include as-is; otherwise wrap
            if denom_token.startswith('{'):
                out.append(s[i:i + cmd_len] + numerator + denom_token)
            else:
                out.append(s[i:i + cmd_len] + numerator + '{' + denom_token + '}')
            # advance pointers to character after denom token
            i = den_idx
            last_appended = i
            changed = True
        else:
            i += 1
    # append the remainder
    if last_appended < n:
        out.append(s[last_appended:])
    new_s = ''.join(out)
    return new_s, changed


def _fix_unbraced_fractions(s: str, max_iters: int = 5) -> str:
    """Iteratively apply one-pass fixer until string is stable or max_iters reached."""
    current = s
    for _ in range(max_iters):
        new_s, changed = _fix_unbraced_fractions_once(current)
        if not changed or new_s == current:
            return new_s
        current = new_s
    return current

def _split_on_connectives(s: str) -> List[str]:
    """Split formulas joined by connective words like 'and', 'or', etc."""
    for pattern in CONNECTIVE_COMPILED:
        if pattern.search(s):
            parts = pattern.split(s)
            return [part.strip() for part in parts if part.strip()]
    return [s]

def _fix_unbraced_scripts(s: str) -> str:
    """Fix unbraced subscripts/superscripts"""
    if not s:
        return s
    s = re.sub(r'([\^_])\\(frac|tfrac|dfrac)\{([^}]*)\}\{([^}]*)\}', r'\1{\\\2{\3}{\4}}', s)
    s = re.sub(r'([\^_])\\sqrt\{([^}]*)\}', r'\1{\\sqrt{\2}}', s)
    s = re.sub(r'([\^_])\\(log|ln|sin|cos|tan|exp|operatorname)\{([^}]*)\}', r'\1{\\\2{\3}}', s)
    s = re.sub(r'([\^_])\\sqrt\s*(\d+)', r'\1{\\sqrt{\2}}', s)
    s = re.sub(r'([\^_])\\(log|ln|sin|cos|tan|exp)\s*(\d+)', r'\1{\\\2{\3}}', s)
    s = re.sub(r'([\^_])\\([a-zA-Z]+)\s*(?![a-zA-Z0-9{])', r'\1{\\\2}', s)
    return s

def _is_safe_boundary(s: str, pos: int) -> bool:
    """Check if position is a safe boundary for tokenization."""
    if pos >= len(s):
        return True
    ch = s[pos]
    if ch.isspace():
        return True
    if ch in ',;:+-*/^)]}%.$\\':
        return True
    return False

def _canonicalize_unbraced_frac_commands(s: str, max_iters: int = 5) -> str:
    """
    Turn occurrences like \\tfrac\\pi4  or \\dfrac\\alpha\\beta  or \\frac a b
    into the braced form \\tfrac{\\pi}{4} so later passes can rely on braced args.
    """
    if not s:
        return s

    def one_pass(text: str) -> Tuple[str, bool]:
        out = []
        i = 0
        n = len(text)
        changed = False
        while i < n:
            # match \tfrac, \dfrac or \frac that are NOT immediately followed by '{'
            if text.startswith(r'\tfrac', i) or text.startswith(r'\dfrac', i) or text.startswith(r'\frac', i):
                # determine command name length
                if text.startswith(r'\tfrac', i):
                    cmd = r'\tfrac'
                elif text.startswith(r'\dfrac', i):
                    cmd = r'\dfrac'
                else:
                    cmd = r'\frac'
                cmd_len = len(cmd)
                # check next non-space char
                j = i + cmd_len
                # if already braced, just copy verbatim
                if j < n and text[j] == '{':
                    out.append(text[i:i + cmd_len])  # append command only; keep rest for future
                    i += cmd_len
                    continue
                # try to parse a numerator token and a denominator token
                # numerator: parse token starting at j
                num_idx, num_tok = _consume_denominator_token(text, j)
                if not num_tok:
                    # nothing to consume, copy the command and advance
                    out.append(cmd)
                    i += cmd_len
                    continue
                # denominator: parse next token starting at num_idx
                den_idx, den_tok = _consume_denominator_token(text, num_idx)
                if not _is_safe_boundary(text, den_idx):
                    # refuse to canonicalize here – copy the command verbatim and advance one char
                    out.append(cmd)
                    i += cmd_len
                    continue
                # we found both -- produce braced form
                out.append(f"{cmd}{{{num_tok}}}{{{den_tok}}}")
                i = den_idx
                changed = True
            else:
                out.append(text[i])
                i += 1
        return ''.join(out), changed
    cur = s
    for _ in range(max_iters):
        cur2, changed = one_pass(cur)
        if not changed or cur2 == cur:
            return cur2
        cur = cur2
    return cur

def _split_respecting_environments(s: str) -> List[str]:
    """Split on \\\\, \\n, \\cr but respect \\begin{...}\\end{...} environments."""
    if not s:
        return []
    parts = []
    current = ""
    i = 0
    n = len(s)
    while i < n:
        # Check for \begin{env}
        if s.startswith(r'\begin{', i):
            env_match = re.match(r'\\begin\{([^\}]+)\}', s[i:])
            if env_match:
                env_name = env_match.group(1)
                end_pattern = r'\end{' + env_name + '}'
                end_pos = s.find(end_pattern, i + env_match.end())
                if end_pos != -1:
                    # Add entire environment to current part (don't split inside it)
                    current += s[i:end_pos + len(end_pattern)]
                    i = end_pos + len(end_pattern)
                    continue
        # Check for line break patterns
        if s[i:i + 2] == '\\\\' or s[i] == '\n' or s.startswith(r'\cr', i):
            # print(f"DEBUG: Found line break at position {i}")
            if s[i:i + 2] == '\\\\':
                next_start = i + 2
                # Skip whitespace
                while next_start < n and s[next_start].isspace():
                    next_start += 1
                if next_start < n:
                    next_part = s[next_start:next_start + 10]
                    # If next line starts with operator/sign, it's continuation
                    if next_part and next_part[0] in '-+':
                        current += ' '  # Replace \\ with space for continuation
                        i += 2
                        continue
                    # If next line looks like separate equation (has = sign early), split
                    elif '=' in s[next_start:next_start + 50]:
                        if current.strip():
                            parts.append(current.strip())
                        current = ""
                        i += 2
                        continue
            if current.strip():
                parts.append(current.strip())

            current = ""
            # Skip the break token
            if s[i:i + 2] == '\\\\':
                i += 2
                # Skip optional spacing like [3mm]
                while i < n and s[i].isspace():
                    i += 1
                if i < n and s[i] == '[':
                    end_bracket = s.find(']', i)
                    if end_bracket != -1:
                        i = end_bracket + 1
            elif s.startswith(r'\cr', i):
                i += 3
            else:  # \n
                i += 1
        else:
            current += s[i]
            i += 1
    if current.strip():
        parts.append(current.strip())
    return parts

def split_main_from_conditions(s: str) -> Tuple[str, Optional[str]]:
    """
    Split a formula into main part and conditions.
    Returns (main_part, conditions) where conditions is None if no conditions found.
    """
    # 1. Text-based condition markers
    for marker in CONDITION_MARKERS:
        match = re.search(marker, s, re.IGNORECASE)
        if match:
            main_part = s[:match.start()].strip()
            condition_part = s[match.start():].strip()
            return main_part, condition_part
    # 2. Parametric/labeling conditions with \quad - ADD THIS
    # Match patterns like "\quad N = 2", "\quad a > 0", "\quad a \in \mathbb{R}^+", etc.
    quad_condition_match = re.search(r'\\quad\s+([A-Za-z]\s*(?:=|>|<|≥|≤|\\in|\\geq|\\leq|\\neq)\s*[^,\\]*)', s)
    if quad_condition_match:
        main_part = s[:quad_condition_match.start()].strip()
        condition_part = s[quad_condition_match.start():].strip()
        return main_part, condition_part
    # 3. Parenthetical conditions at the end
    paren_condition_patterns = [
        r'\s*\([^)]*(?:[<>≤≥]|\\text\s*\{[^}]*\})[^)]*\)\s*$',
        r'\s*\([^)]*(?:\\forall|\\exists|\\in|\\notin|\\subset|\\supset|\\subseteq|\\supseteq)[^)]*\)\s*$',
        r'\s*\([^)]*\\mathbb\{[^}]+\}[^)]*\)\s*$',
        r'\s*\([^)]*(?:k|n|m|i|j)\\in[^)]*\)\s*$',
    ]
    for pattern in paren_condition_patterns:
        match = re.search(pattern, s)
        if match:
            main_part = s[:match.start()].strip()
            condition_part = s[match.start():].strip()
            return main_part, condition_part
    # 4. Spacing-based separation - \quad followed by inequality
    quad_matches = list(re.finditer(r'\\q(?:uad|quad)\s*', s))
    for match in quad_matches:
        after_quad = s[match.end():]
        if re.search(r'[<>≤≥≠]|\\(?:leq?|geq?|neq?)(?![a-zA-Z])', after_quad):
            main_part = s[:match.start()].strip()
            condition_part = s[match.start():].strip()
            return main_part, condition_part
    # 5. Smart comma splitting - parameters like ", N = 2"
    parts = s.split(',')
    if len(parts) > 1:
        first_part = parts[0].strip()
        if '=' in first_part and len(first_part) > 8:
            remaining = ','.join(parts[1:]).strip()
            # Check for parameter assignments like "N = 2" or inequalities
            if re.search(r'([A-Z]\s*=\s*\d+|[<>≤≥≠]|\\(?:leq?|geq?|neq?)(?![a-zA-Z]))', remaining):
                return first_part, remaining
    # No clear separation found
    return s, None

def _is_piecewise_function(s: str) -> bool:
    """Detect piecewise functions more intelligently."""
    if '\\begin{cases}' in s:
        return True
    if '\\matrix{' in s and ('\\left\\{' in s or '\\right.' in s):
        return True
    if '\\begin{array}' in s:
        for indicator in PIECEWISE_INDICATORS:
            if re.search(indicator, s, re.IGNORECASE):
                return True
    # 4. Left brace with multiple conditions (add HTML entity support!)
    if '\\left\\{' in s and ('\\\\' in s or '&lt;' in s or '&gt;' in s or '<' in s or '>' in s):
        return True
    return False


def _canonicalize_over_to_frac(s: str) -> str:
    r"""Convert {numerator \over denominator} to \frac{numerator}{denominator}"""
    if not s:
        return s
    result = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '{':
            brace_end = _match_brace(s, i)
            if brace_end != -1:
                brace_content = s[i + 1:brace_end]

                # FIXED: Use regex to match \over as complete command only
                over_match = re.search(r'\\over(?![a-zA-Z])', brace_content)

                if over_match:
                    over_pos = over_match.start()
                    numerator = brace_content[:over_pos].strip()
                    # Skip past the complete \over command
                    denominator_start = over_match.end()
                    denominator = brace_content[denominator_start:].strip()
                    result.append(f"\\frac{{{numerator}}}{{{denominator}}}")
                    i = brace_end + 1
                    continue

            result.append(s[i:brace_end + 1] if brace_end != -1 else s[i:])
            i = brace_end + 1 if brace_end != -1 else n
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def _contains_matrix_definitions(s: str) -> bool:
    """Check if formula contains matrix or vector definitions."""
    for pattern in MATRIX_PATTERNS:
        if re.search(pattern, s):
            return True
    return False


def _contains_multiple_statements(s: str) -> bool:
    """Check if formula contains multiple mathematical statements."""
    for pattern in CONNECTIVE_COMPILED:
        if pattern.search(s):
            return True
    additional_patterns = [
        r'\\text\s*\{\s*where\s*\}.*=',
        r'\band\b.*=',
    ]
    for pattern in additional_patterns:
        if re.search(pattern, s, re.IGNORECASE):
            return True
    return False


def _contains_inequalities(latex: str) -> bool:
    """Check if LaTeX contains inequality operators."""
    from src.utils.latex_patterns import INEQUALITY_PATTERNS
    for pattern in INEQUALITY_PATTERNS:
        if pattern.search(latex):
            return True
    return False

def _contains_logical_implications(s: str) -> bool:
    """Detect formulas with logical implications (not pure equations)."""
    for pattern in IMPLICATION_PATTERNS:
        if re.search(pattern, s):
            return True
    return False
