import re
from typing import Optional, Tuple, List

from src.utils.latex_parsing import _match_brace
from src.utils.latex_patterns import REL_PATTERNS, INEQUALITY_PATTERNS, ALL_RELATION_PATTERNS

def _parse_stackrel_or_overset_at(s: str, idx: int) -> Optional[Tuple[int, str]]:
    """
    Parse \\stackrel or \\overset, but be conservative about unbraced second arguments.
    Only consume single characters or simple symbols, not full commands.
    """
    n = len(s)
    for cmd in (r'\stackrel', r'\overset'):
        if s.startswith(cmd, idx):
            j = idx + len(cmd)
            # skip whitespace
            while j < n and s[j].isspace():
                j += 1
            # must have first braced arg
            if j >= n or s[j] != '{':
                return None
            first_end = _match_brace(s, j)
            if first_end == -1:
                return None
            j = first_end + 1
            # skip whitespace
            while j < n and s[j].isspace():
                j += 1
            # second arg: braced { ... } OR a single character/symbol
            if j < n and s[j] == '{':
                second_end = _match_brace(s, j)
                if second_end == -1:
                    return None
                return (second_end + 1, s[idx:second_end + 1])

            # Unbraced: only consume single characters, NOT full commands
            if j < n:
                # Only consume single non-command characters like =, <, >, etc.
                if s[j] in '=<>≤≥≠≈∼≡':  # common relation symbols
                    return (j + 1, s[idx:j + 1])
                # Don't consume backslash commands - they're separate
                return None
    return None


def find_top_level_relation(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (lhs, token, rhs) for the FIRST top-level relation token found."""
    if not s:
        return (None, None, None)
    brace_depth = 0
    i = 0
    n = len(s)
    while i < n:
        # Skip comments
        if s[i] == '%':
            nl = s.find('\n', i)
            if nl == -1:
                break
            i = nl + 1
            continue
        # Handle \text{...} - skip entirely
        if s.startswith(r'\text{', i):
            i += len(r'\text{')
            td = 1
            while i < n and td > 0:
                if s[i] == '{':
                    td += 1
                elif s[i] == '}':
                    td -= 1
                i += 1
            continue
        # Handle \begin{env}...\end{env} - skip entirely
        if s.startswith(r'\begin{', i):
            env_match = re.match(r'\\begin\{([^\}]+)\}', s[i:])
            if env_match:
                env_name = env_match.group(1)
                end_pattern = r'\end{' + env_name + '}'
                end_pos = s.find(end_pattern, i + env_match.end())
                if end_pos != -1:
                    i = end_pos + len(end_pattern)
                    continue
                else:
                    return (None, None, None)
        # Track brace depth
        if s[i] == '{':
            brace_depth += 1
            i += 1
            continue
        elif s[i] == '}':
            brace_depth -= 1
            i += 1
            continue
        # Only look for relations at top level
        if brace_depth == 0:
            # Special-case \stackrel / \overset forms
            stack_match = _parse_stackrel_or_overset_at(s, i)
            if stack_match:
                match_end, token_str = stack_match
                lhs = s[:i].strip()
                rhs = s[match_end:].strip()
                return lhs, token_str, rhs
            # Check ALL relation patterns (including inequalities)
            for pattern in ALL_RELATION_PATTERNS:
                match = pattern.match(s, i)
                if match:
                    lhs = s[:i].strip()
                    rhs = s[match.end():].strip()
                    token = match.group(0)
                    return lhs, token, rhs
        i += 1
    return (None, None, None)


def find_all_top_level_relations(s: str) -> List[Tuple[int, int, str]]:
    """
    Find ALL top-level relation tokens in order.
    Returns list of (start_pos, end_pos, token) tuples.
    """
    if not s:
        return []
    relations = []
    brace_depth = 0
    i = 0
    n = len(s)
    while i < n:
        # Skip comments
        if s[i] == '%':
            nl = s.find('\n', i)
            if nl == -1:
                break
            i = nl + 1
            continue
        # Handle \text{...} - skip entirely
        if s.startswith(r'\text{', i):
            i += len(r'\text{')
            td = 1
            while i < n and td > 0:
                if s[i] == '{':
                    td += 1
                elif s[i] == '}':
                    td -= 1
                i += 1
            continue
        # Handle \begin{env}...\end{env} - skip entirely
        if s.startswith(r'\begin{', i):
            env_match = re.match(r'\\begin\{([^\}]+)\}', s[i:])
            if env_match:
                env_name = env_match.group(1)
                end_pattern = r'\end{' + env_name + '}'
                end_pos = s.find(end_pattern, i + env_match.end())
                if end_pos != -1:
                    i = end_pos + len(end_pattern)
                    continue
                else:
                    break  # Malformed environment
        # Track brace depth
        if s[i] == '{':
            brace_depth += 1
            i += 1
            continue
        elif s[i] == '}':
            brace_depth -= 1
            i += 1
            continue
        # Only look for relations at top level
        if brace_depth == 0:
            # Check stackrel/overset first
            stack_match = _parse_stackrel_or_overset_at(s, i)
            if stack_match:
                match_end, token_str = stack_match
                relations.append((i, match_end, token_str))
                i = match_end
                continue
            # Check all relation patterns
            found_match = False
            for pattern in ALL_RELATION_PATTERNS:
                match = pattern.match(s, i)
                if match:
                    token = match.group(0)
                    relations.append((i, match.end(), token))
                    i = match.end()
                    found_match = True
                    break
            if not found_match:
                i += 1
        else:
            i += 1
    return relations

def is_inequality_relation(token: str) -> bool:
    """Check if a relation token represents an inequality."""
    if not token:
        return False
    for pattern in INEQUALITY_PATTERNS:
        if pattern.fullmatch(token):
            return True
    return False


def split_relation_chain(s: str) -> List[Tuple[str, str, str]]:
    """
    Split a chain like A=B=C=D into multiple (lhs, relation, rhs) tuples.
    Strategy: Connect each element to first and last, avoiding duplicates.
    """
    relations = find_all_top_level_relations(s)

    if len(relations) <= 1:
        # No chain, use original logic
        lhs, token, rhs = find_top_level_relation(s)
        if lhs and token and rhs:
            return [(lhs, token, rhs)]
        return []

    # Check if all relations are equalities (reject if mixed with inequalities)
    for _, _, token in relations:
        if is_inequality_relation(token):
            return []  # Don't split chains containing inequalities

    # Extract the expressions between relations
    expressions = []
    last_end = 0

    for start, end, token in relations:
        expr = s[last_end:start].strip()
        if expr:
            expressions.append(expr)
        last_end = end

    # Add the final expression
    final_expr = s[last_end:].strip()
    if final_expr:
        expressions.append(final_expr)

    if len(expressions) < 2:
        return []

    # Generate pairs: each to first, each to last (avoiding duplicates)
    pairs = []
    seen = set()

    first_expr = expressions[0]
    last_expr = expressions[-1]

    for i, expr in enumerate(expressions):
        # Connect to first (unless it is the first)
        if i > 0:
            pair = (first_expr, expr)
            if pair not in seen:
                pairs.append((first_expr, '=', expr))
                seen.add(pair)
                seen.add((expr, first_expr))  # Mark reverse as seen too

        # Connect to last (unless it is the last)
        if i < len(expressions) - 1:
            pair = (expr, last_expr)
            if pair not in seen:
                pairs.append((expr, '=', last_expr))
                seen.add(pair)
                seen.add((last_expr, expr))  # Mark reverse as seen too

    return pairs

