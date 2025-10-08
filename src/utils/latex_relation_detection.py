import re
from typing import Optional, Tuple, List

from src.utils.latex_parsing import _match_brace
from src.utils.latex_patterns import ALL_RELATION_PATTERNS

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

            # unbraced: only consume single characters, NOT full commands
            if j < n:
                # only consume single non-command characters like =, <, >, etc.
                if s[j] in '=<>≤≥≠≈∼≡':  # common relation symbols
                    return (j + 1, s[idx:j + 1])
                # don't consume backslash commands - they're separate
                return None
    return None

def find_all_top_level_relations(s: str) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start_pos, end_pos, token) tuples.
    """
    if not s:
        return []
    relations = []
    brace_depth = 0
    i = 0
    n = len(s)
    while i < n:
        # skip comments
        if s[i] == '%':
            nl = s.find('\n', i)
            if nl == -1:
                break
            i = nl + 1
            continue
        # handle \text{...} - skip entirely
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
        # handle \begin{env}...\end{env} - skip entirely
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
                    break  # malformed enviroment
        # track brace depth
        if s[i] == '{':
            brace_depth += 1
            i += 1
            continue
        elif s[i] == '}':
            brace_depth -= 1
            i += 1
            continue
        # only look for relations at top level
        if brace_depth == 0:
            # check stackrel/overset first
            stack_match = _parse_stackrel_or_overset_at(s, i)
            if stack_match:
                match_end, token_str = stack_match
                relations.append((i, match_end, token_str))
                i = match_end
                continue
            # check all relation patterns
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
