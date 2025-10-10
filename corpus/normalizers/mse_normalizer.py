import html
from typing import Optional, List

from corpus.formula_models import NormalizedFormula
from src.utils.latex_patterns import  SIZE_COMMAND_PATTERN, TAG_PATTERN
from src.utils.latex_parsing import (
    _split_respecting_environments, split_main_from_conditions, _is_piecewise_function,
    _split_on_connectives, _contains_multiple_statements,
)
from src.utils.latex_cleaning import _strip_math_environments, _clean_side
from src.utils.latex_relation_detection import find_all_top_level_relations

class MSENormalizer:
    def __init__(self):
        pass

    def normalize_latex(self, raw: str, source_id: str) -> List[NormalizedFormula]:
        """
        Normalize LaTeX formula(s), handling multiple statements.
        Returns a list of NormalizedFormula objects - one per mathematical statement.
        """
        if not raw:
            return [NormalizedFormula.rejected(source_id, raw, "empty")]
        raw = html.unescape(raw)
        if _is_piecewise_function(raw):
            return [NormalizedFormula.rejected(source_id, raw, "contains_piecewise_function")]
        if _contains_multiple_statements(raw):
            return [NormalizedFormula.rejected(source_id, raw, "multiple_statements")]
        if "\\Rightarrow" in raw:
            return [NormalizedFormula.rejected(source_id, raw, "contains_rightarrow_implication")]
        stripped = _strip_math_environments(raw)
        # check for multiple statements after environment processing
        parts = _split_respecting_environments(stripped)
        if len(parts) > 1:
            # count parts that contain equations (have = signs)
            equation_parts = [part for part in parts if '=' in part]
            if len(equation_parts) > 1:
                # process each equation as a separate formula
                results = []
                for i, part in enumerate(equation_parts):
                    part_id = f"{source_id}-part{i+1}"
                    expanded_parts = _split_on_connectives(part)
                    combined_part = " ".join(p.strip() for p in expanded_parts if p.strip())
                    if combined_part:
                        main_part, conditions = split_main_from_conditions(combined_part)
                        result = _normalize_formula_with_conditions(main_part, part_id, conditions, raw)
                        results.append(result)
                return results if results else [NormalizedFormula.rejected(source_id, raw, "empty_after_parsing")]
        # combine all parts back into a single formula
        expanded_parts = []
        for part in parts:
            expanded_parts.extend(_split_on_connectives(part))
        combined_formula = ""
        for part in expanded_parts:
            part = part.strip()
            if not part:
                continue
            if not combined_formula:
                combined_formula = part
            else:
                combined_formula += " " + part
        if not combined_formula:
            return [NormalizedFormula.rejected(source_id, raw, "empty_after_parsing")]
        main_part, conditions = split_main_from_conditions(combined_formula)
        result = _normalize_formula_with_conditions(main_part, source_id, conditions, raw)
        return [result]


def _normalize_formula_with_conditions(s: str, source_id: str, conditions: Optional[str] = None, raw_latex: str = "") -> NormalizedFormula:
    """Normalize formula keeping first expression as main and extracting equivalent forms from chains"""
    s = TAG_PATTERN.sub('', s)
    s = SIZE_COMMAND_PATTERN.sub('', s)
    s = s.strip()
    if not s:
        return NormalizedFormula.rejected(source_id, raw_latex, "empty_after_cleaning")
    # check for any relations - extract equivalent forms from chains
    all_relations = find_all_top_level_relations(s)
    if all_relations:
        equivalent_forms = _extract_equivalent_forms_from_chain(s)
        if equivalent_forms and len(equivalent_forms) >= 2:
            leading_expr = _clean_side(equivalent_forms[0])
            equiv_forms = [_clean_side(expr) for expr in equivalent_forms[1:]]
            equiv_forms = [expr for expr in equiv_forms if expr and expr != leading_expr]
            if leading_expr:
                return NormalizedFormula.accepted(source_id, raw_latex, leading_expr, equiv_forms, conditions)
        elif equivalent_forms and len(equivalent_forms) == 1:
            leading_expr = _clean_side(equivalent_forms[0])
            if leading_expr:
                return NormalizedFormula.accepted(source_id, raw_latex, leading_expr, [], conditions)
    cleaned = _clean_side(s)
    if cleaned:
        return NormalizedFormula.accepted(source_id, raw_latex, cleaned, [], conditions)
    return NormalizedFormula.rejected(source_id, raw_latex, "empty_after_processing")


def _extract_equivalent_forms_from_chain(s: str) -> List[str]:
    """Extract all expressions from a chain like A=B=C=D into [A, B, C, D]"""
    relations = find_all_top_level_relations(s)
    if not relations:
        return []
    expressions = []
    last_end = 0
    for start, end, token in relations:
        expr = s[last_end:start].strip()
        if expr:
            expressions.append(expr)
        last_end = end
    final_expr = s[last_end:].strip()
    if final_expr:
        expressions.append(final_expr)
    return expressions