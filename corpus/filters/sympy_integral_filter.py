import re, html, json, time, gc
import logging
from pathlib import Path
from typing import List, Dict, Optional
from latex2sympy2_extended import latex2sympy
from sympy import Integral

from corpus.formula_models import NormalizedFormula, IntegralFormula
from src.utils.timeout import with_timeout, TimeoutError
from src.utils.provenance import now_iso
from src.utils.symbol_validation import validate_parsed_symbols

PROCESSING_TIMEOUT_SECONDS = 25
logger = logging.getLogger(__name__)

def _is_single_letter_ij(expr: str) -> bool:
    if not expr:
        return False
    cleaned = expr.strip()
    if cleaned in ['I', 'J']:
        return True
    if cleaned in [r'\mathcal{I}', r'\mathcal{J}']:
        return True
    pattern = r'^(I|J|\\mathcal\{[IJ]\})([_^]\{?[^}]*\}?|\([^)]*\))*$'
    if re.match(pattern, cleaned):
        return True
    return False


def _contains_multiple_integrals(latex_expr: str) -> bool:
    integral_pattern = re.compile(r'\\int(?![a-zA-Z])')
    matches = integral_pattern.findall(latex_expr)
    return len(matches) > 1


class SymPyIntegralFilter:
    def __init__(self, debug: bool = False, progress: bool = False):
        self.debug = debug
        self.progress = progress
        self.integral_pattern = re.compile(r'\\int(?![a-zA-Z])')
        self.differential_pattern = re.compile(r'd\s*[a-zA-Z]')
        self.multidim_patterns = [
            re.compile(r'\\iint'),
            re.compile(r'\\iiint'),
            re.compile(r'\\oint.*\\oint'),
            re.compile(r'\\int.*\\int.*d[a-zA-Z].*d[a-zA-Z]'),
        ]
        self.polylog_patterns = [
            re.compile(r'\\operatorname\{Li\}_'),  # \operatorname{Li}_2
            re.compile(r'\\text\{Li\}_'),           # \text{Li}_2
            re.compile(r'Li_\{?\d+\}?'),            # Li_2 or Li_{2}
        ]
        self.latex_rejections = [
            (re.compile(r'[KIJYfpqFPQ]_'), "subscripted special function not parsed"),
            (re.compile(r'\b[KESHfgh]\('), "unparsed function call"),
            (re.compile(r'\b[KSHAfghE]\b'), "unknown function symbol "),
            (re.compile(r'\b[FGH]\('), "uppercase unknown function "),
            (re.compile(r'\bo[\(\\]'), "little o notation"),
            (re.compile(r'[a-zA-Z]\''), "derivative notation"),
            (re.compile(r'[A-Z][a-z]'), "uppercase + lowercase"),
            (re.compile(r'[LXZTN][\+\-\*/\^\)]'), "uppercase parameter variable"),
            (re.compile(r'\([LXZTN]'), "uppercase parameter variable after paren"),
            (re.compile(r'\d+\.\d+'), "decimal number (prefer fractions)"),
            (re.compile(r'\\lim'), "limit operator"),
            (re.compile(r'\\prod'), "product"),
            (re.compile(r'matrix'), "matrix expression"),
            (re.compile(r'\\\{'), "fractional part function"),
            (re.compile(r'\['), "floor/ceiling function"),
        ]

    def clean_latex(self, latex: str) -> str:
        if not latex:
            return ""
        latex = html.unescape(latex)
        latex = re.sub(r'\s+', ' ', latex).strip()
        return latex

    def should_reject_by_latex(self, latex: str) -> tuple[bool, Optional[str]]:
        """Returns:  (should_reject, rejection_reason)"""
        if not latex:
            return False, None
        has_polylog = any(pattern.search(latex) for pattern in self.polylog_patterns)
        for pattern, reason in self.latex_rejections:
            match = pattern.search(latex)
            if match:
                if pattern.pattern == r'[A-Z][a-z]':
                    if match.group() == 'Li' or has_polylog:
                        continue
                # rejected
                if self.debug:
                    logger.debug(f"latex rejection: {reason} (matched: {match.group() if hasattr(match, 'group') else 'N/A'})")
                return True, f"latex_rejected: {reason}"
        return False, None

    def has_integral(self, latex: str) -> bool:
        if not latex:
            return False
        return bool(self.integral_pattern.search(latex))
    
    def has_integral_pattern_fast(self, formula: "NormalizedFormula") -> bool:
        """Fast check if formula contains integral patterns"""
        try:
            if not formula or not getattr(formula, 'accepted', False):
                return False
            # check leading expression
            leading_expr = getattr(formula, 'leading_expression', None)
            if leading_expr and self.has_integral(leading_expr):
                return True
            # check equivalent forms
            equivalent_forms = getattr(formula, 'equivalent_forms', None)
            if equivalent_forms:
                for expr in equivalent_forms:
                    if expr and self.has_integral(expr):
                        return True
            return False
        except Exception as e:
            logger.warning(f"error in has_integral_pattern_fast: {e}")
            return False

    def is_1d_integral(self, latex: str) -> bool:
        for pattern in self.multidim_patterns:
            if pattern.search(latex):
                return False
        has_int = self.has_integral(latex)
        has_diff = bool(self.differential_pattern.search(latex))
        return has_int and has_diff
    
    def parse_with_sympy(self, latex: str) -> Dict[str, str]:
        result = {
            'parsing_success': False,
            'parsing_error': None,
            'integrand': '',
            'variable': 'x',
            'lower_bound': None,
            'upper_bound': None,
            'integral_type': 'indefinite'
        }
        try:
            cleaned = self.clean_latex(latex)
            if self.debug:
                logger.debug(f"parsing latex: {cleaned[:100]}...")
            # early rejection
            should_reject, rejection_reason = self.should_reject_by_latex(cleaned)
            if should_reject:
                result['parsing_error'] = rejection_reason
                if self.debug:
                    logger.debug(f"early latex rejection: {rejection_reason}")
                return result
            # use timeout wrapper to prevent hangs
            try:
                sympy_expr = with_timeout(latex2sympy, timeout_seconds=PROCESSING_TIMEOUT_SECONDS)(cleaned)
            except TimeoutError as e:
                result['parsing_error'] = f'timeout_rejected: {str(e)}'
                logger.warning(f"REJECTED due to timeout: {cleaned[:200]}...")
                # force cleanup after timeout
                gc.collect()
                return result
            except Exception as e:
                result['parsing_error'] = f'parsing_exception: {str(e)}'
                if self.debug:
                    logger.warning(f"latex2sympy EXCEPTION for expression: {cleaned[:200]}..., error: {str(e)}")
                return result
            if isinstance(sympy_expr, Integral):
                result['parsing_success'] = True
                result['integrand'] = str(sympy_expr.args[0])
                # extract integration limits
                if len(sympy_expr.args) >= 2:
                    var_info = sympy_expr.args[1]
                    if var_info is not None and hasattr(var_info, '__iter__') and len(var_info) >= 1:
                        result['variable'] = str(var_info[0])
                        if len(var_info) >= 2:
                            result['lower_bound'] = str(var_info[1])
                        if len(var_info) >= 3:
                            result['upper_bound'] = str(var_info[2])
                        if result['lower_bound'] or result['upper_bound']:
                            result['integral_type'] = 'definite'
            else:
                result['parsing_error'] = 'not_integral_object'
        except Exception as e:
            result['parsing_error'] = str(e)
        return result
    
    def process_formula(self, formula: NormalizedFormula, skip_parsed=None, blob_map=None) -> List[IntegralFormula]:
        if not formula.accepted:
            return []
        formula_num = 1
        self._debug_formula_processing(formula_num, formula.id, formula.leading_expression)
        # collect all expressions from the chain
        all_expressions = []
        if formula.leading_expression:
            all_expressions.append((formula.leading_expression, 0))  # (expr, position)
        if formula.equivalent_forms:
            for i, expr in enumerate(formula.equivalent_forms):
                all_expressions.append((expr, i + 1))
        # find integral expressions
        integral_expressions = []
        non_integral_expressions = []
        for expr, position in all_expressions:
            cleaned = self.clean_latex(expr)
            if self.is_1d_integral(cleaned):
                integral_expressions.append((expr, cleaned, position))
            else:
                non_integral_expressions.append(expr)
        if not integral_expressions:
            return []
        # extract author info once per formula
        author_name, author_link = None, None
        if blob_map is not None:
            from corpus.scripts.blob_parser import extract_author_from_formula
            author_name, author_link = extract_author_from_formula(formula, blob_map)
        # create IntegralFormula for each integral found
        integral_formulas = []
        for original_expr, cleaned_expr, position in integral_expressions:
            # check if this expression was already successfully parsed
            if skip_parsed and (formula.id, position) in skip_parsed:
                continue
            start_time = time.time()
            sympy_result = self.parse_with_sympy(cleaned_expr)
            parse_time = time.time() - start_time
            self._debug_sympy_parsing(formula_num, cleaned_expr, parse_time, sympy_result)
            # skip problematic parses
            if parse_time > 3.0 or not sympy_result['parsing_success']:
                if 'timeout' in (sympy_result.get('parsing_error') or ''):
                    logger.warning(f"formula #{formula_num} REJECTED due to timeout after {parse_time:.3f}s: {cleaned_expr[:200]}...")
                    continue
                elif parse_time > 3.0:
                    logger.warning(f"formula #{formula_num} took {parse_time:.3f}s to parse, skipping: {cleaned_expr[:200]}...")
                    continue
            if sympy_result['parsing_success']:
                # symbol validation
                is_valid, rejection_reason, params = validate_parsed_symbols(
                    sympy_result['integrand'],
                    sympy_result['variable'],
                    debug=self.debug
                )
                if not is_valid:
                    if self.debug:
                        logger.debug(f"symbol validation failed: {rejection_reason}")
                    continue
                equivalent_forms = []
                if _contains_multiple_integrals(original_expr):
                    equivalent_forms = []
                else:
                    for other_expr, _, _ in integral_expressions:
                        if other_expr != original_expr:
                            equivalent_forms.append(other_expr)
                    equivalent_forms.extend(non_integral_expressions)
                # filter out single letters I, J from equivalent forms
                equivalent_forms = [expr for expr in equivalent_forms
                                  if not _is_single_letter_ij(expr)]
                integral_formula = IntegralFormula.from_sympy_parsing(
                    formula, cleaned_expr, position, sympy_result, equivalent_forms,
                    author_name=author_name, author_link=author_link,
                    auto_canonicalize=False  # skip auto-canonicalization (will do after normalization)
                )
                integral_formulas.append(integral_formula)
        return integral_formulas

    def filter_formulas(self, input_file: str, output_file: str, processed_ids: set = None,
                        force_reprocess: bool = False, limit: Optional[int] = None,
                        existing_integral_count: int = 0, start_line: int = 0) -> Dict[str, int]:
        # load blob path mapping for author extraction
        from corpus.scripts.blob_parser import load_raw_formula_blob_paths
        logger.info("loading blob paths for author extraction...")
        blob_map = load_raw_formula_blob_paths()
        logger.info(f"loaded {len(blob_map):,} blob paths")

        processed_ids = processed_ids or set()
        stats = {
            'total_formulas': 0,
            'skipped_already_processed': 0,
            'skipped_no_integral': 0,
            'integrals_found': existing_integral_count,
            'sympy_success': 0,
            'sympy_failures': 0,
            'processing_errors': 0
        }
        # count total lines for progress
        total_lines = 0
        if self.progress:
            logger.info("counting total formulas...")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        total_lines += 1
            # adjust for start_line offset
            effective_total = total_lines - start_line
            if limit:
                effective_total = min(effective_total, limit)
            if start_line > 0:
                logger.info(f"found {total_lines:,} total formulas, starting from line {start_line+1:,}, processing {effective_total:,}")
            else:
                logger.info(f"found {total_lines:,} total formulas{f', processing {effective_total:,}' if limit else ''}")
        start_time = time.time()
        output_dir = Path(output_file).parent
        stats_file = output_dir / "filter_stats.json"
        with open(output_file, 'a', encoding='utf-8') as output_handle:
            with open(input_file, 'r', encoding='utf-8') as input_handle:
                for line_num, line in enumerate(input_handle, 1):
                    if line_num <= start_line:
                        continue
                    if limit and stats['total_formulas'] >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        try:
                            data = json.loads(line)
                            formula = NormalizedFormula.from_dict(data)
                        except Exception as e:
                            stats['processing_errors'] += 1
                            logger.warning(f"error parsing formula from line {line_num}: {e}")
                            continue
                        stats['total_formulas'] += 1
                        formula_id = getattr(formula, 'id', f'line-{line_num}')
                        # skip if already processed (unless force reprocess)
                        try:
                            if not force_reprocess and formula.id and formula.id in processed_ids:
                                stats['skipped_already_processed'] += 1
                                if self.debug and stats['total_formulas'] % 1000 == 0:
                                    logger.debug(f"Skipped already processed: {formula_id}")
                                continue
                        except Exception as e:
                            logger.warning(f"error checking processed status for {formula_id}: {e}")

                        has_integral_pattern = self.has_integral_pattern_fast(formula)
                        if not has_integral_pattern:
                            stats['skipped_no_integral'] += 1
                            if self.debug and stats['total_formulas'] % 1000 == 0:
                                logger.debug(f"skipped no integral pattern: {formula_id}")
                            continue
                        try:
                            if self.debug:
                                logger.debug(f"processing formula {formula_id}")
                            try:
                                integrals = with_timeout(self.process_formula, timeout_seconds=PROCESSING_TIMEOUT_SECONDS)(formula, blob_map=blob_map)
                            except TimeoutError:
                                stats['processing_errors'] += 1
                                logger.warning(f"timeout processing formula {formula_id} ({PROCESSING_TIMEOUT_SECONDS}s limit)")
                                # timeout occurred
                                gc.collect()  # cleanup after timeout
                                continue
                            if self.debug:
                                logger.debug(f"completed processing formula {formula_id}, found {len(integrals)} integrals")
                        except Exception as e:
                            stats['processing_errors'] += 1
                            logger.warning(f"Error processing formula {formula_id}: {e}")
                            continue
                        if integrals:
                            for integral in integrals:
                                try:
                                    output_handle.write(integral.to_json() + '\n')
                                    output_handle.flush()
                                    stats['integrals_found'] += 1
                                except Exception as e:
                                    logger.error(f"Error writing integral for {formula_id}: {e}")
                                    stats['processing_errors'] += 1

                        try:
                            if formula.id:
                                processed_ids.add(formula.id)
                        except Exception as e:
                            logger.warning(f"Error updating processed set for {formula_id}: {e}")
                        
                        # progress reporting
                        if self.progress and stats['total_formulas'] % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = stats['total_formulas'] / elapsed if elapsed > 0 else 0
                            # calculate ETA
                            if total_lines > 0:
                                current_line = start_line + stats['total_formulas']
                                progress_pct = (current_line / total_lines) * 100
                                remaining = total_lines - current_line
                                eta_seconds = remaining / rate if rate > 0 else 0
                                if eta_seconds >= 3600:
                                    eta_str = f"{eta_seconds/3600:.1f}h"
                                elif eta_seconds >= 60:
                                    eta_str = f"{eta_seconds/60:.1f}m"
                                else:
                                    eta_str = f"{eta_seconds:.0f}s"
                                logger.info(f"Progress: {current_line:,}/{total_lines:,} ({progress_pct:.1f}%) | "
                                           f"Integrals: {stats['integrals_found']:,} | "
                                           f"Rate: {rate:.1f}/s | "
                                           f"ETA: {eta_str} | "
                                           f"Elapsed: {elapsed/60:.1f}m")
                            else:
                                logger.info(f"Processed: {stats['total_formulas']:,} | "
                                           f"Integrals: {stats['integrals_found']:,} | "
                                           f"Rate: {rate:.1f}/s | "
                                           f"Elapsed: {elapsed/60:.1f}m")
                        # more frequent cleanup and stats writing
                        if stats['total_formulas'] % 500 == 0:
                            gc.collect()
                            self._write_stats(stats_file, stats, start_time, total_lines, limit, existing_integral_count, start_line)
                    except Exception as e:
                        stats['processing_errors'] += 1
                        logger.error(f"Critical error processing line {line_num}: {e}")
                        # try to cache critical failure if possible
                        continue
        # final stats
        elapsed = time.time() - start_time
        logger.info(f"\n=== FILTER SUMMARY ===")
        logger.info(f"total formulas processed: {stats['total_formulas']:,}")
        logger.info(f"already processed (skipped): {stats['skipped_already_processed']:,}")
        logger.info(f"no integral pattern (skipped): {stats['skipped_no_integral']:,}")
        logger.info(f"integrals found: {stats['integrals_found']:,}")
        logger.info(f"processing errors: {stats['processing_errors']:,}")
        logger.info(f"processing time: {elapsed/60:.1f} minutes")
        logger.info(f"processing rate: {stats['total_formulas']/elapsed:.1f} formulas/second")
        self._write_final_stats(stats_file, stats, start_time, total_lines, limit, existing_integral_count, start_line)
        return stats

    def _debug_formula_processing(self, formula_num: int, formula_id: str, leading_expr: str = None):
        if not self.debug:
            return
        logger.debug(f"processing formula {formula_num}: {formula_id}")
        if leading_expr:
            logger.debug(f"leading expression: {leading_expr[:200]}...")

    def _debug_sympy_parsing(self, formula_num: int, expr: str, start_time: float, result: dict):
        if not self.debug:
            return
        parse_time = start_time
        logger.debug(
            f"sympy parsing for formula {formula_num} completed in {parse_time:.3f}s, success: {result['parsing_success']}")
        if result['parsing_success']:
            logger.debug(f"successfully parsed: {expr[:100]}...")
        else:
            logger.debug(f"failed to parse: {expr[:100]}... ERROR: {result.get('parsing_error', 'unknown')}")


    @staticmethod
    def _write_stats(stats_file: Path, stats: Dict[str, int], start_time: float, total_lines: int,
                     limit: Optional[int], existing_integral_count: int = 0, start_line: int = 0):
        try:
            elapsed = time.time() - start_time
            new_integrals = stats['integrals_found'] - existing_integral_count
            current_stats = {
                'status': 'in_progress',
                'total_formulas_in_file': total_lines,
                'formulas_processed': stats['total_formulas'],
                'start_line': start_line,
                'current_line': start_line + stats['total_formulas'],
                'skipped_already_processed': stats['skipped_already_processed'],
                'skipped_no_integral': stats['skipped_no_integral'],
                'integrals_found_total': stats['integrals_found'],
                'integrals_found_new': new_integrals,
                'integrals_found_existing': existing_integral_count,
                'processing_errors': stats['processing_errors'],
                'acceptance_rate': new_integrals / max(stats['total_formulas'], 1) if stats[
                                                                                          'total_formulas'] > 0 else 0,
                'limit_applied': limit,
                'elapsed_seconds': elapsed,
                'processing_rate': stats['total_formulas'] / elapsed if elapsed > 0 else 0,
                'last_updated': now_iso()
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(current_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"could not write incremental stats: {e}")

    @staticmethod
    def _write_final_stats(stats_file: Path, stats: Dict[str, int], start_time: float, total_lines: int,
                           limit: Optional[int], existing_integral_count: int = 0, start_line: int = 0):
        try:
            elapsed = time.time() - start_time
            new_integrals = stats['integrals_found'] - existing_integral_count
            final_stats = {
                'status': 'completed',
                'total_formulas_in_file': total_lines,
                'formulas_processed': stats['total_formulas'],
                'start_line': start_line,
                'current_line': start_line + stats['total_formulas'],
                'skipped_already_processed': stats['skipped_already_processed'],
                'skipped_no_integral': stats['skipped_no_integral'],
                'integrals_found_total': stats['integrals_found'],
                'integrals_found_new': new_integrals,
                'integrals_found_existing': existing_integral_count,
                'processing_errors': stats['processing_errors'],
                'acceptance_rate': new_integrals / max(stats['total_formulas'], 1) if stats[
                                                                                          'total_formulas'] > 0 else 0,
                'limit_applied': limit,
                'elapsed_seconds': elapsed,
                'processing_rate': stats['total_formulas'] / elapsed if elapsed > 0 else 0,
                'completed_at': now_iso()
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"could not write final stats: {e}")
    
