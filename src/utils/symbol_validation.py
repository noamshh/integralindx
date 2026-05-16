import re
import logging
from typing import Tuple, Optional, Set
import sympy as sp
from src.models.egen.vocab import SYMPY_TO_PREFIX
from src.utils.safe_math import safe_sympify

logger = logging.getLogger(__name__)

COMMON_PARAMS = {
    'a', 'b', 'c', 'r', 's', 't', 'u', 'z', 'n', 'm',
    'alpha', 'beta',
}
SPURIOUS_SYMBOLS = {
    'dx', 'dy', 'dz', 'dt', 'du', 'dv', 'dw', 'dr', 'ds',
    'eta', 'theta', 'phi', 'psi', 'omega', 'lambda',
    'mu', 'nu', 'xi', 'rho', 'sigma', 'tau', 'chi', 'kappa',
    'epsilon',  'iota', 'omicron', 'upsilon',
}
MATH_CONSTANTS = {'pi', 'E', 'I', 'e'}

def is_subscripted_variable(symbol_name: str) -> bool:
    return bool(re.match(r'^[a-zA-Z]+_\d+$', symbol_name))


def get_function_classes(expr) -> Set[type]:
    function_classes = set()
    for arg in sp.preorder_traversal(expr):
        if isinstance(arg, sp.Function):
            func_class = arg.func
            function_classes.add(func_class)
    return function_classes


def validate_parsed_symbols(sympy_integrand: str, sympy_variable: str, debug: bool = False) -> Tuple[bool, Optional[str], Optional[Set[str]]]:
    try:
        expr = safe_sympify(sympy_integrand)
        # hardcoded edge case rejections
        if sp.I in expr.atoms():
            reason = "contains imaginary unit I (not supported)"
            if debug:
                logger.debug(f"Rejected: {reason}")
            return False, reason, None
        if sp.zoo in expr.atoms():
            reason = "contains complex infinity zoo (not supported)"
            if debug:
                logger.debug(f"Rejected: {reason}")
            return False, reason, None
        if any(isinstance(arg, sp.Order) for arg in sp.preorder_traversal(expr)):
            reason = "contains big-O notation (not supported)"
            if debug:
                logger.debug(f"Rejected: {reason}")
            return False, reason, None
        for arg in sp.preorder_traversal(expr):
            if isinstance(arg, (sp.Min, sp.Max)):
                reason = f"contains {type(arg).__name__} function (not supported)"
                if debug:
                    logger.debug(f"Rejected: {reason}")
                return False, reason, None
        function_classes = get_function_classes(expr)
        if sp.re in function_classes or sp.im in function_classes:
            reason = "contains re/im functions (not supported)"
            if debug:
                logger.debug(f"Rejected: {reason}")
            return False, reason, None
        for arg in sp.preorder_traversal(expr):
            if isinstance(arg, sp.Derivative):
                reason = "contains Derivative operator (not supported)"
                if debug:
                    logger.debug(f"Rejected: {reason}")
                return False, reason, None

        all_symbols = {str(s) for s in expr.free_symbols}
        params = all_symbols - {sympy_variable} - MATH_CONSTANTS
        if debug:
            logger.debug(f"Validating symbols: all={all_symbols}, params={params}")
        spurious = params & SPURIOUS_SYMBOLS
        if spurious:
            reason = f"spurious symbols: {spurious}"
            if debug:
                logger.debug(f"Rejected: {reason}")
            return False, reason, None
        subscripted = [p for p in params if is_subscripted_variable(p)]
        if subscripted:
            reason = f"subscripted variables: {subscripted}"
            if debug:
                logger.debug(f"Rejected: {reason}")
            return False, reason, None
        if len(params) > 3:
            reason = f"too many params: {len(params)} > 3 (found: {params})"
            if debug:
                logger.debug(f"Rejected: {reason}")
            return False, reason, None
        unknown = params - COMMON_PARAMS
        if unknown:
            reason = f"unknown params not in whitelist: {unknown}"
            if debug:
                logger.debug(f"Rejected: {reason}")
            return False, reason, None
        function_classes = get_function_classes(expr)
        if function_classes:
            allowed_functions = set(SYMPY_TO_PREFIX.keys())
            disallowed_functions = function_classes - allowed_functions
            if disallowed_functions:
                func_names = {f.__name__ if hasattr(f, '__name__') else str(f)
                             for f in disallowed_functions}
                reason = f"disallowed functions: {func_names}"
                if debug:
                    logger.debug(f"Rejected: {reason}")
                return False, reason, None
        # all passed
        return True, None, params
    except Exception as e:
        reason = f"failed to parse for symbol validation: {e}"
        if debug:
            logger.warning(f"Symbol validation error: {reason}")
        return False, reason, None


def get_param_count_stats(integrands: list) -> dict:
    stats = {}
    for sympy_integrand, sympy_variable in integrands:
        is_valid, _, params = validate_parsed_symbols(sympy_integrand, sympy_variable)
        if is_valid and params is not None:
            param_count = len(params)
            stats[param_count] = stats.get(param_count, 0) + 1
    return stats
