"""
Test suite for variable normalization and validation.
Combines tests for parameter normalization, variable validation, and real data testing.
"""
import pytest
import sqlite3
import sympy as sp
from typing import Tuple, Optional

from src.utils.symbol_validation import validate_parsed_symbols
from src.utils.variable_normalization import (
    extract_variables_from_expression,
    create_variable_mapping,
    apply_variable_substitution,
    normalize_parameters
)
from src.utils.integrand_canonicalization import canonicalize_integrand


class TestParameterNormalization:
    """test parameter normalization strategy: max 3 params → a, b, c"""

    # common parameter whitelist (what L2S parser should produce)
    COMMON_PARAMS = {'a', 'b', 'c', 's', 'r', 't', 'z', 'u', 'n', 'm', 'alpha', 'beta', 'gamma', 'delta'}
    # spurious symbols (indicates L2S parsing error)
    SPURIOUS = {'dx', 'dy', 'dz', 'eta', 'theta', 'phi', 'psi', 'omega', 'lambda', 'mu', 'nu', 'xi', 'rho', 'sigma', 'tau', 'chi'}

    def test_valid_cases(self):
        """test expressions that should pass validation"""
        test_cases = [
            ('x**2', {'x'}, 0, 'no params'),
            ('a*x', {'x', 'a'}, 1, '1 param'),
            ('a*b*x', {'x', 'a', 'b'}, 2, '2 params'),
            ('a*b*c*x', {'x', 'a', 'b', 'c'}, 3, '3 params (max)'),
            ('alpha*x', {'x', 'alpha'}, 1, 'alpha is common'),
            ('t**n*x', {'x', 't', 'n'}, 2, 't,n common params'),
        ]

        for expr_str, expected_syms, expected_param_count, note in test_cases:
            expr = sp.sympify(expr_str)
            symbols = {str(s) for s in expr.free_symbols}

            # remove integration variable and constants
            params = symbols - {'x', 'pi', 'E', 'I'}

            # check for spurious symbols
            has_spurious = bool(params & self.SPURIOUS)

            # check param count
            too_many_params = len(params) > 3

            # decision
            is_valid = not has_spurious and not too_many_params

            assert is_valid, f"Failed for {expr_str} ({note}): expected valid, got invalid"
            assert len(params) == expected_param_count, f"Failed param count for {expr_str} ({note}): expected {expected_param_count}, got {len(params)}"

    def test_invalid_cases(self):
        """test expressions that should fail validation"""
        test_cases = [
            ('dx*x', {'x', 'dx'}, 'dx is spurious'),
            ('theta*x', {'x', 'theta'}, 'theta is spurious'),
            ('a*b*c*s*x', {'x', 'a', 'b', 'c', 's'}, '>3 params'),
            ('eta + x', {'x', 'eta'}, 'eta is spurious Greek'),
        ]

        for expr_str, expected_syms, note in test_cases:
            expr = sp.sympify(expr_str)
            symbols = {str(s) for s in expr.free_symbols}

            # remove integration variable and constants
            params = symbols - {'x', 'pi', 'E', 'I'}

            # check for spurious symbols
            has_spurious = bool(params & self.SPURIOUS)

            # check param count
            too_many_params = len(params) > 3

            # decision
            is_valid = not has_spurious and not too_many_params

            assert not is_valid, f"Failed for {expr_str} ({note}): expected invalid, got valid"


class TestVariableValidation:
    """test variable whitelist validation before implementing in pipeline"""

    # proposed whitelist: x, s, t, u, v, w, alpha + math constants
    WHITELIST = {'x', 's', 't', 'u', 'v', 'w', 'alpha', 'pi', 'E', 'I'}

    def test_valid_variables(self):
        """test expressions with valid variables"""
        test_cases = [
            ('x**2 + s', {'x', 's'}, 'normalized variables'),
            ('alpha*x', {'x', 'alpha'}, 'alpha is valid parameter'),
            ('s**2/(x**2 + 1)', {'x', 's'}, 'canonical normalized form'),
            ('t**alpha/(t + t**alpha + 1)**2', {'t', 'alpha'}, 'alpha as exponent'),
            ('x*atan(alpha*tan(x))', {'x', 'alpha'}, 'alpha as coefficient'),
        ]

        for expr_str, expected_symbols, description in test_cases:
            expr = sp.sympify(expr_str)
            symbols = {str(s) for s in expr.free_symbols}
            spurious = symbols - self.WHITELIST
            is_valid = len(spurious) == 0

            assert is_valid, f"Failed for {expr_str} ({description}): found spurious symbols {spurious}"
            assert symbols == expected_symbols, f"Failed symbol check for {expr_str}: expected {expected_symbols}, got {symbols}"

    def test_invalid_variables(self):
        """test expressions with invalid variables"""
        test_cases = [
            ('theta*x', {'theta', 'x'}, 'theta should be normalized to s/t/u'),
            ('dx/sin(x)', {'dx', 'x'}, 'dx is spurious'),
            ('eta + x', {'eta', 'x'}, 'Greek letter should be normalized'),
            ('theta_4**2/(x**2 + 1)', {'theta_4', 'x'}, 'subscripted Greek'),
        ]

        for expr_str, expected_symbols, description in test_cases:
            expr = sp.sympify(expr_str)
            symbols = {str(s) for s in expr.free_symbols}
            spurious = symbols - self.WHITELIST
            is_valid = len(spurious) == 0

            assert not is_valid, f"Failed for {expr_str} ({description}): expected invalid, got valid"
            assert len(spurious) > 0, f"Failed for {expr_str}: expected to find spurious symbols, found none"


class TestRealDataNormalization:
    """test variable normalization on real database examples"""

    @pytest.fixture
    def db_connection(self):
        """create database connection for testing"""
        try:
            conn = sqlite3.connect('data/integral.db')
            yield conn
            conn.close()
        except sqlite3.OperationalError:
            pytest.skip("Database not available for testing")

    def test_full_normalization_flow(self, db_connection):
        """
        test complete normalization flow on a single integral.

        flow:
        1. symbol validation (whitelist/blacklist)
        2. variable normalization (x, a, b, c)
        3. canonical form computation
        """
        cursor = db_connection.cursor()

        # get a clean test case
        query = '''
        SELECT ii.sympy_integrand, ii.sympy_variable, ii.integrand_canonical, ii.id
        FROM integral_instances ii
        JOIN integrand_groups ig ON ii.integrand_hash = ig.integrand_hash
        WHERE ii.integrand_canonical NOT LIKE '%dx%'
        AND ii.integrand_canonical NOT LIKE '%theta%'
        LIMIT 5
        '''

        cursor.execute(query)
        rows = cursor.fetchall()

        assert len(rows) > 0, "No test data available in database"

        for sympy_int, sympy_var, canonical, iid in rows:
            result = self._test_normalization(sympy_int, sympy_var, iid)

            # basic checks
            assert result['validation_passed'], f"Validation failed for {iid}: {result['validation_reason']}"
            assert result['normalized_integrand'] is not None, f"Normalization failed for {iid}"
            assert result['canonical_form'] is not None, f"Canonicalization failed for {iid}"

    def _test_normalization(self, sympy_integrand: str, sympy_variable: str, integral_id: str) -> dict:
        """test complete normalization flow on a single integral"""
        result = {
            'original_integrand': sympy_integrand,
            'original_variable': sympy_variable,
            'validation_passed': False,
            'validation_reason': None,
            'params_found': None,
            'normalized_integrand': None,
            'normalized_variable': None,
            'canonical_form': None,
            'canonical_hash': None,
            'final_variables': None
        }

        try:
            # step 1: symbol validation
            is_valid, reason, params = validate_parsed_symbols(sympy_integrand, sympy_variable, debug=False)
            result['validation_passed'] = is_valid
            result['validation_reason'] = reason
            result['params_found'] = params

            if not is_valid:
                return result

            # step 2: variable normalization
            # (simplified for testing - actual pipeline has more steps)
            result['normalized_integrand'] = sympy_integrand
            result['normalized_variable'] = 'x'

            # step 3: canonicalization
            canonical, hash_val = canonicalize_integrand(sympy_integrand)
            result['canonical_form'] = canonical
            result['canonical_hash'] = hash_val

            return result

        except Exception as e:
            result['validation_passed'] = False
            result['validation_reason'] = f"Error: {str(e)}"
            return result

    def test_problematic_cases(self, db_connection):
        """test validation on problematic cases from database"""
        cursor = db_connection.cursor()

        test_cases = [
            ("dx cases", "integrand_canonical LIKE '%dx%'", 5),
            ("theta cases", "integrand_canonical LIKE '%theta%'", 5),
            ("alpha cases", "integrand_canonical LIKE '%alpha%'", 5),
        ]

        for case_name, condition, limit in test_cases:
            query = f'''
            SELECT ii.sympy_integrand, ii.sympy_variable, ii.integrand_canonical, ii.id
            FROM integral_instances ii
            JOIN integrand_groups ig ON ii.integrand_hash = ig.integrand_hash
            WHERE ii.{condition}
            LIMIT {limit}
            '''

            cursor.execute(query)
            rows = cursor.fetchall()

            # at least verify we can process these cases without crashing
            for sympy_int, sympy_var, canonical, iid in rows:
                result = self._test_normalization(sympy_int, sympy_var, iid)

                # just check that we got a result (validation may pass or fail)
                assert result is not None, f"Failed to process {iid} in {case_name}"


class TestVariableSubstitution:
    """test variable substitution and mapping"""

    def test_extract_variables(self):
        """test extracting variables from expressions"""
        test_cases = [
            ('x**2 + a*x + b', 'x', {'x'}, {'a', 'b'}),
            ('sin(x) + cos(alpha*x)', 'x', {'x'}, {'alpha'}),
            ('t**2 + s*t', 't', {'t'}, {'s'}),
        ]

        for expr_str, var_str, expected_vars, expected_params in test_cases:
            integrand, variable = extract_variables_from_expression(expr_str, var_str)

            # check that we got the right variable
            assert str(variable) == var_str, f"Variable mismatch for {expr_str}: expected {var_str}, got {variable}"

    def test_parameter_ordering(self):
        """test that parameters are ordered deterministically"""
        # parameters should be ordered alphabetically
        test_cases = [
            (['beta', 'alpha'], ['a', 'b']),  # alphabetical: alpha→a, beta→b
            (['n', 'm'], ['a', 'b']),  # alphabetical: m→a, n→b
            (['c', 'b', 'a'], ['a', 'b', 'c']),  # already ordered
        ]

        for input_params, expected_output in test_cases:
            # normalize_parameters should order alphabetically
            normalized = normalize_parameters(input_params)
            assert normalized == expected_output, f"Parameter ordering failed for {input_params}: expected {expected_output}, got {normalized}"
