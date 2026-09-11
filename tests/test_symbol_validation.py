"""
Test suite for symbol validation on both synthetic and real database examples.
"""
import pytest
import sqlite3
from src.utils.symbol_validation import validate_parsed_symbols


class TestSymbolValidation:
    """test symbol validation with whitelist/blacklist"""

    def test_valid_symbols(self):
        """test expressions with valid symbols (should pass)"""
        test_cases = [
            # (integrand, variable, description)
            ('x**2', 'x', 'simple polynomial'),
            ('a*x + b', 'x', 'linear with params'),
            ('alpha*sin(x)', 'x', 'alpha is whitelisted'),
            ('a*b*c*x', 'x', '3 params (max allowed)'),
            ('beta*x**2 + alpha*x', 'x', 'alpha and beta whitelisted'),
        ]

        for integrand, variable, description in test_cases:
            is_valid, reason, params = validate_parsed_symbols(integrand, variable, debug=False)

            assert is_valid, f"Failed for {integrand} ({description}): {reason}"
            assert params is not None, f"Expected params for {integrand}, got None"

    def test_spurious_symbols(self):
        """test expressions with spurious symbols (should fail)"""
        test_cases = [
            # (integrand, variable, spurious_symbol, description)
            ('dx*x', 'x', 'dx', 'dx is blacklisted'),
            ('theta*x', 'x', 'theta', 'theta is blacklisted'),
            ('eta + x', 'x', 'eta', 'eta is spurious Greek'),
            ('phi*sin(x)', 'x', 'phi', 'phi is blacklisted'),
        ]

        for integrand, variable, spurious, description in test_cases:
            is_valid, reason, params = validate_parsed_symbols(integrand, variable, debug=False)

            assert not is_valid, f"Failed for {integrand} ({description}): expected invalid, got valid"
            assert spurious in reason.lower() if reason else False, \
                f"Expected reason to mention {spurious} for {integrand}"

    def test_too_many_params(self):
        """test expressions with >3 parameters (should fail)"""
        test_cases = [
            ('a*b*c*s*x', 'x', '4 params'),
            ('a*b*c*s*t*x', 'x', '5 params'),
        ]

        for integrand, variable, description in test_cases:
            is_valid, reason, params = validate_parsed_symbols(integrand, variable, debug=False)

            assert not is_valid, f"Failed for {integrand} ({description}): expected invalid (too many params)"
            assert 'param' in reason.lower() if reason else False, \
                f"Expected reason to mention params for {integrand}"

    def test_subscripted_variables(self):
        """test expressions with subscripted variables (should fail)"""
        test_cases = [
            ('theta_4*x', 'x', 'subscripted Greek letter'),
            ('alpha_1*x + alpha_2', 'x', 'multiple subscripted'),
        ]

        for integrand, variable, description in test_cases:
            is_valid, reason, params = validate_parsed_symbols(integrand, variable, debug=False)

            assert not is_valid, f"Failed for {integrand} ({description}): expected invalid (subscripted)"


class TestRealDatabaseSymbolValidation:
    """test symbol validation on real integrands from database"""

    @pytest.fixture
    def db_connection(self):
        """create database connection for testing"""
        try:
            conn = sqlite3.connect('data/integral.db')
            yield conn
            conn.close()
        except sqlite3.OperationalError:
            pytest.skip("Database not available for testing")

    def test_dx_cases(self, db_connection):
        """test validation on integrands containing 'dx'"""
        cursor = db_connection.cursor()

        query = '''
        SELECT ii.sympy_integrand, ii.sympy_variable, ii.integrand_canonical, ii.id
        FROM integral_instances ii
        WHERE ii.integrand_canonical LIKE '%dx%'
        LIMIT 10
        '''

        cursor.execute(query)
        rows = cursor.fetchall()

        if len(rows) == 0:
            pytest.skip("No dx cases found in database")

        rejected_count = 0
        for sympy_int, sympy_var, canonical, iid in rows:
            is_valid, reason, params = validate_parsed_symbols(sympy_int, sympy_var, debug=False)

            if not is_valid:
                rejected_count += 1
                assert 'dx' in reason.lower() if reason else False, \
                    f"Expected 'dx' in rejection reason for {iid}: {reason}"

        # expect most dx cases to be rejected
        assert rejected_count > 0, "Expected at least some dx cases to be rejected"

    def test_theta_cases(self, db_connection):
        """test validation on integrands containing 'theta'"""
        cursor = db_connection.cursor()

        query = '''
        SELECT ii.sympy_integrand, ii.sympy_variable, ii.integrand_canonical, ii.id
        FROM integral_instances ii
        WHERE ii.integrand_canonical LIKE '%theta%'
        LIMIT 10
        '''

        cursor.execute(query)
        rows = cursor.fetchall()

        if len(rows) == 0:
            pytest.skip("No theta cases found in database")

        rejected_count = 0
        for sympy_int, sympy_var, canonical, iid in rows:
            is_valid, reason, params = validate_parsed_symbols(sympy_int, sympy_var, debug=False)

            if not is_valid:
                rejected_count += 1
                assert 'theta' in reason.lower() if reason else False, \
                    f"Expected 'theta' in rejection reason for {iid}: {reason}"

        # expect most theta cases to be rejected
        assert rejected_count > 0, "Expected at least some theta cases to be rejected"

    def test_clean_cases(self, db_connection):
        """test validation on clean integrands (should mostly pass)"""
        cursor = db_connection.cursor()

        query = '''
        SELECT ii.sympy_integrand, ii.sympy_variable, ii.integrand_canonical, ii.id
        FROM integral_instances ii
        WHERE ii.integrand_canonical NOT LIKE '%dx%'
        AND ii.integrand_canonical NOT LIKE '%theta%'
        AND ii.integrand_canonical NOT LIKE '%eta%'
        LIMIT 20
        '''

        cursor.execute(query)
        rows = cursor.fetchall()

        if len(rows) == 0:
            pytest.skip("No clean cases found in database")

        accepted_count = 0
        rejected_count = 0

        for sympy_int, sympy_var, canonical, iid in rows:
            is_valid, reason, params = validate_parsed_symbols(sympy_int, sympy_var, debug=False)

            if is_valid:
                accepted_count += 1
                # check that params are reasonable
                if params:
                    assert len(params) <= 3, f"Too many params for {iid}: {len(params)}"
            else:
                rejected_count += 1

        # expect most clean cases to pass
        acceptance_rate = accepted_count / len(rows) if rows else 0
        assert acceptance_rate >= 0.5, f"Expected >=50% acceptance rate for clean cases, got {acceptance_rate:.1%}"

    def test_alpha_cases(self, db_connection):
        """test validation on integrands containing 'alpha' (should pass - whitelisted)"""
        cursor = db_connection.cursor()

        query = '''
        SELECT ii.sympy_integrand, ii.sympy_variable, ii.integrand_canonical, ii.id
        FROM integral_instances ii
        WHERE ii.integrand_canonical LIKE '%alpha%'
        LIMIT 10
        '''

        cursor.execute(query)
        rows = cursor.fetchall()

        if len(rows) == 0:
            pytest.skip("No alpha cases found in database")

        accepted_count = 0
        for sympy_int, sympy_var, canonical, iid in rows:
            is_valid, reason, params = validate_parsed_symbols(sympy_int, sympy_var, debug=False)

            if is_valid:
                accepted_count += 1
                # check that alpha is recognized as a parameter
                assert params is not None and 'alpha' in params, \
                    f"Expected 'alpha' in params for {iid}: {params}"

        # expect most alpha cases to be accepted (it's whitelisted)
        acceptance_rate = accepted_count / len(rows) if rows else 0
        assert acceptance_rate >= 0.7, \
            f"Expected >=70% acceptance rate for alpha cases, got {acceptance_rate:.1%}"

    def test_statistics(self, db_connection):
        """collect statistics on validation results across database"""
        cursor = db_connection.cursor()

        # get sample of integrands
        query = '''
        SELECT ii.sympy_integrand, ii.sympy_variable
        FROM integral_instances ii
        LIMIT 100
        '''

        cursor.execute(query)
        rows = cursor.fetchall()

        if len(rows) == 0:
            pytest.skip("No data found in database")

        total = len(rows)
        accepted = 0
        rejected = 0
        rejection_reasons = {}

        for sympy_int, sympy_var in rows:
            is_valid, reason, params = validate_parsed_symbols(sympy_int, sympy_var, debug=False)

            if is_valid:
                accepted += 1
            else:
                rejected += 1
                # categorize rejection reason
                if reason:
                    if 'spurious' in reason.lower():
                        rejection_reasons['spurious_symbols'] = rejection_reasons.get('spurious_symbols', 0) + 1
                    elif 'param' in reason.lower():
                        rejection_reasons['too_many_params'] = rejection_reasons.get('too_many_params', 0) + 1
                    else:
                        rejection_reasons['other'] = rejection_reasons.get('other', 0) + 1

        # basic sanity checks
        assert accepted + rejected == total, "Counts don't match"

        # log statistics (visible in pytest output with -v)
        print(f"\n=== Symbol Validation Statistics ===")
        print(f"Total: {total}")
        print(f"Accepted: {accepted} ({accepted/total:.1%})")
        print(f"Rejected: {rejected} ({rejected/total:.1%})")
        print(f"Rejection reasons: {rejection_reasons}")
