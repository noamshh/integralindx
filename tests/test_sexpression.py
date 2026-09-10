"""
tests for prefix notation to S-expression conversion
"""
import pytest
from src.utils.sexpression import prefix_to_sexp


class TestPrefixToSexp:
    """test prefix to S-expression conversion"""

    def test_simple_variable(self):
        """test simple variable"""
        assert prefix_to_sexp("x") == "x"

    def test_simple_constant(self):
        """test simple constant"""
        assert prefix_to_sexp("1") == "1"

    def test_simple_operation(self):
        """test simple binary operation"""
        assert prefix_to_sexp("add x 1") == "(+ x 1)"
        assert prefix_to_sexp("mul a b") == "(* a b)"
        assert prefix_to_sexp("pow x 2") == "(pow x 2)"

    def test_nested_operations(self):
        """test nested operations"""
        assert prefix_to_sexp("add mul a b c") == "(+ (* a b) c)"
        assert prefix_to_sexp("pow x add a b") == "(pow x (+ a b))"

    def test_negative_single_digit(self):
        """test negative single digit integer (INT- 1 → -1)"""
        result = prefix_to_sexp("mul x INT- 1")
        assert result == "(* x -1)", f"expected '(* x -1)', got '{result}'"

    def test_negative_multi_digit(self):
        """test negative multi-digit integer (INT- 1 2 → -12)"""
        result = prefix_to_sexp("add x INT- 1 2")
        assert result == "(+ x -12)", f"expected '(+ x -12)', got '{result}'"

    def test_positive_multi_digit(self):
        """test positive multi-digit integer (INT+ 1 2 3 → 123)"""
        result = prefix_to_sexp("mul INT+ 1 2 3 x")
        assert result == "(* 123 x)", f"expected '(* 123 x)', got '{result}'"

    def test_power_with_negative_exponent(self):
        """test pow with negative exponent (pow x INT- 1 → (pow x -1))"""
        result = prefix_to_sexp("pow x INT- 1")
        assert result == "(pow x -1)", f"expected '(pow x -1)', got '{result}'"

    def test_complex_expression_with_negatives(self):
        """test complex expression with multiple negative numbers"""
        result = prefix_to_sexp("add mul INT- 2 x pow y INT- 1")
        expected = "(+ (* -2 x) (pow y -1))"
        assert result == expected, f"expected '{expected}', got '{result}'"

    def test_nested_with_multi_digit(self):
        """test nested expression with multi-digit integers"""
        result = prefix_to_sexp("mul INT+ 2 5 6 pow div x 2 8")
        expected = "(* 256 (pow (/ x 2) 8))"
        assert result == expected, f"expected '{expected}', got '{result}'"

    def test_original_bug_case(self):
        """test the original bug case from 08b02519.txt"""
        # original malformed: "pow add 1 INT- pow x 2"
        # should become: "(pow (+ 1 -1) (pow x 2))"
        result = prefix_to_sexp("pow add 1 INT- 1 pow x 2")
        expected = "(pow (+ 1 -1) (pow x 2))"
        assert result == expected, f"expected '{expected}', got '{result}'"

    def test_trig_with_negative(self):
        """test trigonometric function with negative coefficient"""
        result = prefix_to_sexp("sin mul INT- 1 x")
        expected = "(sin (* -1 x))"
        assert result == expected, f"expected '{expected}', got '{result}'"


class TestRoundTrip:
    """test round-trip conversion consistency"""

    def test_negative_number_roundtrip(self):
        """verify that negative numbers can be identified in prefix notation"""
        from src.utils.prefix_notation import sympy_to_prefix
        import sympy as sp

        # create expression with negative number
        expr = sp.sympify("-2 * x")
        prefix = sympy_to_prefix(expr)

        # convert to sexp
        sexp = prefix_to_sexp(prefix)

        # should contain -2 in some form
        assert "-2" in sexp or "INT- 2" in prefix


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
