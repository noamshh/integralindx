"""
test round-trip conversion through E-Gen wrapper
"""
import pytest
from src.utils.sexpression import prefix_to_sexp
from src.egraph.egen_wrapper import _convert_numeric_token


class TestEGenRoundTrip:
    """test round-trip conversion of numeric tokens"""

    def test_convert_negative_one(self):
        """test converting -1 back to INT- 1 format"""
        result = _convert_numeric_token("-1")
        assert result == ["INT-", "1"], f"expected ['INT-', '1'], got {result}"

    def test_convert_negative_multi_digit(self):
        """test converting -42 back to INT- format"""
        result = _convert_numeric_token("-42")
        assert result == ["INT-", "4", "2"], f"expected ['INT-', '4', '2'], got {result}"

    def test_convert_positive_multi_digit(self):
        """test converting 123 back to INT+ format"""
        result = _convert_numeric_token("123")
        assert result == ["INT+", "1", "2", "3"], f"expected ['INT+', '1', '2', '3'], got {result}"

    def test_convert_single_digit(self):
        """test single digit stays as-is"""
        result = _convert_numeric_token("5")
        assert result == ["5"], f"expected ['5'], got {result}"

    def test_convert_zero(self):
        """test zero stays as-is"""
        result = _convert_numeric_token("0")
        assert result == ["0"], f"expected ['0'], got {result}"

    def test_convert_decimal_to_rational(self):
        """test decimal converts to div format"""
        result = _convert_numeric_token("0.5")
        assert result == ["div", "1", "2"], f"expected ['div', '1', '2'], got {result}"

    def test_full_roundtrip_negative(self):
        """test full round-trip: prefix → sexp → egen mock → prefix"""
        # start with prefix notation
        prefix_in = "pow x INT- 1"

        # convert to S-expression (what we send to E-Gen)
        sexp = prefix_to_sexp(prefix_in)
        assert sexp == "(pow x -1)"

        # simulate E-Gen output (returns S-expression tokens)
        # E-Gen would output something like: (pow x -1)
        # our wrapper splits this and converts each token
        egen_output_tokens = ["pow", "x", "-1"]

        # simulate what egen_wrapper does: convert each token
        converted = []
        for token in egen_output_tokens:
            numeric_tokens = _convert_numeric_token(token)
            converted.extend(numeric_tokens)

        # should get back to prefix-like format
        result = " ".join(converted)
        assert result == "pow x INT- 1", f"expected 'pow x INT- 1', got '{result}'"

    def test_full_roundtrip_multi_digit_negative(self):
        """test full round-trip with multi-digit negative"""
        # start with prefix notation
        prefix_in = "mul INT- 4 2 x"

        # convert to S-expression
        sexp = prefix_to_sexp(prefix_in)
        assert sexp == "(* -42 x)"

        # simulate E-Gen output tokens
        egen_output_tokens = ["*", "-42", "x"]

        # convert back
        converted = []
        for token in egen_output_tokens:
            # map operators
            if token == "*":
                token = "mul"
            elif token == "+":
                token = "add"
            # then convert numeric
            numeric_tokens = _convert_numeric_token(token)
            converted.extend(numeric_tokens)

        result = " ".join(converted)
        assert result == "mul INT- 4 2 x", f"expected 'mul INT- 4 2 x', got '{result}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
