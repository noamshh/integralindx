"""
Test integral bounds canonicalization functionality.
"""

import pytest
from src.utils.latex_cleaning import _canonicalize_integral_bounds


class TestIntegralBoundsCanonicalization:
    """Test integral bounds canonicalization with various formats."""

    def test_simple_unbraced_bounds(self):
        r"""Test simple unbraced bounds like \int_a^b."""
        test_cases = [
            (r'\int_0^1', r'\int_{0}^{1}'),
            (r'\int_a^b', r'\int_{a}^{b}'),
            (r'\int_{-1}^1', r'\int_{-1}^{1}'),
            (r'\int_0^{\infty}', r'\int_{0}^{\infty}'),
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"

    def test_complex_bounds_with_nested_braces(self):
        r"""Test complex bounds with nested braces like \frac{\pi}{2}."""
        test_cases = [
            (r'\int_0^{\frac{\pi}{2}}', r'\int_{0}^{\frac{\pi}{2}}'),
            (r'\int_{\frac{\pi}{4}}^{\frac{\pi}{2}}', r'\int_{\frac{\pi}{4}}^{\frac{\pi}{2}}'),  # already canonical
            (r'\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}', r'\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}'),  # already canonical
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"

    def test_reverse_order_bounds(self):
        r"""Test bounds in reverse order \int^b_a -> \int_a^b."""
        test_cases = [
            (r'\int^1_0', r'\int_{0}^{1}'),
            (r'\int^{\infty}_0', r'\int_{0}^{\infty}'),
            (r'\int^{1}_{0}', r'\int_{0}^{1}'),
            (r'\int^{\frac{\pi}{2}}_{0}', r'\int_{0}^{\frac{\pi}{2}}'),
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"

    def test_spaced_bounds(self):
        r"""Test bounds with spaces after \int."""
        test_cases = [
            (r'\int _0^1', r'\int_{0}^{1}'),
            (r'\int _0^{\frac{\pi}{2}}', r'\int_{0}^{\frac{\pi}{2}}'),
            (r'\int ^1_0', r'\int_{0}^{1}'),
            (r'\int ^{\infty}_{0}', r'\int_{0}^{\infty}'),
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"

    def test_single_bounds_subscript_only(self):
        r"""Test single bounds (subscript only) like \int_a."""
        test_cases = [
            (r'\int_0 f(x) dx', r'\int_{0} f(x) dx'),
            (r'\int_a f(x) dx', r'\int_{a} f(x) dx'),
            (r'\int_{\mathbb{R}} f(x) dx', r'\int_{\mathbb{R}} f(x) dx'),  # already canonical
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"

    def test_already_canonical_forms(self):
        """Test expressions that are already in canonical form."""
        test_cases = [
            r'\int_{0}^{1}',
            r'\int_{a}^{b}',
            r'\int_{0}^{\infty}',
            r'\int_{-1}^{1}',
            r'\int_{0}^{\frac{\pi}{2}}',
            r'\int_{\frac{\pi}{4}}^{\frac{\pi}{2}}',
        ]
        
        for input_expr in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == input_expr, f"Canonical form should be unchanged: {input_expr} -> {result}"

    def test_indefinite_integrals_unchanged(self):
        """Test that indefinite integrals (no bounds) are unchanged."""
        test_cases = [
            r'\int f(x) dx',
            r'\int \sin(x) dx',
            r'\int \frac{1}{x} dx',
        ]
        
        for input_expr in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == input_expr, f"Indefinite integral should be unchanged: {input_expr} -> {result}"

    def test_complete_expressions(self):
        """Test canonicalization within complete mathematical expressions."""
        test_cases = [
            (r'I = \int_0^1 x^2 dx = \frac{1}{3}', r'I = \int_{0}^{1} x^2 dx = \frac{1}{3}'),
            (r'\int^{\pi}_0 \sin(x) dx = 2', r'\int_{0}^{\pi} \sin(x) dx = 2'),
            (r'The integral \int_0^{\frac{\pi}{2}} \cos(x) dx equals 1', 
             r'The integral \int_{0}^{\frac{\pi}{2}} \cos(x) dx equals 1'),
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"

    def test_no_integrals(self):
        """Test expressions with no integrals are unchanged."""
        test_cases = [
            r'x^2 + y^2 = 1',
            r'\frac{d}{dx} \sin(x) = \cos(x)',
            r'\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}',
        ]
        
        for input_expr in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == input_expr, f"No integral expression should be unchanged: {input_expr} -> {result}"

    def test_multiple_integrals(self):
        """Test expressions with multiple integrals."""
        test_cases = [
            (r'\int_0^1 f(x) dx + \int^2_1 g(x) dx', 
             r'\int_{0}^{1} f(x) dx + \int_{1}^{2} g(x) dx'),
            (r'\int_a^b f(x) dx = \int_c^d g(y) dy',
             r'\int_{a}^{b} f(x) dx = \int_{c}^{d} g(y) dy'),
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"

    def test_edge_cases(self):
        """Test edge cases and potential problematic inputs."""
        test_cases = [
            ('', ''),  # empty string
            (r'\int', r'\int'),  # just \int
            (r'not an integral', r'not an integral'),  # no \int
            (r'\int_{}^{}', r'\int_{}^{}'),  # empty bounds
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"

    def test_fraction_interference_bug(self):
        """Test that integral canonicalization doesn't interfere with fractions."""
        test_cases = [
            # The specific bug case from MSE data
            (r'\int_{-1}^1\frac{1}{x}', r'\int_{-1}^{1}\frac{1}{x}'),
            (r'\int_0^1\frac{dx}{x}', r'\int_{0}^{1}\frac{dx}{x}'),
            (r'\int_a^b\frac{\sin(x)}{x}dx', r'\int_{a}^{b}\frac{\sin(x)}{x}dx'),
            # Make sure we don't capture \frac as part of bounds
            (r'\int_{-1}^1 \frac{1}{x} \sqrt{\frac{1+x}{1-x}}', r'\int_{-1}^{1} \frac{1}{x} \sqrt{\frac{1+x}{1-x}}'),
        ]
        
        for input_expr, expected in test_cases:
            result = _canonicalize_integral_bounds(input_expr)
            assert result == expected, f"Input: {input_expr} -> Expected: {expected}, Got: {result}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])