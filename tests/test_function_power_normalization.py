"""
Test suite for function power normalization bug fix.

This tests the fix for the critical bug where ln^3(sin(x)) was incorrectly
transformed to ln(sin(x)^3) instead of staying as ln^3(sin(x)).
"""

import pytest
from src.utils.latex_cleaning import _normalize_function_powers


class TestFunctionPowerNormalization:
    """Test function power normalization with real-world examples from MSE data"""

    def test_simple_function_powers_should_transform(self):
        """Test that simple function powers are correctly transformed"""
        test_cases = [
            # Single variable cases
            (r'\sin^2 x', r'\sin(x)^{2}'),
            (r'\cos^3 y', r'\cos(y)^{3}'),
            (r'\tan^4 z', r'\tan(z)^{4}'),
            (r'\ln^2 t', r'\ln(t)^{2}'),
            (r'\log^5 w', r'\log(w)^{5}'),

            # Simple parentheses cases
            (r'\sin^2(x)', r'\sin(x)^{2}'),
            (r'\cos^3(y)', r'\cos(y)^{3}'),
            (r'\log^2(z)', r'\log(z)^{2}'),
            (r'\ln^4(t)', r'\ln(t)^{4}'),

            # Simple expressions in parentheses (no LaTeX commands)
            (r'\cos^3(x + y)', r'\cos(x + y)^{3}'),
            (r'\sin^2(x/2)', r'\sin(x/2)^{2}'),
            (r'\tan^2(2*x)', r'\tan(2*x)^{2}'),
            (r'\log^3(x - 1)', r'\log(x - 1)^{3}'),
        ]

        for input_expr, expected in test_cases:
            result = _normalize_function_powers(input_expr)
            assert result == expected, f"Failed for {input_expr}: got {result}, expected {expected}"

    def test_nested_functions_should_not_transform(self):
        """Test that nested functions are NOT incorrectly transformed"""
        test_cases = [
            # The original bug case
            (r'\ln^3(\sin(x))', r'\ln^3(\sin(x))'),

            # Real MSE examples
            (r'\sin^2(\tan(x))', r'\sin^2(\tan(x))'),
            (r'\cos^4(\log(x))', r'\cos^4(\log(x))'),
            (r'\tan^3(\arcsin(x))', r'\tan^3(\arcsin(x))'),

            # Complex nested cases
            (r'\log^2(\cos(x) + \sin(x))', r'\log^2(\cos(x) + \sin(x))'),
            (r'\sin^5(\tan(x) + \cos(x))', r'\sin^5(\tan(x) + \cos(x))'),
        ]

        for input_expr, expected in test_cases:
            result = _normalize_function_powers(input_expr)
            assert result == expected, f"Failed for {input_expr}: got {result}, expected {expected}"

    def test_latex_commands_should_not_transform(self):
        """Test that expressions with LaTeX commands are preserved"""
        test_cases = [
            # Fractions
            (r'\sin^2(\frac{x}{2})', r'\sin^2(\frac{x}{2})'),
            (r'\log^3(\frac{x + 1}{x - 1})', r'\log^3(\frac{x + 1}{x - 1})'),

            # Square roots
            (r'\cos^4(\sqrt{x})', r'\cos^4(\sqrt{x})'),
            (r'\tan^2(\sqrt{x + 1})', r'\tan^2(\sqrt{x + 1})'),

            # Inverse functions
            (r'\sin^3(\arctan(x))', r'\sin^3(\arctan(x))'),
            (r'\log^2(\arcsin(x))', r'\log^2(\arcsin(x))'),

            # Exponentials (e^x has no backslash so transforms, \exp has backslash so doesn't)
            (r'\cos^2(e^x)', r'\cos(e^x)^{2}'),  # e^x is just letter+caret, no LaTeX command
            (r'\ln^4(\exp(x))', r'\ln^4(\exp(x))'),  # \exp is a LaTeX command

            # Summations/products
            (r'\sin^2(\sum_{i=1}^n x_i)', r'\sin^2(\sum_{i=1}^n x_i)'),
            (r'\log^3(\prod_{i=1}^n x_i)', r'\log^3(\prod_{i=1}^n x_i)'),
        ]

        for input_expr, expected in test_cases:
            result = _normalize_function_powers(input_expr)
            assert result == expected, f"Failed for {input_expr}: got {result}, expected {expected}"

    def test_real_world_mse_examples(self):
        """Test with actual patterns found in MSE integral data"""
        test_cases = [
            # From actual MSE data - should NOT transform
            (r'\sin^2{(\tan{x})}', r'\sin^2{(\tan{x})}'),  # MSE: nested with braces
            (r'\ln^2(\sqrt{x})', r'\ln^2(\sqrt{x})'),      # MSE: log with sqrt

            # From actual MSE data - should transform (simple cases)
            (r'\ln^2 z', r'\ln(z)^{2}'),                   # MSE: simple log squared
            (r'\tanh^2(x)', r'\tanh(x)^{2}'),              # MSE: hyperbolic function

            # Edge cases from real data
            (r'\log(\sin(x)^{3})', r'\log(\sin(x)^{3})'),  # Already processed incorrectly
        ]

        for input_expr, expected in test_cases:
            result = _normalize_function_powers(input_expr)
            assert result == expected, f"Failed for {input_expr}: got {result}, expected {expected}"

    def test_multiple_function_powers_in_expression(self):
        """Test expressions with multiple function powers"""
        test_cases = [
            # Multiple simple powers - should all transform
            (r'\sin^2 x + \cos^2 x', r'\sin(x)^{2} + \cos(x)^{2}'),
            (r'\log^2 x \cdot \sin^3 y', r'\log(x)^{2} \cdot \sin(y)^{3}'),

            # Mixed simple and complex - only simple should transform
            (r'\sin^2 x + \cos^3(\tan(y))', r'\sin(x)^{2} + \cos^3(\tan(y))'),
            (r'\log^2(x) + \ln^3(\sin(z))', r'\log(x)^{2} + \ln^3(\sin(z))'),
        ]

        for input_expr, expected in test_cases:
            result = _normalize_function_powers(input_expr)
            assert result == expected, f"Failed for {input_expr}: got {result}, expected {expected}"

    def test_edge_cases_and_malformed_input(self):
        """Test edge cases and potentially malformed input"""
        test_cases = [
            # Empty and whitespace
            ('', ''),
            ('   ', '   '),

            # No powers
            (r'\sin(x) + \cos(y)', r'\sin(x) + \cos(y)'),

            # Powers without parentheses or variables (should be unchanged)
            (r'\sin^2', r'\sin^2'),
            (r'\log^3', r'\log^3'),

            # Complex power expressions (actually transform since content has no backslash)
            (r'\sin^{2 + 3}(x)', r'\sin(x)^{{2 + 3}}'),  # Complex power but no LaTeX in parentheses

            # Mixed with non-function text
            (r'Let \sin^2 x = y', r'Let \sin(x)^{2} = y'),
        ]

        for input_expr, expected in test_cases:
            result = _normalize_function_powers(input_expr)
            assert result == expected, f"Failed for {input_expr}: got {result}, expected {expected}"

    def test_braced_powers(self):
        """Test function powers with braced exponents"""
        test_cases = [
            # Simple braced powers - should transform
            (r'\sin^{2}(x)', r'\sin(x)^{{2}}'),
            (r'\cos^{10}(y)', r'\cos(y)^{{10}}'),
            (r'\tan^{n}(z)', r'\tan(z)^{{n}}'),

            # Braced powers with nested functions - should NOT transform
            (r'\log^{3}(\sin(x))', r'\log^{3}(\sin(x))'),
            (r'\sin^{4}(\arctan(y))', r'\sin^{4}(\arctan(y))'),
        ]

        for input_expr, expected in test_cases:
            result = _normalize_function_powers(input_expr)
            assert result == expected, f"Failed for {input_expr}: got {result}, expected {expected}"


class TestFunctionPowerIntegration:
    """Test that the function power fix integrates properly with the full normalization pipeline"""

    def test_integration_with_normalization_pipeline(self):
        """Test that the fix works within the complete LaTeX cleaning pipeline"""
        from src.utils.latex_cleaning import _normalize_special_functions

        # Test full pipeline with the bug case
        test_input = r'\ln^3(\sin(x))'

        # Step 1: Function power normalization (should NOT transform)
        step1 = _normalize_function_powers(test_input)
        assert step1 == r'\ln^3(\sin(x))', f"Function power step failed: {step1}"

        # Step 2: Special function normalization (ln -> log)
        step2 = _normalize_special_functions(step1)
        assert r'\log^3(\sin(' in step2, f"Special function step failed: {step2}"

        # The bug was that step1 would produce \ln(\sin(x)^3) instead of \ln^3(\sin(x))
        # With our fix, it should preserve the structure correctly

    def test_mathematical_correctness(self):
        """Test that the fix preserves mathematical correctness"""
        # These should have different mathematical meanings and be handled differently
        cases = [
            # [log(sin(x))]^3 - cube of the logarithm result
            (r'\ln^3(\sin(x))', 'should_stay_unchanged'),

            # log([sin(x)]^3) = log(sin^3(x)) = 3*log(sin(x)) - different meaning
            # Note: \sin^3(x) inside gets transformed to \sin(x)^3, which is correct
            (r'\ln(\sin^3(x))', r'\ln(\sin(x)^{3})'),  # Inner \sin^3 transforms
            (r'\ln(\sin(x)^3)', 'should_stay_unchanged'),  # Already in correct form

            # Simple case: [sin(x)]^2 - square of sine result
            (r'\sin^2(x)', r'\sin(x)^{2}'),
        ]

        for input_expr, expected_behavior in cases:
            result = _normalize_function_powers(input_expr)
            if expected_behavior == 'should_stay_unchanged':
                assert result == input_expr, f"Should preserve: {input_expr}, got: {result}"
            else:
                assert result == expected_behavior, f"Should transform {input_expr} to {expected_behavior}, got: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])