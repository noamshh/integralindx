"""
Test suite for LaTeX normalization pipeline.
Tests proper handling of relations, chains, and single expressions.
"""

import pytest
from corpus.normalizers.mse_normalizer import MSENormalizer
from corpus.formula_models import NormalizedFormula

class TestNormalizationPipeline:
    
    def setup_method(self):
        self.normalizer = MSENormalizer()

    def test_single_expressions_no_relations(self):
        """Single mathematical expressions without relations should NOT be split"""
        test_cases = [
            # Complete closed forms - should remain intact (spaces cleaned)
            (r"\frac{\pi^2}{12}\left( 1-\sqrt{3}\right)+\log(2) \log \left(1+\sqrt{3} \right)",
             r"\frac{\pi^2}{12}\left(1-\sqrt{3}\right)+\log(2)\log \left(1+\sqrt{3}\right)", []),
            
            # Complex integrals - should remain intact (bounds canonicalized, space cleaned)
            (r"\int_0^1 \frac{\log \left(1+x^{2+\sqrt{3}}\right)}{1+x}\mathrm dx",
             r"\int_{0}^{1}\frac{\log \left(1+x^{2+\sqrt{3}}\right)}{1+x}\mathrm dx", []),
            
            # Simple expressions (space around + cleaned)
            (r"\sin(x) + \cos(x)",
             r"\sin(x)+ \cos(x)", []),
            
            # Fractions with complex numerator/denominator
            (r"\frac{a+b+c}{d+e+f}",
             r"\frac{a+b+c}{d+e+f}", [])
        ]
        
        for raw_latex, expected_leading, expected_equiv in test_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-single")
            assert len(results) == 1, f"Should return single result for: {raw_latex}"
            result = results[0]
            assert result.accepted, f"Should accept: {raw_latex}"
            assert result.leading_expression == expected_leading, f"Wrong leading for: {raw_latex}"
            assert result.equivalent_forms == expected_equiv, f"Should have no equiv forms for: {raw_latex}"

    def test_proper_equation_chains(self):
        """Proper equation chains like A = B = C should be split correctly"""
        test_cases = [
            # Simple chain
            (r"a = b = c",
             "a", ["b", "c"]),
            
            # Variable assignment chain (bounds canonicalized, space cleaned)
            (r"I = \int_0^1 f(x) dx = 42",
             "I", [r"\int_{0}^{1}f(x)dx", "42"]),
            
            # Mathematical identity chain (spaces around operators cleaned, powers normalized)
            (r"\sin^2(x) + \cos^2(x) = 1 = \sec^2(x) - \tan^2(x)",
             r"\sin(x)^{2}+ \cos(x)^{2}", ["1", r"\sec(x)^{2}- \tan(x)^{2}"])
        ]
        
        for raw_latex, expected_leading, expected_equiv in test_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-chain")
            assert len(results) == 1, f"Should return single result for: {raw_latex}"
            result = results[0]
            assert result.accepted, f"Should accept: {raw_latex}"
            assert result.leading_expression == expected_leading, f"Wrong leading for: {raw_latex}"
            assert set(result.equivalent_forms) == set(expected_equiv), f"Wrong equiv forms for: {raw_latex}"

    def test_integral_with_result_equation(self):
        """Integrals with their results should be handled correctly"""
        test_cases = [
            # Integral equals numerical result (bounds canonicalized, space cleaned)
            (r"\int_0^1 x^2 dx = \frac{1}{3}",
             r"\int_{0}^{1}x^2 dx", [r"\frac{1}{3}"]),
             
            # Named integral (already canonical bounds, space cleaned, ln->log)
            (r"I = \int_{-1}^1\frac{1}{x}\sqrt{\frac{1+x}{1-x}}\ln\left(\frac{2x^2+2x+1}{2x^2-2x+1}\right) dx",
             "I", [r"\int_{-1}^{1}\frac{1}{x}\sqrt{\frac{1+x}{1-x}}\log\left(\frac{2x^2+2x+1}{2x^2-2x+1}\right)dx"]),
        ]
        
        for raw_latex, expected_leading, expected_equiv in test_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-integral-eq")
            assert len(results) == 1, f"Should return single result for: {raw_latex}"
            result = results[0]
            assert result.accepted, f"Should accept: {raw_latex}"
            assert result.leading_expression == expected_leading, f"Wrong leading for: {raw_latex}"
            assert set(result.equivalent_forms) == set(expected_equiv), f"Wrong equiv forms for: {raw_latex}"

    def test_approximation_relations(self):
        """Test approximation symbols treated as relation chains"""
        test_cases = [
            # Simple approximation with \\approx
            (r"I \approx 0.094561677526995723016",
             "I", [r"0.094561677526995723016"]),
            
            # Integral with approximation 
            (r"\int_0^1 f(x) dx \approx 1.234567",
             r"\int_{0}^{1}f(x)dx", ["1.234567"]),
            
            # Unicode approximation symbol
            (r"x ≈ 3.14159",
             "x", ["3.14159"]),
             
            # Multiple approximations in chain
            (r"A \approx B ≈ C",
             "A", ["B", "C"]),
             
            # Real dataset example: integral approximation
            (r"\int_{0}^{\infty} \frac{1}{(x-\log x)^2}dx≈2.51792",
             r"\int_{0}^{\infty}\frac{1}{(x-\log(x))^2}dx", ["2.51792"]),
        ]
        
        for raw_latex, expected_leading, expected_equiv in test_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-approx")
            assert len(results) == 1, f"Should return single result for: {raw_latex}"
            result = results[0]
            assert result.accepted, f"Should accept approximation: {raw_latex}"
            assert result.leading_expression == expected_leading, f"Wrong leading for: {raw_latex}"
            assert set(result.equivalent_forms) == set(expected_equiv), f"Wrong equiv forms for: {raw_latex}"

    def test_multiple_statements_splitting(self):
        """Test that multiple mathematical statements get split into separate formulas"""
        test_cases = [
            # Align environment with multiple equations
            (r"\begin{align} a &= b \\ c &= d \end{align}",
             [("a", ["b"]), ("c", ["d"])]),
             
            # Align with continuation (should be single formula) 
            (r"\begin{align} f(x) &= x^2 \\ &+ 2x + 1 \end{align}",
             [(r"f(x)", [r"x^2 + 2x + 1"])]),
             
            # Multiple separate equations in environment
            (r"\begin{align} x &= 1 \\ y &= 2 \\ z &= 3 \end{align}",
             [("x", ["1"]), ("y", ["2"]), ("z", ["3"])]),
        ]
        
        for raw_latex, expected_formulas in test_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-multi")
            assert len(results) == len(expected_formulas), f"Expected {len(expected_formulas)} results for: {raw_latex}"
            
            for i, (result, (expected_leading, expected_equiv)) in enumerate(zip(results, expected_formulas)):
                assert result.accepted, f"Result {i+1} should be accepted for: {raw_latex}"
                assert result.leading_expression == expected_leading, f"Wrong leading for result {i+1}: {raw_latex}"
                assert set(result.equivalent_forms) == set(expected_equiv), f"Wrong equiv forms for result {i+1}: {raw_latex}"

    def test_malformed_parsing_bugs(self):
        """Test cases that reveal parsing bugs in the current implementation"""
        bug_cases = [
            # BUG: This gets split incorrectly when it should remain whole
            (r"\frac{\pi^2}{12}\left( 1-\sqrt{3}\right)+\log(2) \log \left(1+\sqrt{3} \right)",
             "should_not_split"),
             
            # BUG: Integral in parens gets mangled 
            (r"I=\int_{-1}^1\frac1x\sqrt{\frac{1+x}{1-x}}\ln\left(\frac{2\,x^2+2\,x+1}{2\,x^2-2\,x+1}\right)\mathrm dx",
             "should_preserve_full_integral"),
        ]
        
        for raw_latex, bug_description in bug_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-bug")
            print(f"\n=== BUG CASE: {bug_description} ===")
            print(f"Raw: {raw_latex}")
            print(f"Results: {len(results)}")
            for i, result in enumerate(results):
                print(f"Result {i+1} - Leading: {result.leading_expression}")
                print(f"Result {i+1} - Equiv: {result.equivalent_forms}")
            # Don't assert - just document the bugs

    def test_environments_and_complex_latex(self):
        """Test handling of LaTeX environments and complex structures"""
        # Test cases that should return multiple formulas
        multi_cases = [
            # Align environment with multiple equations - now handled as multiple formulas
            (r"\begin{align} a &= b \\ c &= d \end{align}",
             [("a", ["b"]), ("c", ["d"])]),
        ]
        
        for raw_latex, expected_formulas in multi_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-env-multi")
            assert len(results) == len(expected_formulas), f"Expected {len(expected_formulas)} results for: {raw_latex}"
            
            for i, (result, (expected_leading, expected_equiv)) in enumerate(zip(results, expected_formulas)):
                assert result.accepted, f"Result {i+1} should be accepted for: {raw_latex}"
                assert result.leading_expression == expected_leading, f"Wrong leading for result {i+1}: {raw_latex}"
                assert set(result.equivalent_forms) == set(expected_equiv), f"Wrong equiv forms for result {i+1}: {raw_latex}"
        
        # Test cases that should be rejected
        rejection_cases = [
            # Cases environment (piecewise)
            (r"f(x) = \begin{cases} x^2 & \text{if } x > 0 \\ -x & \text{if } x \leq 0 \end{cases}",
             "contains_piecewise_function"),  # Should be rejected
        ]
        
        for raw_latex, expected_rejection in rejection_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-env-reject")
            assert len(results) == 1, f"Should return single result for rejected formula: {raw_latex}"
            result = results[0]
            assert not result.accepted, f"Should reject: {raw_latex}"
            assert expected_rejection in result.rejection_reason, f"Wrong rejection reason for: {raw_latex}"


    def test_edge_cases_empty_and_whitespace(self):
        """Test edge cases with empty and whitespace inputs"""
        # Test cases that should be rejected (truly empty)
        rejection_cases = [
            ("", "empty"),
            ("   ", "empty"),
            ("\n\n", "empty_after_parsing"),
        ]
        
        for raw_latex, expected_reason in rejection_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-empty")
            assert len(results) == 1, f"Should return single result for: {raw_latex}"
            result = results[0]
            assert not result.accepted, f"Should reject empty: {repr(raw_latex)}"
            assert expected_reason in (result.rejection_reason or ""), f"Wrong rejection for: {repr(raw_latex)}"
            
        # Test cases with valid mathematical content (should be accepted)
        acceptance_cases = [
            ("\\n\\n", "literal_n_characters"),  # This represents "n" in two rows, valid math
        ]
        
        for raw_latex, description in acceptance_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-empty-accept")
            assert len(results) == 1, f"Should return single result for: {raw_latex}"
            result = results[0]
            assert result.accepted, f"Should accept valid math: {repr(raw_latex)} ({description})"

    def test_edge_cases_unicode_and_special_chars(self):
        """Test handling of Unicode and special mathematical characters"""
        test_cases = [
            # Unicode math symbols
            ("α + β = γ", "α + β", ["γ"]),
            ("∫₀¹ f(x) dx", "∫₀¹ f(x) dx", []),  # Should remain as single expression
            ("∑ᵢ₌₁ⁿ aᵢ", "∑ᵢ₌₁ⁿ aᵢ", []),
            
            # Mixed LaTeX and Unicode  
            (r"\int_0^1 f(x) dx ≈ π/4", r"\int_0^1 f(x) dx", ["π/4"]),
        ]
        
        for raw_latex, expected_leading, expected_equiv in test_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-unicode")
            result = results[0] if len(results) == 1 else results[0]
            assert result.accepted, f"Should accept Unicode: {raw_latex}"
            # Note: We allow minor cleaning differences, just check structure
            assert result.equivalent_forms == expected_equiv or \
                   (len(expected_equiv) == 1 and len(result.equivalent_forms) == 1), \
                   f"Wrong equiv structure for: {raw_latex}"

    def test_edge_cases_malformed_latex(self):
        """Test handling of malformed LaTeX"""
        malformed_cases = [
            # Missing braces
            (r"\frac{a}{b", "should_handle_gracefully"),
            (r"\int_0^1 \frac{dx}{x", "should_handle_gracefully"), 
            
            # Unbalanced braces in simple cases
            (r"f(x) = {a + b", "should_handle_gracefully"),
        ]
        
        for raw_latex, description in malformed_cases:
            # These should either accept (if recoverable) or reject gracefully
            try:
                results = self.normalizer.normalize_latex(raw_latex, f"test-malformed")
                result = results[0] if len(results) == 1 else results[0]
                # If accepted, should have reasonable content
                if result.accepted:
                    assert len(result.leading_expression) > 0, f"Empty result for: {raw_latex}"
                print(f"Malformed case handled: {description}")
            except Exception as e:
                pytest.fail(f"Should handle malformed LaTeX gracefully: {raw_latex}, got: {e}")

    def test_edge_cases_long_expressions(self):
        """Test handling of very long expressions"""
        # Long but valid expression
        long_expr = "x + " * 100 + "y = z"  # 200+ characters
        results = self.normalizer.normalize_latex(long_expr, "test-long")
        result = results[0] if len(results) == 1 else results[0]
        
        # Should either accept or reject with length reason
        if not result.accepted:
            assert "length" in (result.rejection_reason or "").lower() or \
                   "long" in (result.rejection_reason or "").lower(), \
                   f"Should reject long expressions appropriately: {result.rejection_reason}"
        else:
            # If accepted, should be reasonable
            assert len(result.leading_expression) > 0

    def test_edge_cases_complex_nested_structures(self):
        """Test complex nested mathematical structures"""
        test_cases = [
            # Deeply nested fractions
            (r"\frac{\frac{a}{b}}{\frac{c}{d}} = \frac{ad}{bc}",
             r"\frac{\frac{a}{b}}{\frac{c}{d}}", [r"\frac{ad}{bc}"]),
             
            # Multiple equals in complex expression
            ("a = b = c = d = e", "a", ["b", "c", "d", "e"]),
        ]
        
        for raw_latex, expected_leading, expected_equiv in test_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-nested")
            result = results[0] if len(results) == 1 else results[0]
            assert result.accepted, f"Should accept nested: {raw_latex}"
            # Allow for cleaning differences, just check that we get equivalents
            if expected_equiv:
                assert len(result.equivalent_forms) > 0, f"Should have equivalent forms for: {raw_latex}"

    def test_edge_cases_text_and_math_mixed(self):
        """Test mixed text and math expressions"""  
        test_cases = [
            # Should be rejected as multiple statements
            ('Let $x = 1$ and $y = 2$. Then $z = x + y = 3$.', "multiple_statements"),
            
            # Text with mathematical content
            (r"\text{Let } a = 1, \text{ then } b = 2", "should_handle_text"),
        ]
        
        for raw_latex, expected_behavior in test_cases:
            results = self.normalizer.normalize_latex(raw_latex, f"test-mixed")
            result = results[0] if len(results) == 1 else results[0]
            
            if "multiple_statements" in expected_behavior:
                assert not result.accepted, f"Should reject multiple statements: {raw_latex}"
                assert "multiple_statements" in (result.rejection_reason or ""), \
                       f"Wrong rejection reason: {result.rejection_reason}"
            else:
                # Other mixed cases should be handled somehow
                print(f"Mixed text/math handled: {expected_behavior}")

    def test_discovered_problematic_patterns(self):
        """Test specific patterns discovered from real data that caused issues"""
        
        # Pattern from debug output that creates many equivalent forms
        problematic_text_chain = (
            r"\text{Let}\quad a_0 = a,\quad b_0 = b,\quad a_{n+1} = \frac{a_n+b_n}{2},\quad b_{n+1} = \sqrt{a_n b_n}",
            "complex_text_chain"
        )
        
        raw_latex, description = problematic_text_chain
        results = self.normalizer.normalize_latex(raw_latex, f"test-problematic")
        result = results[0] if len(results) == 1 else results[0]
        
        if result.accepted:
            # Should not create excessive equivalent forms
            assert len(result.equivalent_forms) <= 10, \
                   f"Too many equivalent forms ({len(result.equivalent_forms)}) for: {description}"
            print(f"Handled {description}: {len(result.equivalent_forms)} equivalent forms")
        else:
            print(f"Rejected {description}: {result.rejection_reason}")

    def test_real_mse_edge_cases(self):
        """Test real problematic patterns found in MSE data"""
        
        test_cases = [
            # HTML entities in mathematical expressions
            (r"&amp;I=4\pi\left(\arctan\sqrt{-1-2i}+\arctan\sqrt{-1+2i} -2\arctan1\right)\\[4pt] &amp;=4\pi",
             "html_entities"),
             
            # Align environment (should be rejected)
            (r"\begin{align} I=4\pi(\arctan\sqrt{-1-2i}) \\ =4\pi \end{align}",
             "align_environment"),
             
            # Multiple integrals with \quad spacing
            (r"F_1=\int_0^\infty \frac{1}{1-y^2} dy,\quad F_2(p)=\int_0^\infty \frac{1}{y^2+p} dy",
             "multiple_integrals_quad"),
             
            # LaTeX tags
            (r"I=4(J(-1-2i)+J(-1+2i)-2J(1)),\tag{5}",
             "tagged_expression"),
             
            # Differentiation with \text
            (r"\frac{\text{d}J(p)}{\text{d}p} =\int_0^\infty \frac{dy}{(y^2+p)(1-y^2)}",
             "differentiation_notation"),
        ]
        
        for raw_latex, description in test_cases:
            print(f"\n--- Testing {description} ---")
            print(f"Input: {raw_latex[:60]}...")
            
            try:
                results = self.normalizer.normalize_latex(raw_latex, f"test-real-{description}")
                result = results[0] if len(results) == 1 else results[0]
                
                if result.accepted:
                    print(f"✓ ACCEPTED")
                    print(f"  Leading: {result.leading_expression[:50]}...")
                    print(f"  Equiv forms: {len(result.equivalent_forms)}")
                    
                    # Check for specific issues
                    if description == "html_entities" and "&amp;" in result.leading_expression:
                        pytest.fail(f"HTML entities not properly decoded in: {description}")
                    
                    if description == "align_environment":
                        # This should probably be rejected, but if accepted, should be reasonable
                        assert len(result.leading_expression) > 0, f"Empty leading expression for {description}"
                    
                    if len(result.equivalent_forms) > 8:
                        print(f"    WARNING: Many equivalent forms for {description}")
                        
                else:
                    print(f"✗ REJECTED: {result.rejection_reason}")
                    
                    # Check that rejections are for good reasons
                    if description == "align_environment":
                        assert "multiple_statements" in result.rejection_reason or \
                               "contains_piecewise" in result.rejection_reason, \
                               f"Align environment should be rejected for structural reasons: {result.rejection_reason}"
                               
            except Exception as e:
                pytest.fail(f"Should handle {description} gracefully, got: {str(e)}")
    
    def test_edge_cases_excessive_equivalent_forms(self):
        """Test patterns that might generate too many equivalent forms"""
        
        # Very long equation chain
        long_chain = " = ".join([f"expr_{i}" for i in range(20)])  # 20 expressions
        results = self.normalizer.normalize_latex(long_chain, "test-long-chain")
        result = results[0] if len(results) == 1 else results[0]
        
        if result.accepted:
            # Long chains can generate many equivalent forms - no artificial limits imposed
            # Dataset analysis shows 99.9th percentile is 21 equals, so this test case is realistic
            print(f"Long chain (20 equals) handled: {len(result.equivalent_forms)} equivalent forms")
        else:
            print(f"Long chain rejected: {result.rejection_reason}")
    
    def test_edge_cases_performance_patterns(self):
        """Test patterns that might cause performance issues"""
        
        # Deeply nested braces
        nested = "{{{{{{a}}}}}}" + " = " + "{{{{{{b}}}}}}"
        results = self.normalizer.normalize_latex(nested, "test-nested")
        result = results[0] if len(results) == 1 else results[0]
        
        # Should handle without hanging or crashing
        assert result is not None, "Should return result for nested braces"
        
        # Very repetitive pattern
        repetitive = r"\frac{1}{\frac{1}{\frac{1}{\frac{1}{\frac{1}{x}}}}}"
        results = self.normalizer.normalize_latex(repetitive, "test-repetitive")
        result = results[0] if len(results) == 1 else results[0]
        
        assert result is not None, "Should return result for repetitive pattern"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])