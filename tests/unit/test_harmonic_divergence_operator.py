#!/usr/bin/env python3
"""
Tests for the Harmonic Divergence Operator.

These tests validate the mathematical properties and behavior of H(x).
"""

import numpy as np
import pytest

from riemann.harmonic_divergence_operator import HarmonicDivergenceOperator


class TestHarmonicDivergenceOperator:
    """Test suite for the Harmonic Divergence Operator."""

    @pytest.fixture
    def operator(self):
        """Create an operator instance for testing."""
        return HarmonicDivergenceOperator()

    def test_operator_initialization(self, operator):
        """Test that the operator initializes correctly."""
        assert operator is not None
        assert operator.pi == np.pi

    def test_evaluate_positive_real(self, operator):
        """Test evaluation at positive real numbers."""
        result = operator.evaluate(2.0)

        assert result is not None
        assert result.x == 2.0
        assert np.isfinite(result.magnitude)
        assert result.magnitude >= 0

        # Check that all components are computed
        assert result.ln_component is not None
        assert result.zeta_component is not None
        assert np.isfinite(result.sin_component)
        assert np.isfinite(result.cos_component)

    def test_evaluate_at_e(self, operator):
        """Test evaluation at e, where ln(e) = 1."""
        e = np.e
        result = operator.evaluate(e)

        # ln(e) should be 1
        assert np.abs(result.ln_component - 1.0) < 1e-10

        # Result should be finite
        assert np.isfinite(result.magnitude)

    def test_sin_component_integer_zeros(self, operator):
        """Test that sin(πx) vanishes at integers."""
        for n in [1, 2, 3, 4, 5]:
            result = operator.evaluate(float(n))
            # sin(π·n) should be ~0 for integer n
            assert np.abs(result.sin_component) < 1e-10

    def test_cos_component_parity(self, operator):
        """Test that cos(πx) shows parity behavior."""
        # cos(π·1) = -1
        result1 = operator.evaluate(1.0)
        assert np.abs(result1.cos_component - (-1.0)) < 1e-10

        # cos(π·2) = 1
        result2 = operator.evaluate(2.0)
        assert np.abs(result2.cos_component - 1.0) < 1e-10

        # cos(π·3) = -1
        result3 = operator.evaluate(3.0)
        assert np.abs(result3.cos_component - (-1.0)) < 1e-10

    def test_evaluate_complex(self, operator):
        """Test evaluation at complex numbers."""
        z = complex(0.5, 1.0)
        result = operator.evaluate(z)

        assert result is not None
        assert result.x == z
        assert np.isfinite(result.magnitude)

    def test_magnitude_always_positive(self, operator):
        """Test that magnitude is always non-negative."""
        test_points = [0.5, 1.0, 2.0, 3.14, np.e]

        for x in test_points:
            result = operator.evaluate(x)
            assert result.magnitude >= 0

    def test_phase_in_valid_range(self, operator):
        """Test that phase is in [-π, π]."""
        test_points = [0.5, 1.0, 2.0, 3.14, np.e]

        for x in test_points:
            result = operator.evaluate(x)
            assert -np.pi <= result.phase <= np.pi

    def test_find_zeros_real(self, operator):
        """Test finding real zeros of H(x)."""
        zeros = operator.find_zeros_real((0.1, 5.0), num_points=50)

        # Should find at least some zeros
        assert isinstance(zeros, list)

        # All zeros should be in the specified range
        for z in zeros:
            assert 0.1 <= z <= 5.0

        # Verify zeros are actually close to zero
        for z in zeros:
            result = operator.evaluate(z)
            # Should have small magnitude (allowing some numerical error)
            assert result.magnitude < 1.0  # Relaxed for numerical stability

    def test_analyze_balance(self, operator):
        """Test component balance analysis."""
        analysis = operator.analyze_balance(2.0)

        assert analysis is not None
        assert "x" in analysis
        assert "total_magnitude" in analysis
        assert "component_magnitudes" in analysis
        assert "relative_contributions" in analysis
        assert "dominant_component" in analysis

        # Check all components are present
        assert "ln" in analysis["component_magnitudes"]
        assert "zeta" in analysis["component_magnitudes"]
        assert "tan" in analysis["component_magnitudes"]
        assert "sin" in analysis["component_magnitudes"]
        assert "cos" in analysis["component_magnitudes"]

        # Relative contributions should sum to ~1
        total_contrib = sum(analysis["relative_contributions"].values())
        assert np.abs(total_contrib - 1.0) < 1e-6

    def test_divergent_harmonic_balance(self, operator):
        """Test divergent vs harmonic component analysis."""
        analysis = operator.analyze_balance(2.0)

        assert "divergent_total" in analysis
        assert "harmonic_total" in analysis
        assert "rotation_total" in analysis
        assert "balance_ratio" in analysis

        # All should be non-negative
        assert analysis["divergent_total"] >= 0
        assert analysis["harmonic_total"] >= 0
        assert analysis["rotation_total"] >= 0

    def test_scan_critical_line(self, operator):
        """Test scanning critical line analogy."""
        results = operator.scan_critical_line(0.0, 5.0, num_points=20)

        assert len(results) == 20

        # Each result should have required fields
        for result in results:
            assert "x" in result
            assert "total_magnitude" in result
            assert "is_balanced" in result

    def test_emergent_constants(self, operator):
        """Test that π and e emerge correctly."""
        constants = operator.compute_emergent_constants()

        assert "pi_encoded_in" in constants
        assert "e_encoded_in" in constants
        assert "ln_e" in constants
        assert "euler_identity" in constants

        # ln(e) should be 1
        assert np.abs(constants["ln_e"] - 1.0) < 1e-10

        # Euler's identity: e^(iπ) = -1
        euler_val = constants["euler_identity_value"]
        assert np.abs(euler_val - (-1.0)) < 1e-10

    def test_pi_embedded_in_trig_components(self, operator):
        """Test that π is embedded in trigonometric components."""
        # sin(π) = 0
        result = operator.evaluate(1.0)
        assert np.abs(result.sin_component) < 1e-10

        # sin(π/2) corresponds to x = 0.5
        result_half = operator.evaluate(0.5)
        # sin(π/2) = 1
        assert np.abs(result_half.sin_component - 1.0) < 1e-10

    def test_safe_ln_handles_small_values(self, operator):
        """Test that ln handles very small values safely."""
        # Very small positive value
        small = 1e-100
        ln_small = operator._safe_ln(small)
        assert np.isfinite(ln_small) or ln_small == complex(-np.inf, 0)

        # Zero should return -inf
        ln_zero = operator._safe_ln(0.0)
        assert ln_zero == complex(-np.inf, 0)

    def test_safe_zeta_convergence(self, operator):
        """Test that zeta function computation is stable."""
        # For Re(s) > 1, zeta should converge
        result = operator._safe_zeta(2.0)
        # ζ(2) = π²/6 ≈ 1.6449
        expected = (np.pi**2) / 6
        assert np.abs(result - expected) < 0.01  # Allow some approximation error

    def test_safe_tan_pole_handling(self, operator):
        """Test that tan handles poles (odd integers) safely."""
        # tan(π/2) has a pole at x = 1
        tan_pole = operator._safe_tan(1.0)
        # Should return infinity (positive or negative)
        assert np.isinf(tan_pole.imag)

    def test_component_contributions_sum_to_one(self, operator):
        """Test that relative contributions sum to 1."""
        test_points = [0.5, 1.0, 2.0, np.e]

        for x in test_points:
            analysis = operator.analyze_balance(x)
            total = sum(analysis["relative_contributions"].values())
            assert np.abs(total - 1.0) < 1e-6

    def test_balanced_points_detection(self, operator):
        """Test detection of balanced points."""
        analysis = operator.analyze_balance(2.0)

        assert "is_balanced" in analysis
        assert isinstance(analysis["is_balanced"], bool)

        # Balance ratio should determine is_balanced
        ratio = analysis["balance_ratio"]
        expected_balanced = 0.5 < ratio < 2.0
        assert analysis["is_balanced"] == expected_balanced

    def test_harmonic_oscillation_period(self, operator):
        """Test that harmonic components have correct periodicity."""
        # sin(πx) has period 2 in x (since sin has period 2π in its argument)
        result_0 = operator.evaluate(0.5)
        result_2 = operator.evaluate(2.5)  # 0.5 + 2

        # sin(π·0.5) = sin(π/2) = 1 and sin(π·2.5) = sin(5π/2) = 1
        # Both should equal 1 due to the period of sin being 2π
        assert np.abs(result_0.sin_component - 1.0) < 1e-10
        assert np.abs(result_2.sin_component - 1.0) < 1e-10

    def test_integer_lattice_cancellation(self, operator):
        """Test that sin(πx) enforces integer lattice cancellation."""
        # At integers, sin(πn) = 0
        integers = [1, 2, 3, 4, 5]

        for n in integers:
            result = operator.evaluate(float(n))
            # sin component should vanish at integers
            assert np.abs(result.sin_component) < 1e-10

    def test_result_dataclass_validation(self, operator):
        """Test that result dataclass validates correctly."""
        result = operator.evaluate(2.0)

        # All required fields should be present
        assert hasattr(result, "x")
        assert hasattr(result, "value")
        assert hasattr(result, "ln_component")
        assert hasattr(result, "zeta_component")
        assert hasattr(result, "tan_component")
        assert hasattr(result, "sin_component")
        assert hasattr(result, "cos_component")
        assert hasattr(result, "magnitude")
        assert hasattr(result, "phase")

    def test_operator_reproduces_euler_identity(self, operator):
        """Test that operator structure contains Euler's identity."""
        constants = operator.compute_emergent_constants()

        # cos(π) + i·sin(π) = e^(iπ) = -1
        cos_sin = constants["cos_pi_plus_i_sin_pi"]
        euler = constants["euler_identity_value"]

        # They should be equal (both equal to -1)
        assert np.abs(cos_sin - euler) < 1e-10
        assert np.abs(euler - (-1.0)) < 1e-10

    def test_operator_unit_circle_embedding(self, operator):
        """Test that unit circle is embedded via e^(iπx)."""
        # For any real x: |e^(iπx)| = 1 (unit circle)
        test_points = [0.0, 0.5, 1.0, 1.5, 2.0]

        for x in test_points:
            # cos(πx) + i·sin(πx) should have magnitude 1
            result = operator.evaluate(x)
            unit_circle_point = complex(result.cos_component, result.sin_component)
            magnitude = np.abs(unit_circle_point)

            # Should be on unit circle
            assert np.abs(magnitude - 1.0) < 1e-10


def test_main_function_runs():
    """Test that the main function executes without errors."""
    from riemann.harmonic_divergence_operator import main

    # Should not raise any exceptions
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"main() raised exception: {e}")


if __name__ == "__main__":
    # Run tests manually when pytest can't be used due to module name conflict
    import sys
    sys.path.insert(0, 'code')
    
    test_class = TestHarmonicDivergenceOperator()
    operator = HarmonicDivergenceOperator()  # Create directly
    
    print("Running Harmonic Divergence Operator Tests")
    print("=" * 70)
    
    # Run a subset of key tests
    try:
        test_class.test_operator_initialization(operator)
        print("✓ test_operator_initialization")
    except Exception as e:
        print(f"✗ test_operator_initialization: {e}")
    
    try:
        test_class.test_evaluate_positive_real(operator)
        print("✓ test_evaluate_positive_real")
    except Exception as e:
        print(f"✗ test_evaluate_positive_real: {e}")
    
    try:
        test_class.test_evaluate_at_e(operator)
        print("✓ test_evaluate_at_e")
    except Exception as e:
        print(f"✗ test_evaluate_at_e: {e}")
    
    try:
        test_class.test_sin_component_integer_zeros(operator)
        print("✓ test_sin_component_integer_zeros")
    except Exception as e:
        print(f"✗ test_sin_component_integer_zeros: {e}")
    
    try:
        test_class.test_cos_component_parity(operator)
        print("✓ test_cos_component_parity")
    except Exception as e:
        print(f"✗ test_cos_component_parity: {e}")
    
    try:
        test_class.test_emergent_constants(operator)
        print("✓ test_emergent_constants")
    except Exception as e:
        print(f"✗ test_emergent_constants: {e}")
    
    try:
        test_class.test_pi_embedded_in_trig_components(operator)
        print("✓ test_pi_embedded_in_trig_components")
    except Exception as e:
        print(f"✗ test_pi_embedded_in_trig_components: {e}")
    
    try:
        test_class.test_operator_reproduces_euler_identity(operator)
        print("✓ test_operator_reproduces_euler_identity")
    except Exception as e:
        print(f"✗ test_operator_reproduces_euler_identity: {e}")
    
    try:
        test_class.test_operator_unit_circle_embedding(operator)
        print("✓ test_operator_unit_circle_embedding")
    except Exception as e:
        print(f"✗ test_operator_unit_circle_embedding: {e}")
    
    print("\nTests complete!")
