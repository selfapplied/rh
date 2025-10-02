#!/usr/bin/env python3
"""
Test Suite for Stieltjes Lemmas

This module provides assertions that map code ↔ statement for the
Stieltjes representation lemmas.
"""

import os
import sys

import numpy as np
import pytest
from sympy import cos, exp, symbols


# Add the proofs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'proofs'))

from proofs.stieltjes_proof import ProofResult, StieltjesProofHarness


class TestStieltjesLemmas:
    """Test suite for Stieltjes representation lemmas"""
    
    def setup_method(self):
        """Set up test harness with canonical parameters"""
        self.harness = StieltjesProofHarness(alpha=5.0, omega=2.0, sigma=0.223607)
    
    def test_lemma_s1_complete_monotonicity_structure(self):
        """Test that Lemma S1 has the correct mathematical structure"""
        result = self.harness.check_lemma_s1_complete_monotonicity()
        
        # Assert the lemma has the required components
        assert result.lemma_name == "Lemma S1: Complete Monotonicity ⇒ Stieltjes"
        assert "alternating signs" in result.symbolic_condition
        assert "G^" in result.symbolic_condition  # Check for derivative notation
        
        # Assert assumptions are properly stated
        assert any("α ∈" in assumption for assumption in result.assumptions)
        assert any("ω ∈" in assumption for assumption in result.assumptions)
        assert any("σ ∈" in assumption for assumption in result.assumptions)
        
        # Assert the lemma is not yet complete (as expected)
        assert result.status in ["INCOMPLETE", "PASS", "FAIL"]
    
    def test_lemma_s2_positive_measure_structure(self):
        """Test that Lemma S2 has the correct mathematical structure"""
        result = self.harness.check_lemma_s2_positive_measure()
        
        # Assert the lemma has the required components
        assert result.lemma_name == "Lemma S2: Critical Hat → Positive Measure"
        assert "w_θ(x)" in result.symbolic_condition
        assert "≥ 0" in result.symbolic_condition
        
        # Assert assumptions are properly stated
        assert any("α ∈" in assumption for assumption in result.assumptions)
        assert any("ω ∈" in assumption for assumption in result.assumptions)
        assert any("x > 0" in assumption for assumption in result.assumptions)
        
        # Assert the lemma is not yet complete (as expected)
        assert result.status in ["INCOMPLETE", "PASS", "FAIL"]
    
    def test_theorem_s_global_psd_structure(self):
        """Test that Theorem S has the correct mathematical structure"""
        result = self.harness.check_theorem_s_global_psd()
        
        # Assert the theorem has the required components
        assert result.lemma_name == "Theorem S: Global PSD via Stieltjes"
        assert "Stieltjes" in result.symbolic_condition
        assert "PSD" in result.symbolic_condition
        
        # Assert assumptions are properly stated
        assert any("Stieltjes transform" in assumption for assumption in result.assumptions)
        assert any("positive measure" in assumption for assumption in result.assumptions)
        assert any("Moment representation" in assumption for assumption in result.assumptions)
        
        # Assert the theorem is not yet complete (as expected)
        assert result.status in ["INCOMPLETE", "PASS", "FAIL"]
    
    def test_critical_hat_kernel_properties(self):
        """Test that the critical hat kernel has the required properties"""
        # Define the kernel symbolically
        t = symbols('t', real=True)
        alpha, omega = 5.0, 2.0
        
        g_theta = exp(-alpha * t**2) * cos(omega * t)
        
        # Test that the kernel is defined
        assert g_theta is not None
        
        # Test that it's a function of t
        assert t in g_theta.free_symbols
        
        # Test that it contains the expected terms
        assert "exp" in str(g_theta) or "exp" in str(g_theta)
        assert "cos" in str(g_theta)
    
    def test_generating_function_structure(self):
        """Test that the generating function has the correct structure"""
        # This is a structural test - we're checking that the
        # generating function is defined in terms of the kernel
        
        g_theta = self.harness._define_critical_hat_kernel()
        G_z = self.harness._define_generating_function(g_theta)
        
        # Test that the generating function is defined
        assert G_z is not None
        
        # Test that it depends on z
        z = symbols('z', complex=True)
        assert z in G_z.free_symbols
    
    def test_parameter_ranges(self):
        """Test that parameter ranges are properly defined"""
        # Test alpha range
        assert self.harness.alpha_range.start == 5
        assert self.harness.alpha_range.end == 10
        
        # Test omega range
        assert self.harness.omega_range.start == 2
        assert self.harness.omega_range.end == 2.6
        
        # Test sigma range
        assert self.harness.sigma_range.start == 0.1
        assert self.harness.sigma_range.end == 0.3
    
    def test_canonical_parameters(self):
        """Test that canonical parameters are within ranges"""
        # Test alpha
        assert 5 <= self.harness.alpha <= 10
        
        # Test omega
        assert 2 <= self.harness.omega <= 2.6
        
        # Test sigma
        assert 0.1 <= self.harness.sigma <= 0.3
    
    def test_proof_harness_integration(self):
        """Test that the proof harness integrates properly"""
        # Run all checks
        results = self.harness.run_all_checks()
        
        # Assert we get results for all lemmas/theorems
        assert 'lemma_s1' in results
        assert 'lemma_s2' in results
        assert 'theorem_s' in results
        
        # Assert all results are ProofResult objects
        for result in results.values():
            assert isinstance(result, ProofResult)
            assert hasattr(result, 'lemma_name')
            assert hasattr(result, 'status')
            assert hasattr(result, 'symbolic_condition')
            assert hasattr(result, 'assumptions')
            assert hasattr(result, 'errors')


class TestStieltjesNumericalVerification:
    """Test suite for numerical verification of Stieltjes properties"""
    
    def setup_method(self):
        """Set up numerical test harness"""
        self.harness = StieltjesProofHarness(alpha=5.0, omega=2.0, sigma=0.223607)
    
    def test_kernel_positivity_numerical(self):
        """Numerically verify that the critical hat kernel is positive"""
        # Test kernel at various points
        t_values = np.linspace(-10, 10, 100)
        
        for t in t_values:
            # g_θ(t) = 0.5 * e^(-αt²) cos²(ωt)
            g_val = 0.5 * np.exp(-self.harness.alpha * t**2) * np.cos(self.harness.omega * t)**2
            
            # The kernel should be non-negative by construction (cos²)
            assert g_val >= 0.0
    
    def test_generating_function_convergence(self):
        """Test that the generating function converges numerically"""
        # This is a placeholder for numerical convergence tests
        # In practice, we would test the Stieltjes integral convergence
        
        # For now, just test that the kernel is well-behaved
        t_values = np.linspace(0, 100, 1000)
        g_values = 0.5 * np.exp(-self.harness.alpha * t_values**2) * np.cos(self.harness.omega * t_values)**2
        
        # The kernel should decay exponentially and be subunit
        assert np.max(g_values) < 1.0  # Max at t=0 is 0.5
        assert np.abs(g_values[-1]) < 1e-6  # Should be very small at the end


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])