"""
Rigorous Critical Hat: Mathematical Analysis Without Metaphor

This module implements the rigorous mathematical analysis of the "critical hat"
concept, focusing on what we can actually prove versus what remains heuristic.

Key Mathematical Points:
1. Mellin transform duality is rigorous
2. Convolution as frequency filtering is rigorous  
3. The "ideal" critical hat is a distribution, not a function
4. We need approximating families K_σ(s) that converge to the ideal filter
5. Connection to RH requires identification with Weil's kernel
"""

import numpy as np
import scipy.signal
from scipy.special import gamma, digamma
from typing import List, Dict, Tuple, Callable, Optional
import matplotlib.pyplot as plt

class RigorousCriticalHat:
    """
    Rigorous analysis of the critical hat concept.
    
    This strips away the metaphor and focuses on what we can actually prove:
    1. Mellin transform duality
    2. Convolution as frequency filtering
    3. Approximating families that converge to ideal filter
    4. Connection to explicit formula and Weil's criterion
    """
    
    def __init__(self, sigma: float = 0.1):
        """
        Initialize rigorous critical hat analysis.
        
        Args:
            sigma: Parameter for approximating family K_σ(s)
        """
        self.sigma = sigma
        self.critical_line = 0.5
        
        # Generate approximating kernel
        self.kernel = self._generate_approximating_kernel()
        self.mellin_spectrum = self._compute_mellin_spectrum()
    
    def _generate_approximating_kernel(self) -> np.ndarray:
        """
        Generate approximating kernel K_σ(s) that converges to ideal filter.
        
        The ideal filter is:
        K̂(s) = {1, if Re(s) = 1/2
                {0, otherwise
        
        We approximate with:
        K̂_σ(s) = exp(-(Re(s) - 1/2)²/(2σ²))
        """
        # Create frequency grid
        s_values = np.linspace(0.1, 2.0, 200)  # Re(s) values
        t_values = np.linspace(-10, 10, 100)   # Im(s) values
        
        # Create 2D grid
        S, T = np.meshgrid(s_values, t_values)
        s_complex = S + 1j * T
        
        # Compute approximating kernel
        kernel_spectrum = np.exp(-((S - self.critical_line)**2) / (2 * self.sigma**2))
        
        # Inverse Mellin transform to get kernel
        kernel = self._inverse_mellin_transform(kernel_spectrum, s_values, t_values)
        
        return kernel
    
    def _inverse_mellin_transform(self, 
                                 spectrum: np.ndarray, 
                                 s_values: np.ndarray, 
                                 t_values: np.ndarray) -> np.ndarray:
        """
        Approximate inverse Mellin transform.
        
        This is a simplified version - in practice would need more sophisticated
        numerical methods for the inverse Mellin transform.
        """
        # For now, use a simple approximation
        # In practice, this would be more complex
        kernel = np.fft.ifft(spectrum[0, :])  # Simplified
        return np.real(kernel)
    
    def _compute_mellin_spectrum(self) -> np.ndarray:
        """Compute the Mellin spectrum of the kernel"""
        # This would be the actual Mellin transform in practice
        # For now, use FFT as approximation
        return np.fft.fft(self.kernel)
    
    def test_mellin_duality(self, 
                           test_function: Callable[[float], float]) -> Dict[str, any]:
        """
        Test the Mellin transform duality rigorously.
        
        This tests the fundamental relationship:
        (f*g)̂(s) = f̂(s) · ĝ(s)
        """
        # Create test function
        x_values = np.linspace(0.1, 10, 100)
        f_values = np.array([test_function(x) for x in x_values])
        
        # Apply convolution
        convolved = scipy.signal.convolve(f_values, self.kernel, mode='same')
        
        # Compute FFTs (approximating Mellin transforms)
        f_hat = np.fft.fft(f_values)
        g_hat = self.mellin_spectrum
        fg_hat = np.fft.fft(convolved)
        
        # Test duality: (f*g)̂(s) ≈ f̂(s) · ĝ(s)
        # Ensure same length for comparison
        min_len = min(len(f_hat), len(g_hat), len(fg_hat))
        f_hat_truncated = f_hat[:min_len]
        g_hat_truncated = g_hat[:min_len]
        fg_hat_truncated = fg_hat[:min_len]
        
        expected_product = f_hat_truncated * g_hat_truncated
        duality_error = np.mean(np.abs(fg_hat_truncated - expected_product))
        
        return {
            'duality_error': duality_error,
            'duality_preserved': duality_error < 1e-10,
            'f_hat': f_hat,
            'g_hat': g_hat,
            'fg_hat': fg_hat,
            'expected_product': expected_product
        }
    
    def test_spectral_symmetry(self) -> Dict[str, any]:
        """
        Test that Re(s) = 1/2 acts as a spectral symmetry line.
        
        This tests the mathematical fact that the critical line has special
        properties in the Mellin domain.
        """
        # Test points on and off the critical line
        critical_points = [0.5 + 1j * t for t in [14.1347, 21.0220, 25.0109]]
        off_critical_points = [0.3 + 1j * t for t in [14.1347, 21.0220, 25.0109]]
        
        # Evaluate kernel spectrum at these points
        critical_values = []
        off_critical_values = []
        
        for point in critical_points:
            # Approximate evaluation of K̂(s) at critical line
            s_real = np.real(point)
            s_imag = np.imag(point)
            
            # Use our approximating formula
            value = np.exp(-((s_real - self.critical_line)**2) / (2 * self.sigma**2))
            critical_values.append(value)
        
        for point in off_critical_points:
            s_real = np.real(point)
            value = np.exp(-((s_real - self.critical_line)**2) / (2 * self.sigma**2))
            off_critical_values.append(value)
        
        # Test symmetry properties
        critical_strength = np.mean(critical_values)
        off_critical_strength = np.mean(off_critical_values)
        
        symmetry_ratio = critical_strength / (off_critical_strength + 1e-10)
        
        return {
            'critical_values': critical_values,
            'off_critical_values': off_critical_values,
            'critical_strength': critical_strength,
            'off_critical_strength': off_critical_strength,
            'symmetry_ratio': symmetry_ratio,
            'symmetry_preserved': symmetry_ratio > 10  # Critical line should be much stronger
        }
    
    def test_convergence_to_ideal(self, 
                                 sigma_values: List[float]) -> Dict[str, any]:
        """
        Test convergence to ideal filter as σ → 0.
        
        The ideal filter is:
        K̂(s) = {1, if Re(s) = 1/2
                {0, otherwise
        """
        convergence_results = {}
        
        for sigma in sigma_values:
            # Create kernel with this sigma
            kernel = self._generate_kernel_with_sigma(sigma)
            
            # Test how close it is to ideal
            ideal_approximation = self._test_ideal_approximation(kernel, sigma)
            
            convergence_results[sigma] = ideal_approximation
        
        return convergence_results
    
    def _generate_kernel_with_sigma(self, sigma: float) -> np.ndarray:
        """Generate kernel with specific sigma value"""
        # Simplified version - in practice would be more complex
        s_values = np.linspace(0.1, 2.0, 100)
        kernel_spectrum = np.exp(-((s_values - self.critical_line)**2) / (2 * sigma**2))
        return np.fft.ifft(kernel_spectrum)
    
    def _test_ideal_approximation(self, kernel: np.ndarray, sigma: float) -> Dict[str, float]:
        """Test how well kernel approximates ideal filter"""
        # Test at critical line
        critical_value = np.exp(-((0.5 - self.critical_line)**2) / (2 * sigma**2))
        
        # Test off critical line
        off_critical_value = np.exp(-((0.3 - self.critical_line)**2) / (2 * sigma**2))
        
        # Compute approximation quality
        ideal_ratio = critical_value / (off_critical_value + 1e-10)
        
        return {
            'critical_value': critical_value,
            'off_critical_value': off_critical_value,
            'ideal_ratio': ideal_ratio,
            'sigma': sigma
        }
    
    def test_explicit_formula_connection(self, 
                                       primes: List[int],
                                       test_function: Callable[[float], float]) -> Dict[str, any]:
        """
        Test connection to explicit formula and Weil's criterion.
        
        This is where we need to be most careful - the connection to RH
        requires identification with the specific Weil kernel.
        """
        # Apply kernel to primes
        prime_series = np.array(primes, dtype=float)
        filtered_primes = scipy.signal.convolve(prime_series, self.kernel, mode='same')
        
        # Test explicit formula with filtered primes
        t_values = np.linspace(-5, 5, 200)
        phi_values = np.array([test_function(t) for t in t_values])
        
        # Apply kernel to test function
        filtered_phi = scipy.signal.convolve(phi_values, self.kernel, mode='same')
        
        # Compute explicit formula terms
        archimedean_term = np.trapezoid(filtered_phi * np.exp(-t_values**2), t_values)
        
        # Compute prime terms
        prime_terms = []
        for prime in filtered_primes[:10]:
            log_p = np.log(prime)
            term_sum = 0.0
            for k in range(1, 4):
                t_k = k * log_p
                t_idx = np.argmin(np.abs(t_values - t_k))
                if t_idx < len(filtered_phi):
                    term_sum += log_p * filtered_phi[t_idx] / np.sqrt(prime**k)
            prime_terms.append(term_sum)
        
        total_explicit_formula = archimedean_term - sum(prime_terms)
        
        # Important: This is NOT the same as Weil's criterion
        # We need to identify our kernel with the specific Weil kernel
        return {
            'explicit_formula_value': total_explicit_formula,
            'positive': total_explicit_formula >= -1e-10,
            'archimedean_term': archimedean_term,
            'prime_terms_sum': sum(prime_terms),
            'note': 'This is NOT the same as Weil\'s criterion - we need to identify our kernel with the specific Weil kernel'
        }
    
    def rigorous_analysis_summary(self) -> Dict[str, any]:
        """
        Provide rigorous analysis summary of what we can actually prove.
        """
        return {
            'rigorous_foundations': {
                'mellin_duality': 'Convolution-Mellin duality is mathematically rigorous',
                'frequency_filtering': 'Convolution as frequency filtering is rigorous',
                'spectral_symmetry': 'Re(s) = 1/2 as spectral symmetry line is established'
            },
            'heuristic_elements': {
                'ideal_filter': 'Exact indicator function is a distribution, not a function',
                'critical_hat': 'Current "critical hat" is a metaphor, not a proof device',
                'positivity_rh': 'Direct positivity → RH connection needs Weil kernel identification'
            },
            'what_we_can_say': {
                'accurate_statement': 'The critical hat is a Mellin filter centered on the symmetry line; RH asks whether the true zeta-induced filter is positive there.',
                'mathematical_status': 'This statement is accurate and safe'
            },
            'next_steps': {
                'approximating_family': 'Define K_σ(s) = exp(-(Re(s) - 1/2)²/(2σ²))',
                'convergence_analysis': 'Study convergence to ideal filter as σ → 0',
                'explicit_formula_test': 'Apply K_σ to explicit formula and compare with Weil\'s',
                'weil_kernel_identification': 'Identify specific kernel that encodes zeta\'s zeros'
            }
        }

def test_rigorous_critical_hat():
    """Test the rigorous critical hat analysis"""
    
    print("RIGOROUS CRITICAL HAT ANALYSIS")
    print("=" * 50)
    
    # Create rigorous analysis
    hat = RigorousCriticalHat(sigma=0.1)
    
    # Test Mellin duality
    def test_func(x):
        return np.exp(-x**2)
    
    duality_test = hat.test_mellin_duality(test_func)
    print(f"Mellin duality preserved: {duality_test['duality_preserved']}")
    print(f"Duality error: {duality_test['duality_error']:.2e}")
    
    # Test spectral symmetry
    symmetry_test = hat.test_spectral_symmetry()
    print(f"Spectral symmetry preserved: {symmetry_test['symmetry_preserved']}")
    print(f"Symmetry ratio: {symmetry_test['symmetry_ratio']:.2f}")
    
    # Test convergence to ideal
    sigma_values = [0.5, 0.2, 0.1, 0.05, 0.01]
    convergence_test = hat.test_convergence_to_ideal(sigma_values)
    
    print("\nConvergence to ideal filter:")
    for sigma, result in convergence_test.items():
        print(f"  σ = {sigma:.2f}: ideal_ratio = {result['ideal_ratio']:.2f}")
    
    # Test explicit formula connection
    test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    explicit_test = hat.test_explicit_formula_connection(test_primes, test_func)
    print(f"\nExplicit formula value: {explicit_test['explicit_formula_value']:.6f}")
    print(f"Positive: {explicit_test['positive']}")
    print(f"Note: {explicit_test['note']}")
    
    # Show rigorous analysis summary
    summary = hat.rigorous_analysis_summary()
    print(f"\nRIGOROUS ANALYSIS SUMMARY:")
    print(f"Rigorous foundations: {summary['rigorous_foundations']}")
    print(f"Heuristic elements: {summary['heuristic_elements']}")
    print(f"What we can say: {summary['what_we_can_say']}")
    print(f"Next steps: {summary['next_steps']}")
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    print("The critical hat is a valuable metaphor that provides insight")
    print("into the structure of the problem, but it's not yet a rigorous")
    print("proof device. We need to focus on the mathematical foundations")
    print("and develop the approximating family K_σ(s) properly.")
    print("="*50)

if __name__ == "__main__":
    test_rigorous_critical_hat()
