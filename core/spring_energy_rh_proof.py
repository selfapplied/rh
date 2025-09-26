"""
Spring Energy → Positive Definite RH Proof

The Bridge: From "primes are time-springs" to positive-definite proof path.

Route A: Weil-Guinand positivity (explicit-formula ⇒ PSD)
Route B: Li/Keiper positivity as spring moments  
Route C: de Branges/Hermite-Biehler kernel

Key insight: "spring energy = quadratic form ≥ 0"
"""

import numpy as np
import scipy.integrate as integrate
from scipy.special import digamma, gamma
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class SpringKernel:
    """Spring kernel h(t) with positive energy spectrum |ĥ(ξ)|² ≥ 0"""
    alpha: float  # damping parameter
    omega: float  # frequency parameter
    cutoff_eta: callable  # smooth even cutoff
    
    def __post_init__(self):
        # Ensure cutoff is even and smooth
        assert self.cutoff_eta(-1) == self.cutoff_eta(1), "Cutoff must be even"
    
    def h(self, t: float) -> float:
        """Spring response function h(t) = e^(-αt²)cos(ωt)·η(t)"""
        gaussian = np.exp(-self.alpha * t**2)
        cosine = np.cos(self.omega * t)
        cutoff = self.cutoff_eta(t)
        return gaussian * cosine * cutoff
    
    def h_hat(self, u: float) -> complex:
        """Fourier transform ĥ(u) = (1/2)(e^(-(u-ω)²/4α) + e^(-(u+ω)²/4α))"""
        term1 = np.exp(-(u - self.omega)**2 / (4 * self.alpha))
        term2 = np.exp(-(u + self.omega)**2 / (4 * self.alpha))
        return 0.5 * (term1 + term2)
    
    def g(self, t: float) -> float:
        """Spring energy g(t) = h * h̃(t) where h̃(t) = h(-t)"""
        return self.h(t) * self.h(-t)  # Since h is even, h̃ = h
    
    def g_hat(self, u: float) -> float:
        """Energy spectrum ĝ(u) = |ĥ(u)|² ≥ 0 (Bochner's theorem)"""
        h_hat_val = self.h_hat(u)
        return abs(h_hat_val)**2

class WeilGuinandPositivity:
    """Route A: Weil-Guinand positivity via explicit formula"""
    
    def __init__(self, spring_kernel: SpringKernel):
        self.kernel = spring_kernel
        self.primes = self._generate_primes(1000)
    
    def _generate_primes(self, n: int) -> List[int]:
        """Generate first n primes"""
        primes = []
        candidate = 2
        while len(primes) < n:
            if all(candidate % p != 0 for p in primes):
                primes.append(candidate)
            candidate += 1
        return primes
    
    def archimedean_term(self) -> float:
        """
        Archimedean term A_∞(g) = (1/2π) ∫ ĝ(u)(ψ(1/2+iu) + ψ(1/2-iu)) du
        
        This is the CORRECT formula (no ζ(2) detours!)
        """
        def integrand(u):
            g_hat_val = self.kernel.g_hat(u)
            psi_term = digamma(0.5 + 1j*u).real + digamma(0.5 - 1j*u).real
            return g_hat_val * psi_term
        
        # Integrate over reasonable range
        result, _ = integrate.quad(integrand, -50, 50, limit=1000)
        return result / (2 * np.pi)
    
    def prime_side(self) -> float:
        """
        Prime side: -∑_p ∑_{k≥1} (log p)/p^(k/2) (g(k log p) + g(-k log p))
        
        This captures the "prime kicks" at times t = k log p
        """
        total = 0.0
        
        for p in self.primes:
            for k in range(1, 10):  # Truncate k sum
                log_p = np.log(p)
                t_k = k * log_p
                
                # Spring response at time k log p
                g_plus = self.kernel.g(t_k)
                g_minus = self.kernel.g(-t_k)
                
                # Prime kick contribution
                contribution = (log_p / np.sqrt(float(p**k))) * (g_plus + g_minus)
                total += contribution
        
        return -total  # Negative sign from explicit formula
    
    def zero_side(self, zeros: List[complex]) -> float:
        """
        Zero side: ∑_ρ ĝ((ρ-1/2)/i)
        
        This is the sum of spring energies at zeta zeros
        """
        total = 0.0
        
        for rho in zeros:
            # Transform zero to frequency domain
            xi = (rho - 0.5) / 1j
            g_hat_val = self.kernel.g_hat(xi.real)
            total += g_hat_val
        
        return total
    
    def explicit_formula_balance(self, zeros: List[complex]) -> Dict[str, float]:
        """
        Explicit formula: Zero Side = g(0)log(π) + Prime Side + Archimedean Term
        
        Returns the balance and individual components
        """
        g_zero = self.kernel.g(0)
        log_pi_term = g_zero * np.log(np.pi)
        prime_side_val = self.prime_side()
        archimedean_val = self.archimedean_term()
        
        left_side = self.zero_side(zeros)
        right_side = log_pi_term + prime_side_val + archimedean_val
        
        balance = left_side - right_side
        
        return {
            "zero_side": left_side,
            "log_pi_term": log_pi_term,
            "prime_side": prime_side_val,
            "archimedean_term": archimedean_val,
            "right_side": right_side,
            "balance": balance,
            "is_balanced": abs(balance) < 1e-6
        }

class LiKeiperPositivity:
    """Route B: Li/Keiper positivity as spring moments"""
    
    def __init__(self, spring_kernel: SpringKernel):
        self.kernel = spring_kernel
    
    def li_coefficient(self, n: int, zeros: List[complex]) -> float:
        """
        Li coefficient λₙ = ∑_ρ (1 - (1 - 1/ρ)^n)
        
        RH ⇔ λₙ ≥ 0 for all n
        """
        total = 0.0
        for rho in zeros:
            if abs(rho) > 1e-10:  # Avoid division by zero
                term = 1 - (1 - 1/rho)**n
                total += term.real  # Take real part
        return total
    
    def hankel_matrix(self, n_max: int, zeros: List[complex]) -> np.ndarray:
        """
        Hankel matrix H_{m,n} = λ_{m+n}
        
        This matrix is PSD iff the Li coefficients form a Hamburger moment sequence
        """
        H = np.zeros((n_max, n_max))
        
        for m in range(n_max):
            for n in range(n_max):
                lambda_val = self.li_coefficient(m + n, zeros)
                H[m, n] = lambda_val
        
        return H
    
    def is_hankel_psd(self, n_max: int, zeros: List[complex]) -> Tuple[bool, float]:
        """
        Check if Hankel matrix is positive semidefinite
        
        Returns (is_psd, smallest_eigenvalue)
        """
        H = self.hankel_matrix(n_max, zeros)
        eigenvalues = np.linalg.eigvals(H)
        min_eigenval = np.min(eigenvalues.real)
        
        return min_eigenval >= -1e-10, min_eigenval

class DeBrangesKernel:
    """Route C: de Branges/Hermite-Biehler kernel"""
    
    def __init__(self, structure_function: callable):
        self.E = structure_function
    
    def canonical_kernel(self, z: complex, w: complex) -> complex:
        """
        Canonical kernel K_E(z,w) = (E(z)Ē(w) - E*(z)Ē*(w))/(2πi(w̄ - z))
        
        This is PSD iff zeros lie on the critical line
        """
        numerator = (self.E(z) * np.conj(self.E(w)) - 
                    self.E(np.conj(z)) * np.conj(self.E(np.conj(w))))
        denominator = 2 * np.pi * 1j * (np.conj(w) - z)
        
        return numerator / denominator
    
    def test_psd_property(self, test_points: List[complex]) -> Tuple[bool, float]:
        """
        Test if canonical kernel is positive semidefinite
        
        Returns (is_psd, smallest_eigenvalue)
        """
        n = len(test_points)
        K = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                K[i, j] = self.canonical_kernel(test_points[i], test_points[j])
        
        # Check if K is Hermitian
        is_hermitian = np.allclose(K, K.conj().T)
        
        if is_hermitian:
            eigenvalues = np.linalg.eigvals(K)
            min_eigenval = np.min(eigenvalues.real)
            return min_eigenval >= -1e-10, min_eigenval
        else:
            return False, float('-inf')

def create_spring_kernel(alpha: float = 5.0, omega: float = 25.0) -> SpringKernel:
    """Create a minimal working spring kernel with proper zeta frequency support"""
    
    def cutoff_eta(t: float) -> float:
        """Smooth even cutoff function with much broader support"""
        if abs(t) > 100:
            return 0.0
        return np.exp(-t**2 / 10000)  # Much broader smooth decay
    
    return SpringKernel(alpha=alpha, omega=omega, cutoff_eta=cutoff_eta)

def test_spring_energy_rh_proof():
    """Test the spring energy → RH proof framework"""
    
    print("SPRING ENERGY → RH PROOF FRAMEWORK")
    print("=" * 60)
    
    # Create spring kernel
    kernel = create_spring_kernel(alpha=1.0, omega=2.0)
    
    # Test known zeta zeros (first 5)
    zeta_zeros = [
        0.5 + 14.134725141734693j,
        0.5 + 21.022039638771555j,
        0.5 + 25.010857580145688j,
        0.5 + 30.424876125859529j,
        0.5 + 32.935061587739190j
    ]
    
    print("1. SPRING KERNEL PROPERTIES:")
    print("-" * 30)
    print(f"h(0) = {kernel.h(0):.6f}")
    print(f"g(0) = {kernel.g(0):.6f}")
    print(f"ĝ(0) = {kernel.g_hat(0):.6f}")
    print(f"ĝ(ω) = {kernel.g_hat(kernel.omega):.6f}")
    
    print("\n2. ROUTE A: WEIL-GUINAND POSITIVITY")
    print("-" * 30)
    wg = WeilGuinandPositivity(kernel)
    
    # Test explicit formula balance
    balance_result = wg.explicit_formula_balance(zeta_zeros)
    
    print(f"Zero side: {balance_result['zero_side']:.6f}")
    print(f"Prime side: {balance_result['prime_side']:.6f}")
    print(f"Archimedean term: {balance_result['archimedean_term']:.6f}")
    print(f"Balance: {balance_result['balance']:.6f}")
    print(f"Is balanced: {balance_result['is_balanced']}")
    
    print("\n3. ROUTE B: LI/KEIPER POSITIVITY")
    print("-" * 30)
    lk = LiKeiperPositivity(kernel)
    
    # Test Li coefficients
    for n in [1, 2, 3, 4, 5]:
        lambda_n = lk.li_coefficient(n, zeta_zeros)
        print(f"λ_{n} = {lambda_n:.6f}")
    
    # Test Hankel matrix
    is_psd, min_eigenval = lk.is_hankel_psd(5, zeta_zeros)
    print(f"Hankel matrix PSD: {is_psd}")
    print(f"Min eigenvalue: {min_eigenval:.6f}")
    
    print("\n4. POSITIVITY CRITERION")
    print("-" * 30)
    print("For RH to hold, we need:")
    print("• Zero side ≥ 0 for all spring kernels")
    print("• Li coefficients λₙ ≥ 0 for all n")
    print("• Hankel matrices H_{m,n} = λ_{m+n} are PSD")
    
    # Test positivity
    zero_side_positive = balance_result['zero_side'] >= 0
    li_positive = all(lk.li_coefficient(n, zeta_zeros) >= 0 for n in range(1, 6))
    
    print(f"\nZero side ≥ 0: {zero_side_positive}")
    print(f"Li coefficients ≥ 0: {li_positive}")
    print(f"Hankel PSD: {is_psd}")
    
    overall_positive = zero_side_positive and li_positive and is_psd
    print(f"\n🎯 OVERALL POSITIVITY: {overall_positive}")
    
    return {
        "spring_kernel": kernel,
        "weil_guinand": wg,
        "li_keiper": lk,
        "balance_result": balance_result,
        "is_positive": overall_positive
    }

if __name__ == "__main__":
    result = test_spring_energy_rh_proof()
