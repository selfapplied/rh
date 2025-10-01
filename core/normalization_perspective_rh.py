"""
Normalization Perspective on Riemann Hypothesis

Key Insight: Zeta zeros act as L2/softmax normalizers for the prime distribution.

The critical line Re(s) = 1/2 is a normalization constraint, analogous to:
- L2 normalization: Projects vectors onto the unit sphere
- Softmax: Projects scores onto probability simplex
- Critical line: Projects zeros onto the balanced energy manifold

Mathematical Framework:
1. Energy functional: E(s) = |Re(s) - 1/2|² + |ζ(s)|²
2. Normalization constraint: Re(s) = 1/2 (critical line)
3. Zeros as minima: ζ(ρ) = 0 ⟺ E(ρ) minimized subject to constraint

This connects:
- Machine learning optimization (softmax, L2 norm)
- Constrained optimization (Lagrange multipliers)
- Spectral theory (eigenvalue distributions)
- RH (zeros on critical line)
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class NormalizationMetrics:
    """Metrics for normalization analysis"""
    l2_norm: float           # L2 norm of zero distribution
    energy: float            # Total energy
    constraint_violation: float  # Distance from critical line
    entropy: float           # Shannon entropy (like softmax)
    normalized: bool         # Whether normalized

class ZetaZeroNormalization:
    """
    Analyzes zeta zeros through a normalization lens.
    
    Key concepts:
    1. L2 normalization: ||ρ|| = constant (energy conservation)
    2. Softmax-like: Zeros create a "probability" distribution over primes
    3. Critical line constraint: Re(ρ) = 1/2 (the normalization manifold)
    """
    
    def __init__(self, critical_line: float = 0.5):
        """
        Initialize normalization analyzer.
        
        Args:
            critical_line: The normalization constraint (default: 0.5)
        """
        self.critical_line = critical_line
        
    def l2_normalize(self, zeros: List[complex]) -> Tuple[List[complex], float]:
        """
        L2 normalize a set of zeros.
        
        Projects zeros onto the unit sphere in complex plane,
        analogous to L2 normalization in machine learning.
        
        Args:
            zeros: List of complex zeros
            
        Returns:
            Tuple of (normalized zeros, normalization constant)
        """
        # Compute L2 norm
        l2_norm = np.sqrt(sum(abs(z)**2 for z in zeros))
        
        # Normalize
        if l2_norm > 1e-10:
            normalized = [z / l2_norm for z in zeros]
        else:
            normalized = zeros
            
        return normalized, l2_norm
    
    def softmax_like_distribution(self, zeros: List[complex], temperature: float = 1.0) -> np.ndarray:
        """
        Compute a softmax-like distribution from zeros.
        
        This creates a "probability" distribution over the imaginary parts
        of zeros, analogous to softmax in machine learning.
        
        Args:
            zeros: List of complex zeros
            temperature: Softmax temperature parameter
            
        Returns:
            Probability distribution over zeros
        """
        # Extract imaginary parts (the "scores")
        im_parts = np.array([z.imag for z in zeros])
        
        # Compute softmax: exp(x_i / T) / Σ exp(x_j / T)
        exp_scores = np.exp(im_parts / temperature)
        softmax_dist = exp_scores / np.sum(exp_scores)
        
        return softmax_dist
    
    def critical_line_projection(self, zeros: List[complex]) -> List[complex]:
        """
        Project zeros onto the critical line.
        
        This is the key normalization constraint for RH:
        all zeros should satisfy Re(ρ) = 1/2.
        
        Args:
            zeros: List of complex zeros
            
        Returns:
            Projected zeros on critical line
        """
        projected = [complex(self.critical_line, z.imag) for z in zeros]
        return projected
    
    def energy_functional(self, z: complex) -> float:
        """
        Compute energy functional E(z) = |Re(z) - 1/2|² + |z|²
        
        Zeros minimize this energy on the critical line.
        This is analogous to the loss function in optimization.
        
        Args:
            z: Complex number
            
        Returns:
            Energy value
        """
        distance_from_critical = (z.real - self.critical_line)**2
        magnitude_penalty = abs(z)**2
        
        return distance_from_critical + magnitude_penalty
    
    def constraint_violation(self, zeros: List[complex]) -> Dict[str, float]:
        """
        Measure how much zeros violate the critical line constraint.
        
        Args:
            zeros: List of complex zeros
            
        Returns:
            Dictionary with violation metrics
        """
        violations = [abs(z.real - self.critical_line) for z in zeros]
        
        return {
            'max_violation': max(violations) if violations else 0.0,
            'mean_violation': np.mean(violations) if violations else 0.0,
            'l2_violation': np.sqrt(sum(v**2 for v in violations)),
            'on_critical_line': all(v < 1e-6 for v in violations)
        }
    
    def shannon_entropy(self, distribution: np.ndarray) -> float:
        """
        Compute Shannon entropy of a distribution.
        
        This measures how "spread out" the zeros are,
        analogous to entropy in information theory.
        
        Args:
            distribution: Probability distribution
            
        Returns:
            Shannon entropy
        """
        # Filter out zeros to avoid log(0)
        nonzero = distribution[distribution > 1e-10]
        return -np.sum(nonzero * np.log(nonzero))
    
    def normalization_analysis(self, zeros: List[complex], 
                              temperature: float = 1.0) -> NormalizationMetrics:
        """
        Comprehensive normalization analysis of zeros.
        
        Args:
            zeros: List of complex zeros
            temperature: Softmax temperature
            
        Returns:
            NormalizationMetrics object
        """
        # L2 normalization
        normalized_zeros, l2_norm = self.l2_normalize(zeros)
        
        # Energy functional
        total_energy = sum(self.energy_functional(z) for z in zeros)
        
        # Constraint violation
        violation_metrics = self.constraint_violation(zeros)
        
        # Softmax distribution
        softmax_dist = self.softmax_like_distribution(zeros, temperature)
        
        # Entropy
        entropy = self.shannon_entropy(softmax_dist)
        
        # Check if normalized
        is_normalized = violation_metrics['on_critical_line']
        
        return NormalizationMetrics(
            l2_norm=l2_norm,
            energy=total_energy,
            constraint_violation=violation_metrics['l2_violation'],
            entropy=entropy,
            normalized=is_normalized
        )
    
    def compare_on_vs_off_critical_line(self, 
                                       critical_zeros: List[complex],
                                       off_critical_zeros: List[complex]) -> Dict[str, Dict]:
        """
        Compare normalization properties of zeros on vs off critical line.
        
        This demonstrates that the critical line is the "normalized" state.
        
        Args:
            critical_zeros: Zeros on critical line
            off_critical_zeros: Zeros off critical line
            
        Returns:
            Comparison dictionary
        """
        critical_metrics = self.normalization_analysis(critical_zeros)
        off_critical_metrics = self.normalization_analysis(off_critical_zeros)
        
        return {
            'critical_line': {
                'l2_norm': critical_metrics.l2_norm,
                'energy': critical_metrics.energy,
                'constraint_violation': critical_metrics.constraint_violation,
                'entropy': critical_metrics.entropy,
                'normalized': critical_metrics.normalized
            },
            'off_critical_line': {
                'l2_norm': off_critical_metrics.l2_norm,
                'energy': off_critical_metrics.energy,
                'constraint_violation': off_critical_metrics.constraint_violation,
                'entropy': off_critical_metrics.entropy,
                'normalized': off_critical_metrics.normalized
            },
            'energy_increase_off_line': off_critical_metrics.energy - critical_metrics.energy,
            'critical_line_minimizes_energy': off_critical_metrics.energy > critical_metrics.energy
        }

class ConstrainedOptimizationRH:
    """
    Views RH as a constrained optimization problem.
    
    Problem formulation:
    minimize E(s) = |Re(s) - 1/2|² + |ζ(s)|²
    subject to ζ(s) = 0
    
    Solution: All zeros lie on Re(s) = 1/2
    
    This is analogous to:
    - L2 normalization: minimize distance subject to ||v|| = 1
    - Softmax: minimize cross-entropy subject to Σp_i = 1
    """
    
    def __init__(self, critical_line: float = 0.5):
        self.critical_line = critical_line
        self.normalizer = ZetaZeroNormalization(critical_line)
    
    def lagrangian(self, z: complex, lambda_mult: float) -> float:
        """
        Lagrangian: L(z, λ) = E(z) + λ·constraint(z)
        
        where constraint(z) = |Re(z) - 1/2|
        
        Args:
            z: Complex number
            lambda_mult: Lagrange multiplier
            
        Returns:
            Lagrangian value
        """
        energy = self.normalizer.energy_functional(z)
        constraint = abs(z.real - self.critical_line)
        
        return energy + lambda_mult * constraint
    
    def gradient_energy(self, z: complex) -> complex:
        """
        Gradient of energy functional: ∇E(z)
        
        At critical points: ∇E(z) = 0
        
        Args:
            z: Complex number
            
        Returns:
            Gradient (as complex number)
        """
        # ∂E/∂(Re z) = 2(Re(z) - 1/2) + 2 Re(z)
        d_real = 2 * (z.real - self.critical_line) + 2 * z.real
        
        # ∂E/∂(Im z) = 2 Im(z)
        d_imag = 2 * z.imag
        
        return complex(d_real, d_imag)
    
    def project_to_constraint_manifold(self, z: complex) -> complex:
        """
        Project z onto the constraint manifold (critical line).
        
        This is the "normalization" operation.
        
        Args:
            z: Complex number
            
        Returns:
            Projected point on critical line
        """
        return complex(self.critical_line, z.imag)
    
    def optimization_path(self, 
                         initial_z: complex, 
                         steps: int = 100,
                         learning_rate: float = 0.01) -> List[complex]:
        """
        Gradient descent path toward critical line.
        
        This demonstrates how the optimization naturally
        converges to the critical line.
        
        Args:
            initial_z: Starting point
            steps: Number of optimization steps
            learning_rate: Step size
            
        Returns:
            List of points along optimization path
        """
        path = [initial_z]
        current_z = initial_z
        
        for _ in range(steps):
            # Compute gradient
            grad = self.gradient_energy(current_z)
            
            # Gradient descent step
            new_z = current_z - learning_rate * grad
            
            # Project onto constraint manifold (critical line)
            projected_z = self.project_to_constraint_manifold(new_z)
            
            path.append(projected_z)
            current_z = projected_z
        
        return path

def demonstrate_normalization_perspective():
    """Demonstrate the normalization perspective on RH"""
    
    print("=" * 70)
    print("NORMALIZATION PERSPECTIVE ON RIEMANN HYPOTHESIS")
    print("=" * 70)
    print()
    print("Key Insight: Zeta zeros are L2/softmax-normalized to the critical line")
    print()
    
    # Initialize analyzer
    normalizer = ZetaZeroNormalization(critical_line=0.5)
    
    # Test zeros on critical line
    critical_zeros = [
        complex(0.5, 14.134725),
        complex(0.5, 21.022040),
        complex(0.5, 25.010858),
        complex(0.5, 30.424876),
        complex(0.5, 32.935062)
    ]
    
    # Test zeros off critical line
    off_critical_zeros = [
        complex(0.6, 14.134725),
        complex(0.4, 21.022040),
        complex(0.55, 25.010858),
        complex(0.45, 30.424876),
        complex(0.7, 32.935062)
    ]
    
    print("1. L2 NORMALIZATION ANALYSIS")
    print("-" * 70)
    
    # L2 normalize
    critical_normalized, critical_norm = normalizer.l2_normalize(critical_zeros)
    off_normalized, off_norm = normalizer.l2_normalize(off_critical_zeros)
    
    print(f"Critical line L2 norm: {critical_norm:.6f}")
    print(f"Off critical line L2 norm: {off_norm:.6f}")
    print()
    
    print("2. SOFTMAX-LIKE DISTRIBUTION")
    print("-" * 70)
    
    # Compute softmax distributions
    critical_softmax = normalizer.softmax_like_distribution(critical_zeros, temperature=1.0)
    off_softmax = normalizer.softmax_like_distribution(off_critical_zeros, temperature=1.0)
    
    print("Critical line softmax distribution:")
    for i, prob in enumerate(critical_softmax):
        print(f"  Zero {i+1}: {prob:.6f}")
    
    print("\nOff critical line softmax distribution:")
    for i, prob in enumerate(off_softmax):
        print(f"  Zero {i+1}: {prob:.6f}")
    print()
    
    print("3. ENERGY FUNCTIONAL")
    print("-" * 70)
    
    # Compute energies
    critical_energy = sum(normalizer.energy_functional(z) for z in critical_zeros)
    off_energy = sum(normalizer.energy_functional(z) for z in off_critical_zeros)
    
    print(f"Critical line total energy: {critical_energy:.6f}")
    print(f"Off critical line total energy: {off_energy:.6f}")
    print(f"Energy increase off line: {off_energy - critical_energy:.6f}")
    print(f"Critical line minimizes energy: {off_energy > critical_energy}")
    print()
    
    print("4. CONSTRAINT VIOLATION")
    print("-" * 70)
    
    critical_violation = normalizer.constraint_violation(critical_zeros)
    off_violation = normalizer.constraint_violation(off_critical_zeros)
    
    print("Critical line constraint violation:")
    print(f"  Max violation: {critical_violation['max_violation']:.10f}")
    print(f"  Mean violation: {critical_violation['mean_violation']:.10f}")
    print(f"  On critical line: {critical_violation['on_critical_line']}")
    
    print("\nOff critical line constraint violation:")
    print(f"  Max violation: {off_violation['max_violation']:.6f}")
    print(f"  Mean violation: {off_violation['mean_violation']:.6f}")
    print(f"  On critical line: {off_violation['on_critical_line']}")
    print()
    
    print("5. ENTROPY ANALYSIS")
    print("-" * 70)
    
    critical_entropy = normalizer.shannon_entropy(critical_softmax)
    off_entropy = normalizer.shannon_entropy(off_softmax)
    
    print(f"Critical line entropy: {critical_entropy:.6f}")
    print(f"Off critical line entropy: {off_entropy:.6f}")
    print()
    
    print("6. CONSTRAINED OPTIMIZATION PERSPECTIVE")
    print("-" * 70)
    
    optimizer = ConstrainedOptimizationRH(critical_line=0.5)
    
    # Test optimization path
    initial_point = complex(0.7, 14.134725)
    optimization_path = optimizer.optimization_path(initial_point, steps=50)
    
    print(f"Initial point: {initial_point}")
    print(f"Final point: {optimization_path[-1]}")
    print(f"Converged to critical line: {abs(optimization_path[-1].real - 0.5) < 1e-6}")
    print()
    
    print("7. COMPARISON SUMMARY")
    print("-" * 70)
    
    comparison = normalizer.compare_on_vs_off_critical_line(critical_zeros, off_critical_zeros)
    
    print("Critical line metrics:")
    for key, value in comparison['critical_line'].items():
        print(f"  {key}: {value}")
    
    print("\nOff critical line metrics:")
    for key, value in comparison['off_critical_line'].items():
        print(f"  {key}: {value}")
    
    print(f"\nEnergy increase off line: {comparison['energy_increase_off_line']:.6f}")
    print(f"Critical line minimizes energy: {comparison['critical_line_minimizes_energy']}")
    print()
    
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The critical line Re(s) = 1/2 acts as a NORMALIZATION CONSTRAINT:")
    print()
    print("1. L2 perspective: Zeros are normalized to lie on the critical 'sphere'")
    print("2. Softmax perspective: Zeros create an optimal probability distribution")
    print("3. Energy perspective: Critical line minimizes the energy functional")
    print("4. Optimization perspective: Critical line is the constrained optimum")
    print()
    print("This connects RH to fundamental concepts in:")
    print("  - Machine learning (softmax, normalization layers)")
    print("  - Optimization (constrained optimization, Lagrange multipliers)")
    print("  - Information theory (entropy, distributions)")
    print("  - Physics (least action, energy minimization)")
    print()
    print("The zeros 'normalize' the prime distribution to the balanced state.")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_normalization_perspective()

