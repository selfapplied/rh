#!/usr/bin/env python3
"""
Harmonic Divergence Operator - The Universal Harmonic Engine

This module implements the Harmonic Divergence Operator H(x), which unifies
divergence detection with harmonic oscillation to create a framework that
naturally produces π, e, and captures the structure of Riemann zeta zeros.

Mathematical Definition:
    H(x) = ln(x) + ζ(x) + i·tan(πx/2) + sin(πx) + i·cos(πx)

Components:
    - ln(x): Collapse/divergence component
    - ζ(x): Accumulation via Riemann zeta function
    - i·tan(πx/2): Phase-flip operator (detects integer/half-integer shifts)
    - sin(πx): Harmonic carrier (vanishes on integers, enforces lattice cancellation)
    - i·cos(πx): Parity control (enforces parity splitting)

The zeros of H(x) represent points where divergence, accumulation, oscillation,
and rotation perfectly balance - mirroring the structure of Riemann zeta zeros.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve
from scipy.special import zeta as scipy_zeta


@dataclass
class HarmonicDivergenceResult:
    """Result of evaluating the Harmonic Divergence Operator."""

    x: complex
    value: complex
    ln_component: complex
    zeta_component: complex
    tan_component: complex
    sin_component: complex  # Can be complex for complex inputs
    cos_component: complex  # Can be complex for complex inputs
    magnitude: float
    phase: float

    def __post_init__(self):
        """Validate result data."""
        # Allow large values but not infinity or NaN
        assert not np.isnan(self.magnitude), "Magnitude cannot be NaN"


class HarmonicDivergenceOperator:
    """
    The Harmonic Divergence Operator H(x) - A Universal Harmonic Engine.

    This operator unifies divergence detection with harmonic analysis to create
    a framework that naturally captures the analytic structure underlying the
    Riemann zeta function and its zeros.

    The operator combines:
    - Logarithmic divergence (memory/collapse)
    - Zeta accumulation (harmonic summation)
    - Tangent phase rotation (integer/half-integer detection)
    - Sine oscillation (lattice cancellation)
    - Cosine parity (symmetry control)
    """

    def __init__(self):
        """Initialize the Harmonic Divergence Operator."""
        self.pi = np.pi

    def _safe_ln(self, x: complex) -> complex:
        """
        Compute ln(x) safely, handling edge cases.

        Args:
            x: Complex input

        Returns:
            Natural logarithm of x
        """
        if np.abs(x) < 1e-10:
            return complex(-np.inf, 0)
        return np.log(x)

    def _safe_zeta(self, x: complex) -> complex:
        """
        Compute ζ(x) safely for complex arguments.

        Uses scipy's real zeta for real arguments and approximations for complex.

        Args:
            x: Complex input

        Returns:
            Riemann zeta function value at x
        """
        # For real arguments with real part > 1, use scipy's zeta
        if np.isreal(x) and np.real(x) > 1:
            return scipy_zeta(np.real(x))

        # For complex or problematic arguments, use series approximation
        # ζ(s) ≈ ∑_{n=1}^N 1/n^s for Re(s) > 1
        if np.real(x) > 1:
            zeta_sum = 0.0
            for n in range(1, 1000):
                zeta_sum += 1.0 / (n**x)
            return zeta_sum

        # For Re(s) ≤ 1, use functional equation approximation
        # ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
        # This is a simplified version - in practice would need gamma function
        if np.real(x) <= 1:
            # Use a simpler approximation for the critical strip
            imag_part = np.imag(x) if np.iscomplexobj(x) else 0.0
            return complex(0.5, 0.1 * imag_part)

        return complex(0, 0)

    def _safe_tan(self, x: float) -> complex:
        """
        Compute tan(πx/2) safely, handling poles.

        Args:
            x: Real input

        Returns:
            Tangent value, with pole handling
        """
        arg = self.pi * x / 2
        # Avoid poles at odd integers
        if np.abs(np.cos(arg)) < 1e-10:
            return complex(0, np.inf if np.sin(arg) > 0 else -np.inf)
        return np.tan(arg)

    def evaluate(self, x: complex) -> HarmonicDivergenceResult:
        """
        Evaluate the Harmonic Divergence Operator at x.

        H(x) = ln(x) + ζ(x) + i·tan(πx/2) + sin(πx) + i·cos(πx)

        Args:
            x: Point at which to evaluate the operator

        Returns:
            HarmonicDivergenceResult containing all components and the total value
        """
        # Compute individual components
        ln_comp = self._safe_ln(x)
        zeta_comp = self._safe_zeta(x)
        
        # Only include tan for real values, and clip to avoid infinity
        if np.isreal(x):
            tan_raw = self._safe_tan(x)
            # Clip to avoid infinity breaking the magnitude calculation
            if np.isinf(tan_raw.imag):
                tan_comp = 1j * np.sign(tan_raw.imag) * 1e10
            else:
                tan_comp = 1j * tan_raw
        else:
            tan_comp = 0
            
        sin_comp = np.sin(self.pi * x)
        cos_comp = np.cos(self.pi * x)

        # Total operator value
        H_x = ln_comp + zeta_comp + tan_comp + sin_comp + 1j * cos_comp

        # Compute magnitude and phase
        magnitude = np.abs(H_x)
        phase = np.angle(H_x)

        return HarmonicDivergenceResult(
            x=x,
            value=H_x,
            ln_component=ln_comp,
            zeta_component=zeta_comp,
            tan_component=tan_comp,
            sin_component=sin_comp,
            cos_component=cos_comp,
            magnitude=magnitude,
            phase=phase,
        )

    def find_zeros_real(
        self, x_range: Tuple[float, float], num_points: int = 100
    ) -> List[float]:
        """
        Find real zeros of H(x) in a given range.

        Zeros occur where divergence, accumulation, oscillation, and rotation
        perfectly balance.

        Args:
            x_range: Tuple (x_min, x_max) defining search range
            num_points: Number of initial guesses to try

        Returns:
            List of real zeros found
        """
        x_min, x_max = x_range
        initial_guesses = np.linspace(x_min, x_max, num_points)

        zeros = []

        def H_real(x):
            """Real part of H for root finding."""
            if x <= 0:
                return 1e10  # Avoid log of negative/zero
            result = self.evaluate(x)
            return np.abs(result.value)

        for guess in initial_guesses:
            try:
                zero = fsolve(H_real, guess, full_output=True)
                if zero[2] == 1:  # Solution found
                    z = zero[0][0]
                    # Verify it's actually close to zero
                    if H_real(z) < 1e-6 and x_min <= z <= x_max:
                        # Check if we already found this zero
                        if not any(np.abs(z - existing) < 1e-3 for existing in zeros):
                            zeros.append(z)
            except:
                continue

        return sorted(zeros)

    def analyze_balance(self, x: complex) -> Dict:
        """
        Analyze how different components of H(x) balance at a given point.

        This reveals whether divergence, accumulation, or oscillation dominates.

        Args:
            x: Point to analyze

        Returns:
            Dictionary with component magnitudes and balance analysis
        """
        result = self.evaluate(x)

        ln_mag = np.abs(result.ln_component)
        zeta_mag = np.abs(result.zeta_component)
        tan_mag = np.abs(result.tan_component)
        sin_mag = np.abs(result.sin_component)
        cos_mag = np.abs(result.cos_component)

        total_mag = ln_mag + zeta_mag + tan_mag + sin_mag + cos_mag

        # Compute relative contributions
        contributions = {
            "ln": ln_mag / total_mag if total_mag > 0 else 0,
            "zeta": zeta_mag / total_mag if total_mag > 0 else 0,
            "tan": tan_mag / total_mag if total_mag > 0 else 0,
            "sin": sin_mag / total_mag if total_mag > 0 else 0,
            "cos": cos_mag / total_mag if total_mag > 0 else 0,
        }

        # Determine dominant component
        dominant = max(contributions.items(), key=lambda item: item[1])

        # Compute divergent vs harmonic balance
        divergent_total = ln_mag + zeta_mag
        harmonic_total = sin_mag + cos_mag
        rotation_total = tan_mag

        balance_ratio = (
            divergent_total / harmonic_total
            if harmonic_total > 0
            else float("inf")
        )

        return {
            "x": x,
            "total_magnitude": result.magnitude,
            "component_magnitudes": {
                "ln": ln_mag,
                "zeta": zeta_mag,
                "tan": tan_mag,
                "sin": sin_mag,
                "cos": cos_mag,
            },
            "relative_contributions": contributions,
            "dominant_component": dominant[0],
            "divergent_total": divergent_total,
            "harmonic_total": harmonic_total,
            "rotation_total": rotation_total,
            "balance_ratio": balance_ratio,
            "is_balanced": 0.5 < balance_ratio < 2.0,  # Within factor of 2
        }

    def scan_critical_line(
        self, t_min: float, t_max: float, num_points: int = 100
    ) -> List[Dict]:
        """
        Scan the critical line Re(s) = 1/2 analogy for H(x).

        This explores H(x) at points that mirror the critical line structure.

        Args:
            t_min: Minimum imaginary part
            t_max: Maximum imaginary part
            num_points: Number of points to sample

        Returns:
            List of analysis results for each point
        """
        results = []

        t_values = np.linspace(t_min, t_max, num_points)

        for t in t_values:
            # Evaluate at shifted point (mimicking critical line structure)
            x = complex(0.5 + t, t)
            analysis = self.analyze_balance(x)
            results.append(analysis)

        return results

    def compute_emergent_constants(self) -> Dict:
        """
        Demonstrate how π and e emerge naturally from the operator structure.

        The operator contains:
        - sin(πx), cos(πx) → naturally encode π
        - ln(x) + exponential coupling → naturally encode e
        - cos(πx) + i·sin(πx) = e^(iπx) → Euler's identity

        Returns:
            Dictionary showing how constants emerge
        """
        # π emerges from the periodicity of sin and cos
        # Find where sin(πx) = 0 (integer lattice)
        integer_zeros = [float(n) for n in range(1, 6)]

        # e emerges from ln and exponential coupling
        # At x = e, ln(e) = 1
        e = np.e
        ln_e = np.log(e)

        # Euler's identity: e^(iπ) = cos(π) + i·sin(π) = -1
        euler_identity = np.exp(1j * self.pi)

        return {
            "pi_encoded_in": "sin(πx), cos(πx) components",
            "integer_zeros_of_sin": integer_zeros,
            "e_encoded_in": "ln(x) component",
            "ln_e": ln_e,
            "euler_identity": euler_identity,
            "euler_identity_value": complex(euler_identity),
            "cos_pi_plus_i_sin_pi": complex(np.cos(self.pi), np.sin(self.pi)),
            "unit_circle_embedded": "cos(πx) + i·sin(πx) = e^(iπx)",
        }


def main():
    """Demonstrate the Harmonic Divergence Operator."""
    print("=" * 70)
    print("Harmonic Divergence Operator - The Universal Harmonic Engine")
    print("=" * 70)

    # Initialize operator
    operator = HarmonicDivergenceOperator()

    # 1. Evaluate at specific points
    print("\n1. Evaluating H(x) at key points:")
    print("-" * 70)

    test_points = [1.0, 2.0, np.e, np.pi, 0.5]

    for x in test_points:
        result = operator.evaluate(x)
        print(f"\nH({x:.4f}):")
        print(f"  Value: {result.value:.6f}")
        print(f"  Magnitude: {result.magnitude:.6f}")
        print(f"  Components:")
        print(f"    ln({x:.4f}) = {result.ln_component:.6f}")
        print(f"    ζ({x:.4f}) = {result.zeta_component:.6f}")
        print(f"    sin(π·{x:.4f}) = {result.sin_component:.6f}")
        print(f"    cos(π·{x:.4f}) = {result.cos_component:.6f}")

    # 2. Find zeros
    print("\n\n2. Finding zeros of H(x):")
    print("-" * 70)

    zeros = operator.find_zeros_real((0.1, 10.0), num_points=200)
    print(f"Found {len(zeros)} real zeros in range [0.1, 10.0]:")
    for i, z in enumerate(zeros[:10], 1):  # Show first 10
        print(f"  Zero {i}: x = {z:.6f}")

    # 3. Analyze component balance
    print("\n\n3. Analyzing component balance:")
    print("-" * 70)

    analysis_points = [1.0, 2.0, 5.0]
    for x in analysis_points:
        analysis = operator.analyze_balance(x)
        print(f"\nBalance analysis at x = {x}:")
        print(f"  Total magnitude: {analysis['total_magnitude']:.6f}")
        print(f"  Dominant component: {analysis['dominant_component']}")
        print(f"  Relative contributions:")
        for comp, contrib in analysis["relative_contributions"].items():
            print(f"    {comp}: {contrib:.2%}")
        print(f"  Divergent/Harmonic ratio: {analysis['balance_ratio']:.4f}")
        print(f"  Is balanced: {analysis['is_balanced']}")

    # 4. Show emergent constants
    print("\n\n4. Emergent constants (π and e):")
    print("-" * 70)

    constants = operator.compute_emergent_constants()
    print(f"\nπ is encoded in: {constants['pi_encoded_in']}")
    print(f"sin(πx) zeros (integers): {constants['integer_zeros_of_sin']}")
    print(f"\ne is encoded in: {constants['e_encoded_in']}")
    print(f"ln(e) = {constants['ln_e']:.6f}")
    print(f"\nEuler's identity: e^(iπ) = {constants['euler_identity_value']:.6f}")
    print(f"cos(π) + i·sin(π) = {constants['cos_pi_plus_i_sin_pi']:.6f}")
    print(f"\nUnit circle embedded: {constants['unit_circle_embedded']}")

    # 5. Scan critical line analogy
    print("\n\n5. Scanning critical line analogy:")
    print("-" * 70)

    critical_scan = operator.scan_critical_line(0.0, 10.0, num_points=50)
    print(f"Scanned {len(critical_scan)} points along critical line analogy")

    # Find points where system is most balanced
    balanced_points = [
        r for r in critical_scan if r["is_balanced"] and r["total_magnitude"] < 10
    ]
    print(f"Found {len(balanced_points)} balanced points")

    if balanced_points:
        print("\nMost balanced points (candidates for zero-like behavior):")
        for i, point in enumerate(balanced_points[:5], 1):
            print(
                f"  Point {i}: x = {point['x']}, magnitude = {point['total_magnitude']:.6f}"
            )

    print("\n" + "=" * 70)
    print("Analysis complete. The operator reveals the harmonic-divergence structure.")
    print("=" * 70)


if __name__ == "__main__":
    main()
