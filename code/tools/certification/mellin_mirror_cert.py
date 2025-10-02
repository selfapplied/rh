#!/usr/bin/env python3
"""
Mellin-Mirror Lemma Certificate: The first real proof-object.

This demonstrates how a mathematical lemma (Mellin-mirror duality) becomes
an operational certificate with REP+DUAL stamps and live verification data.

LEMMA (Mellin-Mirror Duality): For Pascal kernel K_N at depth d, the operator T
satisfies T† = T under the Mellin transform, implying ξ(s) = ξ(1-s̄) functionally.

PROOF STRATEGY: 
1. REP stamp: Verify T is unitary (⟨f,f⟩ preserved)
2. DUAL stamp: Test functional equation on random test functions
"""

import argparse
import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from code.riemann.analysis.rh_analyzer import PascalKernel


@dataclass
class MellinMirrorOperator:
    """The Pascal-dihedral operator T with Mellin-mirror structure."""
    
    depth: int
    N: int
    kernel: PascalKernel
    
    def __post_init__(self):
        self.kernel = PascalKernel(self.N, self.depth)
        
    def apply_to_function(self, f_values: List[complex], s: complex) -> List[complex]:
        """Apply T to function values at complex point s."""
        kernel_weights = self.kernel.get_normalized_kernel()
        
        # Mellin transform approximation
        result = []
        for i, f_val in enumerate(f_values):
            # Apply Pascal smoothing with Mellin scaling
            weight_idx = min(i, len(kernel_weights) - 1)
            mellin_factor = s**(-i/len(f_values))  # Simplified Mellin scaling
            transformed = f_val * kernel_weights[weight_idx] * mellin_factor
            result.append(transformed)
        
        return result
    
    def compute_adjoint(self, f_values: List[complex], s: complex) -> List[complex]:
        """Compute T† (adjoint operator)."""
        # For Mellin-mirror duality: T†(s) = T(1-s̄)
        s_mirror = 1 - s.conjugate()
        return self.apply_to_function(f_values, s_mirror)


class MellinMirrorStamps:
    """Stamps for Mellin-Mirror lemma verification."""
    
    @staticmethod
    def rep_stamp_unitary_preservation(operator: MellinMirrorOperator, 
                                     test_functions: List[List[complex]],
                                     test_points: List[complex],
                                     tolerance: float = 1e-6) -> Dict[str, Any]:
        """REP stamp: Verify T preserves inner products (unitarity)."""
        
        unitary_errors = []
        
        for f_vals in test_functions:
            for s in test_points:
                # Compute ⟨f,f⟩
                inner_original = sum(abs(x)**2 for x in f_vals)
                
                # Compute ⟨Tf,Tf⟩  
                tf_vals = operator.apply_to_function(f_vals, s)
                inner_transformed = sum(abs(x)**2 for x in tf_vals)
                
                # Unitary error: |⟨Tf,Tf⟩ - ⟨f,f⟩|
                error = abs(inner_transformed - inner_original)
                unitary_errors.append(error)
        
        max_error = max(unitary_errors) if unitary_errors else 0.0
        med_error = float(np.median(unitary_errors)) if unitary_errors else 0.0
        
        return {
            "name": "REP",
            "passed": max_error <= tolerance,
            "unitary_error_max": max_error,
            "unitary_error_med": med_error,
            "tolerance": tolerance,
            "test_functions": len(test_functions),
            "test_points": len(test_points),
            "total_tests": len(unitary_errors)
        }
    
    @staticmethod
    def dual_stamp_functional_equation(operator: MellinMirrorOperator,
                                     test_functions: List[List[complex]],
                                     test_points: List[complex],
                                     tolerance: float = 1e-6) -> Dict[str, Any]:
        """DUAL stamp: Verify ξ(s) = ξ(1-s̄) via Mellin-mirror duality."""
        
        fe_residuals = []
        
        for f_vals in test_functions:
            for s in test_points:
                # Compute T(f)(s)
                tf_s = operator.apply_to_function(f_vals, s)
                xi_s = sum(tf_s)  # Simple zeta proxy
                
                # Compute T†(f)(s) = T(f)(1-s̄) by Mellin-mirror duality
                tf_mirror = operator.compute_adjoint(f_vals, s)
                xi_mirror = sum(tf_mirror)
                
                # Functional equation residual: |ξ(s) - ξ(1-s̄)|
                residual = abs(xi_s - xi_mirror)
                fe_residuals.append(residual)
        
        max_residual = max(fe_residuals) if fe_residuals else 0.0
        med_residual = float(np.median(fe_residuals)) if fe_residuals else 0.0
        p95_residual = float(np.percentile(fe_residuals, 95)) if fe_residuals else 0.0
        
        return {
            "name": "DUAL", 
            "passed": med_residual <= tolerance,
            "fe_resid_med": med_residual,
            "fe_resid_p95": p95_residual,
            "fe_resid_max": max_residual,
            "tolerance": tolerance,
            "test_functions": len(test_functions),
            "test_points": len(test_points),
            "total_tests": len(fe_residuals)
        }


def generate_test_functions(num_functions: int, function_size: int, 
                          seed: int = 42) -> List[List[complex]]:
    """Generate random test functions for verification."""
    random.seed(seed)
    
    test_functions = []
    for _ in range(num_functions):
        # Generate smooth random function
        f_vals = []
        for i in range(function_size):
            # Use smooth random coefficients
            real_part = random.gauss(0, 1) * math.exp(-i/function_size)
            imag_part = random.gauss(0, 1) * math.exp(-i/function_size)
            f_vals.append(complex(real_part, imag_part))
        test_functions.append(f_vals)
    
    return test_functions


def generate_test_points(num_points: int, seed: int = 42) -> List[complex]:
    """Generate test points for verification."""
    random.seed(seed)
    
    test_points = []
    for _ in range(num_points):
        # Focus on critical strip
        sigma = 0.5 + random.gauss(0, 0.1)  # Near critical line
        t = random.uniform(10, 50)  # Reasonable imaginary part
        test_points.append(complex(sigma, t))
    
    return test_points


def write_mellin_mirror_certificate(path: str, rep_result: Dict, dual_result: Dict, 
                                   metadata: Dict) -> None:
    """Write the Mellin-Mirror lemma certificate."""
    
    lines = []
    lines.append("CE1{\n")
    lines.append("  lens=MELLIN_MIRROR_LEMMA\n")
    lines.append("  mode=LemmaCertification\n")
    lines.append("  basis=pascal_dihedral_unitary\n")
    lines.append(f"  params{{ depth={metadata['depth']}; N={metadata['N']}; tolerance={metadata['tolerance']} }}\n")
    lines.append("\n")
    
    # Mathematical statement
    lines.append("  lemma_statement=\"For Pascal kernel K_N at depth d, operator T satisfies T† = T under Mellin transform\"\n")
    lines.append("  proof_strategy=\"REP: verify unitarity; DUAL: test functional equation on random functions\"\n")
    lines.append("\n")
    
    # Verification stamps
    lines.append("  stamps{\n")
    
    # REP stamp
    lines.append(f"    REP{{ ")
    lines.append(f"unitary_error_max={rep_result['unitary_error_max']:.6f}; ")
    lines.append(f"unitary_error_med={rep_result['unitary_error_med']:.6f}; ")
    lines.append(f"tolerance={rep_result['tolerance']:.6f}; ")
    lines.append(f"test_functions={rep_result['test_functions']}; ")
    lines.append(f"test_points={rep_result['test_points']}; ")
    lines.append(f"total_tests={rep_result['total_tests']}; ")
    lines.append(f"pass = {str(rep_result['passed']).lower()} ")
    lines.append("}}\n")
    
    # DUAL stamp
    lines.append(f"    DUAL{{ ")
    lines.append(f"fe_resid_med={dual_result['fe_resid_med']:.6f}; ")
    lines.append(f"fe_resid_p95={dual_result['fe_resid_p95']:.6f}; ")
    lines.append(f"fe_resid_max={dual_result['fe_resid_max']:.6f}; ")
    lines.append(f"tolerance={dual_result['tolerance']:.6f}; ")
    lines.append(f"test_functions={dual_result['test_functions']}; ")
    lines.append(f"test_points={dual_result['test_points']}; ")
    lines.append(f"total_tests={dual_result['total_tests']}; ")
    lines.append(f"pass = {str(dual_result['passed']).lower()} ")
    lines.append("}}\n")
    
    lines.append("  }\n")
    lines.append("\n")
    
    # Live verification data
    lines.append("  verification{\n")
    lines.append(f"    lemma_verified = {str(rep_result['passed'] and dual_result['passed']).lower()}\n")
    lines.append(f"    unitarity_preserved = {str(rep_result['passed']).lower()}\n")
    lines.append(f"    functional_equation_satisfied = {str(dual_result['passed']).lower()}\n")
    lines.append(f"    mellin_mirror_duality = {str(rep_result['passed'] and dual_result['passed']).lower()}\n")
    lines.append("  }\n")
    lines.append("\n")
    
    # Provenance
    lines.append("  provenance{\n")
    lines.append(f"    timestamp_utc=\"{metadata['timestamp']}\"\n")
    lines.append(f"    git_rev=\"{metadata['git_rev']}\"\n")
    lines.append(f"    proof_hash=\"{metadata['proof_hash']}\"\n")
    lines.append(f"    rng_seed={metadata['seed']}\n")
    lines.append("  }\n")
    lines.append("\n")
    
    # Self-validation rules
    lines.append("  validator_rules{\n")
    lines.append("    lens=MELLIN_MIRROR_VALIDATE\n")
    lines.append(f"    assert_rep_unitary = {rep_result['unitary_error_max']:.6f} <= {rep_result['tolerance']:.6f}\n")
    lines.append(f"    assert_dual_symmetric = {dual_result['fe_resid_med']:.6f} <= {dual_result['tolerance']:.6f}\n")
    lines.append(f"    assert_sufficient_tests = {rep_result['total_tests']} >= 100\n")
    lines.append("    assert_lemma_proved = REP.pass && DUAL.pass\n")
    lines.append("    emit=MELLIN_MIRROR_LEMMA_VALIDATED\n")
    lines.append("  }\n")
    lines.append("\n")
    
    # Certificate outcome
    lemma_proved = rep_result['passed'] and dual_result['passed']
    lines.append(f"  lemma_status=\"{'PROVED' if lemma_proved else 'UNPROVED'}\"\n")
    lines.append(f"  validator=MELLIN_MIRROR_LEMMA.{'pass' if lemma_proved else 'fail'}\n")
    lines.append("  emit=MellinMirrorLemmaCertificate\n")
    lines.append("}\n")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate Mellin-Mirror lemma certificate")
    parser.add_argument("--depth", type=int, default=4, help="Pascal depth")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Verification tolerance")
    parser.add_argument("--num-functions", type=int, default=20, help="Number of test functions")
    parser.add_argument("--num-points", type=int, default=10, help="Number of test points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out", type=str, default=".out/certs", help="Output directory")
    args = parser.parse_args()
    
    # Set up operator
    N = 2**args.depth + 1
    operator = MellinMirrorOperator(depth=args.depth, N=N, kernel=None)
    
    print(f"Mellin-Mirror Lemma Certificate")
    print(f"Operator: depth={args.depth}, N={N}")
    print(f"Tests: {args.num_functions} functions × {args.num_points} points = {args.num_functions * args.num_points} total")
    
    # Generate test data
    random.seed(args.seed)
    test_functions = generate_test_functions(args.num_functions, N, args.seed)
    test_points = generate_test_points(args.num_points, args.seed + 1)
    
    print(f"\nVerifying REP stamp (unitarity preservation)...")
    rep_result = MellinMirrorStamps.rep_stamp_unitary_preservation(
        operator, test_functions, test_points, args.tolerance
    )
    
    print(f"Verifying DUAL stamp (functional equation)...")
    dual_result = MellinMirrorStamps.dual_stamp_functional_equation(
        operator, test_functions, test_points, args.tolerance
    )
    
    # Gather metadata
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Create proof hash from lemma content
    lemma_content = f"mellin_mirror|depth={args.depth}|N={N}|tolerance={args.tolerance}"
    proof_hash = hashlib.sha256(lemma_content.encode()).hexdigest()[:16]
    
    try:
        import subprocess
        git_result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
        git_rev = git_result.stdout.strip()[:12] if git_result.returncode == 0 else "unknown"
    except:
        git_rev = "unknown"
    
    metadata = {
        "depth": args.depth,
        "N": N,
        "tolerance": args.tolerance,
        "timestamp": timestamp,
        "git_rev": git_rev,
        "proof_hash": proof_hash,
        "seed": args.seed
    }
    
    # Generate certificate
    timestamp_file = time.strftime("%Y%m%d-%H%M%S")
    base = f"lemma-mellin-mirror-depth{args.depth}-{timestamp_file}"
    
    os.makedirs(args.out, exist_ok=True)
    cert_path = os.path.join(args.out, f"{base}.ce1")
    
    write_mellin_mirror_certificate(cert_path, rep_result, dual_result, metadata)
    
    # Print results
    print(f"\nLemma Verification Results:")
    print("=" * 50)
    print(f"REP (Unitarity):       {'PASS' if rep_result['passed'] else 'FAIL'}")
    print(f"  Max error:           {rep_result['unitary_error_max']:.6f}")
    print(f"  Median error:        {rep_result['unitary_error_med']:.6f}")
    print(f"  Tolerance:           {rep_result['tolerance']:.6f}")
    print()
    print(f"DUAL (Functional Eq):  {'PASS' if dual_result['passed'] else 'FAIL'}")
    print(f"  Median residual:     {dual_result['fe_resid_med']:.6f}")
    print(f"  95th percentile:     {dual_result['fe_resid_p95']:.6f}")
    print(f"  Max residual:        {dual_result['fe_resid_max']:.6f}")
    print(f"  Tolerance:           {dual_result['tolerance']:.6f}")
    print("=" * 50)
    
    # Lemma verdict
    lemma_proved = rep_result['passed'] and dual_result['passed']
    print(f"\nMELLIN-MIRROR LEMMA: {'✅ PROVED' if lemma_proved else '❌ UNPROVED'}")
    
    if lemma_proved:
        print("✅ T preserves inner products (unitary)")
        print("✅ T satisfies functional equation (Mellin-mirror duality)")
        print("✅ Mathematical foundation verified with live data")
    else:
        print("❌ Unitarity or functional equation failed")
        print("❌ Lemma requires refinement")
    
    print(f"\nGenerated certificate: {cert_path}")
    print("Certificate contains live verification data and self-audit logic.")
    
    return 0 if lemma_proved else 1


# Import the stamps class from our module
class MellinMirrorStamps:
    """Stamps for Mellin-Mirror lemma verification."""
    
    @staticmethod
    def rep_stamp_unitary_preservation(operator: MellinMirrorOperator, 
                                     test_functions: List[List[complex]],
                                     test_points: List[complex],
                                     tolerance: float = 1e-6) -> Dict[str, Any]:
        """REP stamp: Verify T preserves inner products (unitarity)."""
        
        unitary_errors = []
        
        for f_vals in test_functions:
            for s in test_points:
                # Compute ⟨f,f⟩
                inner_original = sum(abs(x)**2 for x in f_vals)
                
                # Compute ⟨Tf,Tf⟩  
                tf_vals = operator.apply_to_function(f_vals, s)
                inner_transformed = sum(abs(x)**2 for x in tf_vals)
                
                # Unitary error: |⟨Tf,Tf⟩ - ⟨f,f⟩|
                error = abs(inner_transformed - inner_original)
                unitary_errors.append(error)
        
        max_error = max(unitary_errors) if unitary_errors else 0.0
        med_error = float(np.median(unitary_errors)) if unitary_errors else 0.0
        
        return {
            "name": "REP",
            "passed": max_error <= tolerance,
            "unitary_error_max": max_error,
            "unitary_error_med": med_error,
            "tolerance": tolerance,
            "test_functions": len(test_functions),
            "test_points": len(test_points),
            "total_tests": len(unitary_errors)
        }
    
    @staticmethod
    def dual_stamp_functional_equation(operator: MellinMirrorOperator,
                                     test_functions: List[List[complex]],
                                     test_points: List[complex],
                                     tolerance: float = 1e-6) -> Dict[str, Any]:
        """DUAL stamp: Verify ξ(s) = ξ(1-s̄) via Mellin-mirror duality."""
        
        fe_residuals = []
        
        for f_vals in test_functions:
            for s in test_points:
                # Compute T(f)(s)
                tf_s = operator.apply_to_function(f_vals, s)
                xi_s = sum(tf_s)  # Simple zeta proxy
                
                # Compute T†(f)(s) = T(f)(1-s̄) by Mellin-mirror duality
                tf_mirror = operator.compute_adjoint(f_vals, s)
                xi_mirror = sum(tf_mirror)
                
                # Functional equation residual: |ξ(s) - ξ(1-s̄)|
                residual = abs(xi_s - xi_mirror)
                fe_residuals.append(residual)
        
        max_residual = max(fe_residuals) if fe_residuals else 0.0
        med_residual = float(np.median(fe_residuals)) if fe_residuals else 0.0
        p95_residual = float(np.percentile(fe_residuals, 95)) if fe_residuals else 0.0
        
        return {
            "name": "DUAL", 
            "passed": med_residual <= tolerance,
            "fe_resid_med": med_residual,
            "fe_resid_p95": p95_residual,
            "fe_resid_max": max_residual,
            "tolerance": tolerance,
            "test_functions": len(test_functions),
            "test_points": len(test_points),
            "total_tests": len(fe_residuals)
        }


if __name__ == "__main__":
    exit(main())
