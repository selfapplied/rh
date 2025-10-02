#!/usr/bin/env python3
"""
Stress test certification: ≥33 windows + depth=6 to test caps and monotonicity rules.
"""

import argparse

from riemann.verification.validation import CertificationStamper
from tools.certification.stamp_cert import write_stamped_ce1, write_stamped_toml


def create_stress_test_params():
    """Create stress test parameters with ≥33 windows and depth=6."""
    
    # Extended zero list for ≥33 windows (first 35 zeros for robust statistics)
    extended_zeros = [
        # First 35 non-trivial zeros of ζ(s)
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        52.970321, 56.446248, 59.347044, 60.831778, 65.112544,
        67.079811, 69.546402, 72.067158, 75.704690, 77.144840,
        79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
        92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
        103.725538, 105.446623, 107.168611, 111.029535, 111.874659
    ]
    
    return {
        "depth": 6,           # N = 2^6 + 1 = 65 (stress test)
        "N": 65,
        "gamma": 3,           # Pipeline gamma
        "d": 0.05,
        "window": 0.5,
        "step": 0.1,
        "zeros": extended_zeros
    }


def main():
    parser = argparse.ArgumentParser(description="Generate stress test RH certification")
    parser.add_argument("--out", type=str, default=".out/certs", help="Output directory")
    parser.add_argument("--depth", type=int, default=6, help="Stress test depth")
    args = parser.parse_args()
    
    # Create stress test parameters
    params = create_stress_test_params()
    if args.depth != 6:
        params["depth"] = args.depth
        params["N"] = 2**args.depth + 1
    
    print(f"Stress test certification: depth={params['depth']}, N={params['N']}, zeros={len(params['zeros'])}")
    
    # Apply stamps with stress test parameters
    stamper = CertificationStamper(depth=params["depth"])
    print("\nApplying stress test certification stamps...")
    stamp_results = stamper.stamp_certification(params)
    
    # Format for output
    stamps_formatted = stamper.format_stamps_for_ce1(stamp_results)
    
    # Generate output paths
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"cert-depth{params['depth']}-N{params['N']}-stress_{timestamp}"
    
    import os
    os.makedirs(args.out, exist_ok=True)
    toml_path = os.path.join(args.out, f"{base}.toml")
    ce1_path = os.path.join(args.out, f"{base}.ce1")
    
    # Write outputs
    write_stamped_toml(toml_path, params, stamps_formatted)
    write_stamped_ce1(ce1_path, params, stamps_formatted)
    
    print(f"\nGenerated stress test certification:")
    print(f"  TOML: {toml_path}")
    print(f"  CE1:  {ce1_path}")
    
    # Print stamp results
    print(f"\nStress Test Stamp Results:")
    print("=" * 70)
    implemented_count = 0
    passed_count = 0
    
    for name, stamp in stamp_results.items():
        status = "PASS" if stamp.passed else "FAIL"
        implemented = not (stamp.error_max == float('inf'))
        
        if implemented:
            implemented_count += 1
            if stamp.passed:
                passed_count += 1
            print(f"{name:12} | {status:4} | err_max={stamp.error_max:.6f} | err_med={stamp.error_med:.6f}")
        else:
            print(f"{name:12} | TODO | (not yet implemented)")
    
    print("=" * 70)
    print(f"Stress test results: {passed_count}/{implemented_count} stamps passed")
    
    # Analyze adaptive threshold behavior
    line_lock = stamp_results.get("LINE_LOCK")
    if line_lock and line_lock.details:
        depth = line_lock.details.get("depth", params["depth"])
        thresh_med = line_lock.details.get("adaptive_dist_med_threshold", 0.01)
        thresh_max = line_lock.details.get("adaptive_dist_max_threshold", 0.02)
        base_med = line_lock.details.get("base_th_med", 0.01)
        base_max = line_lock.details.get("base_th_max", 0.02)
        
        print(f"\nAdaptive Threshold Analysis:")
        print(f"  depth={depth}: thresh_med={thresh_med:.3f} (vs base={base_med:.3f})")
        print(f"  depth={depth}: thresh_max={thresh_max:.3f} (vs base={base_max:.3f})")
        print(f"  Scaling factor: {thresh_med/base_med:.1f}x for median")
        
        # Check caps
        if thresh_med >= 0.09:
            print(f"  ⚠️  Near median cap (0.100)")
        if thresh_max >= 0.055:
            print(f"  ⚠️  Near max cap (0.060)")
    
    # Check Λ stability
    lambda_stamp = stamp_results.get("LAMBDA")
    if lambda_stamp and lambda_stamp.details:
        lambda_bound = lambda_stamp.details.get("lower_bound", 0)
        print(f"\nΛ Stability Check:")
        print(f"  Lower bound: {lambda_bound:.6f}")
        if lambda_bound > 0.04:
            print(f"  ✅ Λ stable and positive")
        elif lambda_bound > 0.0:
            print(f"  ⚠️  Λ positive but declining")
        else:
            print(f"  ❌ Λ negative - system under stress")
    
    # Overall assessment
    if implemented_count == 8 and passed_count >= 7:
        print("\n🎉 STRESS TEST PASSED - System scales well!")
        print("Ready for production CE1 paper!")
    elif passed_count >= 6:
        print("\n✅ STRESS TEST MOSTLY PASSED - Good scalability")
        print("Minor issues at high depth (expected)")
    else:
        print("\n❌ STRESS TEST CHALLENGING - System limits reached")
        print("Consider depth=5 as practical limit")
    
    return 0


if __name__ == "__main__":
    exit(main())