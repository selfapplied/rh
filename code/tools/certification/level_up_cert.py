#!/usr/bin/env python3
"""
Level-up certification: depth=5, N=33, extended zeros for stronger statistics.
"""

import argparse

from stamp_cert import write_stamped_ce1, write_stamped_toml
from stamps import CertificationStamper


def create_level_up_params():
    """Create level-up parameters for stronger certification."""
    
    # Extended zero list for ≥33 windows (first 11 zeros)
    extended_zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        52.970321
    ]
    
    return {
        "depth": 5,           # N = 2^5 + 1 = 33
        "N": 33,
        "gamma": 3,           # Pipeline gamma
        "d": 0.05,
        "window": 0.5,
        "step": 0.1,
        "zeros": extended_zeros
    }


def main():
    parser = argparse.ArgumentParser(description="Generate level-up RH certification")
    parser.add_argument("--out", type=str, default=".out/certs", help="Output directory")
    args = parser.parse_args()
    
    # Create level-up parameters
    params = create_level_up_params()
    
    print(f"Level-up certification: depth={params['depth']}, N={params['N']}, zeros={len(params['zeros'])}")
    
    # Apply stamps with level-up parameters
    stamper = CertificationStamper(depth=params["depth"])
    print("\nApplying level-up certification stamps...")
    stamp_results = stamper.stamp_certification(params)
    
    # Format for output
    stamps_formatted = stamper.format_stamps_for_ce1(stamp_results)
    
    # Generate output paths
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"cert-depth{params['depth']}-N{params['N']}-levelup_{timestamp}"
    
    import os
    os.makedirs(args.out, exist_ok=True)
    toml_path = os.path.join(args.out, f"{base}.toml")
    ce1_path = os.path.join(args.out, f"{base}.ce1")
    
    # Write outputs
    write_stamped_toml(toml_path, params, stamps_formatted)
    write_stamped_ce1(ce1_path, params, stamps_formatted)
    
    print(f"\nGenerated level-up certification:")
    print(f"  TOML: {toml_path}")
    print(f"  CE1:  {ce1_path}")
    
    # Print stamp results
    print(f"\nLevel-up Stamp Results:")
    print("=" * 60)
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
    
    print("=" * 60)
    print(f"Level-up results: {passed_count}/{implemented_count} stamps passed")
    
    # Check if Λ improved
    lambda_stamp = stamp_results.get("LAMBDA")
    if lambda_stamp and lambda_stamp.details:
        lambda_bound = lambda_stamp.details.get("lower_bound", 0)
        print(f"Λ lower bound: {lambda_bound:.6f} (target: higher than depth=4)")
    
    if implemented_count == 8 and passed_count == 8:
        print("🎉 LEVEL-UP CERTIFICATION ACHIEVED!")
        print("Ready for Pascal-Dihedral Tickets paper draft!")
    elif passed_count >= 6:
        print("✅ STRONG LEVEL-UP - most stamps passed")
    else:
        print("❌ LEVEL-UP NEEDS WORK - several stamps failing")
    
    return 0


if __name__ == "__main__":
    exit(main())
