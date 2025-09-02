#!/usr/bin/env python3
"""Simple demo of current RH analysis capabilities"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rh import RHIntegerAnalyzer, DihedralAction

def simple_demo():
    """Simple demonstration of what's currently working"""
    print("🔍 RH Analysis System - Current Status")
    print("=" * 50)
    
    # Test basic functionality
    try:
        # Create analyzer
        analyzer = RHIntegerAnalyzer(depth=2)
        print(f"✓ Created analyzer with depth={analyzer.depth}, N={analyzer.N}")
        
        # Test with a simple point
        s = complex(0.5, 14.134725)  # Known RH zero
        coeffs = [1.0, 0.5, 0.25, 0.125, 0.0625]  # Simple geometric series
        
        print(f"\n📊 Testing point s = {s}")
        print(f"   Coefficients: {coeffs}")
        
        # Analyze the point
        result = analyzer.analyze_point(s, coeffs)
        
        print(f"\n📈 Analysis Results:")
        print(f"   Mask: {result['mask']}")
        print(f"   Best action: {result['best_action']}")
        print(f"   Locked: {result['is_locked']}")
        print(f"   Gap: {result['gap']}")
        print(f"   Lock reason: {result['lock_reason']}")
        
        # Test different depths
        print(f"\n🔍 Testing Different Depths:")
        for depth in [1, 2, 3]:
            try:
                test_analyzer = RHIntegerAnalyzer(depth=depth)
                N = 2**depth + 1
                test_coeffs = [1.0 / (i + 1) for i in range(N)]
                
                test_result = test_analyzer.analyze_point(s, test_coeffs)
                print(f"   Depth {depth} (N={N}): locked={test_result['is_locked']}, gap={test_result['gap']:.3f}")
                
            except Exception as e:
                print(f"   Depth {depth}: Error - {e}")
        
        print(f"\n✅ Basic system is working!")
        
        # Metanion-informed phaselock demo
        print(f"\n🧭 Metanion-informed phaselock demo:")
        zeros = [0.5+14.134725j, 0.5+21.022040j, 0.5+25.010858j]
        s = complex(0.5, 14.134725)
        meta = analyzer.analyze_point_metanion(s, zeros)
        print(f"   Locked={meta['is_locked']}, gap={meta['gap']}, best={meta['best_action']}, method={meta['method']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_demo()

