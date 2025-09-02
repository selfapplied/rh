#!/usr/bin/env python3
"""Simple demo of current RH analysis capabilities"""

import sys
import os
import argparse
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rh import RHIntegerAnalyzer, DihedralAction
from pascal import pascal_nested_brackets

def simple_demo(depth: int = 2, d: float = 0.05):
    """Simple demonstration of what's currently working"""
    print("🔍 RH Analysis System - Current Status")
    print("=" * 50)
    
    # Test basic functionality
    try:
        # Create analyzer
        analyzer = RHIntegerAnalyzer(depth=depth)
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
        
        # Visualize dihedral correlations (basic)
        try:
            # Rebuild template to compute correlations
            template = analyzer.mask_builder.build_template(coeffs, result['intervals'])
            scores = analyzer.correlator.correlate_all_actions_weighted(result['mask'], template)
            rotations = scores['rotations']
            reflections = scores['reflections']
            # Render bar plot (best-effort; skip if matplotlib missing)
            try:
                import importlib
                import math
                plt = importlib.import_module('matplotlib.pyplot')
                os.makedirs('.out/rieman', exist_ok=True)
                ts = time.strftime('%Y%m%d-%H%M%S')
                out_png = f".out/rieman/rieman-basic-N{analyzer.N}-t{result['s'].imag:.6f}-{ts}.png"
                fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
                x = list(range(len(rotations)))
                ax.bar(x, rotations, color='#1f77b4', alpha=0.7, label='rotations')
                ax.bar(x, reflections, color='#ff7f0e', alpha=0.4, label='reflections')
                ax.set_title(f"Dihedral correlation (weighted) N={analyzer.N} t≈{result['s'].imag:.3f}")
                ax.set_xlabel('shift index')
                ax.set_ylabel('score')
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_png)
                plt.close(fig)
                print(f"   📷 Saved basic correlation PNG: {out_png}")
            except Exception:
                pass
        except Exception:
            pass

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
        s_on = complex(0.5, 14.134725)
        s_off = complex(0.5 + d, 14.134725)
        meta_on = analyzer.analyze_point_metanion(s_on, zeros)
        meta_off = analyzer.analyze_point_metanion(s_off, zeros)
        print(f"   ON-LINE:  locked={meta_on['is_locked']}, gap={meta_on['gap']}, best={meta_on['best_action']}")
        print(f"   OFF-LINE: locked={meta_off['is_locked']}, gap={meta_off['gap']}, best={meta_off['best_action']}")
        # Visualize dihedral correlations (metanion mask/template)
        try:
            scores = analyzer.correlator.correlate_all_actions_weighted(meta_on['mask'], meta_on['template'])
            rotations = scores['rotations']
            reflections = scores['reflections']
            try:
                import importlib
                plt = importlib.import_module('matplotlib.pyplot')
                os.makedirs('.out/rieman', exist_ok=True)
                ts = time.strftime('%Y%m%d-%H%M%S')
                out_png = f".out/rieman/rieman-metanion-online-N{analyzer.N}-t{meta_on['s'].imag:.6f}-{ts}.png"
                fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
                x = list(range(len(rotations)))
                ax.bar(x, rotations, color='#2ca02c', alpha=0.7, label='rotations')
                ax.bar(x, reflections, color='#d62728', alpha=0.4, label='reflections')
                ax.set_title(f"Metanion dihedral correlation (on-line) N={analyzer.N} t≈{meta_on['s'].imag:.3f}")
                ax.set_xlabel('shift index')
                ax.set_ylabel('score')
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_png)
                plt.close(fig)
                print(f"   📷 Saved metanion (on-line) correlation PNG: {out_png}")
            except Exception:
                pass
        except Exception:
            pass

        # Visualize off-line as well
        try:
            scores = analyzer.correlator.correlate_all_actions_weighted(meta_off['mask'], meta_off['template'])
            rotations = scores['rotations']
            reflections = scores['reflections']
            try:
                import importlib
                plt = importlib.import_module('matplotlib.pyplot')
                os.makedirs('.out/rieman', exist_ok=True)
                ts = time.strftime('%Y%m%d-%H%M%S')
                out_png = f".out/rieman/rieman-metanion-offline-N{analyzer.N}-t{meta_off['s'].imag:.6f}-d{d:.3f}-{ts}.png"
                fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
                x = list(range(len(rotations)))
                ax.bar(x, rotations, color='#17becf', alpha=0.7, label='rotations')
                ax.bar(x, reflections, color='#9467bd', alpha=0.4, label='reflections')
                ax.set_title(f"Metanion dihedral correlation (off-line d={d}) N={analyzer.N} t≈{meta_off['s'].imag:.3f}")
                ax.set_xlabel('shift index')
                ax.set_ylabel('score')
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_png)
                plt.close(fig)
                print(f"   📷 Saved metanion (off-line) correlation PNG: {out_png}")
            except Exception:
                pass
        except Exception:
            pass

        # Pascal-space demo (migrated from main.py)
        print("\n📐 Pascal-Space Bracket Demo:")
        x = 0.5
        brackets = pascal_nested_brackets(x, 2, 4)
        if brackets:
            b = brackets[-1]
            print(f"  Bracket for x={x:.6f} at depth={b.depth}:")
            print(f"    Index: {b.cell_index}")
            print(f"    Interval: [{b.lower_bound:.6f}, {b.upper_bound:.6f}]")
            print(f"    Weight: C({b.N},{b.cell_index})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Riemann analysis demo with optional visualization")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--d", type=float, default=0.05, help="Offset from critical line for off-line demo")
    args = ap.parse_args()
    simple_demo(depth=args.depth, d=args.d)

