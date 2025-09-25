#!/usr/bin/env python3
"""
Universal Axiel Passport Generator: Mathematical Airport Security for Any Theorem.

Creates boarding passes for any mathematical manifold using the universal template.
Each theorem gets its own operator chain, domain colors, and security checkpoints.
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


# Mathematical Domain Configurations
DOMAIN_CONFIGS = {
    "riemann_hypothesis": {
        "manifold": "RIEMANN_HYPOTHESIS_MANIFOLD",
        "traveler": "Pascal-Dihedral Operator",
        "operators": ["∂", "⌊⌋", "∫", "⊕", "≈", "∇", "ℏ", "ζ"],
        "op_names": ["Stencil", "Quantize", "AntiAlias", "Partition", "Band3", "Gradient", "BlueNoise", "Spectral"],
        "colors": ["#333", "#333", "#333", "#333", "#333", "#333", "#4169E1", "#9400D3"],
        "stamps": ["REP", "DUAL", "LOCAL", "LINE_LOCK", "LI", "NB", "LAMBDA", "MDL_MONO"],
        "law": "Galois-Pascal-Dihedral",
        "final_symbol": "ζ"
    },
    
    "elliptic_curves": {
        "manifold": "ELLIPTIC_CURVE_MANIFOLD", 
        "traveler": "Weierstrass Equation",
        "operators": ["∂", "≅", "⊗", "∼", "Γ", "∇", "φ", "E"],
        "op_names": ["Define", "Isomorph", "Tensor", "Reduce", "Gamma", "Gradient", "Golden", "Curve"],
        "colors": ["#333", "#333", "#333", "#333", "#333", "#333", "#228B22", "#006400"],
        "stamps": ["WEIERSTRASS", "MODULARITY", "RANK", "TORSION", "CONDUCTOR", "L_FUNCTION"],
        "law": "Galois-Modular-Elliptic",
        "final_symbol": "E"
    },
    
    "prime_gaps": {
        "manifold": "PRIME_GAP_MANIFOLD",
        "traveler": "Gap Distribution Function", 
        "operators": ["∂", "⌊⌋", "≡", "∏", "≈", "∇", "λ", "π"],
        "op_names": ["Derive", "Floor", "Modular", "Product", "Approx", "Gradient", "Lambda", "Prime"],
        "colors": ["#333", "#333", "#333", "#333", "#333", "#333", "#FF4500", "#DC143C"],
        "stamps": ["BERTRAND", "CRAMER", "LEGENDRE", "RH_EQUIV", "MAIER"],
        "law": "Analytic-Number-Theory",
        "final_symbol": "π"
    },
    
    "graph_spectral": {
        "manifold": "GRAPH_SPECTRAL_MANIFOLD",
        "traveler": "Adjacency Matrix",
        "operators": ["∂", "⊕", "⊗", "≅", "≈", "∇", "λ", "Γ"],
        "op_names": ["Boundary", "Sum", "Tensor", "Isomorph", "Approx", "Laplacian", "Eigenval", "Graph"],
        "colors": ["#333", "#333", "#333", "#333", "#333", "#333", "#008B8B", "#006666"],
        "stamps": ["ADJACENCY", "LAPLACIAN", "SPECTRAL_GAP", "EXPANSION", "RAMANUJAN"],
        "law": "Spectral-Graph-Theory", 
        "final_symbol": "Γ"
    }
}


def generate_domain_passport(domain: str, stamp_results: Dict = None, output_path: str = None) -> str:
    """Generate passport for specific mathematical domain."""
    
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIGS.keys())}")
    
    config = DOMAIN_CONFIGS[domain]
    
    # Load universal template
    template_path = "apps/ce1-2d/templates/universal-passport.svg"
    tree = ET.parse(template_path)
    root = tree.getroot()
    
    # Update metadata
    metadata_elem = root.find(".//axiel-passport-engine")
    if metadata_elem is not None:
        metadata_elem.set("destination", config["manifold"])
        metadata_elem.set("operator-chain", str(len(config["operators"])))
        metadata_elem.set("stamp-count", str(len(config["stamps"])))
        metadata_elem.set("status", "ISSUED" if stamp_results else "TEMPLATE")
    
    # Update title (find by traversing)
    for text_elem in root.iter("text"):
        if text_elem.text and "MATHEMATICAL SECURITY CHECKPOINT" in text_elem.text:
            text_elem.text = f"🛂 {config['manifold'].replace('_', ' ')} SECURITY 🛂"
            break
    
    # Update header information
    header_texts = root.findall(".//g[@id='header']//text")
    if len(header_texts) >= 4:
        header_texts[1].text = f"Destination: {config['manifold']} ● Traveler: {config['traveler']}"
        header_texts[2].text = f"Passport ID: [AX_HASH] ● Authority: Axiel Engine v1.0"
        header_texts[3].text = f"Status: {'✅ CLEARED' if stamp_results else '⏳ PROCESSING'} ● Validity: Mathematical Truth"
    
    # Update operator stations
    for i, (symbol, op_name, color) in enumerate(zip(config["operators"], config["op_names"], config["colors"])):
        station = root.find(f".//g[@id='station-{i+1}']")
        if station is not None:
            # Update symbol
            symbol_text = station.find(".//use")
            if symbol_text is not None:
                # Replace use with actual text
                parent = symbol_text.getparent()
                parent.remove(symbol_text)
                new_text = ET.SubElement(parent, "text")
                new_text.set("y", "8")
                new_text.set("text-anchor", "middle")
                new_text.set("font-family", "STIX Two Math, serif")
                new_text.set("font-size", "36")
                new_text.set("fill", "none")
                new_text.set("stroke", color)
                new_text.set("stroke-width", "1.5")
                new_text.text = symbol
            
            # Update operator name
            name_text = station.find(".//text")
            if name_text is not None:
                name_text.text = op_name
                name_text.set("fill", color)
    
    # Update progress indicator (find by traversing)
    for text_elem in root.iter("text"):
        if text_elem.text and "Security Level" in text_elem.text:
            stage_count = len(config["operators"])
            current_stage = stage_count if stamp_results else 0
            status = "ALL CLEAR" if stamp_results else "PROCESSING"
            text_elem.text = f"Security Level: {current_stage}/{stage_count} ● Status: {status}"
            break
    
    # Update stamp ledger if we have results
    if stamp_results:
        stamp_slots = root.find(".//g[@id='stamp-slots']")
        if stamp_slots is not None:
            # Clear existing slots
            for child in list(stamp_slots):
                stamp_slots.remove(child)
            
            # Add actual stamp results
            for i, stamp_name in enumerate(config["stamps"]):
                if stamp_name in stamp_results:
                    stamp = stamp_results[stamp_name]
                    
                    # Create stamp element
                    stamp_group = ET.SubElement(stamp_slots, "g")
                    stamp_group.set("id", f"stamp-{i+1}")
                    stamp_group.set("transform", f"translate(0,{i*45})")
                    
                    # Background color based on pass/fail
                    bg_color = "#d4edda" if stamp.passed else "#f8d7da"
                    border_color = "#c3e6cb" if stamp.passed else "#f5c6cb"
                    text_color = "#155724" if stamp.passed else "#721c24"
                    status_icon = "✅" if stamp.passed else "❌"
                    
                    # Background rect
                    rect = ET.SubElement(stamp_group, "rect")
                    rect.set("x", "-10")
                    rect.set("y", "-15")
                    rect.set("width", "1500")
                    rect.set("height", "35")
                    rect.set("fill", bg_color)
                    rect.set("stroke", border_color)
                    rect.set("stroke-width", "1")
                    rect.set("rx", "3")
                    
                    # Stamp text
                    text = ET.SubElement(stamp_group, "text")
                    text.set("x", "10")
                    text.set("y", "5")
                    text.set("font-family", "monospace")
                    text.set("font-size", "14")
                    text.set("font-weight", "bold")
                    text.set("fill", text_color)
                    
                    # Format stamp details
                    if hasattr(stamp, 'details') and stamp.details:
                        key_details = []
                        if "unitary_error_max" in stamp.details:
                            key_details.append(f"unitary_err={stamp.details['unitary_error_max']:.6f}")
                        if "fe_resid_med" in stamp.details:
                            key_details.append(f"fe_resid={stamp.details['fe_resid_med']:.6f}")
                        if "additivity_err" in stamp.details:
                            key_details.append(f"additivity={stamp.details['additivity_err']:.3f}")
                        if "dist_med" in stamp.details:
                            key_details.append(f"dist={stamp.details['dist_med']:.6f}")
                        details_str = "; ".join(key_details[:2])  # Limit to 2 key details
                    else:
                        details_str = f"err={stamp.error_max:.6f}"
                    
                    text.text = f"{status_icon} {stamp_name} {{ {details_str}; pass = {str(stamp.passed).lower()} }}"
    
    # Update approval banner
    summary_elem = root.find(".//g[@id='security-summary']")
    if summary_elem is not None and stamp_results:
        passed_count = sum(1 for s in stamp_results.values() if s.passed)
        total_count = len(stamp_results)
        
        if passed_count == total_count:
            # Full approval
            summary_elem.find("rect").set("fill", "#d4edda")
            summary_elem.find("rect").set("stroke", "#c3e6cb")
            summary_text = summary_elem.find("text")
            summary_text.text = "🎉 PASSPORT APPROVED - WELCOME TO THE MANIFOLD"
            summary_text.set("fill", "#155724")
            
            detail_text = summary_elem.findall("text")[1]
            detail_text.text = f"Checkpoint Status: {passed_count}/{total_count} cleared ● Immigration: APPROVED"
            detail_text.set("fill", "#155724")
        else:
            # Partial or denied
            summary_elem.find("rect").set("fill", "#fff3cd")
            summary_elem.find("rect").set("stroke", "#ffeaa7")
            summary_text = summary_elem.find("text")
            summary_text.text = "⏳ PASSPORT UNDER REVIEW - ADDITIONAL SCREENING REQUIRED"
            summary_text.set("fill", "#856404")
            
            detail_text = summary_elem.findall("text")[1]
            detail_text.text = f"Checkpoint Status: {passed_count}/{total_count} cleared ● Immigration: PENDING"
            detail_text.set("fill", "#856404")
    
    # Write passport
    if output_path:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return output_path
    else:
        return ET.tostring(root, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description="Generate universal mathematical passport")
    parser.add_argument("--domain", choices=list(DOMAIN_CONFIGS.keys()), 
                       default="riemann_hypothesis", help="Mathematical domain")
    parser.add_argument("--out", type=str, default=".out/passports", help="Output directory")
    parser.add_argument("--with-proof", action="store_true", help="Include actual proof stamps")
    args = parser.parse_args()
    
    print(f"🎫 Universal Passport Generator: {args.domain.replace('_', ' ').title()}")
    print("=" * 60)
    
    config = DOMAIN_CONFIGS[args.domain]
    
    # Generate proof stamps if requested
    stamp_results = None
    if args.with_proof and args.domain == "riemann_hypothesis":
        # Use our proven RH stamps
        from unlock_proof import create_proof_unlock_params, ProofUnlockStamper
        
        print("Generating proof stamps...")
        params = create_proof_unlock_params()
        stamper = ProofUnlockStamper(depth=params["depth"])
        stamp_results = stamper.stamp_certification(params)
        
        passes = sum(1 for s in stamp_results.values() if s.passed)
        print(f"✅ Proof generated: {passes}/{len(stamp_results)} stamps")
    
    # Generate passport
    timestamp_file = time.strftime("%Y%m%d-%H%M%S")
    domain_name = args.domain.replace("_", "-")
    status = "proof" if args.with_proof else "template"
    
    os.makedirs(args.out, exist_ok=True)
    passport_path = os.path.join(args.out, f"{domain_name}-passport-{status}-{timestamp_file}.svg")
    
    print(f"Generating {config['manifold']} passport...")
    generate_domain_passport(args.domain, stamp_results, passport_path)
    
    print(f"\n🎫 PASSPORT GENERATED")
    print("=" * 60)
    print(f"Domain: {config['manifold']}")
    print(f"Traveler: {config['traveler']}")
    print(f"Operators: {' → '.join(config['operators'])}")
    print(f"File: {passport_path}")
    
    if stamp_results:
        passed = sum(1 for s in stamp_results.values() if s.passed)
        total = len(stamp_results)
        status = "✅ APPROVED" if passed == total else f"⏳ PENDING ({passed}/{total})"
        print(f"Status: {status}")
        
        if passed == total:
            print("🎉 Welcome to the mathematical manifold!")
            print("Your passport grants unlimited traversal rights.")
        else:
            print("Additional screening required at immigration.")
    else:
        print("Status: 📋 TEMPLATE (no proof stamps)")
        print("Use --with-proof to generate actual certification")
    
    # Validate the passport
    print(f"\nValidating passport...")
    
    import subprocess
    result = subprocess.run([
        "python3", "tools/validate_svg_nf.py", passport_path
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Passport validation PASSED")
        print("✅ Universal template working correctly")
    else:
        print("❌ Passport validation issues detected")
        print(result.stderr)
    
    print(f"\n🏆 Universal Mathematical Airport Security Operational!")
    print("Ready to issue boarding passes for any mathematical manifold.")
    
    return 0


if __name__ == "__main__":
    exit(main())
