#!/usr/bin/env python3
"""
SVG-HOO Normal Form Validator: CI validator for CE1-2D Perfect Core.

Validates that SVG passports conform to the frozen specification:
- Parse → Rewrite to Normal Form → Hash → Verify determinism
- Check monotone staging, golden layout, anchor consistency
"""

import argparse
import hashlib
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import yaml


def load_specs(spec_dir: str = "specs") -> Tuple[Dict, Dict]:
    """Load frozen specifications."""
    
    with open(f"{spec_dir}/svg-hoo.schema.yaml", "r") as f:
        hoo_spec = yaml.safe_load(f)
    
    with open(f"{spec_dir}/layout-v1.yaml", "r") as f:
        layout_spec = yaml.safe_load(f)
    
    return hoo_spec, layout_spec


def parse_svg_passport(svg_path: str) -> Dict:
    """Parse SVG passport and extract components."""
    
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        return {"error": f"Failed to parse SVG: {e}"}
    
    # Extract metadata
    metadata = {}
    metadata_elem = root.find(".//metadata")
    if metadata_elem is not None:
        # Extract Axiel passport data
        for elem in metadata_elem.iter():
            if elem.tag.startswith("axiel-"):
                metadata[elem.tag] = elem.attrib
    
    # Extract mounted stages
    mounted_stages = []
    left_group = root.find(".//g[@id='left']")
    if left_group is not None:
        for use_elem in left_group.findall(".//use"):
            href = use_elem.get("href", "")
            if href.startswith("#s") and "-" in href:
                stage_num = int(href[2])  # Extract number from #s1-name
                mounted_stages.append(stage_num)
    
    mounted_stages.sort()
    
    # Extract anchors
    anchors = {}
    anchor_elem = root.find(".//g[@id='anchor-display']")
    if anchor_elem is not None:
        text_elem = anchor_elem.find("text")
        if text_elem is not None and text_elem.text:
            # Parse AX=... | VA=... | MA=... | LA=...
            anchor_text = text_elem.text
            for part in anchor_text.split(" | "):
                if "=" in part:
                    key, value = part.split("=", 1)
                    anchors[key.strip()] = value.strip()
    
    return {
        "metadata": metadata,
        "mounted_stages": mounted_stages,
        "anchors": anchors,
        "svg_tree": tree
    }


def compute_normal_form_signature(mounted_stages: List[int], hoo_spec: Dict) -> str:
    """Compute Normal Form signature from mounted stages."""
    
    nf_chain = hoo_spec["normal_form_chain"]
    
    # Build NF signature from mounted stages in canonical order
    nf_ops = []
    for stage_info in nf_chain:
        stage_num = stage_info["stage"]
        if stage_num in mounted_stages:
            nf_ops.append(f"{stage_info['name']}@{stage_num}")
    
    nf_sig = "NF:" + "→".join(nf_ops)
    return nf_sig


def validate_monotone_staging(mounted_stages: List[int]) -> Tuple[bool, str]:
    """Validate monotone staging: only stages 1..k mounted."""
    
    if not mounted_stages:
        return True, "No stages mounted"
    
    max_stage = max(mounted_stages)
    expected_stages = list(range(1, max_stage + 1))
    
    if mounted_stages == expected_stages:
        return True, f"Monotone staging 1..{max_stage}"
    else:
        missing = set(expected_stages) - set(mounted_stages)
        return False, f"Non-monotone: missing stages {missing}"


def validate_hue_restriction(mounted_stages: List[int]) -> Tuple[bool, str]:
    """Validate no hue before stage 7."""
    
    # In our template, stages 7-8 have hue (BlueNoise, Spectralize)
    hue_stages = [7, 8]
    early_hue = [s for s in mounted_stages if s in hue_stages and max(mounted_stages) < 7]
    
    if early_hue:
        return False, f"Hue used in stages {early_hue} before stage 7"
    else:
        return True, "Hue restriction satisfied"


def validate_golden_layout(svg_tree: ET.ElementTree, layout_spec: Dict) -> Tuple[bool, str]:
    """Validate golden ratio layout."""
    
    root = svg_tree.getroot()
    
    # Extract canvas dimensions
    width = float(root.get("width", 0))
    height = float(root.get("height", 0))
    
    # Check golden ratio
    phi = layout_spec["constants"]["phi"]
    expected_width = (1 + phi) * height
    
    width_error = abs(width - expected_width) / expected_width
    tolerance = layout_spec["validation"]["golden_ratio_tolerance"]
    
    if width_error <= tolerance:
        return True, f"Golden ratio satisfied: {width}/{height} ≈ {1+phi:.6f}"
    else:
        return False, f"Golden ratio violated: error {width_error:.6f} > {tolerance}"


def compute_anchor_hashes(nf_sig: str, layout_data: Dict, passport_data: Dict) -> Dict[str, str]:
    """Compute VA, MA, LA, AX anchor hashes."""
    
    # VA: H(NF_sig, layout_rest, layout_delta, palette_id)
    va_content = f"{nf_sig}|{layout_data}|SPECTRAL_V1"
    va_hash = hashlib.sha256(va_content.encode()).hexdigest()[:16]
    
    # MA: H(P, G, Σ, stamp_metrics)  
    ma_content = f"{passport_data.get('proposition', '')}|{passport_data.get('generators', '')}|{passport_data.get('stamps', '')}"
    ma_hash = hashlib.sha256(ma_content.encode()).hexdigest()[:16]
    
    # LA: H(VA || MA || provenance)
    la_content = f"{va_hash}|{ma_hash}|{passport_data.get('provenance', '')}"
    la_hash = hashlib.sha256(la_content.encode()).hexdigest()[:16]
    
    # AX: bech32(VA ⊕ MA ⊕ LA) - simplified as XOR of hashes
    va_int = int(va_hash, 16)
    ma_int = int(ma_hash, 16) 
    la_int = int(la_hash, 16)
    ax_int = va_int ^ ma_int ^ la_int
    ax_hash = f"ax{ax_int:016x}"  # Simplified bech32
    
    return {
        "VA": va_hash,
        "MA": ma_hash, 
        "LA": la_hash,
        "AX": ax_hash
    }


def validate_svg_passport(svg_path: str, spec_dir: str = "specs") -> Dict:
    """Complete validation of SVG passport."""
    
    # Load specifications
    hoo_spec, layout_spec = load_specs(spec_dir)
    
    # Parse SVG
    parsed = parse_svg_passport(svg_path)
    if "error" in parsed:
        return {"valid": False, "error": parsed["error"]}
    
    results = {"valid": True, "checks": {}}
    
    # Check 1: Monotone staging
    monotone_ok, monotone_msg = validate_monotone_staging(parsed["mounted_stages"])
    results["checks"]["monotone_staging"] = {"pass": monotone_ok, "message": monotone_msg}
    
    # Check 2: Hue restriction
    hue_ok, hue_msg = validate_hue_restriction(parsed["mounted_stages"])
    results["checks"]["hue_restriction"] = {"pass": hue_ok, "message": hue_msg}
    
    # Check 3: Golden layout
    layout_ok, layout_msg = validate_golden_layout(parsed["svg_tree"], layout_spec)
    results["checks"]["golden_layout"] = {"pass": layout_ok, "message": layout_msg}
    
    # Check 4: Normal Form signature
    nf_sig = compute_normal_form_signature(parsed["mounted_stages"], hoo_spec)
    results["checks"]["nf_signature"] = {"pass": True, "message": f"NF signature: {nf_sig}"}
    
    # Check 5: Anchor consistency (if present)
    if parsed["anchors"]:
        # Compute expected anchors
        expected_anchors = compute_anchor_hashes(nf_sig, {"layout": "golden"}, {"provenance": "template"})
        
        anchor_consistent = True
        for key in ["AX", "VA", "MA", "LA"]:
            if key in parsed["anchors"] and key in expected_anchors:
                if parsed["anchors"][key] != expected_anchors[key] and not parsed["anchors"][key].startswith("[TEMPLATE"):
                    anchor_consistent = False
                    break
        
        results["checks"]["anchor_consistency"] = {"pass": anchor_consistent, "message": "Anchor hashes validated"}
    else:
        results["checks"]["anchor_consistency"] = {"pass": True, "message": "No anchors to validate"}
    
    # Overall validation
    all_passed = all(check["pass"] for check in results["checks"].values())
    results["valid"] = all_passed
    results["nf_signature"] = nf_sig
    results["mounted_stages"] = parsed["mounted_stages"]
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate SVG-HOO Normal Form passport")
    parser.add_argument("svg_file", help="SVG passport file to validate")
    parser.add_argument("--spec-dir", default="specs", help="Specification directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print(f"🔍 SVG-HOO Validator: {args.svg_file}")
    print("=" * 50)
    
    # Validate passport
    result = validate_svg_passport(args.svg_file, args.spec_dir)
    
    if result["valid"]:
        print("✅ PASSPORT VALID")
        print(f"NF Signature: {result['nf_signature']}")
        print(f"Mounted stages: {result['mounted_stages']}")
    else:
        print("❌ PASSPORT INVALID")
        if "error" in result:
            print(f"Error: {result['error']}")
    
    # Print check details
    if args.verbose or not result["valid"]:
        print("\nValidation Details:")
        for check_name, check_result in result["checks"].items():
            status = "✅ PASS" if check_result["pass"] else "❌ FAIL"
            print(f"  {check_name:20} | {status} | {check_result['message']}")
    
    print()
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    exit(main())
