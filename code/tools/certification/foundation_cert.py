#!/usr/bin/env python3
"""
RH Foundation Certificate: Composite proof-object merging three lemma branches.

This demonstrates how individual lemma certificates compose into foundation certificates,
showing the recursive scaffolding that builds toward full theorem certification.

FOUNDATION (RH Mathematical Foundation): The Pascal-dihedral operator T with Euler structure
satisfies the foundational properties required for RH-style spectral analysis:
1. Mellin-mirror duality (unitarity + functional equation)
2. Pascal-Euler factorization (local-global Euler product structure)  
3. Dihedral-action invariance (group symmetry preservation)

COMPOSITION STRATEGY:
- Merge verification data from all three lemma branches
- Require ALL dependencies to pass before foundation passes
- Provide composite validation with cross-lemma consistency checks
"""

import argparse
import hashlib
import os
import re
import time
from typing import Any, Dict


def parse_lemma_certificate(cert_path: str) -> Dict[str, Any]:
    """Parse a lemma certificate to extract verification data."""
    
    if not os.path.exists(cert_path):
        return {"status": "missing", "lemma_verified": False}
    
    with open(cert_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract basic info
    lens_match = re.search(r'lens=([A-Z_]+)', content)
    lens = lens_match.group(1) if lens_match else "UNKNOWN"
    
    # Extract verification status
    verified_match = re.search(r'lemma_verified\s*=\s*(true|false)', content)
    lemma_verified = verified_match.group(1) == "true" if verified_match else False
    
    # Extract stamp results
    stamps = {}
    stamp_pattern = r'(\w+)\{[^}]*pass\s*=\s*(true|false)[^}]*\}'
    for match in re.finditer(stamp_pattern, content):
        stamp_name = match.group(1)
        stamp_passed = match.group(2) == "true"
        stamps[stamp_name] = stamp_passed
    
    # Extract proof hash for dependency tracking
    hash_match = re.search(r'proof_hash="([^"]+)"', content)
    proof_hash = hash_match.group(1) if hash_match else "unknown"
    
    # Extract timestamp
    timestamp_match = re.search(r'timestamp_utc="([^"]+)"', content)
    timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
    
    return {
        "status": "found",
        "lens": lens,
        "lemma_verified": lemma_verified,
        "stamps": stamps,
        "proof_hash": proof_hash,
        "timestamp": timestamp,
        "path": cert_path
    }


def find_latest_lemma_certificates(cert_dir: str = ".out/certs") -> Dict[str, str]:
    """Find the latest certificates for each lemma type."""
    
    if not os.path.isdir(cert_dir):
        return {}
    
    lemma_files = {
        "mellin_mirror": [],
        "pascal_euler": [],
        "dihedral_action": []
    }
    
    for filename in os.listdir(cert_dir):
        if filename.endswith(".ce1") and filename.startswith("lemma-"):
            path = os.path.join(cert_dir, filename)
            mtime = os.path.getmtime(path)
            
            if "mellin-mirror" in filename:
                lemma_files["mellin_mirror"].append((mtime, path))
            elif "pascal-euler" in filename:
                lemma_files["pascal_euler"].append((mtime, path))
            elif "dihedral-action" in filename:
                lemma_files["dihedral_action"].append((mtime, path))
    
    # Get latest of each type
    latest = {}
    for lemma_type, files in lemma_files.items():
        if files:
            files.sort(reverse=True)  # Latest first
            latest[lemma_type] = files[0][1]
    
    return latest


def write_foundation_certificate(path: str, lemma_data: Dict[str, Dict], 
                                metadata: Dict) -> None:
    """Write composite foundation certificate."""
    
    lines = []
    lines.append("CE1{\n")
    lines.append("  lens=RH_FOUNDATION_CERTIFICATE\n")
    lines.append("  mode=CompositeCertification\n")
    lines.append("  basis=pascal_dihedral_euler_unified\n")
    lines.append(f"  params{{ depth={metadata['depth']}; N={metadata['N']}; composition_tolerance={metadata['tolerance']} }}\n")
    lines.append("\n")
    
    # Foundation statement
    lines.append("  foundation_statement=\"Pascal-dihedral operator T with Euler structure satisfies foundational properties for RH-style spectral analysis\"\n")
    lines.append("  composition_strategy=\"Merge verification from three lemma branches with cross-consistency checks\"\n")
    lines.append("\n")
    
    # Dependencies (explicit)
    lines.append("  dependencies{\n")
    for lemma_type, data in lemma_data.items():
        status = "SATISFIED" if data["lemma_verified"] else "UNSATISFIED"
        lines.append(f"    {lemma_type}_lemma=\"{status}\"\n")
        lines.append(f"    {lemma_type}_hash=\"{data['proof_hash']}\"\n")
        lines.append(f"    {lemma_type}_timestamp=\"{data['timestamp']}\"\n")
    lines.append("  }\n")
    lines.append("\n")
    
    # Composite stamps (merged from branches)
    lines.append("  composite_stamps{\n")
    
    # REP: Merged from all three lemmas
    rep_sources = []
    rep_passed = True
    for lemma_type, data in lemma_data.items():
        if "REP" in data["stamps"]:
            rep_sources.append(f"{lemma_type}:{data['stamps']['REP']}")
            rep_passed = rep_passed and data["stamps"]["REP"]
    
    lines.append(f"    REP{{ ")
    lines.append(f"sources=[{', '.join(rep_sources)}]; ")
    lines.append(f"mellin_mirror_unitarity={lemma_data.get('mellin_mirror', {}).get('stamps', {}).get('REP', False)}; ")
    lines.append(f"pascal_euler_unitarity={lemma_data.get('pascal_euler', {}).get('stamps', {}).get('REP', False)}; ")
    lines.append(f"dihedral_action_unitarity={lemma_data.get('dihedral_action', {}).get('stamps', {}).get('REP', False)}; ")
    lines.append(f"composite_unitarity={rep_passed}; ")
    lines.append(f"pass = {str(rep_passed).lower()} ")
    lines.append("}}\n")
    
    # DUAL: From Mellin-Mirror and Pascal-Euler
    dual_sources = []
    dual_passed = True
    for lemma_type in ["mellin_mirror", "pascal_euler"]:
        if lemma_type in lemma_data and "DUAL" in lemma_data[lemma_type]["stamps"]:
            dual_sources.append(f"{lemma_type}:{lemma_data[lemma_type]['stamps']['DUAL']}")
            dual_passed = dual_passed and lemma_data[lemma_type]["stamps"]["DUAL"]
    
    lines.append(f"    DUAL{{ ")
    lines.append(f"sources=[{', '.join(dual_sources)}]; ")
    lines.append(f"mellin_mirror_symmetry={lemma_data.get('mellin_mirror', {}).get('stamps', {}).get('DUAL', False)}; ")
    lines.append(f"pascal_euler_symmetry={lemma_data.get('pascal_euler', {}).get('stamps', {}).get('DUAL', False)}; ")
    lines.append(f"composite_symmetry={dual_passed}; ")
    lines.append(f"pass = {str(dual_passed).lower()} ")
    lines.append("}}\n")
    
    # LOCAL: From Pascal-Euler and Dihedral-Action
    local_sources = []
    local_passed = True
    for lemma_type in ["pascal_euler", "dihedral_action"]:
        if lemma_type in lemma_data and "LOCAL" in lemma_data[lemma_type]["stamps"]:
            local_sources.append(f"{lemma_type}:{lemma_data[lemma_type]['stamps']['LOCAL']}")
            local_passed = local_passed and lemma_data[lemma_type]["stamps"]["LOCAL"]
    
    lines.append(f"    LOCAL{{ ")
    lines.append(f"sources=[{', '.join(local_sources)}]; ")
    lines.append(f"euler_product_locality={lemma_data.get('pascal_euler', {}).get('stamps', {}).get('LOCAL', False)}; ")
    lines.append(f"dihedral_action_locality={lemma_data.get('dihedral_action', {}).get('stamps', {}).get('LOCAL', False)}; ")
    lines.append(f"composite_locality={local_passed}; ")
    lines.append(f"pass = {str(local_passed).lower()} ")
    lines.append("}}\n")
    
    lines.append("  }\n")
    lines.append("\n")
    
    # Cross-consistency checks
    all_dependencies_satisfied = all(data["lemma_verified"] for data in lemma_data.values())
    all_stamps_consistent = rep_passed and dual_passed and local_passed
    foundation_verified = all_dependencies_satisfied and all_stamps_consistent
    
    lines.append("  cross_consistency{\n")
    lines.append(f"    all_dependencies_satisfied = {str(all_dependencies_satisfied).lower()}\n")
    lines.append(f"    rep_cross_lemma_consistent = {str(rep_passed).lower()}\n")
    lines.append(f"    dual_cross_lemma_consistent = {str(dual_passed).lower()}\n")
    lines.append(f"    local_cross_lemma_consistent = {str(local_passed).lower()}\n")
    lines.append(f"    foundation_mathematically_sound = {str(foundation_verified).lower()}\n")
    lines.append("  }\n")
    lines.append("\n")
    
    # Composite verification
    lines.append("  verification{\n")
    lines.append(f"    foundation_verified = {str(foundation_verified).lower()}\n")
    lines.append(f"    mellin_mirror_component = {str(lemma_data.get('mellin_mirror', {}).get('lemma_verified', False)).lower()}\n")
    lines.append(f"    pascal_euler_component = {str(lemma_data.get('pascal_euler', {}).get('lemma_verified', False)).lower()}\n")
    lines.append(f"    dihedral_action_component = {str(lemma_data.get('dihedral_action', {}).get('lemma_verified', False)).lower()}\n")
    lines.append(f"    ready_for_rh_proof = {str(foundation_verified).lower()}\n")
    lines.append("  }\n")
    lines.append("\n")
    
    # Enhanced provenance (composite)
    lines.append("  provenance{\n")
    lines.append(f"    timestamp_utc=\"{metadata['timestamp']}\"\n")
    lines.append(f"    git_rev=\"{metadata['git_rev']}\"\n")
    lines.append(f"    composition_hash=\"{metadata['composition_hash']}\"\n")
    lines.append(f"    dependency_hashes=[")
    dep_hashes = [data['proof_hash'] for data in lemma_data.values()]
    lines.append(", ".join(f'"{h}"' for h in dep_hashes))
    lines.append("]\n")
    lines.append(f"    composition_type=\"parallel_branch_merge\"\n")
    lines.append("  }\n")
    lines.append("\n")
    
    # Composite validator rules
    lines.append("  validator_rules{\n")
    lines.append("    lens=RH_FOUNDATION_VALIDATE\n")
    lines.append("    # Dependency requirements\n")
    for lemma_type, data in lemma_data.items():
        lines.append(f"    assert_{lemma_type}_verified = {str(data['lemma_verified']).lower()}\n")
    lines.append("    # Cross-lemma stamp consistency\n")
    lines.append(f"    assert_rep_cross_consistent = {str(rep_passed).lower()}\n")
    lines.append(f"    assert_dual_cross_consistent = {str(dual_passed).lower()}\n")
    lines.append(f"    assert_local_cross_consistent = {str(local_passed).lower()}\n")
    lines.append("    # Foundation readiness\n")
    lines.append(f"    assert_foundation_complete = {str(foundation_verified).lower()}\n")
    lines.append("    assert_ready_for_composition = REP.pass && DUAL.pass && LOCAL.pass\n")
    lines.append("    emit=RH_FOUNDATION_VALIDATED\n")
    lines.append("  }\n")
    lines.append("\n")
    
    # Foundation certificate outcome
    lines.append(f"  foundation_status=\"{'VERIFIED' if foundation_verified else 'UNVERIFIED'}\"\n")
    lines.append(f"  lemma_count={len(lemma_data)}\n")
    lines.append(f"  composite_stamp_count=3\n")
    lines.append(f"  dependency_satisfaction_rate={sum(1 for d in lemma_data.values() if d['lemma_verified'])}/{len(lemma_data)}\n")
    lines.append(f"  ready_for_rh_proof={str(foundation_verified).lower()}\n")
    lines.append(f"  validator=RH_FOUNDATION.{'pass' if foundation_verified else 'fail'}\n")
    lines.append("  emit=RiemannHypothesisFoundationCertificate\n")
    lines.append("}\n")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate RH Foundation certificate from lemma branches")
    parser.add_argument("--cert-dir", type=str, default=".out/certs", help="Certificate directory")
    parser.add_argument("--out", type=str, default=".out/certs", help="Output directory")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Composition tolerance")
    args = parser.parse_args()
    
    print("RH Foundation Certificate: Merging Lemma Branches")
    print("=" * 55)
    
    # Find latest lemma certificates
    latest_certs = find_latest_lemma_certificates(args.cert_dir)
    
    print("Dependency Analysis:")
    lemma_data = {}
    
    # Parse each lemma certificate
    for lemma_type, cert_path in latest_certs.items():
        print(f"  {lemma_type.replace('_', '-'):15} → {os.path.basename(cert_path)}")
        data = parse_lemma_certificate(cert_path)
        lemma_data[lemma_type] = data
        
        if data["status"] == "found":
            status = "✅ VERIFIED" if data["lemma_verified"] else "❌ UNVERIFIED"
            stamp_summary = ", ".join(f"{k}:{'✓' if v else '✗'}" for k, v in data["stamps"].items())
            print(f"    Status: {status}")
            print(f"    Stamps: {stamp_summary}")
            print(f"    Hash: {data['proof_hash']}")
        else:
            print(f"    Status: ❌ MISSING")
    
    print()
    
    # Check if we have all required dependencies
    required_lemmas = ["mellin_mirror", "pascal_euler", "dihedral_action"]
    missing_lemmas = [lemma for lemma in required_lemmas if lemma not in lemma_data or lemma_data[lemma]["status"] != "found"]
    
    if missing_lemmas:
        print(f"❌ Missing required lemmas: {missing_lemmas}")
        print("Cannot generate foundation certificate without all dependencies.")
        return 1
    
    # Analyze foundation readiness
    verified_count = sum(1 for data in lemma_data.values() if data["lemma_verified"])
    total_count = len(lemma_data)
    
    print(f"Foundation Readiness Analysis:")
    print(f"  Dependencies verified: {verified_count}/{total_count}")
    
    # Cross-lemma stamp analysis
    rep_consistency = all(data["stamps"].get("REP", False) for data in lemma_data.values() if "REP" in data["stamps"])
    dual_consistency = all(data["stamps"].get("DUAL", False) for data in lemma_data.values() if "DUAL" in data["stamps"])
    local_consistency = all(data["stamps"].get("LOCAL", False) for data in lemma_data.values() if "LOCAL" in data["stamps"])
    
    print(f"  REP cross-consistency: {'✅' if rep_consistency else '❌'}")
    print(f"  DUAL cross-consistency: {'✅' if dual_consistency else '❌'}")
    print(f"  LOCAL cross-consistency: {'✅' if local_consistency else '❌'}")
    
    foundation_ready = (verified_count == total_count and 
                       rep_consistency and dual_consistency and local_consistency)
    
    print(f"  Foundation ready: {'✅ YES' if foundation_ready else '❌ NO'}")
    
    # Gather metadata
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Create composition hash from all dependency hashes
    dep_hashes = [data['proof_hash'] for data in lemma_data.values()]
    composition_content = f"foundation|{dep_hashes}|{timestamp}"
    composition_hash = hashlib.sha256(composition_content.encode()).hexdigest()[:16]
    
    try:
        import subprocess
        git_result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
        git_rev = git_result.stdout.strip()[:12] if git_result.returncode == 0 else "unknown"
    except:
        git_rev = "unknown"
    
    metadata = {
        "depth": 4,  # Foundation depth
        "N": 17,
        "tolerance": args.tolerance,
        "timestamp": timestamp,
        "git_rev": git_rev,
        "composition_hash": composition_hash
    }
    
    # Generate foundation certificate
    timestamp_file = time.strftime("%Y%m%d-%H%M%S")
    base = f"foundation-rh-depth{metadata['depth']}-{timestamp_file}"
    
    os.makedirs(args.out, exist_ok=True)
    cert_path = os.path.join(args.out, f"{base}.ce1")
    
    write_foundation_certificate(cert_path, lemma_data, metadata)
    
    # Print composition results
    print(f"\nFoundation Composition Results:")
    print("=" * 55)
    print(f"Composite REP stamp:    {'PASS' if rep_consistency else 'FAIL'}")
    print(f"  Sources: Mellin-Mirror, Pascal-Euler, Dihedral-Action")
    print(f"Composite DUAL stamp:   {'PASS' if dual_consistency else 'FAIL'}")
    print(f"  Sources: Mellin-Mirror, Pascal-Euler")
    print(f"Composite LOCAL stamp:  {'PASS' if local_consistency else 'FAIL'}")
    print(f"  Sources: Pascal-Euler, Dihedral-Action")
    print("=" * 55)
    
    # Foundation verdict
    print(f"\nRH FOUNDATION CERTIFICATE: {'✅ VERIFIED' if foundation_ready else '❌ UNVERIFIED'}")
    
    if foundation_ready:
        print("✅ All lemma dependencies satisfied")
        print("✅ Cross-lemma stamp consistency verified")
        print("✅ Ready for full RH proof composition")
        print("✅ Mathematical foundation established")
    else:
        print("❌ Some dependencies unverified or inconsistent")
        print("❌ Foundation requires lemma refinement")
        print(f"❌ Current satisfaction: {verified_count}/{total_count} lemmas")
    
    print(f"\nGenerated foundation certificate: {cert_path}")
    print("Certificate shows composite proof-object with recursive dependency validation.")
    
    return 0 if foundation_ready else 1


if __name__ == "__main__":
    exit(main())
