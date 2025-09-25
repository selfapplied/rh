#!/usr/bin/env python3
"""
Stamped Certification Generator - Transform romantic RH assertions into respectable certifications.

This script reads existing certification artifacts and applies the 8 surgical stamps
to produce a properly verified RH_CERT with numerical evidence.
"""

import argparse
import os
import time
import re
from typing import Dict, Any, List
from core.validation import CertificationStamper
from core.certification import write_toml


def find_latest_cert_toml(cert_dir: str = ".out/certs") -> str:
    """Find the latest certification TOML file."""
    if not os.path.isdir(cert_dir):
        raise FileNotFoundError(f"Certificate directory not found: {cert_dir}")
    
    candidates = []
    for name in os.listdir(cert_dir):
        if name.endswith(".toml") and name.startswith("cert-"):
            path = os.path.join(cert_dir, name)
            mtime = os.path.getmtime(path)
            candidates.append((mtime, path))
    
    if not candidates:
        raise FileNotFoundError(f"No certification TOML files found in {cert_dir}")
    
    candidates.sort(reverse=True)
    return candidates[0][1]


def parse_cert_toml(toml_path: str) -> Dict[str, Any]:
    """Parse certification TOML file."""
    params = {}
    
    with open(toml_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract scalar parameters
    for key in ["depth", "N", "gamma", "window", "step", "d"]:
        match = re.search(rf"^{key}\s*=\s*(.+)$", content, re.MULTILINE)
        if match:
            value = match.group(1).strip()
            try:
                if key in ["depth", "N", "gamma"]:
                    params[key] = int(value)
                else:
                    params[key] = float(value)
            except ValueError:
                print(f"Warning: Could not parse {key} = {value}")
    
    # Extract zeros array
    zeros_match = re.search(r"zeros\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if zeros_match:
        zeros_str = zeros_match.group(1)
        zeros = []
        for num_str in re.findall(r"[\d.]+", zeros_str):
            try:
                zeros.append(float(num_str))
            except ValueError:
                continue
        params["zeros"] = zeros
    
    # Extract summary statistics
    summary = {}
    summary_match = re.search(r"\[summary\](.*?)(?=\[|\Z)", content, re.DOTALL)
    if summary_match:
        summary_content = summary_match.group(1)
        for key in ["online_locked_rate", "offline_locked_rate", "online_total", "offline_total", "online_locked", "offline_locked"]:
            match = re.search(rf"^{key}\s*=\s*(.+)$", summary_content, re.MULTILINE)
            if match:
                value = match.group(1).strip()
                try:
                    if "rate" in key:
                        summary[key] = float(value)
                    else:
                        summary[key] = int(value)
                except ValueError:
                    print(f"Warning: Could not parse summary {key} = {value}")
    
    params["summary"] = summary
    return params


def write_stamped_ce1(path: str, params: Dict[str, Any], stamps: Dict[str, Any]) -> None:
    """Write stamped CE1 certification block."""
    zeros_list = "; ".join(f"{z}" for z in params.get("zeros", []))
    summary = params.get("summary", {})
    
    # Build params string
    param_items = []
    for key in ["depth", "N", "gamma", "d", "window", "step"]:
        if key in params:
            param_items.append(f"{key}={params[key]}")
    params_str = "; ".join(param_items)
    
    lines = []
    lines.append("CE1{\n")
    lines.append("  lens=RH_CERT\n")
    lines.append("  mode=Certification\n")
    lines.append("  basis=metanion:pascal_dihedral\n")
    lines.append(f"  params{{ {params_str} }}\n")
    lines.append("\n")
    
    # Add stamps block
    lines.append("  stamps{\n")
    
    # REP stamp
    if "REP" in stamps:
        rep = stamps["REP"]
        lines.append(f"    REP{{ unitary_error_max={rep['unitary_error_max']:.6f}; unitary_error_med={rep['unitary_error_med']:.6f}; pass = {str(rep['pass']).lower()} }}\n")
    
    # DUAL stamp
    if "DUAL" in stamps:
        dual = stamps["DUAL"]
        lines.append(f"    DUAL{{ fe_resid_med={dual['fe_resid_med']:.6f}; fe_resid_p95={dual['fe_resid_p95']:.6f}; pass = {str(dual['pass']).lower()} }}\n")
    
    # LOCAL stamp
    if "LOCAL" in stamps:
        local = stamps["LOCAL"]
        buckets_str = ",".join(map(str, local.get('buckets', [])))
        lines.append(f"    LOCAL{{ buckets=[{buckets_str}]; additivity_err={local['additivity_err']:.3f}; seed={local.get('seed', 42)}; trials={local.get('trials', 100)}; mean_err={local.get('mean_err', 0.0):.3f}; sd={local.get('sd', 0.0):.3f}; pass = {str(local['pass']).lower()} }}\n")
    
    # LINE_LOCK stamp
    if "LINE_LOCK" in stamps:
        line_lock = stamps["LINE_LOCK"]
        shuffle_kind = line_lock.get('shuffle_kind', 'within-window permute')
        adaptive_med = line_lock.get('adaptive_dist_med_threshold', 0.01)
        adaptive_max = line_lock.get('adaptive_dist_max_threshold', 0.02)
        depth = line_lock.get('depth', 4)
        formula_med = line_lock.get('thresh_formula_med', 'th_med = 0.01*(1+4*max(0,depth-4))')
        base_med = line_lock.get('base_th_med', 0.01)
        base_max = line_lock.get('base_th_max', 0.02)
        eps = line_lock.get('eps', 1e-6)
        
        lines.append(f"    LINE_LOCK{{ ")
        lines.append(f"locus=\"{line_lock['locus']}\"; ")
        lines.append(f"windows_total={line_lock.get('windows_total', line_lock['windows'])}; ")
        lines.append(f"locked_total={line_lock.get('locked_total', line_lock['locked'])}; ")
        lines.append(f"dist_med={line_lock['dist_med']:.6f}; dist_max={line_lock['dist_max']:.6f}; ")
        lines.append(f"thresh_med={adaptive_med:.3f}; thresh_max={adaptive_max:.3f}; ")
        lines.append(f"base_th_med={base_med:.3f}; base_th_max={base_max:.3f}; ")
        lines.append(f"thresh_formula=\"{formula_med}\"; ")
        lines.append(f"depth={depth}; eps={eps:.0e}; ")
        lines.append(f"null_drop={line_lock.get('null_drop', 0.0):.3f}; ")
        lines.append(f"shuffle_kind=\"{shuffle_kind}\"; ")
        lines.append(f"pass = {str(line_lock['pass']).lower()} ")
        lines.append("}}\n")
    
    # LI stamp
    if "LI" in stamps:
        li = stamps["LI"]
        lines.append(f"    LI{{ up_to_N={li['up_to_N']}; min_lambda={li['min_lambda']:.6f}; violations={li['violations']}; pass = {str(li['pass']).lower()} }}\n")
    
    # NB stamp
    if "NB" in stamps:
        nb = stamps["NB"]
        lines.append(f"    NB{{ L2_error={nb['L2_error']:.6f}; basis_size={nb['basis_size']}; pass = {str(nb['pass']).lower()} }}\n")
    
    # LAMBDA stamp
    if "LAMBDA" in stamps:
        lambda_stamp = stamps["LAMBDA"]
        ci = lambda_stamp.get('ci', [0, 0, 0])
        ci_str = f"[{ci[0]:.3f},{ci[1]:.3f},{ci[2]:.3f}]"
        lines.append(f"    LAMBDA{{ lower_bound={lambda_stamp['lower_bound']:.6f}; ci={ci_str}; gamma_eval={lambda_stamp.get('gamma', 1.0)}; window={lambda_stamp.get('window', 0.5)}; method=\"heat-flow\"; pass = {str(lambda_stamp['pass']).lower()} }}\n")
    
    # MDL_MONO stamp
    if "MDL_MONO" in stamps:
        mdl = stamps["MDL_MONO"]
        depth_str = ",".join(map(str, mdl['depth']))
        gains_str = ",".join(f"{g:.3f}" for g in mdl['gains'][:4])  # First 4 gains
        lines.append(f"    MDL_MONO{{ depth=[{depth_str}]; gains=[{gains_str}]; monotone={str(mdl['monotone']).lower()}; pass = {str(mdl['pass']).lower()} }}\n")
    
    lines.append("  }\n")
    lines.append("\n")
    
    # Add provenance hash
    import hashlib
    stream_data = f"{params_str}|{zeros_list}"
    hash_obj = hashlib.sha256(stream_data.encode())
    provenance_hash = hash_obj.hexdigest()[:16]  # First 16 chars
    
    lines.append("  provenance{ ")
    lines.append(f"hash={provenance_hash}; ")
    lines.append("version=\"ce1.rhc.v0.4\" ")
    lines.append("}\n")
    lines.append("\n")
    
    # Add original data
    lines.append(f"  zeros_ref=[{zeros_list}]  # benchmark only\n")
    lines.append(f"  artifact={path}\n")
    lines.append("  emit=RiemannHypothesisCertification\n")
    lines.append("\n")
    
    # Validator outcome with adaptive threshold rules
    all_passed = all(stamp["pass"] for stamp in stamps.values())
    validator_outcome = "RH_CERT_VALIDATE.pass" if all_passed else "RH_CERT_VALIDATE.fail"
    
    # Add adaptive threshold validation rules
    lines.append(f"  validator={validator_outcome}\n")
    lines.append("\n")
    
    # CE1 validator rules for adaptive thresholds
    lines.append("  validator_rules{\n")
    lines.append("    lens=RH_CERT_VALIDATE_THRESH\n")
    if "LINE_LOCK" in stamps:
        ll = stamps["LINE_LOCK"]
        lines.append(f"    assert_thresh_med_ge_base = {ll.get('adaptive_dist_med_threshold', 0.01):.3f} >= {ll.get('base_th_med', 0.01):.3f}\n")
        lines.append(f"    assert_thresh_max_ge_base = {ll.get('adaptive_dist_max_threshold', 0.02):.3f} >= {ll.get('base_th_max', 0.02):.3f}\n")
        lines.append(f"    assert_thresh_med_capped = {ll.get('adaptive_dist_med_threshold', 0.01):.3f} <= 0.100\n")
        lines.append(f"    assert_thresh_max_capped = {ll.get('adaptive_dist_max_threshold', 0.02):.3f} <= 0.060\n")
        lines.append(f"    assert_null_rule_enforced = {str(ll.get('null_rule_met', False)).lower()}\n")
        lines.append(f"    assert_windows_sufficient = {str(ll.get('windows_sufficient', False)).lower()}\n")
    lines.append("    emit=RHCERT_ValidateAdaptive\n")
    lines.append("  }\n")
    lines.append("}\n")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def write_stamped_toml(path: str, original_params: Dict[str, Any], stamps: Dict[str, Any]) -> None:
    """Write stamped TOML with original data plus stamp results."""
    
    # Start with original parameters
    toml_data = {
        "depth": original_params.get("depth", 4),
        "N": original_params.get("N", 17),
        "gamma": original_params.get("gamma", 3),
        "window": original_params.get("window", 0.5),
        "step": original_params.get("step", 0.1),
        "d": original_params.get("d", 0.05),
        "zeros": original_params.get("zeros", []),
    }
    
    # Add stamp results
    stamp_summary = {}
    for name, stamp in stamps.items():
        stamp_summary[f"{name.lower()}_passed"] = stamp["pass"]
        stamp_summary[f"{name.lower()}_error_max"] = stamp.get("unitary_error_max", stamp.get("fe_resid_med", stamp.get("dist_max", 0.0)))
        stamp_summary[f"{name.lower()}_error_med"] = stamp.get("unitary_error_med", stamp.get("fe_resid_p95", stamp.get("dist_med", 0.0)))
    
    toml_data["stamp_summary"] = stamp_summary
    
    # Add original summary if available
    if "summary" in original_params:
        toml_data["original_summary"] = original_params["summary"]
    
    # Add certification status
    all_implemented = all(not (stamp.get("unitary_error_max", 0) == float('inf') and 
                              stamp.get("fe_resid_med", 0) == float('inf') and
                              stamp.get("dist_max", 0) == float('inf'))
                         for stamp in stamps.values())
    
    all_passed = all(stamp["pass"] for stamp in stamps.values())
    
    toml_data["certification"] = {
        "all_stamps_implemented": all_implemented,
        "all_stamps_passed": all_passed,
        "respectable": all_implemented and all_passed,
        "status": "respectable" if (all_implemented and all_passed) else "romantic"
    }
    
    write_toml(path, toml_data)


def main():
    parser = argparse.ArgumentParser(description="Generate stamped RH certification")
    parser.add_argument("--cert", type=str, help="Path to certification TOML (default: latest)")
    parser.add_argument("--out", type=str, default=".out/certs", help="Output directory")
    parser.add_argument("--depth", type=int, help="Override depth parameter")
    args = parser.parse_args()
    
    # Find and parse input certification
    cert_path = args.cert or find_latest_cert_toml()
    print(f"Reading certification from: {cert_path}")
    
    params = parse_cert_toml(cert_path)
    if args.depth:
        params["depth"] = args.depth
    
    print(f"Parsed parameters: depth={params.get('depth')}, N={params.get('N')}, zeros={len(params.get('zeros', []))}")
    
    # Apply stamps
    stamper = CertificationStamper(depth=params.get("depth", 4))
    print("\nApplying certification stamps...")
    stamp_results = stamper.stamp_certification(params)
    
    # Format for output
    stamps_formatted = stamper.format_stamps_for_ce1(stamp_results)
    
    # Generate output paths
    base_name = os.path.basename(cert_path).replace(".toml", "")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stamped_base = f"{base_name}_stamped_{timestamp}"
    
    os.makedirs(args.out, exist_ok=True)
    stamped_toml_path = os.path.join(args.out, f"{stamped_base}.toml")
    stamped_ce1_path = os.path.join(args.out, f"{stamped_base}.ce1")
    
    # Write outputs
    write_stamped_toml(stamped_toml_path, params, stamps_formatted)
    write_stamped_ce1(stamped_ce1_path, params, stamps_formatted)
    
    print(f"\nGenerated stamped certification:")
    print(f"  TOML: {stamped_toml_path}")
    print(f"  CE1:  {stamped_ce1_path}")
    
    # Print stamp results
    print(f"\nStamp Results:")
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
    print(f"Implemented: {implemented_count}/8 stamps")
    print(f"Passed: {passed_count}/{implemented_count} implemented stamps")
    
    if implemented_count == 8 and passed_count == 8:
        print("🎉 RESPECTABLE CERTIFICATION ACHIEVED!")
    elif implemented_count >= 4 and passed_count == implemented_count:
        print("✅ PARTIALLY RESPECTABLE - fast wins validated")
    else:
        print("❌ STILL ROMANTIC - needs more stamps")
    
    return 0


if __name__ == "__main__":
    exit(main())
