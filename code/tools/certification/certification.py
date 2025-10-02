#!/usr/bin/env python3
"""
Certification sweep for Metanion-informed phaselock.

Writes TOML reports under .out/certs/<timestamp>/ with summary statistics.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List

from .rh_analyzer import RHIntegerAnalyzer


def makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_toml_kv(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}\n"
    if isinstance(value, (int, float)):
        return f"{key} = {value}\n"
    if isinstance(value, str):
        escaped = value.replace("\n", "\\n").replace('"', '\\"')
        return f"{key} = \"{escaped}\"\n"
    if isinstance(value, list):
        items = []
        for v in value:
            if isinstance(v, str):
                items.append(f'"{v.replace("\n", "\\n").replace("\"", "\\\"")}"')
            elif isinstance(v, bool):
                items.append('true' if v else 'false')
            else:
                items.append(str(v))
        return f"{key} = [{', '.join(items)}]\n"
    if isinstance(value, dict):
        lines = [f"[{key}]\n"]
        for k, v in value.items():
            lines.append(format_toml_kv(k, v))
        return "".join(lines)
    return f"{key} = \"{str(value)}\"\n"


def write_toml(path: str, data: Dict[str, Any]) -> None:
    lines: List[str] = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"[{k}]\n")
            for kk, vv in v.items():
                lines.append(format_toml_kv(kk, vv))
            lines.append("\n")
        else:
            lines.append(format_toml_kv(k, v))
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def write_ce1(path: str, meta: Dict[str, Any]) -> None:
    """Write a CE1 block capturing the certification result."""
    zeros_list = "; ".join(f"{z}" for z in meta["zeros"]) if isinstance(
        meta.get("zeros"), list) else str(meta.get("zeros"))
    summary = meta.get("summary", {})
    params = meta.get("params", {})
    params_str = "; ".join(f"{k}={v}" for k, v in params.items())
    ce1 = []
    ce1.append("CE1{\n")
    ce1.append("  lens=RH_CERT\n")
    ce1.append("  mode=Certification\n")
    ce1.append("  basis=metanion:pascal_dihedral\n")
    ce1.append(f"  params{{ {params_str} }}\n")
    ce1.append(f"  zeros=[{zeros_list}]\n")
    total = summary.get('online_total', 0)
    online_locked = summary.get('online_locked', 0)
    online_ratio = (online_locked / total) if total else 0
    ce1.append("  summary{ ")
    ce1.append("; ".join(
        [
            f"total={total}",
            f"online_locked={online_locked}",
            f"online_ratio={online_ratio}",
        ]
    ))
    ce1.append(" }\n")
    ce1.append(f"  artifact={meta.get('artifact', '')}\n")
    ce1.append("  emit=RiemannHypothesisCertification\n")
    ce1.append("}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(ce1))


def sweep_cert(depth: int, zeros: List[float], window: float, step: float, d: float, gamma: int, out_dir: str) -> Dict[str, Any]:
    analyzer = RHIntegerAnalyzer(depth=depth)
    N = analyzer.N
    results = {
        "depth": depth,
        "N": N,
        "gamma": gamma,
        "window": window,
        "step": step,
        "d": d,
        "zeros": zeros,
        "by_zero": {},
        "summary": {},
    }

    total_online = 0
    total_offline = 0
    total_online_locked = 0
    total_offline_locked = 0

    for tz in zeros:
        t_start = tz - window
        t_end = tz + window
        ts = []
        t = t_start
        while t <= t_end + 1e-12:
            ts.append(round(t, 12))
            t += step

        per = {
            "t_zero": tz,
            "count_online": 0,
            "count_offline": 0,
            "locked_online": 0,
            "locked_offline": 0,
            "gap_online": [],
            "gap_offline": [],
        }

        for t in ts:
            # On-line (sigma = 1/2)
            s_on = complex(0.5, t)
            r_on = analyzer.analyze_point_metanion(s_on, [0.5 + tz*1j for tz in zeros])
            per["count_online"] += 1
            total_online += 1
            if r_on["gap"] >= gamma:
                per["locked_online"] += 1
                total_online_locked += 1
            per["gap_online"].append(int(r_on["gap"]))

            # Off-line (sigma = 1/2 + d)
            s_off = complex(0.5 + d, t)
            r_off = analyzer.analyze_point_metanion(s_off, [0.5 + tz*1j for tz in zeros])
            per["count_offline"] += 1
            total_offline += 1
            if r_off["gap"] >= gamma:
                per["locked_offline"] += 1
                total_offline_locked += 1
            per["gap_offline"].append(int(r_off["gap"]))

        results["by_zero"][str(tz)] = per

    results["summary"] = {
        "online_locked_rate": (total_online_locked / total_online) if total_online else 0.0,
        "offline_locked_rate": (total_offline_locked / total_offline) if total_offline else 0.0,
        "online_total": total_online,
        "offline_total": total_offline,
        "online_locked": total_online_locked,
        "offline_locked": total_offline_locked,
    }

    ts = time.strftime("%Y%m%d-%H%M%S")
    makedirs(out_dir)
    base = f"cert-depth{depth}-N{N}-{ts}"
    out_toml = os.path.join(out_dir, f"{base}.toml")
    write_toml(out_toml, results)

    # CE1 companion
    ce1_meta = {
        "zeros": zeros,
        "summary": results["summary"],
        "artifact": out_toml,
        "params": {
            "depth": depth,
            "N": N,
            "gamma": gamma,
            "d": d,
            "window": window,
            "step": step,
        },
    }
    out_ce1 = os.path.join(out_dir, f"{base}.ce1")
    write_ce1(out_ce1, ce1_meta)

    return {"out_path": out_toml, "out_ce1": out_ce1, "results": results}


def main():
    p = argparse.ArgumentParser(description="Certification sweep for Metanion phaselock")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--zeros", type=float, nargs="+", default=[14.134725, 21.022040, 25.010858])
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--d", type=float, default=0.05)
    p.add_argument("--gamma", type=int, default=3)
    p.add_argument("--out", type=str, default=".out/certs")
    p.add_argument("--require-online-rate", type=float, default=0.95, help="Min on-line locked rate")
    p.add_argument("--require-offline-rate", type=float, default=0.05, help="Max off-line locked rate")
    args = p.parse_args()

    res = sweep_cert(
        depth=args.depth,
        zeros=args.zeros,
        window=args.window,
        step=args.step,
        d=args.d,
        gamma=args.gamma,
        out_dir=args.out,
    )
    print(f"Wrote certificate: {res['out_path']}")
    summary = res['results']['summary']
    print(f"Summary: on-line locked {summary['online_locked']} / {summary['online_total']}, "
          f"off-line locked {summary['offline_locked']} / {summary['offline_total']}")

    # Assertions for CI/guard
    if summary['online_total'] > 0:
        if summary['online_locked_rate'] < args.require_online_rate:
            print(f"CI-FAIL: online rate {summary['online_locked_rate']:.3f} < {args.require_online_rate}")
            return 1
    if summary['offline_total'] > 0:
        if summary['offline_locked_rate'] > args.require_offline_rate:
            print(f"CI-FAIL: offline rate {summary['offline_locked_rate']:.3f} > {args.require_offline_rate}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


