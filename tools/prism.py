#!/usr/bin/env python3
"""
Prism RefractionScan around a target zero using dihedral dispersion.

Outputs a CE1v block with:
- spectral_map: facet-binned, gamma-normalized intensities aggregated over the t-window
- locked_orientation: anchor facet index from on-line lock at t≈zero
- seed_id: user-provided identifier
"""

import argparse
import os
import time
import math
import importlib
import re
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import collections


from typing import Any, Dict, List, Tuple, cast, Optional

from core.rh_analyzer import (
    RHIntegerAnalyzer,
    DihedralCorrelator,
    PascalKernel,
    QuantitativeGapAnalyzer,
    IntegerSandwich,
    NTTProcessor,
    mate,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_scan_params(
    *,
    zero_t: float,
    depth: int,
    facets: int,
    window: float,
    step: float,
    d: float,
    gamma: float,
) -> None:
    _require(isinstance(depth, int), "depth must be an integer")
    _require(depth >= 1, "depth must be >= 1")
    _require(isinstance(facets, int), "facet count must be an integer")
    _require(facets >= 3, "facet count must be >= 3")
    _require(window > 0.0, "window must be > 0")
    _require(step > 0.0, "step must be > 0")
    _require(step <= 2.0 * window + 1e-12, "step must not exceed total window span")
    _require(math.isfinite(zero_t), "zero must be finite")
    _require(math.isfinite(d), "d must be finite")
    _require(gamma is not None and float(gamma) > 0.0, "gamma must be > 0")


def _makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_ce1v(path: str, payload: Dict[str, Any]) -> None:
    # Minimal CE1v writer following style in certify.py
    spectrum_obj: Any = payload.get("spectral_map", [])
    if isinstance(spectrum_obj, (list, tuple)):
        spectrum_str = ", ".join(f"{float(x):.6f}" for x in spectrum_obj)
    else:
        spectrum_str = str(spectrum_obj)
    locked_orientation = int(payload.get("locked_orientation", -1))
    seed_id = str(payload.get("seed_id", ""))
    meta_obj: Any = payload.get("meta", {})
    meta: Dict[str, Any] = meta_obj if isinstance(meta_obj, dict) else {}

    lines: List[str] = []
    lines.append("CE1v{\n")
    # Lens description
    lens = meta.get("lens", "Prism[pascal_dihedral]")
    mode = meta.get("mode", "RefractionScan")
    lines.append(f"  lens={lens}\n")
    lines.append(f"  mode={mode}\n")
    # Params block (flat key=value; semicolon-separated)
    params_obj: Any = payload.get("params", {})
    if isinstance(params_obj, dict):
        params_str = "; ".join(f"{k}={v}" for k, v in params_obj.items())
        lines.append(f"  params{{ {params_str} }}\n")
    # Core payload
    lines.append(f"  spectral_map=[{spectrum_str}]\n")
    lines.append(f"  locked_orientation={locked_orientation}\n")
    lines.append(f"  seed_id=\"{seed_id}\"\n")
    # Artifact path for traceability
    artifact = str(payload.get("artifact", ""))
    if artifact:
        lines.append(f"  artifact={artifact}\n")
    lines.append("}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def _bin_index(shift: int, N: int, facets: int) -> int:
    # Map dihedral shift index to facet bin [0, facets)
    # Use rotation shift only for angular position; reflections share same bin
    # floor(facets * shift / N)
    # Use integer math; avoid numpy
    if N <= 0:
        return 0
    value = (facets * shift) // N
    if value < 0:
        value = 0
    if value >= facets:
        value = facets - 1
    return int(value)


def _normalize_intensities(values: List[float], gamma: float) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v
    if span > 0:
        norm = [(v - min_v) / span for v in values]
    else:
        norm = [0.0 for _ in values]
    if gamma is None:
        gamma = 1.0
    norm = [pow(x, float(gamma)) for x in norm]
    s = sum(norm)
    if s > 0:
        norm = [x / s for x in norm]
    return norm


# def _aggregate_spectrum(
#     correlator: DihedralCorrelator,
#     mask: List[int],
#     template: List[int],
#     facets: int,
#     gamma: float,
#     steps: List[float],
#     N: int,
# ) -> List[float]:
#     # Accumulate facet intensities across all t steps
#     facet_accum = [0.0 for _ in range(facets)]

#     for _ in steps:
#         scores = correlator.correlate_all_actions_weighted(mask, template)
#         rotations = scores["rotations"]
#         reflections = scores["reflections"]

#         # Build 2N action intensities (rotations + reflections)
#         all_actions: List[Tuple[int, float]] = []
#         for s, score in enumerate(rotations):
#             all_actions.append((s, float(score)))
#         for s, score in enumerate(reflections):
#             all_actions.append((s, float(score)))

#         # Normalize per-step to [0,1] by min-max across all actions
#         action_vals = [v for (_, v) in all_actions]
#         norm_vals = _normalize_intensities(action_vals, gamma=1.0)  # pre-normalization before binning

#         # Bin by shift index
#         for (idx, (shift, _)) in enumerate(all_actions):
#             b = _bin_index(shift, N, facets)
#             facet_accum[b] += float(norm_vals[idx])

#     # Final gamma normalization across facets
#     spectrum = _normalize_intensities(facet_accum, gamma=gamma)
#     return spectrum


def _aggregate_spectrum_both(
    correlator: DihedralCorrelator,
    mask: List[int],
    template: List[int],
    facets: int,
    gamma: float,
    steps: List[float],
    N: int,
) -> Tuple[List[float], List[float]]:
    """Return (raw_accum, stepnorm_gamma_spectrum).

    raw_accum: sums raw action scores into facet bins across steps (no per-step normalization).
    stepnorm_gamma_spectrum: original behavior (per-step min-max, then gamma + facet normalize).
    """
    facet_accum_raw = [0.0 for _ in range(facets)]
    facet_accum_stepnorm = [0.0 for _ in range(facets)]

    for _ in steps:
        scores = correlator.correlate_all_actions_weighted(mask, template)
        rotations = scores["rotations"]
        reflections = scores["reflections"]

        all_actions: List[Tuple[int, float]] = []
        for s, score in enumerate(rotations):
            all_actions.append((s, float(score)))
        for s, score in enumerate(reflections):
            all_actions.append((s, float(score)))

        # Raw accumulation (no per-step normalization)
        for (shift, val) in all_actions:
            b = _bin_index(shift, N, facets)
            facet_accum_raw[b] += val

        # Step-normalized accumulation (as before)
        action_vals = [v for (_, v) in all_actions]
        norm_vals = _normalize_intensities(action_vals, gamma=1.0)
        for (idx, (shift, _)) in enumerate(all_actions):
            b = _bin_index(shift, N, facets)
            facet_accum_stepnorm[b] += float(norm_vals[idx])

    spectrum_stepnorm = _normalize_intensities(facet_accum_stepnorm, gamma=gamma)
    return facet_accum_raw, spectrum_stepnorm


def run_refraction_scan(
    zero_t: float,
    depth: int,
    facets: int,
    window: float,
    step: float,
    d: float,
    gamma: float,
) -> Dict[str, Any]:
    _validate_scan_params(
        zero_t=zero_t,
        depth=depth,
        facets=facets,
        window=window,
        step=step,
        d=d,
        gamma=gamma,
    )
    analyzer = RHIntegerAnalyzer(depth=depth)
    N = analyzer.N
    kernel = PascalKernel(N, depth)
    correlator = analyzer.correlator

    # Anchor Ω at on-line s = 1/2 + i*zero using metanion lock
    zeros_meta = [0.5 + 1j * zero_t]
    anchor_meta = analyzer.analyze_point_metanion(complex(0.5, zero_t), zeros_meta)
    anchor_action = anchor_meta["best_action"]  # DihedralAction
    locked_orientation = _bin_index(anchor_action.shift, N, facets)

    # Build steps in t
    t_start = zero_t - window
    t_end = zero_t + window
    ts: List[float] = []
    t = t_start
    while t <= t_end + 1e-12:
        ts.append(round(t, 12))
        t += step
    _require(len(ts) > 0, "no t samples generated; check window and step")

    # For refraction, use off-line sigma = 1/2 + d to measure dispersion
    # Create a representative mask/template at mid window for stability
    sigma_ref = 0.5 + d
    t_mid = zero_t
    mask, template = QuantitativeGapAnalyzer.create_e_n_based_mask(
        sigma_ref, t_mid, N, zeros_meta, kernel
    )

    # Aggregate spectrum across t window using the fixed mask/template
    raw_accum, spectrum = _aggregate_spectrum_both(
        correlator=correlator,
        mask=mask,
        template=template,
        facets=facets,
        gamma=gamma,
        steps=ts,
        N=N,
    )

    _require(len(mask) == N, "mask length must equal N")
    _require(len(template) == N, "template length must equal N")
    _require(len(spectrum) == facets, "spectrum facets mismatch")
    _require(0 <= int(locked_orientation) < facets, "locked_orientation out of range")

    return {
        "spectrum": spectrum,
        "locked_orientation": int(locked_orientation),
        "N": N,
        "mask": mask,
        "template": template,
        "t_mid": t_mid,
        "spectrum_raw": raw_accum,
    }


def _render_polar_png(spectrum: List[float], locked_orientation: int, out_png: str, title: str = "Prism RefractionScan", raw_mode: bool = False) -> bool:
    plt = importlib.import_module("matplotlib.pyplot")
    # mcolors = importlib.import_module("matplotlib.colors")
    cm = importlib.import_module("matplotlib.cm")
    from matplotlib import cm

    facets = len(spectrum)
    if facets <= 0:
        return False

    # Normalize intensities for colormap (unless raw_mode, then use modulo mapping)
    if raw_mode:
        # Map raw values via modulo to [0,1) for color; use signed magnitude for radial offset
        # Choose a modulus based on 95th percentile to keep extremes bounded
        vals = list(spectrum)
        sorted_vals = sorted(abs(v) for v in vals)
        idx = max(0, min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1))))
        mod = max(1.0, float(sorted_vals[idx]))
        color_vals = [((v % mod) / mod) if mod > 0 else 0.0 for v in vals]
        # Radial offset by tanh scaling of magnitude to tame outliers
        def _tanh_scale(x: float) -> float:
            return math.tanh(abs(x) / (mod if mod > 0 else 1.0))
        norm_vals = [_tanh_scale(v) for v in vals]
        color_norm = color_vals
    else:
        min_v = min(spectrum)
        max_v = max(spectrum)
        span = max_v - min_v
        norm_vals = [(v - min_v) / span if span > 0 else 0.0 for v in spectrum]
        color_norm = norm_vals
    cmap = plt.get_cmap("viridis")

    # Polar wedge ring
    fig = plt.figure(figsize=(6, 6), dpi=180)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticks([2 * math.pi * i / facets for i in range(facets)])
    ax.set_xticklabels([str(i) for i in range(facets)])

    width = 2 * math.pi / facets
    base = 0.6
    height = 0.35
    radii = [base + 0.2 * nv for nv in norm_vals]
    angles = [i * width for i in range(facets)]

    for i in range(facets):
        color = cmap(color_norm[i])
        edgecolor = (0, 0, 0, 0.15)
        ax.bar(
            x=angles[i],
            height=height,
            width=width*0.95,
            bottom=radii[i],
            color=color,
            edgecolor=edgecolor,
            linewidth=0.8,
            align='edge'
        )

    # Draw a subtle inner ring
    ax.bar(x=0, height=0.005, width=2*math.pi, bottom=base, color=(0,0,0,0.06), align='edge')

    # Highlight locked orientation with a radial marker and emphasized wedge edge
    locked_idx = max(0, min(facets - 1, int(locked_orientation)))
    locked_theta_center = (locked_idx + 0.5) * width
    max_r = base + height + 0.25
    ax.plot([locked_theta_center, locked_theta_center], [base - 0.1, max_r], color="#d62728", linewidth=2)

    # Add an outer ring arc on the locked wedge
    ax.bar(
        x=locked_idx * width,
        height=0.01,
        width=width*0.98,
        bottom=base + height + 0.02,
        color="#d62728",
        edgecolor="#d62728",
        linewidth=1.2,
        align='edge',
        alpha=0.8,
    )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def _compute_exact_scores(mask: List[int], template: List[int], N: int) -> Tuple[int, int, int, int, int]:
    """Return (winner_idx, runner_up_idx, winner_score, runner_up_score, gap) using exact integer sandwich."""
    # Use IntegerSandwich internals to compute exact rotations/reflections
    _require(N > 0, "N must be > 0")
    _require(len(mask) == N, "mask length must equal N")
    _require(len(template) == N, "template length must equal N")
    A_pad, V_pad, N_pad = IntegerSandwich._prepare_ntt_arrays(mask, template, N)
    mod = 2**31 - 1
    ntt = NTTProcessor(N_pad)
    A_ntt = ntt.ntt(A_pad)
    V_ntt = ntt.ntt(V_pad)
    # Rotations with reversed V
    V_rev = V_pad[::-1]
    V_rev_ntt = ntt.ntt(V_rev)
    rot_prod = [(x * y) % mod for x, y in zip(A_ntt, V_rev_ntt)]
    rotations = ntt.intt(rot_prod)[:N]
    # Reflections: corr(A, V)
    ref_prod = [(x * y) % mod for x, y in zip(A_ntt, V_ntt)]
    reflections = ntt.intt(ref_prod)[:N]
    all_scores = rotations + reflections
    winner_idx = int(all_scores.index(max(all_scores)))
    winner_score = int(all_scores[winner_idx])
    # Mate index exclusion for runner-up
    is_reflection = winner_idx >= N
    shift = winner_idx % N
    mate_shift, mate_reflection = mate(shift, is_reflection, N)
    mate_idx = mate_shift + (N if mate_reflection else 0)
    rival_scores = [all_scores[i] for i in range(len(all_scores)) if i != winner_idx and i != mate_idx]
    if not rival_scores:
        return winner_idx, winner_idx, winner_score, winner_score, 0
    runner_up_score = int(max(rival_scores))
    gap = int(winner_score - runner_up_score)
    runner_up_idx = int(all_scores.index(runner_up_score))
    return winner_idx, runner_up_idx, winner_score, runner_up_score, gap


def _gaussian_circular_weights(center: float, facets: int, sigma: float) -> List[float]:
    # Compute circular Gaussian weights over facet indices for a real-valued center index
    # distance on circle: min(|i-center|, facets - |i-center|)
    if sigma <= 0:
        sigma = 1.0
    two_sigma2 = 2.0 * (sigma * sigma)
    weights: List[float] = []
    for i in range(facets):
        d = abs(i - center)
        d = min(d, facets - d)
        w = 1.0
        w = math.exp(-(d * d) / two_sigma2)
        weights.append(w)
    s = sum(weights) or 1.0
    return [w / s for w in weights]


def _render_polygon_png(
    spectrum: List[float],
    out_png: str,
    title: str = "Prism Polygon",
    raw_mode: bool = False,
    sigma: Optional[float] = None,
) -> bool:
    facets = len(spectrum)
    if facets <= 2:
        return False

    # Prepare normalized values for coloring and thickness
    vals = list(spectrum)
    if raw_mode:
        sorted_vals = sorted(abs(v) for v in vals)
        idx = max(0, min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1))))
        mod = max(1.0, float(sorted_vals[idx]))
        color_base = [((v % mod) / mod) if mod > 0 else 0.0 for v in vals]
        # thickness based on tanh of magnitude
        thick_base = [math.tanh(abs(v) / (mod if mod > 0 else 1.0)) for v in vals]
    else:
        vmin = min(vals)
        vmax = max(vals)
        span = vmax - vmin
        color_base = [((v - vmin) / span) if span > 0 else 0.0 for v in vals]
        thick_base = color_base[:]

    # Gaussian interpolation across edges
    if sigma is None:
        sigma = max(1.0, facets / 6.0)

    cmap = plt.get_cmap("viridis")

    # Polygon vertices on unit circle
    R = 1.0
    vertices = [(R * math.cos(2 * math.pi * i / facets), R * math.sin(2 * math.pi * i / facets)) for i in range(facets)]
    vertices.append(vertices[0])

    fig = plt.figure(figsize=(6, 6), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

    # Build colored line segments for each edge using LineCollection
    segments = []
    colors = []
    linewidths = []
    samples_per_edge = 64
    for i in range(facets):
        p0 = vertices[i]
        p1 = vertices[i + 1]
        for k in range(samples_per_edge):
            t0 = k / samples_per_edge
            t1 = (k + 1) / samples_per_edge
            x0 = p0[0] * (1 - t0) + p1[0] * t0
            y0 = p0[1] * (1 - t0) + p1[1] * t0
            x1 = p0[0] * (1 - t1) + p1[0] * t1
            y1 = p0[1] * (1 - t1) + p1[1] * t1
            segments.append([(x0, y0), (x1, y1)])
            # Continuous facet index for this sample
            center_idx = (i + 0.5) % facets
            weights = _gaussian_circular_weights(center_idx, facets, sigma)
            col_val = sum(w * color_base[j] for j, w in enumerate(weights))
            th_val = sum(w * thick_base[j] for j, w in enumerate(weights))
            colors.append(cmap(col_val))
            linewidths.append(0.5 + 3.0 * th_val)

    lc = collections.LineCollection(segments, colors=colors, linewidths=linewidths, capstyle='round')
    ax.add_collection(lc)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def _render_polygon_surface_png(
    spectrum: List[float],
    out_png: str,
    title: str = "Prism Surface",
    raw_mode: bool = False,
    sigma: Optional[float] = None,
    color_gamma: float = 1.0,
) -> bool:
    facets = len(spectrum)
    if facets <= 2:
        return False

    # Prepare normalized values for coloring (same mapping as edge renderer)
    vals = list(spectrum)
    if raw_mode:
        sorted_vals = sorted(abs(v) for v in vals)
        idx = max(0, min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1))))
        mod = max(1.0, float(sorted_vals[idx]))
        color_base = [((v % mod) / mod) if mod > 0 else 0.0 for v in vals]
    else:
        vmin = min(vals)
        vmax = max(vals)
        span = vmax - vmin
        color_base = [((v - vmin) / span) if span > 0 else 0.0 for v in vals]

    # Optional contrast boost via gamma
    if color_gamma is None:
        color_gamma = 1.0
    if abs(color_gamma - 1.0) > 1e-9:
        color_base = [pow(max(0.0, min(1.0, c)), float(color_gamma)) for c in color_base]

    if sigma is None:
        sigma = max(1.0, facets / 8.0)

    cmap = plt.get_cmap("viridis")

    # Polygon vertices on unit circle
    R = 1.0
    vertices = [(R * math.cos(2 * math.pi * i / facets), R * math.sin(2 * math.pi * i / facets)) for i in range(facets)]
    vertices.append(vertices[0])

    fig = plt.figure(figsize=(6, 6), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

    # Build fine triangulation via triangle fans (center->edge i->edge i+1)
    center = (0.0, 0.0)
    tris = []
    colors = []
    # Subdivision parameters
    radial_steps = 40
    edge_steps = 60
    for i in range(facets):
        A = vertices[i]
        B = vertices[i + 1]
        for er in range(edge_steps):
            v0 = er / edge_steps
            v1 = (er + 1) / edge_steps
            # edge interpolation points on AB
            E0 = (A[0] * (1 - v0) + B[0] * v0, A[1] * (1 - v0) + B[1] * v0)
            E1 = (A[0] * (1 - v1) + B[0] * v1, A[1] * (1 - v1) + B[1] * v1)
            for rr in range(radial_steps):
                u0 = rr / radial_steps
                u1 = (rr + 1) / radial_steps
                # Quad corners in the triangle strip
                P00 = (center[0] * (1 - u0) + E0[0] * u0, center[1] * (1 - u0) + E0[1] * u0)
                P01 = (center[0] * (1 - u0) + E1[0] * u0, center[1] * (1 - u0) + E1[1] * u0)
                P10 = (center[0] * (1 - u1) + E0[0] * u1, center[1] * (1 - u1) + E0[1] * u1)
                P11 = (center[0] * (1 - u1) + E1[0] * u1, center[1] * (1 - u1) + E1[1] * u1)
                # Two triangles per quad
                tris.append([P00, P10, P11])
                tris.append([P00, P11, P01])
                # Color at quad center from Gaussian blend around facet index i with offset v_mid
                v_mid = 0.5 * (v0 + v1)
                center_idx = (i + v_mid) % facets
                weights = _gaussian_circular_weights(center_idx, facets, sigma)
                col_val = sum(w * color_base[j] for j, w in enumerate(weights))
                c = cmap(col_val)
                colors.append(c)
                colors.append(c)

    pc = collections.PolyCollection(tris, facecolors=colors, edgecolors='none', antialiaseds=True)
    ax.add_collection(pc)

    # Overlay facet boundaries to preserve perceived shape
    R = 1.0
    boundary = [(R * math.cos(2 * math.pi * i / facets), R * math.sin(2 * math.pi * i / facets)) for i in range(facets)]
    boundary.append(boundary[0])
    bx = [p[0] for p in boundary]
    by = [p[1] for p in boundary]
    ax.plot(bx, by, color=(0, 0, 0, 0.35), linewidth=1.0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def _render_polygon_surface_smooth_png(
    spectrum: List[float],
    out_png: str,
    title: str = "Prism Surface (smooth)",
    raw_mode: bool = False,
    sigma: Optional[float] = None,
    color_gamma: float = 1.0,
) -> bool:
    facets = len(spectrum)
    if facets <= 2:
        return False

    vals = list(spectrum)
    if raw_mode:
        sorted_vals = sorted(abs(v) for v in vals)
        idx = max(0, min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1))))
        mod = max(1.0, float(sorted_vals[idx]))
        color_base = [((v % mod) / mod) if mod > 0 else 0.0 for v in vals]
    else:
        vmin = min(vals)
        vmax = max(vals)
        span = vmax - vmin
        color_base = [((v - vmin) / span) if span > 0 else 0.0 for v in vals]

    # Optional contrast boost via gamma
    if color_gamma is None:
        color_gamma = 1.0
    if abs(color_gamma - 1.0) > 1e-9:
        color_base = [pow(max(0.0, min(1.0, c)), float(color_gamma)) for c in color_base]

    if sigma is None:
        sigma = max(1.0, facets / 8.0)

    # Build triangulated fan geometry with per-vertex scalar values, then Gouraud shade
    import matplotlib.tri as mtri
    cmap = plt.get_cmap("viridis")

    R = 1.0
    # Sampling resolution
    radial_steps = 80
    edge_steps = 120

    xs: List[float] = []
    ys: List[float] = []
    cs: List[float] = []
    tris: List[Tuple[int, int, int]] = []

    def color_at(theta_frac: float) -> float:
        # theta_frac in [0, facets)
        weights = _gaussian_circular_weights(theta_frac, facets, sigma)
        return float(sum(w * color_base[j] for j, w in enumerate(weights)))

    # Generate vertices ring by ring per wedge to keep indexing simple
    for i in range(facets):
        theta0 = 2 * math.pi * i / facets
        theta1 = 2 * math.pi * (i + 1) / facets
        # For each small quad in the wedge, add its two triangles
        # We'll generate a grid of (edge_steps+1) along edge and (radial_steps+1) radial levels
        # But to keep continuity across wedges, just process triangles locally and append to global lists
        # Local indexing offset
        base_idx = len(xs)
        # Create vertex grid for this wedge
        for er in range(edge_steps + 1):
            v = er / edge_steps
            theta = theta0 * (1 - v) + theta1 * v
            # center-to-edge radial samples
            for rr in range(radial_steps + 1):
                u = rr / radial_steps
                r = R * u
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                xs.append(x)
                ys.append(y)
                # For coloring, use the continuous facet index corresponding to theta
                theta_idx = (i + v) % facets
                cs.append(color_at(theta_idx))
        # Build triangles using the local grid indexing
        def idx_local(er: int, rr: int) -> int:
            return base_idx + er * (radial_steps + 1) + rr
        for er in range(edge_steps):
            for rr in range(radial_steps):
                p00 = idx_local(er, rr)
                p10 = idx_local(er + 1, rr)
                p01 = idx_local(er, rr + 1)
                p11 = idx_local(er + 1, rr + 1)
                tris.append((p00, p10, p11))
                tris.append((p00, p11, p01))

    triang = mtri.Triangulation(xs, ys, tris)

    fig = plt.figure(figsize=(6, 6), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

    tpc = ax.tripcolor(triang, cs, shading='gouraud', cmap=cmap)
    # Overlay facet boundary to maintain the original polygon silhouette
    R = 1.0
    boundary = [(R * math.cos(2 * math.pi * i / facets), R * math.sin(2 * math.pi * i / facets)) for i in range(facets)]
    boundary.append(boundary[0])
    bx = [p[0] for p in boundary]
    by = [p[1] for p in boundary]
    ax.plot(bx, by, color=(0, 0, 0, 0.35), linewidth=1.0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True

def _circular_laplacian(values: List[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    out: List[float] = [0.0] * n
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        out[i] = values[im1] - 2.0 * values[i] + values[ip1]
    return out


def _render_polygon_curved_png(
    spectrum: List[float],
    out_png: str,
    title: str = "Prism Curved Polygon",
    curvature_alpha: float = 0.3,
    use_raw_color: bool = True,
) -> bool:
    facets = len(spectrum)
    if facets <= 2:
        return False

    vals = list(spectrum)
    # Curvature driver: use raw values directly; scale-invariant by z-score-ish normalization
    mean_v = sum(vals) / len(vals)
    var_v = sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals))
    std_v = math.sqrt(var_v) or 1.0
    norm_vals = [(v - mean_v) / std_v for v in vals]
    kappa = _circular_laplacian(norm_vals)

    # Clamp curvature to avoid extremes
    max_abs_k = max(1e-9, max(abs(x) for x in kappa))
    kappa = [max(-1.0, min(1.0, x / max_abs_k)) for x in kappa]

    # Base radius profile
    R0 = 1.0
    radii = [R0 * (1.0 + curvature_alpha * k) for k in kappa]

    # Colors: raw or normalized
    if use_raw_color:
        sorted_vals = sorted(abs(v) for v in vals)
        idx = max(0, min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1))))
        mod = max(1.0, float(sorted_vals[idx]))
        color_base = [((v % mod) / mod) if mod > 0 else 0.0 for v in vals]
    else:
        vmin = min(vals)
        vmax = max(vals)
        span = vmax - vmin
        color_base = [((v - vmin) / span) if span > 0 else 0.0 for v in vals]

    cmap = plt.get_cmap("viridis")

    # Build warped vertices on unit circle with per-vertex radius
    vertices = []
    for i in range(facets):
        theta = 2 * math.pi * i / facets
        r = radii[i]
        vertices.append((r * math.cos(theta), r * math.sin(theta)))
    vertices.append(vertices[0])

    fig = plt.figure(figsize=(6, 6), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

    # Draw filled polygon with gradient approximated along edges via segment sampling
    segments = []
    colors = []
    linewidths = []
    samples_per_edge = 64
    for i in range(facets):
        p0 = vertices[i]
        p1 = vertices[i + 1]
        c0 = cmap(color_base[i])
        c1 = cmap(color_base[(i + 1) % facets])
        for k in range(samples_per_edge):
            t0 = k / samples_per_edge
            t1 = (k + 1) / samples_per_edge
            x0 = p0[0] * (1 - t0) + p1[0] * t0
            y0 = p0[1] * (1 - t0) + p1[1] * t0
            x1 = p0[0] * (1 - t1) + p1[0] * t1
            y1 = p0[1] * (1 - t1) + p1[1] * t1
            segments.append([(x0, y0), (x1, y1)])
            # simple color lerp along the edge
            w = (t0 + t1) * 0.5
            colors.append(tuple(c0[j] * (1 - w) + c1[j] * w for j in range(4)))
            linewidths.append(2.0)

    lc = collections.LineCollection(segments, colors=colors, linewidths=linewidths, capstyle='round')
    ax.add_collection(lc)

    # Light interior shading via a radial alpha wash
    from matplotlib import patches as mpatches
    wash = mpatches.Circle((0, 0), radius=max(max(abs(x), abs(y)) for x, y in vertices), color=(0, 0, 0, 0.04))
    ax.add_patch(wash)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True

def _find_latest_cert_toml(cert_dir: str = ".out/certs") -> Optional[str]:
    if not os.path.isdir(cert_dir):
        return None
    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(cert_dir):
        if name.endswith(".toml") and name.startswith("cert-"):
            path = os.path.join(cert_dir, name)
            mtime = os.path.getmtime(path)
            candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _parse_cert_defaults(toml_path: str) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}

    with open(toml_path, "r", encoding="utf-8") as f:
        text = f.read()

    def _find_number(key: str) -> Optional[float]:
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*$", text, re.MULTILINE)
        if m:
            return float(m.group(1))
        return None

    def _find_int(key: str) -> Optional[int]:
        v = _find_number(key)
        return int(v) if v is not None else None

    def _find_zeros() -> List[float]:
        m = re.search(r"^\s*zeros\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
        if not m:
            return []
        inner = m.group(1)
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", inner)
        out: List[float] = []
        for s in nums:
            out.append(float(s))
        return out

    depth = _find_int("depth")
    gamma = _find_int("gamma")
    d = _find_number("d")
    window = _find_number("window")
    step = _find_number("step")
    zeros = _find_zeros()

    if depth is not None: defaults["depth"] = depth
    if gamma is not None: defaults["gamma"] = gamma
    if d is not None: defaults["d"] = d
    if window is not None: defaults["window"] = window
    if step is not None: defaults["step"] = step
    if zeros: defaults["zero"] = zeros[0]
    return defaults


def main() -> int:
    p = argparse.ArgumentParser(description="Prism RefractionScan for dihedral dispersion")
    p.add_argument("--zero", type=float, default=None, help="Imaginary part t of target zero")
    p.add_argument("--depth", type=int, default=None, help="Pascal depth (N = 2^depth + 1)")
    p.add_argument("--facet", type=int, default=8, help="Number of prism facets for binning")
    p.add_argument("--window", type=float, default=None, help="Half-width of t window")
    p.add_argument("--step", type=float, default=None, help="Step size in t")
    p.add_argument("--d", type=float, default=None, help="Offset from critical line for refraction")
    p.add_argument("--gamma", type=float, default=None, help="Gamma for intensity normalization")
    p.add_argument("--out", type=str, default=".out/prism", help="Output directory")
    p.add_argument("--seed-id", type=str, default="prism_rh1_d4", help="Seed identifier")
    p.add_argument("--cert", type=str, default=None, help="Path to certification TOML; defaults to latest under .out/certs")
    # Curved polygon options
    p.add_argument("--curvature-alpha", type=float, default=0.3, help="Strength of curvature warping (0 disables)")
    p.add_argument("--curvature-source", type=str, default="raw", choices=["raw", "norm"], help="Curvature color source: raw or normalized")
    # Surface tuning
    p.add_argument("--surface-sigma", type=float, default=None, help="Gaussian blend sigma (facets/8 default)")
    p.add_argument("--surface-gamma", type=float, default=1.2, help="Gamma to boost surface color contrast")
    args = p.parse_args()

    # Load defaults from certification TOML if available
    cert_path = args.cert or _find_latest_cert_toml()
    cert_defaults = _parse_cert_defaults(cert_path) if cert_path else {}

    def _val(name: str, fallback: Any) -> Any:
        arg_val = getattr(args, name)
        if arg_val is not None:
            return arg_val
        if name in cert_defaults:
            return cert_defaults[name]
        return fallback

    depth_val: int = int(_val("depth", 4))
    zero_val: float = float(_val("zero", 14.134725))
    window_val: float = float(_val("window", 0.5))
    step_val: float = float(_val("step", 0.1))
    d_val: float = float(_val("d", 0.05))
    gamma_val: float = float(_val("gamma", 3.0))

    res = run_refraction_scan(
        zero_t=zero_val,
        depth=depth_val,
        facets=args.facet,
        window=window_val,
        step=step_val,
        d=d_val,
        gamma=gamma_val,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    _makedirs(args.out)
    N_val: int = cast(int, res.get("N", 0))
    base = f"prism-depth{depth_val}-N{N_val}-F{args.facet}-t{zero_val:.6f}-{ts}"
    out_ce1v = os.path.join(args.out, f"{base}.ce1v")
    out_png = os.path.join(args.out, f"{base}.png")

    # Compute exact CE1vCard from integer-sandwich scores
    winner_idx, runner_idx, winner_score, runner_score, exact_gap = _compute_exact_scores(
        res["mask"], res["template"], N_val
    )
    is_reflection = winner_idx >= N_val
    s_shift = int(winner_idx % N_val)
    lock_tuple = (s_shift, bool(is_reflection))
    G = int(winner_score)
    G2 = int(runner_score)
    gap_val = int(exact_gap)
    tau = 1  # simple secondary threshold for runner-up separation
    passed = (G >= int(gamma_val)) and (gap_val >= tau)

    # Simple mate map hash surrogate: hash of mask/template tuple lengths and N
    mates_hash = hash((len(res["mask"]), len(res["template"]), N_val)) & 0xFFFFFFFF

    payload = {
        "spectral_map": res["spectrum"],
        "locked_orientation": res["locked_orientation"],
        "seed_id": args.seed_id,
        "artifact": out_ce1v,
        "params": {
            "depth": depth_val,
            "N": N_val,
            "facets": args.facet,
            "gamma": gamma_val,
            "d": d_val,
            "window": window_val,
            "step": step_val,
            "zero": zero_val,
        },
        "meta": {
            "lens": f"Prism[pascal_dihedral;facet={args.facet}]",
            "mode": "RefractionScan",
        },
        "CE1vCard": {
            "lock": lock_tuple,
            "G": G,
            "runner_up": G2,
            "gap": gap_val,
            "gamma": int(gamma_val),
            "passed": bool(passed),
            "context": {
                "N": N_val,
                "t_approx": float(zero_val),
                "facet": int(args.facet),
                "mates": int(mates_hash),
            },
        },
    }
    _write_ce1v(out_ce1v, payload)

    png_ok = False
    out_png_raw = os.path.join(args.out, f"{base}-raw.png")
    out_poly = os.path.join(args.out, f"{base}-poly.png")
    out_poly_raw = os.path.join(args.out, f"{base}-poly-raw.png")
    out_poly_surface = os.path.join(args.out, f"{base}-poly-surface.png")
    out_poly_surface_raw = os.path.join(args.out, f"{base}-poly-surface-raw.png")
    out_poly_curved = os.path.join(args.out, f"{base}-poly-curved.png")
    out_poly_surface_smooth = os.path.join(args.out, f"{base}-poly-surface-smooth.png")
    out_poly_surface_smooth_raw = os.path.join(args.out, f"{base}-poly-surface-smooth-raw.png")

    # Only render images if CE1vCard passed (lock achieved with sufficient gap)
    passed = bool(payload.get("CE1vCard", {}).get("passed", False))
    if passed:
        # Render visualization (best-effort)
        png_ok = _render_polar_png(
            res["spectrum"], int(res["locked_orientation"]), out_png, title=f"Prism F={args.facet} N={N_val} t≈{zero_val}", raw_mode=False
        )
        # Also render raw-mode PNG
        _ = _render_polar_png(
            res.get("spectrum_raw", res["spectrum"]), int(res["locked_orientation"]), out_png_raw, title=f"Prism (raw) F={args.facet} N={N_val} t≈{zero_val}", raw_mode=True
        )
        # Polygon renderings
        _render_polygon_png(res["spectrum"], out_poly, title=f"Polygon F={args.facet} N={N_val} t≈{zero_val}", raw_mode=False)
        _render_polygon_png(res.get("spectrum_raw", res["spectrum"]), out_poly_raw, title=f"Polygon (raw) F={args.facet} N={N_val} t≈{zero_val}", raw_mode=True)
        # Shaded polygon surface renderings
        _render_polygon_surface_png(
            res["spectrum"], out_poly_surface,
            title=f"Polygon Surface F={args.facet} N={N_val} t≈{zero_val}",
            raw_mode=False, sigma=args.surface_sigma, color_gamma=args.surface_gamma,
        )
        _render_polygon_surface_png(
            res.get("spectrum_raw", res["spectrum"]), out_poly_surface_raw,
            title=f"Polygon Surface (raw) F={args.facet} N={N_val} t≈{zero_val}",
            raw_mode=True, sigma=args.surface_sigma, color_gamma=args.surface_gamma,
        )
        _render_polygon_surface_smooth_png(
            res["spectrum"], out_poly_surface_smooth,
            title=f"Polygon Surface (smooth) F={args.facet} N={N_val} t≈{zero_val}",
            raw_mode=False, sigma=args.surface_sigma, color_gamma=args.surface_gamma,
        )
        _render_polygon_surface_smooth_png(
            res.get("spectrum_raw", res["spectrum"]), out_poly_surface_smooth_raw,
            title=f"Polygon Surface (smooth, raw) F={args.facet} N={N_val} t≈{zero_val}",
            raw_mode=True, sigma=args.surface_sigma, color_gamma=args.surface_gamma,
        )
        # Curved boundary polygon (default raw color)
        if args.curvature_alpha and abs(args.curvature_alpha) > 1e-6:
            use_raw_color = (args.curvature_source == "raw")
            _render_polygon_curved_png(
                res.get("spectrum_raw", res["spectrum"]) if use_raw_color else res["spectrum"],
                out_poly_curved,
                title=f"Polygon Curved F={args.facet} N={N_val} t≈{zero_val}",
                curvature_alpha=float(args.curvature_alpha),
                use_raw_color=use_raw_color,
            )
    else:
        # Rename base to include -nolock for any future outputs we might add
        base_nolock = f"{base}-nolock"
        out_png = os.path.join(args.out, f"{base_nolock}.png")
        out_png_raw = os.path.join(args.out, f"{base_nolock}-raw.png")
        out_poly = os.path.join(args.out, f"{base_nolock}-poly.png")
        out_poly_raw = os.path.join(args.out, f"{base_nolock}-poly-raw.png")
        out_poly_surface = os.path.join(args.out, f"{base_nolock}-poly-surface.png")
        out_poly_surface_raw = os.path.join(args.out, f"{base_nolock}-poly-surface-raw.png")
        out_poly_curved = os.path.join(args.out, f"{base_nolock}-poly-curved.png")
        out_poly_surface_smooth = os.path.join(args.out, f"{base_nolock}-poly-surface-smooth.png")
        out_poly_surface_smooth_raw = os.path.join(args.out, f"{base_nolock}-poly-surface-smooth-raw.png")

    print(f"Wrote CE1v: {out_ce1v}")
    if passed:
        if png_ok and os.path.exists(out_png):
            print(f"Wrote PNG:  {out_png}")
            if os.path.exists(out_png_raw):
                print(f"Wrote PNG:  {out_png_raw}")
            if os.path.exists(out_poly):
                print(f"Wrote PNG:  {out_poly}")
            if os.path.exists(out_poly_raw):
                print(f"Wrote PNG:  {out_poly_raw}")
            if os.path.exists(out_poly_surface):
                print(f"Wrote PNG:  {out_poly_surface}")
            if os.path.exists(out_poly_surface_raw):
                print(f"Wrote PNG:  {out_poly_surface_raw}")
            if os.path.exists(out_poly_curved):
                print(f"Wrote PNG:  {out_poly_curved}")
            if os.path.exists(out_poly_surface_smooth):
                print(f"Wrote PNG:  {out_poly_surface_smooth}")
            if os.path.exists(out_poly_surface_smooth_raw):
                print(f"Wrote PNG:  {out_poly_surface_smooth_raw}")
        else:
            print("PNG not created (matplotlib missing or render error). Install: pip install matplotlib")
    else:
        print("Lock not achieved or below thresholds; images skipped.")
    print(f"Locked orientation (facet index): {res['locked_orientation']}")
    spectrum_list = res.get('spectrum', [])
    if isinstance(spectrum_list, list):
        print(f"Spectrum: {[float(x) for x in spectrum_list]}")
    else:
        print(f"Spectrum: {spectrum_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


