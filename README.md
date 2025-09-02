# RH

Standalone extraction of the `rh` research code from `iontheprize`.

## Contents
- `rh.py`: core functions
- `twoadic.py`: 2-adic utilities
- `pascal.py`: Pascal-related helpers
- `deep_rh_analysis.py`: analysis scripts
- `rieman.py`: consolidated demo entry
- `test_*.py`: tests

## Getting started

```bash
uv run python rieman.py
```

## Certification

Generate a certification report (defaults: depth=4, gamma=3, d=0.05):

```bash
uv run python certify.py --out .out/certs
```

This writes a TOML file under `.out/certs/` with on-line/off-line lock rates. CI runs the same check and uploads artifacts.

## Proof anchors (accepted theorems → our construction)

- Explicit formula (Weil/Guinand–Weil): connects zeros/prime sums to test-function transforms. Our `GyroscopeLoss` uses a smoothed linear functional of ∂σ log|ξ|; with an admissible even kernel `K_N`, this fits the explicit-formula framework.
- Beurling–Selberg extremal majorants/minorants: integer sandwich mirrors constructing discrete majorant/minorant around a target even kernel. Our `IntegerSandwich.kernel_sandwich` is an L1-mass–preserving discretization toward extremal pairs.
- Paley–Wiener / Poisson summation (bandlimited smoothing): Pascal (binomial) kernels are compactly supported discrete smoothers; their continuous analogs yield controlled tails and moment cancellations. We exploit first-moment cancellation on σ=1/2 and second-moment dominance off-line.
- Large sieve–type inequalities: dihedral correlation energy bounds with weights relate to weighted autocorrelations; provide A_N-type lower bounds and ε_N tails under mild smoothness.
- Stability of argmax under Lipschitz perturbations: if the winning action’s margin exceeds γ and perturbations are ≤ ε < γ/2, the argmax is stable. This underpins the certification when mapping analytical contrast → integer gaps.

How it ties together:
1) Smooth drift: E_N(σ,t) = (∂σ log|ξ| · K_N)(σ,t). On σ=1/2, oddness + even K_N ⇒ first moment cancels, |E_N| ≤ ε_N. Off σ=1/2±d, Taylor + second moment ⇒ |E_N| ≥ c_N d − ε_N.
2) Integer sandwich: discretize λK_N into W_± with equal mass and monotone fixes; map |E_N| contrast to mask/template contrast. Extremal logic gives linear-in-contrast correlation separation.
3) Dihedral gap: weighted dihedral correlation + sandwich exact scoring ⇒ integer margin gap ≥ A_N d − ε′_N. Choose N,γ,d so margin ≥ γ on-line and < γ off-line, uniformly on the window.

## Formal grammar (precise definitions and theorem template)

Definitions
- Kernel: Fix an even kernel K_N: ℝ → ℝ with support in [−Δ, Δ], normalized so ∫K_N = 1 and second moment μ₂(K_N) exists. Discrete version uses Pascal row of length 2^depth+1, normalized.
- Smoothed drift: For σ ∈ ℝ, t ∈ ℝ, define E_N(σ,t) = (∂_σ log|ξ| * K_N)(σ,t), convolution in t with the even kernel lifted to the (σ,t) chart.
- Integer lift: Let λ = 2^q (q∈ℕ). Define W_± ∈ ℕ^N such that W_- ≤ λK_N ≤ W_+ (componentwise) and ∑W_- = ∑W_+ = λ. (Integer sandwich.)
- Mask/template map Φ: Given E := E_N(σ,t), define (M_N, T_N) = Φ(E) ∈ {0,1}^N × {0,1}^N by a rule monotone in |E| and odd in sign(E) (e.g., block size and cyclic shift proportional to |E| and sign(E)). Φ is Lipschitz in E in the Hamming metric up to scale c_Φ/N.
- Dihedral gap: For A = 2M_N−1, V = 2T_N−1, define rotation/reflection scores S_rot[s] = ⟨A, V∘τ_s⟩, S_ref[s] = ⟨A, (V^R)∘τ_s⟩, gap G_N = max(S) − second_max(S), S over 2N actions (mate excluded for runner-up).
- Mate exclusion: The mate of (s,r) is (−s, ¬r). Uniqueness excludes the mate from the runner-up.

Lemmas (targets)
1) First-moment cancellation: For even K_N and odd ∂_σ log|ξ|(1/2,·) in the window, |E_N(1/2,t)| ≤ ε_N uniformly on W_t, with ε_N → 0 as N grows (or Δ shrinks).
2) Off-line linear growth: For σ = 1/2 ± d with small d, |E_N(σ,t)| ≥ c_N d − ε_N uniformly on W_t, with c_N ≈ μ₂(K_N)^{1/2} and ε_N as above.
3) Integer correlation separation: For (M_N,T_N) = Φ(E), there exists A_N > 0 and δ_N ≥ 0 such that G_N ≥ A_N·|E| − δ_N; typically A_N scales like λ/√N for this sandwich.
4) Stability (Lipschitz/argmax): If |E_off| − |E_on| ≥ ΔE and A_N·ΔE − 2δ_N ≥ γ, then the winning action is stable and G_N ≥ γ.
5) Two-scale consistency: Winners at N and 2N satisfy lift (2s+c, r) with preserved reflection r and c ∈ {0,1}; mates map accordingly.

Theorem (certificate without sampling; template)
Fix N, γ, d, and a t-window W_t. Suppose Lemmas 1–4 hold with constants (ε_N, c_N, A_N, δ_N). If A_N (c_N d − 2ε_N) − 2δ_N ≥ γ and A_N ε_N + δ_N < γ, then for all t ∈ W_t:
- On-line: G_N(1/2,t) ≥ γ (locks)
- Off-line: G_N(1/2±d,t) < γ (fails)
Hence the “succeeds-on / fails-off” certificate holds uniformly on W_t. Lemma 5 yields uniqueness/robustness across N and excludes mate ambiguity.

### Latest generated certification (example)

Path:

```
.out/certs/cert-depth4-N17-20250902-031805.toml
```

Key summary from that run:

```toml
[summary]
online_locked_rate = 1.0
offline_locked_rate = 0.0
online_total = 33
offline_total = 33
online_locked = 33
offline_locked = 0
```

### What the numbers mean (plain language)

- On-line: we test points on the critical line (σ = 1/2) over a small window in t. Locked means the integer sandwich gap ≥ γ using the Metanion–Pascal phaselock. A good certificate has a high on-line locked rate (near 100%).
- Off-line: we test points slightly off the line (σ = 1/2 + d) over the same window. A good certificate has a low off-line locked rate (near 0%).

Parameters at defaults:
- depth=4 → N = 2^depth + 1 = 17 (resolution)
- γ=3 → required integer gap to count as locked
- d=0.05 → how far off the line we test
- window, step → size and granularity of the t-sweep around zeros

In short: the certificate succeeds if it locks on the line and fails off the line across the window.

## License
MIT unless noted otherwise.
