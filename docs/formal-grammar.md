## Formal grammar for the certification construction

Definitions

- Kernel: Fix an even kernel K_N: ℝ → ℝ with support in [−Δ, Δ], normalized so ∫K_N = 1 and second moment μ_2(K_N) exists. Discrete version uses Pascal row of length 2^depth+1, normalized.

- Smoothed drift: For σ ∈ ℝ, t ∈ ℝ, define E_N(σ,t) = (∂_σ log|ξ| * K_N)(σ,t) where convolution is in t with the even kernel lifted to the (σ,t) chart.

- Integer lift: Let λ = 2^q (q∈ℕ). Define W_± ∈ ℕ^N such that W_- ≤ λK_N ≤ W_+ (componentwise) and ∑W_- = ∑W_+ = λ. (Integer sandwich.)

- Mask/template map Φ: Given E := E_N(σ,t), define (M_N, T_N) = Φ(E) ∈ {0,1}^N × {0,1}^N by a rule monotone in |E| and odd in sign(E) (e.g., block size and cyclic shift proportional to |E| and sign(E)). Φ is 1-Lipschitz in E in the Hamming metric up to scale c_Φ/N.

- Dihedral gap: For A = 2M_N−1, V = 2T_N−1, define rotation/reflection scores S_rot[s] = ⟨A, V∘τ_s⟩, S_ref[s] = ⟨A, (V^R)∘τ_s⟩, gap G_N = max(S) − second_max(S), S over 2N actions. Exact integer sandwich scoring computes these correlations exactly.

- Mate exclusion: The mate of (s,r) is (−s, ¬r). Uniqueness requires excluding the mate when defining the runner-up for the gap.

Lemmas (targets)

1) First-moment cancellation: For even K_N and odd ∂_σ log|ξ|(1/2,·) in the window, |E_N(1/2,t)| ≤ ε_N uniformly on W_t, with ε_N → 0 as N grows (or Δ shrinks).

2) Off-line linear growth: For σ = 1/2 ± d with small d, |E_N(σ,t)| ≥ c_N d − ε_N uniformly on W_t, with c_N ≈ μ_2(K_N)^{1/2} and ε_N as above.

3) Integer correlation separation: For (M_N,T_N) = Φ(E), there exists A_N > 0 and δ_N ≥ 0 such that G_N ≥ A_N·|E| − δ_N. Moreover, A_N scales with λ and N as A_N ≍ λ/√N for the sandwich choice.

4) Stability (Lipschitz/argmax): If |E_off| − |E_on| ≥ ΔE and A_N·ΔE − 2δ_N ≥ γ, then the winning action is stable and the gap satisfies G_N ≥ γ.

5) Two-scale consistency: Winners at N and 2N satisfy the lift relation (2s+c, r) with preserved reflection r and c ∈ {0,1}; mates map accordingly. Violations are penalized (zero or small probability under the model).

Theorem (certificate without sampling; template):

Fix N, γ, d, and a t-window W_t. Suppose Lemmas 1–4 hold with constants (ε_N, c_N, A_N, δ_N). If A_N (c_N d − 2ε_N) − 2δ_N ≥ γ and A_N ε_N + δ_N < γ, then for all t ∈ W_t,

- On-line: G_N(1/2,t) ≥ γ (locks);
- Off-line: G_N(1/2±d,t) < γ (fails).

Hence the “succeeds-on / fails-off” certificate holds uniformly on W_t.

Notes: Lemma 5 yields uniqueness/robustness across N and excludes mate ambiguity.

