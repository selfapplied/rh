## Proof anchors (accepted theorems → our construction)

- Explicit formula (Weil/Guinand–Weil): connects zeros/prime sums to test-function transforms. Our `GyroscopeLoss` uses a smoothed linear functional of ∂σ log|ξ|; with an admissible even kernel `K_N`, this fits the explicit-formula framework.

- Beurling–Selberg extremal majorants/minorants: integer sandwich mirrors constructing discrete majorant/minorant around a target even kernel. Our `IntegerSandwich.kernel_sandwich` is an L1-mass–preserving discretization toward extremal pairs.

- Paley–Wiener / Poisson summation (bandlimited smoothing): Pascal (binomial) kernels are compactly supported discrete smoothers; their continuous analogs yield controlled tails and moment cancellations. We exploit first-moment cancellation on σ=1/2 and second-moment dominance off-line.

- Large sieve–type inequalities: dihedral correlation energy bounds with weights relate to weighted autocorrelations; provide A_N-type lower bounds and ε_N tails under mild smoothness.

- Stability of argmax under Lipschitz perturbations: if the winning action’s margin exceeds γ and perturbations are ≤ ε < γ/2, the argmax is stable. This underpins the certification when mapping analytical contrast → integer gaps.

How it ties together

1) Smooth drift: E_N(σ,t) = (∂σ log|ξ| * K_N)(σ,t). On σ=1/2, oddness + even K_N ⇒ first moment cancels, |E_N| ≤ ε_N. Off σ=1/2±d, Taylor + second moment ⇒ |E_N| ≥ c_N d − ε_N.

2) Integer sandwich: discretize λK_N into W_± with equal mass and monotone fixes; map |E_N| contrast to mask/template contrast. Extremal logic gives linear-in-contrast correlation separation.

3) Dihedral gap: weighted dihedral correlation + sandwich exact scoring ⇒ integer margin gap ≥ A_N d − ε′_N. Choose N,γ,d so margin ≥ γ on-line and < γ off-line, uniformly on the window.

Next steps to formalize

- Specify admissible `K_N`, compute its second moment and tail bounds; extract explicit c_N, ε_N.
- Prove W_± mass/ordering properties and a quantitative correlation separation lemma.
- Lift to two-scale (N, 2N) with mate exclusion for uniqueness and robustness.

