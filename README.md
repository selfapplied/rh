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

We align with standard tools: explicit formula (Weil/Guinand–Weil), Beurling–Selberg extremals, Paley–Wiener/Poisson summation, large-sieve–type bounds, and argmax stability under Lipschitz perturbations. The detailed algebra sits below in a formal style.

## Formal presentation

Definition (kernel).
Let K_N: $\mathbb{R}\to\mathbb{R}$ be even, supported on $[{-}\Delta,\Delta]$, with
$$ \int_{\mathbb{R}} K_N(u)\,du = 1,\qquad \mu_2(K_N) = \int u^2 K_N(u)\,du < \infty. $$
Discrete K_N uses the normalized Pascal row of length \(N=2^{\text{depth}}+1\).

Definition (smoothed drift).
For $\sigma,t\in\mathbb{R}$, set
$$ E_N(\sigma,t) \;=\; (\partial_\sigma \log\lvert \xi\rvert * K_N)(\sigma,t)
   \,=\, \int_{\mathbb{R}} \partial_\sigma \log\lvert \xi(\sigma,t-u)\rvert\, K_N(u)\,du. $$

Definition (integer sandwich).
Fix $\lambda=2^q$. Choose $W_\pm\in\mathbb{N}^N$ with
$$ W_- \le \lambda K_N \le W_+, \qquad \sum_i W_-^{(i)} = \sum_i W_+^{(i)} = \lambda. $$

Definition (mask/template map).
Let $(M_N,T_N)=\Phi(E_N)\in\{0,1\}^N\times\{0,1\}^N$ be monotone in $|E_N|$ and odd in $\operatorname{sign}(E_N)$. Assume a Lipschitz property: for nearby drifts $E,E'$, 
$$ d_\mathrm{H}\big(\Phi(E),\Phi(E')\big) \;\le\; \tfrac{c_\Phi}{N}\,\lVert E-E'\rVert, $$
where $d_\mathrm{H}$ is Hamming distance.

Definition (dihedral gap).
Let $A=2M_N{-}1$, $V=2T_N{-}1$. For shift $s$,
$$ S_\mathrm{rot}[s]=\langle A, V\circ\tau_s\rangle,\qquad S_\mathrm{ref}[s]=\langle A, (V^R)\circ\tau_s\rangle. $$
With mate excluded, the gap is
$$ G_N\;=\; \max S\; -\; \operatorname{second\_max} S. $$

Lemma (first-moment cancellation).
On $\sigma=\tfrac12$, with even $K_N$ and odd $\partial_\sigma\log|\xi|$ in the window $W_t$, there exists $\varepsilon_N\to0$ with
$$ \sup_{t\in W_t} \big\lvert E_N(\tfrac12,t) \big\rvert \;\le\; \varepsilon_N. $$

Lemma (off-line linear growth).
For $\sigma=\tfrac12\pm d$ and small $d>0$, there exists $c_N>0$ and $\varepsilon_N$ such that
$$ \inf_{t\in W_t} \big\lvert E_N(\tfrac12\pm d, t) \big\rvert \;\ge\; c_N\, d\; -\; \varepsilon_N. $$

Lemma (integer correlation separation).
For $(M_N,T_N)=\Phi(E)$, there exist $A_N>0$, $\delta_N\ge0$ with
$$ G_N \;\ge\; A_N\,\lvert E\rvert \; -\; \delta_N, \qquad A_N \asymp \frac{\lambda}{\sqrt{N}}. $$

Lemma (stability of argmax).
If $|E_\mathrm{off}|-|E_\mathrm{on}|\ge \Delta E$ and $A_N\,\Delta E - 2\delta_N \ge \gamma$, then the winner is stable and $G_N\ge\gamma$.

Lemma (two-scale consistency).
Winners at $N$ and $2N$ obey $(s,r)\mapsto(2s{+}c, r)$ with $c\in\{0,1\}$; mates map accordingly.

Theorem (uniform “succeeds-on / fails-off” certificate).
Fix $N,\gamma,d$ and a window $W_t$. If the lemmas hold with constants $(\varepsilon_N,c_N, A_N,\delta_N)$ and
$$ A_N\,(c_N d - 2\varepsilon_N) - 2\delta_N \;\ge\; \gamma,\qquad A_N\,\varepsilon_N + \delta_N \;<\; \gamma, $$
then for all $t\in W_t$,
$$ G_N(\tfrac12,t)\;\ge\;\gamma,\qquad G_N(\tfrac12\pm d, t)\;<\;\gamma. $$
Thus the certificate holds uniformly; two-scale consistency yields uniqueness and robustness.

### Latest generated certification (example)

Path:

```
.out/certs/cert-depth4-N17-20250902-042232.ce1
```

CE1 block:

```
CE1{
  lens=RH_CERT
  mode=Certification
  basis=metanion:pascal_dihedral
  params{ depth=4; N=17; gamma=3; d=0.05; window=0.5; step=0.1 }
  zeros=[14.134725; 21.02204; 25.010858]
  summary{ online_locked_rate=1.0; offline_locked_rate=0.0; online_total=33; offline_total=33; online_locked=33; offline_locked=0 }
  artifact=.out/certs/cert-depth4-N17-20250902-042232.toml
  emit=RiemannHypothesisCertification
}
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
