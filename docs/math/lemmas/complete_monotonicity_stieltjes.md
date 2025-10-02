# Lemma S1: Complete Monotonicity ⇒ Stieltjes Transform<a name="lemma-s1-complete-monotonicity-%E2%87%92-stieltjes-transform"></a>

<!-- mdformat-toc start --slug=github --maxlevel=6 --minlevel=1 -->

- [Lemma S1: Complete Monotonicity ⇒ Stieltjes Transform](#lemma-s1-complete-monotonicity-%E2%87%92-stieltjes-transform)
  - [Proof Strategy](#proof-strategy)
  - [Explicit Construction](#explicit-construction)
  - [Assumptions](#assumptions)
  - [Bridge Claim](#bridge-claim)
  - [Status](#status)

<!-- mdformat-toc end -->

**Lemma S1** (Complete Monotonicity ⇒ Stieltjes): Let $G(z)$ be analytic on $\\mathbb{C} \\setminus \[0,\\infty)$ and suppose that for all $k \\geq 0$ and all $x < 0$:

$$(-1)^k G^{(k)}(x) \\geq 0$$

Then $G(z)$ is a Stieltjes transform:

$$G(z) = \\int_0^{\\infty} \\frac{d\\mu(x)}{1 - xz}$$

for some positive measure $\\mu$ on $\[0,\\infty)$.

## Proof Strategy<a name="proof-strategy"></a>

**Step 1**: Express $G(z)$ as a Laplace-Mellin transform of the critical hat kernel $g\_\\theta(t)$.

**Step 2**: Show complete monotonicity via $g\_\\theta$'s positivity and decay properties.

**Step 3**: Apply Bernstein's theorem to conclude Stieltjes representation.

## Explicit Construction<a name="explicit-construction"></a>

For the critical hat kernel $g\_\\theta(t) = 0.5 \\cdot e^{-\\alpha t^2} \\cos^2(\\omega t)$:

$$G(z) = \\int_0^{\\infty} \\frac{g\_\\theta(\\log x)}{x} \\cdot \\frac{1}{1 - xz} , dx$$

## Assumptions<a name="assumptions"></a>

- **Kernel class**: Hermite-Gaussian family with $\\alpha > 0$, $\\omega \\geq 0$
- **Parameter ranges**: $(\\alpha, \\omega) \\in [\\alpha\_{\\min}, \\alpha\_{\\max}] \\times [0, \\omega\_{\\max}]$
- **Cutoff function**: $\\eta(t)$ smooth, even, with exponential decay
- **Positivity**: $g\_\\theta \\geq 0$ by construction ($\\cos^2$)

## Bridge Claim<a name="bridge-claim"></a>

This lemma connects our kernel-weighted generating function to classical Stieltjes moment theory, establishing the foundation for all-orders PSD Hankel matrices.

## Status<a name="status"></a>

**TODO**: Complete the symbolic derivation showing complete monotonicity of $G^{(k)}(x)$ for $x < 0$.
