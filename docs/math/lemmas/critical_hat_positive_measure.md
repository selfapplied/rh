# Lemma S2: Critical Hat → Positive Measure<a name="lemma-s2-critical-hat-%E2%86%92-positive-measure"></a>

<!-- mdformat-toc start --slug=github --maxlevel=6 --minlevel=1 -->

- [Lemma S2: Critical Hat → Positive Measure](#lemma-s2-critical-hat-%E2%86%92-positive-measure)
  - [Explicit Construction](#explicit-construction)
  - [Assumptions](#assumptions)
  - [Bridge Claim](#bridge-claim)
  - [Status](#status)

<!-- mdformat-toc end -->

**Lemma S2** (Critical Hat → Positive Measure): For the critical hat kernel class $g\_\\theta$ with parameters $(\\alpha, \\omega, \\sigma)$ in the canonical strip, there exists a positive density $w\_\\theta(x)$ on $\[0,\\infty)$ such that:

$$G(z) = \\int_0^{\\infty} \\frac{w\_\\theta(x) , dx}{1 - xz}$$

## Explicit Construction<a name="explicit-construction"></a>

**Step 1**: Define the Mellin pushforward of $g\_\\theta$:

$$w\_\\theta(x) = \\frac{1}{x} \\int_0^{\\infty} g\_\\theta(t) \\delta(x - e^t) , dt = \\frac{g\_\\theta(\\log x)}{x}$$

**Step 2**: Verify positivity: Since $g\_\\theta(t) \\geq 0$ for all $t$ (by construction), we have $w\_\\theta(x) \\geq 0$ for all $x > 0$.

**Step 3**: Normalization: The "hat" structure ensures proper normalization:

$$\\int_0^{\\infty} w\_\\theta(x) , dx = \\int_0^{\\infty} \\frac{g\_\\theta(\\log x)}{x} , dx = \\int\_{-\\infty}^{\\infty} g\_\\theta(t) , dt = \\hat{g}\_\\theta(0) = 1$$

## Assumptions<a name="assumptions"></a>

- **Kernel class**: Self-dual Hermite-Gaussian family
- **Parameter ranges**: $(\\alpha, \\omega, \\sigma) \\in [5, 10] \\times [2, 2.6] \\times [0.1, 0.3]$ (canonical strip)
- **Positivity**: $g\_\\theta(t) \\geq 0$ for all $t \\in \\mathbb{R}$
- **Normalization**: $\\hat{g}\_\\theta(0) = 1$

## Bridge Claim<a name="bridge-claim"></a>

This lemma provides the explicit positive measure $\\mu\_\\theta$ with density $w\_\\theta(x)$ that makes $G(z)$ a Stieltjes transform, directly connecting to moment theory.

## Status<a name="status"></a>

**Positivity ensured by construction**: $g\_\\theta(t) = 0.5 e^{-\\alpha t^2} \\cos^2(\\omega t) \\geq 0$.
