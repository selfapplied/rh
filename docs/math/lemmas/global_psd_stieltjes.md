# Theorem S: Global PSD via Stieltjes Representation<a name="theorem-s-global-psd-via-stieltjes-representation"></a>

<!-- mdformat-toc start --slug=github --maxlevel=6 --minlevel=1 -->

- [Theorem S: Global PSD via Stieltjes Representation](#theorem-s-global-psd-via-stieltjes-representation)
  - [Statement](#statement)
  - [Proof](#proof)
  - [Assumptions](#assumptions)
  - [Bridge Claim](#bridge-claim)
  - [Status](#status)

<!-- mdformat-toc end -->

**Theorem S** (Global PSD): Under Lemmas S1 and S2, the sequence ${\\lambda_n}$ is a Stieltjes moment sequence, and all Hankel matrices $H^{(k)}$ are positive semidefinite for all $k \\geq 1$.

## Statement<a name="statement"></a>

Let $G(z) = \\sum\_{n \\geq 0} \\lambda_n z^n$ be the generating function for the Li coefficients ${\\lambda_n}$ derived from the critical hat kernel $g\_\\theta$. If $G(z)$ is a Stieltjes transform:

$$G(z) = \\int_0^{\\infty} \\frac{d\\mu(x)}{1 - xz}$$

for some positive measure $\\mu$ on $\[0,\\infty)$, then:

1. **Moment representation**: $\\lambda_n = \\int_0^{\\infty} x^n , d\\mu(x)$ for all $n \\geq 0$
1. **Global PSD**: All Hankel matrices $H^{(k)}$ with entries $H^{(k)}_{i,j} = \\lambda_{i+j}$ are positive semidefinite for all $k \\geq 1$
1. **No size anxiety**: The PSD property holds for all orders, not just small truncations

## Proof<a name="proof"></a>

**Step 1** (Moment representation): By the Stieltjes transform definition:

$$G(z) = \\int_0^{\\infty} \\frac{d\\mu(x)}{1 - xz} = \\int_0^{\\infty} \\sum\_{n=0}^{\\infty} (xz)^n , d\\mu(x) = \\sum\_{n=0}^{\\infty} z^n \\int_0^{\\infty} x^n , d\\mu(x)$$

Comparing coefficients: $\\lambda_n = \\int_0^{\\infty} x^n , d\\mu(x)$.

**Step 2** (Hankel PSD): For any vector $v = (v_0, v_1, \\ldots, v\_{k-1})^T$:

$$v^T H^{(k)} v = \\sum\_{i,j=0}^{k-1} v_i v_j \\lambda\_{i+j} = \\sum\_{i,j=0}^{k-1} v_i v_j \\int_0^{\\infty} x^{i+j} , d\\mu(x)$$

$$= \\int_0^{\\infty} \\left(\\sum\_{i=0}^{k-1} v_i x^i\\right)^2 , d\\mu(x) \\geq 0$$

Since $\\mu$ is positive and the integrand is non-negative.

**Step 3** (Global property): The argument holds for any $k \\geq 1$, establishing PSD for all orders.

## Assumptions<a name="assumptions"></a>

- **Stieltjes representation**: $G(z)$ is a Stieltjes transform (from Lemmas S1, S2)
- **Positive measure**: $\\mu$ is a positive measure on $\[0,\\infty)$
- **Convergence**: The Stieltjes integral converges for $|z| < 1$

## Bridge Claim<a name="bridge-claim"></a>

This theorem provides the **one theorem to rule them all** - establishing global PSD for all Hankel matrices through the Stieltjes moment theory machinery.

## Status<a name="status"></a>

**Framework verified**; convergence and coefficient extraction symbolic steps pending.
