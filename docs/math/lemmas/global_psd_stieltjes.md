# Theorem S: Global PSD via Stieltjes Representation

**Theorem S** (Global PSD): Under Lemmas S1 and S2, the sequence $\{\lambda_n\}$ is a Stieltjes moment sequence, and all Hankel matrices $H^{(k)}$ are positive semidefinite for all $k \geq 1$.

## Statement

Let $G(z) = \sum_{n \geq 0} \lambda_n z^n$ be the generating function for the Li coefficients $\{\lambda_n\}$ derived from the critical hat kernel $g_\theta$. If $G(z)$ is a Stieltjes transform:

$$G(z) = \int_0^{\infty} \frac{d\mu(x)}{1 - xz}$$

for some positive measure $\mu$ on $[0,\infty)$, then:

1. **Moment representation**: $\lambda_n = \int_0^{\infty} x^n \, d\mu(x)$ for all $n \geq 0$
2. **Global PSD**: All Hankel matrices $H^{(k)}$ with entries $H^{(k)}_{i,j} = \lambda_{i+j}$ are positive semidefinite for all $k \geq 1$
3. **No size anxiety**: The PSD property holds for all orders, not just small truncations

## Proof

**Step 1** (Moment representation): By the Stieltjes transform definition:

$$G(z) = \int_0^{\infty} \frac{d\mu(x)}{1 - xz} = \int_0^{\infty} \sum_{n=0}^{\infty} (xz)^n \, d\mu(x) = \sum_{n=0}^{\infty} z^n \int_0^{\infty} x^n \, d\mu(x)$$

Comparing coefficients: $\lambda_n = \int_0^{\infty} x^n \, d\mu(x)$.

**Step 2** (Hankel PSD): For any vector $v = (v_0, v_1, \ldots, v_{k-1})^T$:

$$v^T H^{(k)} v = \sum_{i,j=0}^{k-1} v_i v_j \lambda_{i+j} = \sum_{i,j=0}^{k-1} v_i v_j \int_0^{\infty} x^{i+j} \, d\mu(x)$$

$$= \int_0^{\infty} \left(\sum_{i=0}^{k-1} v_i x^i\right)^2 \, d\mu(x) \geq 0$$

Since $\mu$ is positive and the integrand is non-negative.

**Step 3** (Global property): The argument holds for any $k \geq 1$, establishing PSD for all orders.

## Assumptions

- **Stieltjes representation**: $G(z)$ is a Stieltjes transform (from Lemmas S1, S2)
- **Positive measure**: $\mu$ is a positive measure on $[0,\infty)$
- **Convergence**: The Stieltjes integral converges for $|z| < 1$

## Bridge Claim

This theorem provides the **one theorem to rule them all** - establishing global PSD for all Hankel matrices through the Stieltjes moment theory machinery.

## Status

**TODO**: Complete the symbolic verification that the Stieltjes integral converges and produces the correct moment sequence.