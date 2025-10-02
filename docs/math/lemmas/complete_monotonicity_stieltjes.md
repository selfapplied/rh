# Lemma S1: Complete Monotonicity ⇒ Stieltjes Transform

**Lemma S1** (Complete Monotonicity ⇒ Stieltjes): Let $G(z)$ be analytic on $\mathbb{C} \setminus [0,\infty)$ and suppose that for all $k \geq 0$ and all $x < 0$:

$$(-1)^k G^{(k)}(x) \geq 0$$

Then $G(z)$ is a Stieltjes transform:

$$G(z) = \int_0^{\infty} \frac{d\mu(x)}{1 - xz}$$

for some positive measure $\mu$ on $[0,\infty)$.

## Proof Strategy

**Step 1**: Express $G(z)$ as a Laplace-Mellin transform of the critical hat kernel $g_\theta(t)$.

**Step 2**: Show complete monotonicity via $g_\theta$'s positivity and decay properties.

**Step 3**: Apply Bernstein's theorem to conclude Stieltjes representation.

## Explicit Construction

For the critical hat kernel $g_\theta(t) = 0.5 \cdot e^{-\alpha t^2} \cos^2(\omega t)$:

$$G(z) = \int_0^{\infty} \frac{g_\theta(\log x)}{x} \cdot \frac{1}{1 - xz} \, dx$$

## Assumptions

- **Kernel class**: Hermite-Gaussian family with $\alpha > 0$, $\omega \geq 0$
- **Parameter ranges**: $(\alpha, \omega) \in [\alpha_{\min}, \alpha_{\max}] \times [0, \omega_{\max}]$
- **Cutoff function**: $\eta(t)$ smooth, even, with exponential decay
- **Positivity**: $g_\theta \geq 0$ by construction ($\cos^2$)

## Bridge Claim

This lemma connects our kernel-weighted generating function to classical Stieltjes moment theory, establishing the foundation for all-orders PSD Hankel matrices.

## Status

**TODO**: Complete the symbolic derivation showing complete monotonicity of $G^{(k)}(x)$ for $x < 0$.
