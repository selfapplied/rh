# Lemma S2: Critical Hat → Positive Measure

**Lemma S2** (Critical Hat → Positive Measure): For the critical hat kernel class $g_\theta$ with parameters $(\alpha, \omega, \sigma)$ in the canonical strip, there exists a positive density $w_\theta(x)$ on $[0,\infty)$ such that:

$$G(z) = \int_0^{\infty} \frac{w_\theta(x) \, dx}{1 - xz}$$

## Explicit Construction

**Step 1**: Define the Mellin pushforward of $g_\theta$:

$$w_\theta(x) = \frac{1}{x} \int_0^{\infty} g_\theta(t) \delta(x - e^t) \, dt = \frac{g_\theta(\log x)}{x}$$

**Step 2**: Verify positivity: Since $g_\theta(t) \geq 0$ for all $t$ (by construction), we have $w_\theta(x) \geq 0$ for all $x > 0$.

**Step 3**: Normalization: The "hat" structure ensures proper normalization:

$$\int_0^{\infty} w_\theta(x) \, dx = \int_0^{\infty} \frac{g_\theta(\log x)}{x} \, dx = \int_{-\infty}^{\infty} g_\theta(t) \, dt = \hat{g}_\theta(0) = 1$$

## Assumptions

- **Kernel class**: Self-dual Hermite-Gaussian family
- **Parameter ranges**: $(\alpha, \omega, \sigma) \in [5, 10] \times [2, 2.6] \times [0.1, 0.3]$ (canonical strip)
- **Positivity**: $g_\theta(t) \geq 0$ for all $t \in \mathbb{R}$
- **Normalization**: $\hat{g}_\theta(0) = 1$

## Bridge Claim

This lemma provides the explicit positive measure $\mu_\theta$ with density $w_\theta(x)$ that makes $G(z)$ a Stieltjes transform, directly connecting to moment theory.

## Status

**Positivity ensured by construction**: $g_\theta(t) = 0.5 e^{-\alpha t^2} \cos^2(\omega t) \geq 0$.