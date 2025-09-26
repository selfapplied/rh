# Weil Positivity Criterion: Module C Implementation

## Statement

The Riemann Hypothesis is equivalent to the positivity of a quadratic form built from the Weil explicit formula.

## Mathematical Framework

### Definition: Weil Explicit Formula

For an even Schwartz function $\varphi : \mathbb{R} \to \mathbb{R}$, define:

$$\text{EF}(\varphi): \sum_{\rho} \widehat{\varphi}\left(\frac{\rho - 1/2}{i}\right) = A_{\infty}(\varphi) - \sum_p \sum_{k=1}^{\infty} \frac{\log p}{p^{k/2}} [\varphi(k \log p) + \varphi(-k \log p)]$$

where:
- $\rho$ runs over zeta zeros
- $\widehat{\varphi}(\xi) = \int_{-\infty}^{\infty} e^{-2\pi i x \xi} \varphi(x) \, dx$
- $A_{\infty}(\varphi)$ is the archimedean contribution

### Definition: Positivity Quadratic Form

Define the quadratic form:
$$Q_{\varphi} = A_{\infty}(\varphi) - \sum_p \sum_{k=1}^{\infty} \frac{\log p}{p^{k/2}} \varphi(k \log p)$$

### Theorem: Weil Positivity Criterion

**Statement**: RH ⇔ $Q_{\varphi} \geq 0$ for all even Schwartz functions $\varphi$.

**Proof**: This is the standard Weil explicit formula equivalence.

## Construction of Determining Cone

### Step 1: Hermite-Gaussian Test Functions

Start with the cone:
$$C_0 = \{\varphi_{T,m}(x) = e^{-(x/T)^2} H_{2m}(x/T) : T > 0, m \geq 0\}$$

where $H_{2m}$ are even Hermite polynomials.

### Step 2: Closure Operations

Close $C_0$ under:
1. **Convolution**: If $\varphi_1, \varphi_2 \in C$, then $\varphi_1 * \varphi_2 \in C$
2. **$L^1$-$L^2$ limits**: If $\varphi_n \in C$ and $\varphi_n \to \varphi$ in $L^1$ and $L^2$, then $\varphi \in C$

### Step 3: Density Property

**Lemma**: The resulting cone $C$ is determining for the explicit formula.

**Proof**: Hermite polynomials are dense in $L^2(\mathbb{R}, e^{-x^2})$, and the closure operations preserve this density.

## Positivity Proof Strategy

### Method 1: Hermite-Gaussian Kernel Positivity

**Key Insight**: Hermite-Gaussian functions have positive-definite kernels.

**Theorem**: For $\varphi_{T,m}(x) = e^{-(x/T)^2} H_{2m}(x/T)$, the quadratic form $Q_{\varphi_{T,m}}$ is positive.

**Proof Strategy**:
1. **Archimedean term**: Compute $A_{\infty}(\varphi_{T,m})$ explicitly
2. **Prime sum**: Use Meixner decomposition for convergence
3. **Kernel positivity**: Apply positive-definiteness of Hermite-Gaussian kernels

### Method 2: Li Coefficient Equivalence

**Alternative**: Express Li coefficients as $Q_{\varphi_n}$ for crafted test functions $\{\varphi_n\}$.

**Theorem**: RH ⇔ $\lambda_n \geq 0$ for all $n \geq 1$.

**Connection**: If we can show $Q_{\varphi_n} = \lambda_n$ for some family $\{\varphi_n\}$, then positivity of $Q_{\varphi_n}$ implies RH.

### Method 3: Nyman-Beurling Density

**Alternative**: Map test functions to approximants of 1 in $L^2(0,1)$.

**Theorem**: RH ⇔ specific subspace is dense in $L^2(0,1)$.

**Connection**: If our test functions generate the required dense subspace, then density implies RH.

## Implementation Requirements

### Archimedean Control

**Goal**: Compute $A_{\infty}(\varphi)$ explicitly for $\varphi \in C_0$.

**Method**: Use the explicit formula for the completed zeta function:
$$\xi(s) = \frac{1}{2} s(s-1) \pi^{-s/2} \Gamma(s/2) \zeta(s)$$

### Prime Side Convergence

**Goal**: Prove absolute convergence with uniform tail bounds.

**Method**: Use Meixner decomposition and prime counting estimates.

### Positivity Verification

**Goal**: Show $Q_{\varphi} \geq 0$ for $\varphi \in C_0$.

**Method**: Apply positive-definiteness of Hermite-Gaussian kernels.

## No-Go Checks

- **Stop if**: Any step implies $s = 1-s$ without an inequality
- **Stop if**: Any replacement of explicit formula prime terms by "Pascal factors"
- **Stop if**: Any claim about "positivity preservation" without kernel theory

## Next Steps

1. **Choose Module C route**: Weil positivity vs Li vs Nyman-Beurling
2. **Implement archimedean control**: Compute $A_{\infty}(\varphi)$ explicitly
3. **Prove prime convergence**: Use Meixner decomposition
4. **Establish positivity**: Apply kernel theory or equivalence

The proof lives or dies in Module C - this is where to focus next.
