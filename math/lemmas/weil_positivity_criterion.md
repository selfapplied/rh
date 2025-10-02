# Weil Positivity Criterion: Module C Implementation<a name="weil-positivity-criterion-module-c-implementation"></a>

<!-- mdformat-toc start --slug=github --maxlevel=6 --minlevel=1 -->

- [Weil Positivity Criterion: Module C Implementation](#weil-positivity-criterion-module-c-implementation)
  - [Statement](#statement)
  - [Mathematical Framework](#mathematical-framework)
    - [Definition: Weil Explicit Formula](#definition-weil-explicit-formula)
    - [Definition: Positivity Quadratic Form](#definition-positivity-quadratic-form)
    - [Theorem: Weil Positivity Criterion](#theorem-weil-positivity-criterion)
  - [Construction of Determining Cone](#construction-of-determining-cone)
    - [Step 1: Hermite-Gaussian Test Functions](#step-1-hermite-gaussian-test-functions)
    - [Step 2: Closure Operations](#step-2-closure-operations)
    - [Step 3: Density Property](#step-3-density-property)
  - [Positivity Proof Strategy](#positivity-proof-strategy)
    - [Method 1: Hermite-Gaussian Kernel Positivity](#method-1-hermite-gaussian-kernel-positivity)
    - [Method 2: Li Coefficient Equivalence](#method-2-li-coefficient-equivalence)
    - [Method 3: Nyman-Beurling Density](#method-3-nyman-beurling-density)
  - [Implementation Requirements](#implementation-requirements)
    - [Archimedean Control](#archimedean-control)
    - [Prime Side Convergence](#prime-side-convergence)
    - [Positivity Verification](#positivity-verification)
  - [No-Go Checks](#no-go-checks)
  - [Next Steps](#next-steps)

<!-- mdformat-toc end -->

## Statement<a name="statement"></a>

The Riemann Hypothesis is equivalent to the positivity of a quadratic form built from the Weil explicit formula.

## Mathematical Framework<a name="mathematical-framework"></a>

### Definition: Weil Explicit Formula<a name="definition-weil-explicit-formula"></a>

For an even Schwartz function $\\varphi : \\mathbb{R} \\to \\mathbb{R}$, define:

$$\\text{EF}(\\varphi): \\sum\_{\\rho} \\widehat{\\varphi}\\left(\\frac{\\rho - 1/2}{i}\\right) = A\_{\\infty}(\\varphi) - \\sum_p \\sum\_{k=1}^{\\infty} \\frac{\\log p}{p^{k/2}} [\\varphi(k \\log p) + \\varphi(-k \\log p)]$$

where:

- $\\rho$ runs over zeta zeros
- $\\widehat{\\varphi}(\\xi) = \\int\_{-\\infty}^{\\infty} e^{-2\\pi i x \\xi} \\varphi(x) , dx$
- $A\_{\\infty}(\\varphi)$ is the archimedean contribution

### Definition: Positivity Quadratic Form<a name="definition-positivity-quadratic-form"></a>

Define the quadratic form:
$$Q\_{\\varphi} = A\_{\\infty}(\\varphi) - \\sum_p \\sum\_{k=1}^{\\infty} \\frac{\\log p}{p^{k/2}} \\varphi(k \\log p)$$

### Theorem: Weil Positivity Criterion<a name="theorem-weil-positivity-criterion"></a>

**Statement**: RH ⇔ $Q\_{\\varphi} \\geq 0$ for all even Schwartz functions $\\varphi$.

**Proof**: This is the standard Weil explicit formula equivalence.

## Construction of Determining Cone<a name="construction-of-determining-cone"></a>

### Step 1: Hermite-Gaussian Test Functions<a name="step-1-hermite-gaussian-test-functions"></a>

Start with the cone:
$$C_0 = {\\varphi\_{T,m}(x) = e^{-(x/T)^2} H\_{2m}(x/T) : T > 0, m \\geq 0}$$

where $H\_{2m}$ are even Hermite polynomials.

### Step 2: Closure Operations<a name="step-2-closure-operations"></a>

Close $C_0$ under:

1. **Convolution**: If $\\varphi_1, \\varphi_2 \\in C$, then $\\varphi_1 * \\varphi_2 \\in C$
1. **$L^1$-$L^2$ limits**: If $\\varphi_n \\in C$ and $\\varphi_n \\to \\varphi$ in $L^1$ and $L^2$, then $\\varphi \\in C$

### Step 3: Density Property<a name="step-3-density-property"></a>

**Lemma**: The resulting cone $C$ is determining for the explicit formula.

**Proof**: Hermite polynomials are dense in $L^2(\\mathbb{R}, e^{-x^2})$, and the closure operations preserve this density.

## Positivity Proof Strategy<a name="positivity-proof-strategy"></a>

### Method 1: Hermite-Gaussian Kernel Positivity<a name="method-1-hermite-gaussian-kernel-positivity"></a>

**Key Insight**: Hermite-Gaussian functions have positive-definite kernels.

**Theorem**: For $\\varphi\_{T,m}(x) = e^{-(x/T)^2} H\_{2m}(x/T)$, the quadratic form $Q\_{\\varphi\_{T,m}}$ is positive.

**Proof Strategy**:

1. **Archimedean term**: Compute $A\_{\\infty}(\\varphi\_{T,m})$ explicitly
1. **Prime sum**: Use Meixner decomposition for convergence
1. **Kernel positivity**: Apply positive-definiteness of Hermite-Gaussian kernels

### Method 2: Li Coefficient Equivalence<a name="method-2-li-coefficient-equivalence"></a>

**Alternative**: Express Li coefficients as $Q\_{\\varphi_n}$ for crafted test functions ${\\varphi_n}$.

**Theorem**: RH ⇔ $\\lambda_n \\geq 0$ for all $n \\geq 1$.

**Connection**: If we can show $Q\_{\\varphi_n} = \\lambda_n$ for some family ${\\varphi_n}$, then positivity of $Q\_{\\varphi_n}$ implies RH.

### Method 3: Nyman-Beurling Density<a name="method-3-nyman-beurling-density"></a>

**Alternative**: Map test functions to approximants of 1 in $L^2(0,1)$.

**Theorem**: RH ⇔ specific subspace is dense in $L^2(0,1)$.

**Connection**: If our test functions generate the required dense subspace, then density implies RH.

## Implementation Requirements<a name="implementation-requirements"></a>

### Archimedean Control<a name="archimedean-control"></a>

**Goal**: Compute $A\_{\\infty}(\\varphi)$ explicitly for $\\varphi \\in C_0$.

**Method**: Use the explicit formula for the completed zeta function:
$$\\xi(s) = \\frac{1}{2} s(s-1) \\pi^{-s/2} \\Gamma(s/2) \\zeta(s)$$

### Prime Side Convergence<a name="prime-side-convergence"></a>

**Goal**: Prove absolute convergence with uniform tail bounds.

**Method**: Use Meixner decomposition and prime counting estimates.

### Positivity Verification<a name="positivity-verification"></a>

**Goal**: Show $Q\_{\\varphi} \\geq 0$ for $\\varphi \\in C_0$.

**Method**: Apply positive-definiteness of Hermite-Gaussian kernels.

## No-Go Checks<a name="no-go-checks"></a>

- **Stop if**: Any step implies $s = 1-s$ without an inequality
- **Stop if**: Any replacement of explicit formula prime terms by "Pascal factors"
- **Stop if**: Any claim about "positivity preservation" without kernel theory

## Next Steps<a name="next-steps"></a>

1. **Choose Module C route**: Weil positivity vs Li vs Nyman-Beurling
1. **Implement archimedean control**: Compute $A\_{\\infty}(\\varphi)$ explicitly
1. **Prove prime convergence**: Use Meixner decomposition
1. **Establish positivity**: Apply kernel theory or equivalence

The proof lives or dies in Module C - this is where to focus next.
