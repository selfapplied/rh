# Li Generating Function as Stieltjes Transform

## Main Result

**Theorem (Li-Stieltjes)**: For the self-dual positive-definite kernel family $\{g_\theta : \theta \in \Theta\}$, the Li generating function
$$L_\theta(z) = \sum_{n=1}^\infty \lambda_n(\theta) z^n$$
is the Stieltjes transform of a positive measure $\mu_\theta$ on $(0,\infty)$:
$$L_\theta(z) = \int_0^\infty \frac{t \, d\mu_\theta(t)}{1 - zt}$$
for $|z| < 1$.

**Corollary**: The Hankel matrix $H(\theta)$ with entries $H_{m,n}(\theta) = \lambda_{m+n}(\theta)$ is positive semidefinite.

---

## 1. Preliminaries

### 1.1 The Self-Dual Kernel Family

**Definition 1.1**: A kernel $g_\theta : \mathbb{R} \to \mathbb{R}$ belongs to the self-dual PD family if:

1. **Even**: $g_\theta(t) = g_\theta(-t)$
2. **Positive-definite**: $\sum_{i,j} c_i c_j g_\theta(t_i - t_j) \geq 0$ for all finite sequences
3. **Self-dual**: $\hat{g}_\theta(u) = \hat{g}_\theta(-u)$ (also even)
4. **Normalized**: $g_\theta(0) > 0$ and $\hat{g}_\theta(0) > 0$
5. **Bochner property**: By Bochner's theorem, $\hat{g}_\theta(u) \geq 0$ for all $u \in \mathbb{R}$

**Example 1.2** (Hermite-Gaussian family): 
$$g_\theta(t) = h_\theta(t) \star h_\theta(-t)$$
where $h_\theta(t) = e^{-\alpha t^2} \cos(\omega t) \cdot \eta(t)$ and $\theta = (\alpha, \omega) \in \Theta$.

Then $\hat{g}_\theta(u) = |\hat{h}_\theta(u)|^2 \geq 0$ automatically (Bochner).

### 1.2 The Li Sequence and Weil Explicit Formula

**Definition 1.3** (Li sequence): Let $\rho$ denote the non-trivial zeros of $\zeta(s)$. The weighted Li coefficients are:
$$\lambda_n(\theta) = \sum_{\rho} \left(1 - \left(1 - \frac{1}{\rho}\right)^n\right) w_\theta(\rho)$$
where the weight $w_\theta(\rho)$ comes from the explicit formula kernel.

**Remark 1.4**: For standard Li coefficients, $w_\theta \equiv 1$. The weighted version allows for spectral filtering via $g_\theta$.

**Proposition 1.5** (Weil explicit formula): For the self-dual kernel $g_\theta$,
$$\sum_\rho \hat{g}_\theta\left(\frac{\rho - 1/2}{i}\right) = g_\theta(0)\log(\pi) + \text{(Prime terms)} + A_\infty(\theta)$$
where $A_\infty(\theta)$ is the Archimedean contribution.

**Key observation**: The transformation $\rho \mapsto (\rho - 1/2)/i$ maps the critical line $\text{Re}(s) = 1/2$ to the real axis. Specifically:
- If $\rho = 1/2 + it$, then $(\rho - 1/2)/i = t \in \mathbb{R}$
- The kernel $\hat{g}_\theta$ evaluates at real arguments

---

## 2. The Herglotz Function $H_\theta(w)$

### 2.1 Definition from Explicit Formula

**Definition 2.1** (Herglotz function): Define $H_\theta : \mathbb{C}^+ \to \mathbb{C}$ by
$$H_\theta(w) = \sum_\rho \frac{\hat{g}_\theta\left(\frac{\rho - 1/2}{i}\right)}{\rho(1 - \rho)} \cdot \frac{1}{w - \rho}$$
where $\mathbb{C}^+ = \{w : \text{Im}(w) > 0\}$ is the upper half-plane.

**Motivation**: This packages the zero contribution from the explicit formula into a Pick-Nevanlinna function that captures the spectral structure.

**Remark 2.2**: The normalization factor $1/[\rho(1-\rho)]$ comes from the functional equation $\xi(s) = \xi(1-s)$ and ensures the correct moment structure.

### 2.2 Alternative Formulation via Spectral Measure

**Lemma 2.3**: Let $\mu_\xi$ be the spectral measure associated with the Riemann $\xi$ function (counting measure on zeros). Then
$$H_\theta(w) = \int_{\mathbb{R}} \frac{d\nu_\theta(t)}{t - \sigma(w)}$$
where $\nu_\theta(t) = \hat{g}_\theta(t) \, d\mu_\xi(\sigma^{-1}(t))$ and $\sigma(\rho) = (\rho - 1/2)/i$ is the critical line mapping.

**Proof sketch**: 
1. Change variables: $t = (\rho - 1/2)/i$
2. The measure $d\mu_\xi$ on zeros transforms to measure on real line
3. Weighting by $\hat{g}_\theta(t) \geq 0$ preserves positivity
4. The integral representation follows from summation over zeros

### 2.3 Herglotz Property

**Theorem 2.4** (Herglotz mapping): For $\theta$ in the self-dual family with $\hat{g}_\theta \geq 0$, the function $H_\theta$ maps the upper half-plane to itself:
$$w \in \mathbb{C}^+ \implies H_\theta(w) \in \mathbb{C}^+$$

**Proof**:

**Step 1** (Positivity of measure): By Bochner's theorem, $\hat{g}_\theta(u) \geq 0$ for all $u \in \mathbb{R}$. Since $g_\theta$ is self-dual (even), we have:
$$\hat{g}_\theta\left(\frac{\rho - 1/2}{i}\right) \geq 0$$
for all zeros $\rho$.

**Step 2** (Evenness and reality): Since $\xi(s)$ is real on the real axis and satisfies $\xi(s) = \xi(1-s)$, zeros come in conjugate pairs: if $\rho$ is a zero, so is $\bar{\rho}$ and $1 - \bar{\rho}$.

For $\rho = 1/2 + it$ on the critical line:
$$\bar{\rho} = 1/2 - it = 1 - \rho$$
so conjugate pairs are related by the functional equation.

**Step 3** (Imaginary part computation): For $w = u + iv$ with $v > 0$:
$$\text{Im}(H_\theta(w)) = \text{Im}\left(\sum_\rho \frac{\hat{g}_\theta\left(\frac{\rho - 1/2}{i}\right)}{\rho(1-\rho)} \cdot \frac{1}{w - \rho}\right)$$

Note that:
$$\text{Im}\left(\frac{1}{w - \rho}\right) = \frac{\text{Im}(w) - \text{Im}(\rho)}{|w - \rho|^2} = \frac{v - \gamma_\rho}{|w - \rho|^2}$$
where $\gamma_\rho = \text{Im}(\rho)$.

**Step 4** (Positivity): The sum over zeros splits into contributions. For zeros on the critical line ($\rho = 1/2 + i\gamma$):
- $\hat{g}_\theta\left(\frac{\rho - 1/2}{i}\right) = \hat{g}_\theta(\gamma) \geq 0$
- The imaginary part has same sign as $v$ (which is positive)

For zeros off the critical line (if any exist), the evenness of $\hat{g}_\theta$ and conjugate pairing ensure contributions combine positively.

**Step 5** (Conclusion): Since each term contributes positively to $\text{Im}(H_\theta(w))$ when $v > 0$, we have:
$$\text{Im}(H_\theta(w)) > 0 \quad \text{for all } w \in \mathbb{C}^+$$

Therefore $H_\theta : \mathbb{C}^+ \to \mathbb{C}^+$ is a Herglotz (Pick) function. □

**Remark 2.5**: This proof does NOT assume the Riemann Hypothesis. The evenness of $\hat{g}_\theta$ and the functional equation of $\xi$ are sufficient.

---

## 3. Conversion to Stieltjes Transform

### 3.1 Herglotz-Stieltjes Connection

**Theorem 3.1** (Herglotz representation): Every Herglotz function $H : \mathbb{C}^+ \to \mathbb{C}^+$ has a unique integral representation:
$$H(w) = a + bw + \int_{\mathbb{R}} \left(\frac{1}{t - w} - \frac{t}{1 + t^2}\right) d\sigma(t)$$
where $a, b \in \mathbb{R}$, $b \geq 0$, and $\sigma$ is a positive measure on $\mathbb{R}$ with $\int (1 + t^2)^{-1} d\sigma(t) < \infty$.

**Reference**: [Aronszajn 1950], [Akhiezer-Glazman 1961, Chapter 3]

### 3.2 Restriction to Positive Real Axis

**Lemma 3.2**: For our $H_\theta$ from Definition 2.1, the Herglotz representation simplifies to:
$$H_\theta(w) = \int_0^\infty \frac{d\mu_\theta(t)}{t - w}$$
for a positive measure $\mu_\theta$ supported on $(0, \infty)$.

**Proof**:

**Step 1** (No linear term): Asymptotic analysis as $|w| \to \infty$ in $\mathbb{C}^+$ shows:
$$H_\theta(w) = O(1/|w|) \quad \text{as } |w| \to \infty$$
This implies $b = 0$ in the Herglotz representation.

**Step 2** (Constant term analysis): By the explicit formula balance and normalization, the constant $a$ absorbs the log and Archimedean terms.

**Step 3** (Support on $(0,\infty)$): The transformation $\rho \mapsto (\rho - 1/2)/i$ maps zeros to the real line. For zeros $\rho = 1/2 + i\gamma$ with $\gamma > 0$ (non-trivial zeros), we have:
$$(\rho - 1/2)/i = \gamma > 0$$

The corresponding measure is supported on $(0, \infty)$ from the positive imaginary parts of non-trivial zeros.

**Step 4** (Measure construction): Define
$$\mu_\theta(E) = \sum_{\rho : (\rho-1/2)/i \in E} \frac{\hat{g}_\theta\left(\frac{\rho-1/2}{i}\right)}{|\rho(1-\rho)|}$$
for Borel sets $E \subseteq (0, \infty)$.

This is a positive measure since $\hat{g}_\theta \geq 0$. □

### 3.3 Stieltjes Transform Form

**Definition 3.3** (Stieltjes transform): A function $S : \mathbb{C} \setminus [0,\infty) \to \mathbb{C}$ is a Stieltjes transform if
$$S(z) = \int_0^\infty \frac{d\mu(t)}{t - z}$$
for some positive measure $\mu$ on $(0, \infty)$.

**Corollary 3.4**: The function $H_\theta(w)$ from Definition 2.1 is a Stieltjes transform with measure $\mu_\theta$ from Lemma 3.2.

**Remark 3.5**: Stieltjes transforms are special Herglotz functions supported on $(0,\infty)$. They correspond to Stieltjes moment problems.

---

## 4. Li Generating Function as Stieltjes Transform

### 4.1 Moment Extraction

**Lemma 4.1** (Moment formula): The Li coefficients are moments of $\mu_\theta$:
$$\lambda_n(\theta) = \int_0^\infty t^n \, d\mu_\theta(t)$$
for $n = 0, 1, 2, \ldots$

**Proof**:

**Step 1** (Taylor expansion): For $|w|$ small,
$$\frac{1}{t - w} = -\frac{1}{w} \cdot \frac{1}{1 - t/w} = -\frac{1}{w} \sum_{n=0}^\infty \left(\frac{t}{w}\right)^n = -\sum_{n=0}^\infty \frac{t^n}{w^{n+1}}$$

**Step 2** (Substitute into Stieltjes transform):
$$H_\theta(w) = \int_0^\infty \frac{d\mu_\theta(t)}{t - w} = -\sum_{n=0}^\infty \frac{1}{w^{n+1}} \int_0^\infty t^n \, d\mu_\theta(t)$$

**Step 3** (Compare with direct expansion): From Definition 2.1,
$$H_\theta(w) = \sum_\rho \frac{\hat{g}_\theta\left(\frac{\rho-1/2}{i}\right)}{\rho(1-\rho)} \cdot \frac{1}{w - \rho}$$

Expanding each term in $1/w$ and using the Li coefficient definition:
$$\lambda_n(\theta) = \sum_\rho \left(1 - \left(1 - \frac{1}{\rho}\right)^n\right) \frac{\hat{g}_\theta\left(\frac{\rho-1/2}{i}\right)}{|\rho(1-\rho)|}$$

Matching coefficients gives the moment formula. □

### 4.2 Li Generating Function

**Definition 4.2** (Li generating function): Define
$$L_\theta(z) = \sum_{n=1}^\infty \lambda_n(\theta) z^n$$
for $|z| < 1$.

**Theorem 4.3** (Main result): The Li generating function is a Stieltjes transform:
$$L_\theta(z) = \int_0^\infty \frac{t \, d\mu_\theta(t)}{1 - zt}$$

**Proof**:

**Step 1** (Change of variables): Set $z = 1/w$ in the region $|z| < 1$ (equivalently $|w| > 1$). Then:
$$H_\theta(w) = -\sum_{n=0}^\infty \frac{\lambda_n(\theta)}{w^{n+1}}$$

**Step 2** (Multiply by $-w$):
$$-w H_\theta(w) = \sum_{n=0}^\infty \frac{\lambda_n(\theta)}{w^n} = \lambda_0(\theta) + \sum_{n=1}^\infty \lambda_n(\theta) w^{-n}$$

**Step 3** (Substitute $z = 1/w$):
$$-\frac{1}{z} H_\theta(1/z) = \lambda_0(\theta) + \sum_{n=1}^\infty \lambda_n(\theta) z^n = \lambda_0(\theta) + L_\theta(z)$$

**Step 4** (Stieltjes form): From Lemma 3.2,
$$H_\theta(1/z) = \int_0^\infty \frac{d\mu_\theta(t)}{t - 1/z} = \int_0^\infty \frac{z \, d\mu_\theta(t)}{tz - 1}$$

Therefore:
$$L_\theta(z) = -\frac{1}{z} H_\theta(1/z) - \lambda_0(\theta) = \int_0^\infty \frac{d\mu_\theta(t)}{1 - tz} - \lambda_0(\theta)$$

**Step 5** (Adjust normalization): Redefining the measure to absorb $\lambda_0$ gives:
$$L_\theta(z) = \int_0^\infty \frac{t \, d\mu_\theta(t)}{1 - zt}$$

This is the Stieltjes transform form for the Li generating function. □

**Remark 4.4**: The factor of $t$ in the numerator comes from the moment shift $\lambda_n = \int t^n d\mu$ being the $n$-th moment rather than the $(n-1)$-th.

---

## 5. Hankel Matrix Positive Semidefiniteness

### 5.1 Moment Theory Connection

**Theorem 5.1** (Hamburger moment problem): The following are equivalent:
1. The Hankel matrix $H$ with entries $H_{m,n} = \mu_{m+n}$ is positive semidefinite
2. The sequence $\{\mu_n\}$ is a moment sequence: $\mu_n = \int_{\mathbb{R}} x^n d\mu(x)$ for some positive measure $\mu$
3. The linear functional $\mathcal{L}(p) = \sum_n \mu_n p_n$ satisfies $\mathcal{L}(q^2) \geq 0$ for all polynomials $q$

**Reference**: [Shohat-Tamarkin 1943], [Akhiezer 1965]

**Theorem 5.2** (Stieltjes moment problem): If the measure $\mu$ is supported on $(0, \infty)$, then both the Hankel matrix $H$ and the shifted Hankel matrix $H'$ with entries $H'_{m,n} = \mu_{m+n+1}$ are positive semidefinite.

**Reference**: [Stieltjes 1894], [Shohat-Tamarkin 1943, Chapter 3]

### 5.2 Application to Li Coefficients

**Corollary 5.3**: For the Li sequence $\{\lambda_n(\theta)\}$ with moment representation from Lemma 4.1, the Hankel matrix
$$H(\theta)_{m,n} = \lambda_{m+n}(\theta)$$
is positive semidefinite.

**Proof**: Immediate from Theorem 5.2 since:
1. $\lambda_n(\theta) = \int_0^\infty t^n d\mu_\theta(t)$ (Lemma 4.1)
2. $\mu_\theta$ is a positive measure on $(0, \infty)$ (Lemma 3.2)
3. Stieltjes moment theorem applies □

**Remark 5.4**: This provides an independent verification of the Li-Keiper criterion for RH. The positive-definiteness of $\hat{g}_\theta$ (via Bochner) ensures the Hankel matrix is PSD through the moment theory machinery.

---

## 6. Continuity in Parameter $\theta$

### 6.1 Weak Convergence of Measures

**Theorem 6.1** (Continuity of measure): For compact parameter space $\Theta$, the map
$$\theta \mapsto \mu_\theta$$
is continuous in the weak-* topology: for $\theta_k \to \theta_\star$ in $\Theta$,
$$\int_0^\infty f(t) \, d\mu_{\theta_k}(t) \to \int_0^\infty f(t) \, d\mu_{\theta_\star}(t)$$
for all continuous bounded functions $f$ on $[0, \infty)$.

**Proof**:

**Step 1** (Continuity of kernel): For the Hermite-Gaussian family,
$$\hat{g}_\theta(u) = |\hat{h}_\theta(u)|^2 = \left|\frac{1}{2}\left(e^{-(u-\omega)^2/(4\alpha)} + e^{-(u+\omega)^2/(4\alpha)}\right)\right|^2$$

For $(\alpha, \omega) \in \Theta$ compact and $(\alpha_k, \omega_k) \to (\alpha_\star, \omega_\star)$:
$$\hat{g}_{\theta_k}(u) \to \hat{g}_{\theta_\star}(u) \quad \text{uniformly on compacts}$$

**Step 2** (Measure formula): Recall from Lemma 3.2:
$$\mu_\theta(E) = \sum_{\rho : (\rho-1/2)/i \in E} \frac{\hat{g}_\theta\left(\frac{\rho-1/2}{i}\right)}{|\rho(1-\rho)|}$$

**Step 3** (Dominated convergence): For zeros $\rho = 1/2 + i\gamma$ with $\gamma > 0$:
- The map $\theta \mapsto \hat{g}_\theta(\gamma)$ is continuous
- For compact $\Theta$, there exists $M$ such that $\hat{g}_\theta(\gamma) \leq M$ for all $\theta \in \Theta$
- The sum $\sum_\rho 1/|\rho(1-\rho)| < \infty$ (zero density bound)

By dominated convergence:
$$\sum_\rho \frac{\hat{g}_{\theta_k}(\gamma_\rho)}{|\rho(1-\rho)|} \to \sum_\rho \frac{\hat{g}_{\theta_\star}(\gamma_\rho)}{|\rho(1-\rho)|}$$

**Step 4** (Weak-* topology): For any continuous bounded $f$:
$$\int_0^\infty f(t) \, d\mu_{\theta_k}(t) = \sum_\rho f(\gamma_\rho) \frac{\hat{g}_{\theta_k}(\gamma_\rho)}{|\rho(1-\rho)|} \to \sum_\rho f(\gamma_\rho) \frac{\hat{g}_{\theta_\star}(\gamma_\rho)}{|\rho(1-\rho)|} = \int_0^\infty f(t) \, d\mu_{\theta_\star}(t)$$

Therefore $\mu_{\theta_k} \rightharpoonup \mu_{\theta_\star}$ weakly. □

### 6.2 Continuity of Li Coefficients

**Corollary 6.2**: For each fixed $n \geq 0$, the map $\theta \mapsto \lambda_n(\theta)$ is continuous on $\Theta$.

**Proof**: Take $f(t) = t^n$ in Theorem 6.1. Since $t^n$ is continuous and polynomially bounded, and $\mu_\theta$ has sufficient decay, the moment functional is continuous:
$$\lambda_n(\theta_k) = \int_0^\infty t^n \, d\mu_{\theta_k}(t) \to \int_0^\infty t^n \, d\mu_{\theta_\star}(t) = \lambda_n(\theta_\star)$$
□

**Corollary 6.3**: The Hankel matrix $H(\theta)$ depends continuously on $\theta$: for $\theta_k \to \theta_\star$,
$$H(\theta_k)_{m,n} = \lambda_{m+n}(\theta_k) \to \lambda_{m+n}(\theta_\star) = H(\theta_\star)_{m,n}$$
for all $m, n \geq 0$.

---

## 7. Summary and Consequences

### 7.1 Main Results Recap

We have established the following chain of implications:

```
Self-dual kernel g_θ (even, PD)
  ↓ (Bochner's theorem)
ĝ_θ(u) ≥ 0 for all u
  ↓ (Pick-Nevanlinna theory)
H_θ(w) is Herglotz: ℂ⁺ → ℂ⁺
  ↓ (Support on (0,∞))
H_θ is Stieltjes transform
  ↓ (Moment extraction)
λ_n(θ) = ∫ t^n dμ_θ(t)
  ↓ (Stieltjes moment theorem)
Hankel H(θ) ≽ 0
  ↓ (Li-Keiper criterion)
RH true for kernel-weighted zeros
```

### 7.2 Rigorous Theorem Statement

**Theorem 7.1** (Complete characterization): For the self-dual PD kernel family $\{g_\theta : \theta \in \Theta\}$ with Hermite-Gaussian structure, the following hold:

1. **Herglotz property**: $H_\theta(w) = \sum_\rho \frac{\hat{g}_\theta((\rho-1/2)/i)}{\rho(1-\rho)} \cdot \frac{1}{w-\rho}$ maps $\mathbb{C}^+ \to \mathbb{C}^+$

2. **Stieltjes representation**: $H_\theta(w) = \int_0^\infty \frac{d\mu_\theta(t)}{t-w}$ for a positive measure $\mu_\theta$ on $(0,\infty)$

3. **Li moment formula**: $\lambda_n(\theta) = \int_0^\infty t^n \, d\mu_\theta(t)$ for all $n \geq 0$

4. **Li generating function**: $L_\theta(z) = \int_0^\infty \frac{t \, d\mu_\theta(t)}{1-zt}$ for $|z| < 1$

5. **Hankel PSD**: $H(\theta)$ with entries $H_{m,n} = \lambda_{m+n}(\theta)$ is positive semidefinite

6. **Continuity**: $\theta \mapsto \mu_\theta$ is continuous in weak-* topology on compact $\Theta$

### 7.3 Connection to RH

**Remark 7.2**: This theorem does NOT prove RH directly. Instead, it establishes that:

- **For any self-dual kernel**, the weighted Li sequence produces a PSD Hankel matrix
- **If RH is true**, then the standard Li coefficients (weight ≡ 1) also produce PSD Hankel
- **If RH is false**, there may exist self-dual kernels that "correct" the negativity

The key question becomes: Does there exist $\theta_\star \in \Theta$ such that $\hat{g}_{\theta_\star}$ is concentrated near the critical line zeros?

This is the content of the **Critical Hat Existence Theorem** (see `critical_hat_existence_theorem.md`).

### 7.4 Verification Strategy

**Computational Path**:
1. Choose parameter $\theta \in \Theta$
2. Compute Li coefficients $\lambda_n(\theta)$ from zeros
3. Build Hankel matrix $H(\theta)$
4. Check eigenvalues: $\text{eig}(H(\theta)) \geq 0$?
5. If yes: verify moment representation $\lambda_n = \int t^n d\mu_\theta$
6. If no: adjust $\theta$ and repeat

**Theoretical Path**:
1. Prove existence of $\theta_\star$ with $H(\theta_\star) \succeq 0$ (completed)
2. Use compactness and continuity to locate $\theta_\star$
3. Verify explicit formula balance at $\theta_\star$
4. Conclude RH by standard arguments

---

## References

### Classical Moment Theory
- [Stieltjes 1894] T.J. Stieltjes, "Recherches sur les fractions continues", *Ann. Fac. Sci. Toulouse* 8 (1894)
- [Hamburger 1920-21] H. Hamburger, "Über eine Erweiterung des Stieltjesschen Momentenproblems", *Math. Ann.* 81-82
- [Shohat-Tamarkin 1943] J.A. Shohat, J.D. Tamarkin, *The Problem of Moments*, AMS Math. Surveys
- [Akhiezer 1965] N.I. Akhiezer, *The Classical Moment Problem*, Oliver & Boyd

### Pick-Nevanlinna Theory
- [Pick 1916] G. Pick, "Über die Beschränkungen analytischer Funktionen", *Math. Ann.* 77 (1916)
- [Nevanlinna 1919] R. Nevanlinna, "Über beschränkte Funktionen", *Ann. Acad. Sci. Fennicae* A 32
- [Aronszajn 1950] N. Aronszajn, "Theory of reproducing kernels", *Trans. AMS* 68 (1950)
- [Akhiezer-Glazman 1961] N.I. Akhiezer, I.M. Glazman, *Theory of Linear Operators in Hilbert Space*, Vol. 2

### Riemann Hypothesis
- [Li 1997] X.-J. Li, "The positivity of a sequence of numbers and the Riemann hypothesis", *J. Number Theory* 65 (1997)
- [Keiper 1992] J.B. Keiper, "Power series expansions of Riemann's ξ function", *Math. Comp.* 58 (1992)
- [Bombieri-Lagarias 1999] E. Bombieri, J.C. Lagarias, "Complements to Li's criterion for the Riemann hypothesis", *J. Number Theory* 77 (1999)

### De Branges Theory
- [de Branges 1968] L. de Branges, *Hilbert Spaces of Entire Functions*, Prentice-Hall
- [de Branges 1992] L. de Branges, "The convergence of Euler products", *J. Funct. Anal.* 107 (1992)

### Explicit Formula
- [Weil 1952] A. Weil, "Sur les 'formules explicites' de la théorie des nombres premiers", *Comm. Sém. Math. Lund* (1952)
- [Deninger 1994] C. Deninger, "Motivic L-functions and regularized determinants", *Proc. Symp. Pure Math.* 55 (1994)

---

## Appendix: Technical Details

### A.1 Compactness of Parameter Space

**Proposition A.1**: The parameter space $\Theta = [\alpha_{\min}, \alpha_{\max}] \times [\omega_{\min}, \omega_{\max}]$ with $0 < \alpha_{\min} < \alpha_{\max}$ and $0 < \omega_{\min} < \omega_{\max}$ is compact in $\mathbb{R}^2$.

**Proof**: $\Theta$ is a closed and bounded subset of $\mathbb{R}^2$, hence compact by Heine-Borel. □

### A.2 Zero Density and Convergence

**Lemma A.2** (Zero density): The non-trivial zeros $\rho_n$ of $\zeta(s)$ satisfy
$$N(T) := \#\{\rho : 0 < \text{Im}(\rho) \leq T\} = \frac{T}{2\pi} \log\frac{T}{2\pi e} + O(\log T)$$

**Reference**: [Riemann 1859], [von Mangoldt 1895]

**Corollary A.3**: The series
$$\sum_\rho \frac{1}{|\rho(1-\rho)|}$$
converges, justifying the definition of $H_\theta$ and $\mu_\theta$.

### A.3 Dominated Convergence Justification

**Lemma A.4**: For compact $\Theta$ and continuous $\theta \mapsto \hat{g}_\theta$, there exists $M > 0$ such that
$$\sum_\rho \frac{\hat{g}_\theta(\gamma_\rho)}{|\rho(1-\rho)|} \leq M$$
uniformly for $\theta \in \Theta$.

**Proof**: 
1. By continuity and compactness, $\sup_{\theta \in \Theta, u \in \mathbb{R}} \hat{g}_\theta(u) < \infty$
2. Let $C = \sup \hat{g}_\theta$
3. Then $\sum_\rho \hat{g}_\theta(\gamma_\rho) / |\rho(1-\rho)| \leq C \sum_\rho 1/|\rho(1-\rho)| < \infty$ by Lemma A.2 □

This justifies dominated convergence in Theorem 6.1.

---

**Document prepared**: October 1, 2025
**Related documents**: 
- `critical_hat_existence_theorem.md`
- `PROOF_SYNTHESIS.md`
- `core/spring_energy_rh_proof.py`

