# Harmonic Divergence Operator - Usage Examples

This document provides practical examples of using the Harmonic Divergence Operator to explore harmonic-divergence structures.

## Basic Usage

### Creating an Operator

```python
from riemann.harmonic_divergence_operator import HarmonicDivergenceOperator

# Initialize the operator
operator = HarmonicDivergenceOperator()
```

### Evaluating at a Point

```python
# Evaluate at x = 2
result = operator.evaluate(2.0)

print(f"H(2) = {result.value}")
print(f"Magnitude: {result.magnitude}")
print(f"Phase: {result.phase}")

# Access individual components
print(f"ln(2) = {result.ln_component}")
print(f"ζ(2) = {result.zeta_component}")
print(f"sin(π·2) = {result.sin_component}")
print(f"cos(π·2) = {result.cos_component}")
```

Output:
```
H(2) = 2.338081+1.000000j
Magnitude: 2.542956
Phase: 0.403518
ln(2) = 0.693147
ζ(2) = 1.644934
sin(π·2) = -0.000000
cos(π·2) = 1.000000
```

## Exploring Emergent Constants

### Demonstrating π Embedding

```python
# π is encoded in the trigonometric components
constants = operator.compute_emergent_constants()

print(f"π encoded in: {constants['pi_encoded_in']}")
print(f"Integer zeros of sin(πx): {constants['integer_zeros_of_sin']}")

# Verify sin vanishes at integers
for n in [1, 2, 3, 4, 5]:
    result = operator.evaluate(float(n))
    print(f"sin(π·{n}) = {result.sin_component:.10f}")
```

Output:
```
π encoded in: sin(πx), cos(πx) components
Integer zeros of sin(πx): [1.0, 2.0, 3.0, 4.0, 5.0]
sin(π·1) = 0.0000000000
sin(π·2) = -0.0000000000
sin(π·3) = -0.0000000000
sin(π·4) = 0.0000000000
sin(π·5) = 0.0000000000
```

### Demonstrating e Embedding

```python
import numpy as np

# e is encoded in the logarithmic component
e = np.e
result_e = operator.evaluate(e)

print(f"ln(e) = {result_e.ln_component}")  # Should be 1

# Euler's identity
constants = operator.compute_emergent_constants()
print(f"e^(iπ) = {constants['euler_identity_value']}")
print(f"cos(π) + i·sin(π) = {constants['cos_pi_plus_i_sin_pi']}")
```

Output:
```
ln(e) = 1.000000
e^(iπ) = -1.000000+0.000000j
cos(π) + i·sin(π) = -1.000000+0.000000j
```

## Component Balance Analysis

### Analyzing a Single Point

```python
# Analyze balance at x = 2
analysis = operator.analyze_balance(2.0)

print(f"Total magnitude: {analysis['total_magnitude']:.6f}")
print(f"Dominant component: {analysis['dominant_component']}")
print(f"\nRelative contributions:")
for component, contribution in analysis['relative_contributions'].items():
    print(f"  {component}: {contribution:.2%}")

print(f"\nBalance metrics:")
print(f"  Divergent total: {analysis['divergent_total']:.6f}")
print(f"  Harmonic total: {analysis['harmonic_total']:.6f}")
print(f"  Balance ratio: {analysis['balance_ratio']:.4f}")
print(f"  Is balanced: {analysis['is_balanced']}")
```

Output:
```
Total magnitude: 2.542956
Dominant component: zeta

Relative contributions:
  ln: 20.76%
  zeta: 49.28%
  tan: 0.00%
  sin: 0.00%
  cos: 29.96%

Balance metrics:
  Divergent total: 2.338081
  Harmonic total: 1.000000
  Balance ratio: 2.3381
  Is balanced: False
```

### Finding Balanced Points

```python
# Scan a range to find balanced points
test_points = np.linspace(0.1, 5.0, 100)
balanced_points = []

for x in test_points:
    analysis = operator.analyze_balance(x)
    if analysis['is_balanced']:
        balanced_points.append((x, analysis['balance_ratio']))

print(f"Found {len(balanced_points)} balanced points")
for x, ratio in balanced_points[:5]:
    print(f"  x = {x:.4f}, balance ratio = {ratio:.4f}")
```

## Finding Zeros

### Searching for Real Zeros

```python
# Find real zeros in a range
zeros = operator.find_zeros_real((0.1, 10.0), num_points=200)

print(f"Found {len(zeros)} zeros in range [0.1, 10.0]:")
for i, z in enumerate(zeros[:10], 1):
    # Verify it's close to zero
    result = operator.evaluate(z)
    print(f"  Zero {i}: x = {z:.6f}, |H(x)| = {result.magnitude:.6e}")
```

### Zero Verification

```python
# Verify a candidate zero
candidate = 2.5  # Example candidate

result = operator.evaluate(candidate)
analysis = operator.analyze_balance(candidate)

print(f"Testing x = {candidate}")
print(f"  |H(x)| = {result.magnitude:.6e}")
print(f"  Balance ratio = {analysis['balance_ratio']:.4f}")
print(f"  Is balanced = {analysis['is_balanced']}")

# A true zero should have very small magnitude
is_zero = result.magnitude < 0.01
print(f"  Is likely zero: {is_zero}")
```

## Critical Line Scanning

### Scanning the Critical Line Analogy

```python
# Scan a range analogous to the critical line
results = operator.scan_critical_line(0.0, 20.0, num_points=100)

# Find points with smallest magnitude (closest to zero)
sorted_results = sorted(results, key=lambda r: r['total_magnitude'])

print("Top 10 points with smallest magnitude:")
for i, result in enumerate(sorted_results[:10], 1):
    print(f"  {i}. x = {result['x']}, "
          f"|H(x)| = {result['total_magnitude']:.6f}, "
          f"balanced = {result['is_balanced']}")
```

### Visualizing Critical Line Behavior

```python
import matplotlib.pyplot as plt

# Scan critical line
t_values = np.linspace(0.0, 50.0, 500)
magnitudes = []

for t in t_values:
    x = complex(0.5 + t, t)
    result = operator.evaluate(x)
    magnitudes.append(result.magnitude)

# Plot magnitude along critical line
plt.figure(figsize=(12, 6))
plt.plot(t_values, magnitudes)
plt.xlabel('t')
plt.ylabel('|H(0.5+t, t)|')
plt.title('Harmonic Divergence Operator Magnitude along Critical Line Analogy')
plt.grid(True)
plt.savefig('critical_line_scan.png')
```

## Component Analysis

### Comparing Components at Different Points

```python
# Compare component contributions at different points
test_points = [0.5, 1.0, 2.0, np.e, np.pi]

print("Component analysis across points:")
print(f"{'x':<10} {'ln':<12} {'ζ':<12} {'sin':<12} {'cos':<12}")
print("-" * 60)

for x in test_points:
    result = operator.evaluate(x)
    print(f"{x:<10.4f} "
          f"{abs(result.ln_component):<12.6f} "
          f"{abs(result.zeta_component):<12.6f} "
          f"{abs(result.sin_component):<12.6f} "
          f"{abs(result.cos_component):<12.6f}")
```

### Tracking Dominant Component

```python
# Track which component dominates across a range
x_range = np.linspace(0.1, 5.0, 100)
dominant_components = []

for x in x_range:
    analysis = operator.analyze_balance(x)
    dominant_components.append(analysis['dominant_component'])

# Count occurrences
from collections import Counter
component_counts = Counter(dominant_components)

print("Dominant component frequency:")
for component, count in component_counts.items():
    print(f"  {component}: {count/len(x_range):.1%}")
```

## Advanced Analysis

### Harmonic vs Divergent Dominance

```python
# Classify regions by dominant behavior
x_range = np.linspace(0.1, 10.0, 200)

harmonic_dominant = []
divergent_dominant = []
balanced_region = []

for x in x_range:
    analysis = operator.analyze_balance(x)
    ratio = analysis['balance_ratio']
    
    if ratio < 0.5:
        harmonic_dominant.append(x)
    elif ratio > 2.0:
        divergent_dominant.append(x)
    else:
        balanced_region.append(x)

print(f"Classification of range [0.1, 10.0]:")
print(f"  Harmonic dominant: {len(harmonic_dominant)} points ({len(harmonic_dominant)/200:.1%})")
print(f"  Balanced: {len(balanced_region)} points ({len(balanced_region)/200:.1%})")
print(f"  Divergent dominant: {len(divergent_dominant)} points ({len(divergent_dominant)/200:.1%})")
```

### Phase Analysis

```python
# Analyze phase behavior
x_range = np.linspace(0.1, 5.0, 100)
phases = []

for x in x_range:
    result = operator.evaluate(x)
    phases.append(result.phase)

# Plot phase vs x
plt.figure(figsize=(12, 6))
plt.plot(x_range, phases)
plt.xlabel('x')
plt.ylabel('Phase(H(x))')
plt.title('Phase Behavior of Harmonic Divergence Operator')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.savefig('phase_analysis.png')
```

## Complete Analysis Example

```python
def comprehensive_analysis(x):
    """Perform comprehensive analysis at a point."""
    operator = HarmonicDivergenceOperator()
    
    # Evaluate
    result = operator.evaluate(x)
    
    # Analyze balance
    analysis = operator.analyze_balance(x)
    
    # Print report
    print(f"Comprehensive Analysis at x = {x}")
    print("=" * 60)
    
    print(f"\n1. Operator Value:")
    print(f"   H({x}) = {result.value}")
    print(f"   Magnitude: {result.magnitude:.6f}")
    print(f"   Phase: {result.phase:.6f} rad ({np.degrees(result.phase):.2f}°)")
    
    print(f"\n2. Components:")
    print(f"   ln({x}) = {result.ln_component}")
    print(f"   ζ({x}) = {result.zeta_component}")
    print(f"   tan(π·{x}/2) = {result.tan_component}")
    print(f"   sin(π·{x}) = {result.sin_component:.6f}")
    print(f"   cos(π·{x}) = {result.cos_component:.6f}")
    
    print(f"\n3. Balance Analysis:")
    print(f"   Dominant component: {analysis['dominant_component']}")
    print(f"   Divergent total: {analysis['divergent_total']:.6f}")
    print(f"   Harmonic total: {analysis['harmonic_total']:.6f}")
    print(f"   Balance ratio: {analysis['balance_ratio']:.4f}")
    print(f"   Is balanced: {analysis['is_balanced']}")
    
    print(f"\n4. Relative Contributions:")
    for component, contribution in analysis['relative_contributions'].items():
        bar = '█' * int(contribution * 50)
        print(f"   {component:5s}: {bar} {contribution:.1%}")
    
    return result, analysis

# Run comprehensive analysis
result, analysis = comprehensive_analysis(2.0)
```

## Tips and Best Practices

1. **Handling Singularities**: Be aware that tan(πx/2) has poles at odd integers. The operator clips these to large but finite values.

2. **Zeta Convergence**: For Re(s) > 1, the zeta series converges well. For Re(s) ≤ 1, the operator uses approximations.

3. **Zero Finding**: Use fine-grained sampling (`num_points=200+`) for accurate zero detection.

4. **Balance Interpretation**: A balance ratio between 0.5 and 2.0 indicates a "critical" region where harmonic and divergent forces are comparable.

5. **Complex Evaluation**: While the operator supports complex arguments, the tan component is only computed for real values to avoid complications.

## Further Reading

- See `docs/insights/HARMONIC_DIVERGENCE_OPERATOR.md` for mathematical background
- See `code/riemann/harmonic_divergence_operator.py` for implementation details
- See `tests/unit/test_harmonic_divergence_operator.py` for validation tests
