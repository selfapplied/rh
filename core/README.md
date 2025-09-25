# Shared Branch: Common Dependencies

*Dedicated to shared utilities and cross-branch dependencies that enable modular, well-encapsulated projects*

## Core Philosophy

Following the playbook's **modular design** principle: create semantically coupled dependencies that can be shared between projects, with bonus points for using other projects that already exist and provide the needed bridging.

## Branch Structure

### Core Utilities
- **math/**: Mathematical utilities and common functions
- **computation/**: Computational primitives and algorithms
- **visualization/**: Common visualization and plotting utilities
- **data/**: Data structures and serialization formats
- **validation/**: Common validation and testing utilities
- **integration/**: Integration with external projects

### Key Principles
1. **Semantic Coupling**: Dependencies are meaningfully related, not just convenient
2. **Minimal Interface**: Clean, simple APIs that hide complexity
3. **Cross-Branch Compatibility**: Works seamlessly across math/, ce1/, and badge/
4. **External Integration**: Leverages existing projects when possible

## Shared Components

### Mathematical Utilities
```python
# Common mathematical functions
from shared.math import zeta_function, gamma_function, completed_zeta
from shared.math import pascal_kernel, dihedral_action, gap_analysis

# Mathematical constants and precision
from shared.math import CRITICAL_LINE, ZETA_ZEROS, PRECISION_LEVELS
```

### Computational Primitives
```python
# RISC-like computational primitives
from shared.computation import add, multiply, compare, branch
from shared.computation import convolution, fourier_transform, wavelet

# Memory and data structures
from shared.computation import hex_lattice, semantic_memory, constraint_solver
```

### Visualization Framework
```python
# Common visualization utilities
from shared.visualization import plot_zeta_landscape, plot_certificate
from shared.visualization import generate_svg, render_passport, create_badge

# Color and styling
from shared.visualization import DOMAIN_COLORS, MATHEMATICAL_SYMBOLS, TYPOGRAPHY
```

### Data Formats
```python
# Common data structures
from shared.data import CE1Certificate, MathematicalProof, BadgeTemplate
from shared.data import serialize_ce1, deserialize_ce1, validate_format

# Cross-project integration
from shared.data import export_to_aedificare, import_from_discograph
```

## External Project Integration

### aedificare Integration
- **λ-calculus grammar**: Provides compositional structure for mathematical field theory
- **CE1 specifications**: Shared CE1 format definitions
- **Learning systems**: Adaptive learning for mathematical patterns

### discograph Integration  
- **Constellation mapping**: Reveals how equilibrium geometry organizes the multiverse
- **CE1 specifications**: Shared CE1 format and validation
- **Visualization**: Common visualization frameworks

### metanion Integration
- **Field theory**: Autoverse field theory underlies mirror reality
- **Symbol encoding**: Mathematical relationships encoded as symbols
- **Computational primitives**: Shared computational foundations

## Usage Patterns

### Cross-Branch Dependencies
```python
# In math/ branch
from shared.math import zeta_function
from shared.computation import convolution
from shared.visualization import plot_zeta_landscape

# In ce1/ branch  
from shared.math import pascal_kernel
from shared.computation import dihedral_action
from shared.data import CE1Certificate

# In badge/ branch
from shared.visualization import generate_svg
from shared.data import BadgeTemplate
from shared.validation import validate_certificate
```

### External Project Bridging
```python
# Bridge to aedificare
from shared.integration.aedificare import export_ce1_spec
from shared.integration.aedificare import import_learning_data

# Bridge to discograph
from shared.integration.discograph import export_constellation_map
from shared.integration.discograph import import_ce1_spec

# Bridge to metanion
from shared.integration.metanion import export_field_theory
from shared.integration.metanion import import_computational_primitives
```

## Design Principles

### Semantic Coupling
Dependencies are grouped by **meaning**, not convenience:
- **math/**: Pure mathematical functions and constants
- **computation/**: Algorithmic and computational primitives
- **visualization/**: Visual representation and rendering
- **data/**: Data structures and serialization

### Minimal Interface
Each shared component provides a **clean, simple API**:
```python
# Good: Simple, focused interface
def zeta_function(s, precision=100):
    """Compute Riemann zeta function at point s with given precision."""
    return compute_zeta(s, precision)

# Bad: Complex, unfocused interface  
def math_utils(s, precision=100, method='auto', cache=True, validate=True, ...):
    """Do everything mathematical."""
```

### Cross-Branch Compatibility
Shared components work seamlessly across all branches:
- **No branch-specific dependencies** in shared code
- **Consistent interfaces** across all usage contexts
- **Backward compatibility** when interfaces evolve

---

*Shared branch: Where modular design meets semantic coupling to create living, breathing mathematical systems.*

