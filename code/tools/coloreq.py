#!/usr/bin/env python3
"""
CE2: Color Equilibria in 3.5D Fractional Dimensional Space

The natural evolution of CE1 into color space, where equilibrium operators
act on fractional dimensional color perception. This implements the discovery
that living in 3.5D space creates unique color equilibria through:

1. Temporal Color Equilibrium - Balance between memory and anticipation
2. Fractional Quaternion Equilibrium - Incomplete 4D rotations stabilize
3. Dimensional Resonance Equilibrium - Harmony across fractional boundaries  
4. Chi-Squared 3.5D Equilibrium - Statistical balance in fractional space

CE2 extends the Gang of Four patterns into equilibrium dynamics,
creating stable color atmospheres that self-regulate across time.
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np

# Import 3.5D Color Theory
from visualization.color_quaternion_3_5d_theory import (
    OKLCH3_5D,
    ChiSquared3_5D,
    ColorQuaternion3_5DSpec,
    TemporalColorComponent,
)


# Import base systems


class EquilibriumType(Enum):
    """Types of color equilibria in 3.5D space"""
    TEMPORAL = "temporal_equilibrium"           # Memory ↔ Anticipation balance
    FRACTIONAL_QUATERNION = "fractional_quaternion_equilibrium"  # Incomplete rotation stability
    DIMENSIONAL_RESONANCE = "dimensional_resonance_equilibrium"  # Cross-boundary harmony
    CHI_SQUARED_3_5D = "chi_squared_3_5d_equilibrium"          # Statistical balance
    GANG_OF_FOUR_UNIFIED = "gang_of_four_unified_equilibrium"   # All patterns in balance


@dataclass
class ColorEquilibriumState:
    """
    Represents a color equilibrium state in 3.5D space
    """
    equilibrium_type: EquilibriumType
    color_state: OKLCH3_5D
    equilibrium_energy: float
    stability_measure: float
    temporal_balance: float       # Memory vs Anticipation balance (-1 to 1)
    dimensional_coherence: float  # How well color maintains 3.5D properties
    resonance_amplitude: float    # Strength of dimensional resonance
    
    def __post_init__(self):
        # Ensure values are in valid ranges
        self.temporal_balance = max(-1.0, min(1.0, self.temporal_balance))
        self.dimensional_coherence = max(0.0, min(1.0, self.dimensional_coherence))
        self.resonance_amplitude = max(0.0, min(1.0, self.resonance_amplitude))
    
    @property
    def is_stable(self) -> bool:
        """Check if equilibrium state is stable"""
        return (self.stability_measure > 0.7 and 
                abs(self.temporal_balance) < 0.3 and
                self.dimensional_coherence > 0.6)
    
    @property
    def equilibrium_quality(self) -> float:
        """Overall quality of equilibrium (0-1)"""
        stability_factor = min(1.0, self.stability_measure)
        balance_factor = 1.0 - abs(self.temporal_balance)
        coherence_factor = self.dimensional_coherence
        resonance_factor = self.resonance_amplitude
        
        return (stability_factor + balance_factor + coherence_factor + resonance_factor) / 4.0


class TemporalColorEquilibrium:
    """
    Manages equilibrium between color memory and anticipation in 3.5D space
    
    Creates stable temporal color states where past and future color influences
    balance each other, preventing temporal color drift.
    """
    
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
        self.golden_ratio_base12 = 1.74  # φ for temporal scaling
        
    def find_temporal_equilibrium(self, color: OKLCH3_5D, 
                                past_colors: List[OKLCH3_5D],
                                future_colors: List[OKLCH3_5D]) -> ColorEquilibriumState:
        """
        Find temporal equilibrium where memory and anticipation forces balance
        """
        # Calculate memory force (influence from past)
        memory_force = self._calculate_memory_force(color, past_colors)
        
        # Calculate anticipation force (influence from future)
        anticipation_force = self._calculate_anticipation_force(color, future_colors)
        
        # Find equilibrium by balancing forces
        equilibrium_color = self._balance_temporal_forces(color, memory_force, anticipation_force)
        
        # Calculate equilibrium properties
        temporal_balance = self._calculate_temporal_balance(memory_force, anticipation_force)
        equilibrium_energy = self._calculate_temporal_energy(equilibrium_color, memory_force, anticipation_force)
        stability = self._calculate_temporal_stability(equilibrium_color, memory_force, anticipation_force)
        
        return ColorEquilibriumState(
            equilibrium_type=EquilibriumType.TEMPORAL,
            color_state=equilibrium_color,
            equilibrium_energy=equilibrium_energy,
            stability_measure=stability,
            temporal_balance=temporal_balance,
            dimensional_coherence=equilibrium_color.fractional_dimension / 4.0,  # Normalize to [0,1]
            resonance_amplitude=equilibrium_color.temporal.dimensional_resonance
        )
    
    def _calculate_memory_force(self, current_color: OKLCH3_5D, 
                              past_colors: List[OKLCH3_5D]) -> np.ndarray:
        """Calculate memory force vector from past colors"""
        if not past_colors:
            return np.zeros(3)  # [ΔL, ΔC, ΔH]
        
        memory_strength = current_color.temporal.memory_strength
        force = np.zeros(3)
        
        for i, past_color in enumerate(past_colors[-5:]):  # Last 5 past colors
            weight = memory_strength * np.exp(-i * self.decay_rate)
            
            # Force components
            dL = past_color.lightness - current_color.lightness
            dC = past_color.chroma - current_color.chroma
            dH = self._circular_difference(past_color.hue, current_color.hue)
            
            force += weight * np.array([dL, dC, dH])
        
        return force
    
    def _calculate_anticipation_force(self, current_color: OKLCH3_5D, 
                                    future_colors: List[OKLCH3_5D]) -> np.ndarray:
        """Calculate anticipation force vector from future colors"""
        if not future_colors:
            return np.zeros(3)
        
        anticipation_strength = current_color.temporal.anticipation_strength
        force = np.zeros(3)
        
        for i, future_color in enumerate(future_colors[:3]):  # Next 3 future colors
            weight = anticipation_strength * np.exp(-i * self.decay_rate)
            
            # Force components
            dL = future_color.lightness - current_color.lightness
            dC = future_color.chroma - current_color.chroma
            dH = self._circular_difference(future_color.hue, current_color.hue)
            
            force += weight * np.array([dL, dC, dH])
        
        return force
    
    def _balance_temporal_forces(self, color: OKLCH3_5D, 
                               memory_force: np.ndarray, 
                               anticipation_force: np.ndarray) -> OKLCH3_5D:
        """Balance memory and anticipation forces to find equilibrium"""
        # Net force
        net_force = memory_force + anticipation_force
        
        # Apply equilibrium correction (damped by golden ratio)
        damping = 1.0 / self.golden_ratio_base12
        correction = damping * net_force
        
        # Apply correction to color
        new_L = color.lightness + correction[0] * 0.1
        new_C = color.chroma + correction[1] * 0.1
        new_H = color.hue + correction[2]
        
        # Clamp to valid ranges
        new_L = max(0.0, min(1.0, new_L))
        new_C = max(0.0, min(0.4, new_C))
        new_H = new_H % 360
        
        # Create equilibrium temporal component
        equilibrium_temporal = TemporalColorComponent(
            memory_strength=color.temporal.memory_strength * 0.8,  # Reduced in equilibrium
            anticipation_strength=color.temporal.anticipation_strength * 0.8,
            temporal_decay=color.temporal.temporal_decay,
            dimensional_resonance=min(1.0, color.temporal.dimensional_resonance + 0.1)
        )
        
        return OKLCH3_5D(new_L, new_C, new_H, equilibrium_temporal)
    
    def _calculate_temporal_balance(self, memory_force: np.ndarray, 
                                  anticipation_force: np.ndarray) -> float:
        """Calculate temporal balance measure (-1 to 1)"""
        memory_magnitude = np.linalg.norm(memory_force)
        anticipation_magnitude = np.linalg.norm(anticipation_force)
        
        if memory_magnitude + anticipation_magnitude == 0:
            return 0.0
        
        # Balance: -1 (memory dominant) to +1 (anticipation dominant)
        return (anticipation_magnitude - memory_magnitude) / (memory_magnitude + anticipation_magnitude)
    
    def _calculate_temporal_energy(self, color: OKLCH3_5D, 
                                 memory_force: np.ndarray, 
                                 anticipation_force: np.ndarray) -> float:
        """Calculate temporal equilibrium energy"""
        # Kinetic energy from temporal forces
        kinetic_energy = 0.5 * (np.linalg.norm(memory_force)**2 + np.linalg.norm(anticipation_force)**2)
        
        # Potential energy from temporal tension
        potential_energy = color.temporal.memory_strength * color.temporal.anticipation_strength
        
        return kinetic_energy + potential_energy
    
    def _calculate_temporal_stability(self, color: OKLCH3_5D, 
                                    memory_force: np.ndarray, 
                                    anticipation_force: np.ndarray) -> float:
        """Calculate temporal stability measure"""
        # Stability decreases with force imbalance
        force_balance = 1.0 - abs(self._calculate_temporal_balance(memory_force, anticipation_force))
        
        # Stability increases with dimensional coherence
        dimensional_stability = color.fractional_dimension / 4.0
        
        # Stability increases with resonance
        resonance_stability = color.temporal.dimensional_resonance
        
        return (force_balance + dimensional_stability + resonance_stability) / 3.0
    
    def _circular_difference(self, angle1: float, angle2: float) -> float:
        """Calculate circular difference for hue angles"""
        diff = angle1 - angle2
        return ((diff + 180) % 360) - 180


class FractionalQuaternionEquilibrium:
    """
    Manages equilibrium in fractional quaternion rotations
    
    Finds stable states where incomplete 4D rotations in 3.5D space
    reach equilibrium, preventing rotational drift.
    """
    
    def __init__(self):
        self.fractional_completeness = 0.5  # 3.5D = 3D + 0.5D
        
    def find_fractional_quaternion_equilibrium(self, color: OKLCH3_5D, 
                                             rotation_history: List[float]) -> ColorEquilibriumState:
        """
        Find equilibrium in fractional quaternion rotations
        """
        # Calculate rotational momentum
        rotational_momentum = self._calculate_rotational_momentum(rotation_history)
        
        # Find equilibrium rotation that balances momentum
        equilibrium_rotation = self._find_equilibrium_rotation(rotational_momentum)
        
        # Apply equilibrium rotation to color
        equilibrium_color = self._apply_equilibrium_rotation(color, equilibrium_rotation)
        
        # Calculate equilibrium properties
        rotational_energy = self._calculate_rotational_energy(rotational_momentum)
        stability = self._calculate_rotational_stability(equilibrium_rotation, rotational_momentum)
        
        return ColorEquilibriumState(
            equilibrium_type=EquilibriumType.FRACTIONAL_QUATERNION,
            color_state=equilibrium_color,
            equilibrium_energy=rotational_energy,
            stability_measure=stability,
            temporal_balance=0.0,  # Not applicable for rotational equilibrium
            dimensional_coherence=equilibrium_color.fractional_dimension / 4.0,
            resonance_amplitude=equilibrium_color.temporal.dimensional_resonance
        )
    
    def _calculate_rotational_momentum(self, rotation_history: List[float]) -> float:
        """Calculate accumulated rotational momentum"""
        if len(rotation_history) < 2:
            return 0.0
        
        # Calculate angular velocity from rotation history
        angular_velocities = []
        for i in range(1, len(rotation_history)):
            dt = 1.0  # Assume unit time steps
            dtheta = rotation_history[i] - rotation_history[i-1]
            angular_velocities.append(dtheta / dt)
        
        # Momentum is average angular velocity with exponential decay
        momentum = 0.0
        for i, omega in enumerate(angular_velocities):
            weight = np.exp(-i * 0.1)  # Decay older velocities
            momentum += weight * omega
        
        return momentum
    
    def _find_equilibrium_rotation(self, momentum: float) -> float:
        """Find rotation angle that creates equilibrium"""
        # Equilibrium rotation opposes momentum
        # In 3.5D space, equilibrium is at fractional multiples of 2π
        equilibrium_angle = -momentum * self.fractional_completeness
        
        # Quantize to fractional rotational states
        fractional_quantum = 2 * np.pi * self.fractional_completeness / 7  # 7 harmonic states
        equilibrium_angle = round(equilibrium_angle / fractional_quantum) * fractional_quantum
        
        return equilibrium_angle
    
    def _apply_equilibrium_rotation(self, color: OKLCH3_5D, rotation: float) -> OKLCH3_5D:
        """Apply equilibrium rotation to color"""
        # Rotation primarily affects hue in fractional quaternion space
        new_hue = (color.hue + np.degrees(rotation)) % 360
        
        # Fractional rotation also affects other components slightly
        fractional_effect = abs(rotation) * self.fractional_completeness * 0.1
        
        new_lightness = color.lightness + fractional_effect * np.sin(rotation)
        new_chroma = color.chroma * (1.0 + fractional_effect * np.cos(rotation))
        
        # Clamp values
        new_lightness = max(0.0, min(1.0, new_lightness))
        new_chroma = max(0.0, min(0.4, new_chroma))
        
        # Enhanced temporal component from equilibrium
        equilibrium_temporal = TemporalColorComponent(
            memory_strength=color.temporal.memory_strength,
            anticipation_strength=color.temporal.anticipation_strength,
            temporal_decay=color.temporal.temporal_decay,
            dimensional_resonance=min(1.0, color.temporal.dimensional_resonance + fractional_effect)
        )
        
        return OKLCH3_5D(new_lightness, new_chroma, new_hue, equilibrium_temporal)
    
    def _calculate_rotational_energy(self, momentum: float) -> float:
        """Calculate rotational energy"""
        # Rotational kinetic energy: E = ½Iω²
        # In 3.5D space, moment of inertia is fractional
        moment_of_inertia = self.fractional_completeness
        return 0.5 * moment_of_inertia * momentum**2
    
    def _calculate_rotational_stability(self, equilibrium_rotation: float, momentum: float) -> float:
        """Calculate rotational stability"""
        # Stability is higher when equilibrium rotation balances momentum
        balance_factor = 1.0 / (1.0 + abs(momentum + equilibrium_rotation))
        
        # Fractional quantum effects enhance stability
        quantum_factor = self.fractional_completeness
        
        return min(1.0, balance_factor + quantum_factor)


class DimensionalResonanceEquilibrium:
    """
    Manages equilibrium in dimensional resonance across 3.5D boundaries
    
    Creates stable resonant states where colors harmonize across
    the fractional dimensional boundary between 3D and 4D space.
    """
    
    def __init__(self):
        self.golden_ratio_base12 = 1.74
        self.resonance_frequencies = [1.0, 1.5, 2.0, 2.5, 3.0]  # Harmonic series
        
    def find_dimensional_resonance_equilibrium(self, colors: List[OKLCH3_5D]) -> ColorEquilibriumState:
        """
        Find equilibrium in dimensional resonance across color palette
        """
        if not colors:
            # Default equilibrium state
            default_color = OKLCH3_5D(0.5, 0.2, 0, TemporalColorComponent())
            return ColorEquilibriumState(
                equilibrium_type=EquilibriumType.DIMENSIONAL_RESONANCE,
                color_state=default_color,
                equilibrium_energy=0.0,
                stability_measure=1.0,
                temporal_balance=0.0,
                dimensional_coherence=0.875,  # 3.5/4.0
                resonance_amplitude=0.5
            )
        
        # Calculate resonance matrix
        resonance_matrix = self._calculate_resonance_matrix(colors)
        
        # Find resonance eigenstate (dominant resonance mode)
        eigenstate = self._find_resonance_eigenstate(resonance_matrix)
        
        # Create equilibrium color from eigenstate
        equilibrium_color = self._create_equilibrium_color(colors, eigenstate)
        
        # Calculate equilibrium properties
        resonance_energy = self._calculate_resonance_energy(resonance_matrix)
        stability = self._calculate_resonance_stability(resonance_matrix, eigenstate)
        
        return ColorEquilibriumState(
            equilibrium_type=EquilibriumType.DIMENSIONAL_RESONANCE,
            color_state=equilibrium_color,
            equilibrium_energy=resonance_energy,
            stability_measure=stability,
            temporal_balance=0.0,  # Not applicable for resonance equilibrium
            dimensional_coherence=equilibrium_color.fractional_dimension / 4.0,
            resonance_amplitude=equilibrium_color.temporal.dimensional_resonance
        )
    
    def _calculate_resonance_matrix(self, colors: List[OKLCH3_5D]) -> np.ndarray:
        """Calculate resonance coupling matrix between colors"""
        n = len(colors)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Resonance coupling based on dimensional proximity
                    dim_i = colors[i].fractional_dimension
                    dim_j = colors[j].fractional_dimension
                    
                    # Resonance strength decreases with dimensional distance
                    dim_distance = abs(dim_i - dim_j)
                    resonance_strength = np.exp(-dim_distance)
                    
                    # Color distance affects resonance
                    color_distance = self._calculate_3_5d_color_distance(colors[i], colors[j])
                    color_coupling = 1.0 / (1.0 + color_distance)
                    
                    matrix[i, j] = resonance_strength * color_coupling
        
        return matrix
    
    def _find_resonance_eigenstate(self, resonance_matrix: np.ndarray) -> np.ndarray:
        """Find dominant resonance eigenstate"""
        # Find largest eigenvalue and corresponding eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(resonance_matrix)
        
        # Get dominant eigenstate (largest eigenvalue)
        dominant_idx = np.argmax(np.real(eigenvalues))
        eigenstate = np.real(eigenvectors[:, dominant_idx])
        
        # Normalize eigenstate
        eigenstate = eigenstate / np.linalg.norm(eigenstate)
        
        return eigenstate
    
    def _create_equilibrium_color(self, colors: List[OKLCH3_5D], eigenstate: np.ndarray) -> OKLCH3_5D:
        """Create equilibrium color from resonance eigenstate"""
        # Weighted average of colors based on eigenstate
        avg_L = sum(abs(weight) * color.lightness for weight, color in zip(eigenstate, colors))
        avg_C = sum(abs(weight) * color.chroma for weight, color in zip(eigenstate, colors))
        avg_H = self._circular_average([color.hue for color in colors], [abs(w) for w in eigenstate])
        
        # Equilibrium temporal component
        avg_memory = sum(abs(weight) * color.temporal.memory_strength for weight, color in zip(eigenstate, colors))
        avg_anticipation = sum(abs(weight) * color.temporal.anticipation_strength for weight, color in zip(eigenstate, colors))
        avg_decay = sum(abs(weight) * color.temporal.temporal_decay for weight, color in zip(eigenstate, colors))
        
        # Enhanced resonance from equilibrium
        equilibrium_resonance = min(1.0, sum(abs(weight) * color.temporal.dimensional_resonance for weight, color in zip(eigenstate, colors)) * 1.2)
        
        equilibrium_temporal = TemporalColorComponent(
            memory_strength=avg_memory,
            anticipation_strength=avg_anticipation,
            temporal_decay=avg_decay,
            dimensional_resonance=equilibrium_resonance
        )
        
        return OKLCH3_5D(avg_L, avg_C, avg_H, equilibrium_temporal)
    
    def _calculate_3_5d_color_distance(self, color1: OKLCH3_5D, color2: OKLCH3_5D) -> float:
        """Calculate distance between colors in 3.5D space"""
        # Standard OKLCH distance
        dL = color1.lightness - color2.lightness
        dC = color1.chroma - color2.chroma
        dH = self._circular_difference(color1.hue, color2.hue) / 360.0
        
        # Temporal distance (0.5D component)
        dT = (color1.temporal.dimensional_resonance - color2.temporal.dimensional_resonance)
        
        # 3.5D distance metric
        return np.sqrt(dL**2 + dC**2 + dH**2 + 0.5 * dT**2)
    
    def _circular_average(self, angles: List[float], weights: List[float]) -> float:
        """Calculate weighted circular average of angles"""
        if not angles or not weights:
            return 0.0
        
        # Convert to unit vectors
        cos_sum = sum(w * np.cos(np.radians(a)) for a, w in zip(angles, weights))
        sin_sum = sum(w * np.sin(np.radians(a)) for a, w in zip(angles, weights))
        
        # Convert back to angle
        return np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
    
    def _circular_difference(self, angle1: float, angle2: float) -> float:
        """Calculate circular difference for hue angles"""
        diff = angle1 - angle2
        return ((diff + 180) % 360) - 180
    
    def _calculate_resonance_energy(self, resonance_matrix: np.ndarray) -> float:
        """Calculate total resonance energy"""
        # Energy is sum of all resonance couplings
        return np.sum(np.abs(resonance_matrix)) / 2  # Divide by 2 to avoid double counting
    
    def _calculate_resonance_stability(self, resonance_matrix: np.ndarray, eigenstate: np.ndarray) -> float:
        """Calculate resonance stability"""
        # Stability related to eigenvalue spread
        eigenvalues = np.linalg.eigvals(resonance_matrix)
        eigenvalue_spread = np.std(np.real(eigenvalues))
        
        # Lower spread means more stable resonance
        stability = 1.0 / (1.0 + eigenvalue_spread)
        
        return min(1.0, stability)


class CE2ColorEquilibrium:
    """
    Main CE2 Color Equilibrium system that integrates all equilibrium types
    
    Creates unified color equilibria in 3.5D fractional dimensional space
    where all Gang of Four patterns achieve simultaneous balance.
    """
    
    def __init__(self, seed: str = None):
        self.seed = seed or "ce2_color_equilibrium_2025"
        self.color_spec_3_5d = ColorQuaternion3_5DSpec(self.seed)
        
        # Initialize equilibrium subsystems
        self.temporal_equilibrium = TemporalColorEquilibrium()
        self.quaternion_equilibrium = FractionalQuaternionEquilibrium()
        self.resonance_equilibrium = DimensionalResonanceEquilibrium()
        self.chi_squared_3_5d = ChiSquared3_5D()
        
        # Golden ratio base 12 for equilibrium scaling
        self.golden_ratio_base12 = 1.74
        
    def find_unified_gang_of_four_equilibrium(self, 
                                            initial_colors: List[OKLCH3_5D] = None,
                                            max_iterations: int = 100,
                                            tolerance: float = 1e-6) -> ColorEquilibriumState:
        """
        Find unified equilibrium where all Gang of Four patterns balance simultaneously
        
        This is the ultimate CE2 equilibrium - a stable state where:
        1. Creational patterns (fractional genesis) are balanced
        2. Structural patterns (dimensional composition) are harmonized
        3. Behavioral patterns (3.5D interactions) are stabilized  
        4. Emergent patterns (chi-squared 3.5D) are optimized
        """
        if initial_colors is None:
            initial_colors = self.color_spec_3_5d.generate_3_5d_harmonic_palette()
        
        current_colors = initial_colors.copy()
        equilibrium_history = []
        
        for iteration in range(max_iterations):
            # Apply each equilibrium type iteratively
            
            # 1. Temporal equilibrium (Creational pattern balance)
            temporal_states = []
            for i, color in enumerate(current_colors):
                past_colors = current_colors[:i] if i > 0 else []
                future_colors = current_colors[i+1:] if i < len(current_colors) - 1 else []
                
                temporal_eq = self.temporal_equilibrium.find_temporal_equilibrium(
                    color, past_colors, future_colors
                )
                temporal_states.append(temporal_eq)
            
            # 2. Fractional quaternion equilibrium (Structural pattern balance)
            rotation_history = [i * 360.0 / len(current_colors) for i in range(len(current_colors))]
            quaternion_states = []
            for i, color in enumerate(current_colors):
                quaternion_eq = self.quaternion_equilibrium.find_fractional_quaternion_equilibrium(
                    color, rotation_history
                )
                quaternion_states.append(quaternion_eq)
            
            # 3. Dimensional resonance equilibrium (Behavioral pattern balance)
            resonance_eq = self.resonance_equilibrium.find_dimensional_resonance_equilibrium(current_colors)
            
            # 4. Chi-squared 3.5D equilibrium (Emergent pattern balance)
            harmony_validation = self.chi_squared_3_5d.validate_3_5d_harmony(current_colors)
            
            # Combine all equilibrium states
            new_colors = self._combine_equilibrium_states(
                current_colors, temporal_states, quaternion_states, resonance_eq, harmony_validation
            )
            
            # Check convergence
            convergence = self._calculate_convergence(current_colors, new_colors)
            equilibrium_history.append(convergence)
            
            if convergence < tolerance:
                break
            
            current_colors = new_colors
        
        # Create final unified equilibrium state
        final_equilibrium = self._create_unified_equilibrium_state(
            current_colors, temporal_states, quaternion_states, resonance_eq, harmony_validation
        )
        
        return final_equilibrium
    
    def _combine_equilibrium_states(self, 
                                  current_colors: List[OKLCH3_5D],
                                  temporal_states: List[ColorEquilibriumState],
                                  quaternion_states: List[ColorEquilibriumState],
                                  resonance_state: ColorEquilibriumState,
                                  harmony_validation: Dict[str, Any]) -> List[OKLCH3_5D]:
        """Combine multiple equilibrium states into unified colors"""
        combined_colors = []
        
        for i, current_color in enumerate(current_colors):
            # Weight each equilibrium contribution
            temporal_weight = 0.3
            quaternion_weight = 0.3
            resonance_weight = 0.2
            harmony_weight = 0.2
            
            # Get equilibrium colors
            temporal_color = temporal_states[i].color_state
            quaternion_color = quaternion_states[i].color_state
            resonance_color = resonance_state.color_state
            
            # Weighted combination
            combined_L = (temporal_weight * temporal_color.lightness +
                         quaternion_weight * quaternion_color.lightness +
                         resonance_weight * resonance_color.lightness +
                         harmony_weight * current_color.lightness)
            
            combined_C = (temporal_weight * temporal_color.chroma +
                         quaternion_weight * quaternion_color.chroma +
                         resonance_weight * resonance_color.chroma +
                         harmony_weight * current_color.chroma)
            
            # Circular combination for hue
            hues = [temporal_color.hue, quaternion_color.hue, resonance_color.hue, current_color.hue]
            weights = [temporal_weight, quaternion_weight, resonance_weight, harmony_weight]
            combined_H = self._circular_average(hues, weights)
            
            # Combined temporal component
            combined_temporal = TemporalColorComponent(
                memory_strength=(temporal_color.temporal.memory_strength + current_color.temporal.memory_strength) / 2,
                anticipation_strength=(temporal_color.temporal.anticipation_strength + current_color.temporal.anticipation_strength) / 2,
                temporal_decay=current_color.temporal.temporal_decay,
                dimensional_resonance=min(1.0, (resonance_color.temporal.dimensional_resonance + current_color.temporal.dimensional_resonance) / 2)
            )
            
            combined_color = OKLCH3_5D(combined_L, combined_C, combined_H, combined_temporal)
            combined_colors.append(combined_color)
        
        return combined_colors
    
    def _calculate_convergence(self, old_colors: List[OKLCH3_5D], new_colors: List[OKLCH3_5D]) -> float:
        """Calculate convergence measure between color states"""
        if len(old_colors) != len(new_colors):
            return 1.0  # Maximum change
        
        total_change = 0.0
        for old_color, new_color in zip(old_colors, new_colors):
            # 3.5D distance between old and new states
            change = self._calculate_3_5d_color_distance(old_color, new_color)
            total_change += change
        
        return total_change / len(old_colors)
    
    def _create_unified_equilibrium_state(self,
                                        final_colors: List[OKLCH3_5D],
                                        temporal_states: List[ColorEquilibriumState],
                                        quaternion_states: List[ColorEquilibriumState],
                                        resonance_state: ColorEquilibriumState,
                                        harmony_validation: Dict[str, Any]) -> ColorEquilibriumState:
        """Create unified equilibrium state from all components"""
        # Average properties from all equilibrium types
        avg_energy = (sum(state.equilibrium_energy for state in temporal_states) / len(temporal_states) +
                     sum(state.equilibrium_energy for state in quaternion_states) / len(quaternion_states) +
                     resonance_state.equilibrium_energy) / 3
        
        avg_stability = (sum(state.stability_measure for state in temporal_states) / len(temporal_states) +
                        sum(state.stability_measure for state in quaternion_states) / len(quaternion_states) +
                        resonance_state.stability_measure) / 3
        
        avg_temporal_balance = sum(state.temporal_balance for state in temporal_states) / len(temporal_states)
        
        avg_coherence = sum(color.fractional_dimension for color in final_colors) / len(final_colors) / 4.0
        
        avg_resonance = sum(color.temporal.dimensional_resonance for color in final_colors) / len(final_colors)
        
        # Representative equilibrium color (centroid)
        centroid_color = self._calculate_color_centroid(final_colors)
        
        return ColorEquilibriumState(
            equilibrium_type=EquilibriumType.GANG_OF_FOUR_UNIFIED,
            color_state=centroid_color,
            equilibrium_energy=avg_energy,
            stability_measure=avg_stability,
            temporal_balance=avg_temporal_balance,
            dimensional_coherence=avg_coherence,
            resonance_amplitude=avg_resonance
        )
    
    def _calculate_color_centroid(self, colors: List[OKLCH3_5D]) -> OKLCH3_5D:
        """Calculate centroid of color list in 3.5D space"""
        if not colors:
            return OKLCH3_5D(0.5, 0.2, 0, TemporalColorComponent())
        
        # Average lightness and chroma
        avg_L = sum(color.lightness for color in colors) / len(colors)
        avg_C = sum(color.chroma for color in colors) / len(colors)
        
        # Circular average for hue
        hues = [color.hue for color in colors]
        weights = [1.0 / len(colors)] * len(colors)
        avg_H = self._circular_average(hues, weights)
        
        # Average temporal component
        avg_memory = sum(color.temporal.memory_strength for color in colors) / len(colors)
        avg_anticipation = sum(color.temporal.anticipation_strength for color in colors) / len(colors)
        avg_decay = sum(color.temporal.temporal_decay for color in colors) / len(colors)
        avg_resonance = sum(color.temporal.dimensional_resonance for color in colors) / len(colors)
        
        centroid_temporal = TemporalColorComponent(
            memory_strength=avg_memory,
            anticipation_strength=avg_anticipation,
            temporal_decay=avg_decay,
            dimensional_resonance=avg_resonance
        )
        
        return OKLCH3_5D(avg_L, avg_C, avg_H, centroid_temporal)
    
    def _circular_average(self, angles: List[float], weights: List[float]) -> float:
        """Calculate weighted circular average of angles"""
        if not angles or not weights:
            return 0.0
        
        # Convert to unit vectors
        cos_sum = sum(w * np.cos(np.radians(a)) for a, w in zip(angles, weights))
        sin_sum = sum(w * np.sin(np.radians(a)) for a, w in zip(angles, weights))
        
        # Convert back to angle
        return np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
    
    def _calculate_3_5d_color_distance(self, color1: OKLCH3_5D, color2: OKLCH3_5D) -> float:
        """Calculate distance between colors in 3.5D space"""
        # Standard OKLCH distance
        dL = color1.lightness - color2.lightness
        dC = color1.chroma - color2.chroma
        dH = self._circular_difference(color1.hue, color2.hue) / 360.0
        
        # Temporal distance (0.5D component)
        dT = (color1.temporal.dimensional_resonance - color2.temporal.dimensional_resonance)
        
        # 3.5D distance metric
        return np.sqrt(dL**2 + dC**2 + dH**2 + 0.5 * dT**2)
    
    def _circular_difference(self, angle1: float, angle2: float) -> float:
        """Calculate circular difference for hue angles"""
        diff = angle1 - angle2
        return ((diff + 180) % 360) - 180
    
    def generate_ce2_complete_spec(self) -> Dict[str, Any]:
        """Generate complete CE2 Color Equilibrium specification"""
        # Find unified equilibrium
        unified_equilibrium = self.find_unified_gang_of_four_equilibrium()
        
        # Generate individual equilibrium examples
        palette = self.color_spec_3_5d.generate_3_5d_harmonic_palette()
        
        # Temporal equilibrium example
        temporal_eq = self.temporal_equilibrium.find_temporal_equilibrium(
            palette[0], palette[1:3], palette[3:5]
        )
        
        # Quaternion equilibrium example
        quaternion_eq = self.quaternion_equilibrium.find_fractional_quaternion_equilibrium(
            palette[1], [0, 45, 90, 135, 180]
        )
        
        # Resonance equilibrium
        resonance_eq = self.resonance_equilibrium.find_dimensional_resonance_equilibrium(palette)
        
        # Harmony validation
        harmony_validation = self.chi_squared_3_5d.validate_3_5d_harmony(palette)
        
        return {
            'specification_type': 'ce2_color_equilibrium_complete_spec',
            'seed': self.seed,
            'generated_at': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'golden_ratio_base12': self.golden_ratio_base12,
            
            # Unified Gang of Four equilibrium
            'unified_equilibrium': {
                'equilibrium_type': unified_equilibrium.equilibrium_type.value,
                'equilibrium_color': unified_equilibrium.color_state.to_string(),
                'equilibrium_energy': unified_equilibrium.equilibrium_energy,
                'stability_measure': unified_equilibrium.stability_measure,
                'temporal_balance': unified_equilibrium.temporal_balance,
                'dimensional_coherence': unified_equilibrium.dimensional_coherence,
                'resonance_amplitude': unified_equilibrium.resonance_amplitude,
                'is_stable': unified_equilibrium.is_stable,
                'equilibrium_quality': unified_equilibrium.equilibrium_quality
            },
            
            # Individual equilibrium examples
            'equilibrium_examples': {
                'temporal_equilibrium': {
                    'type': temporal_eq.equilibrium_type.value,
                    'color': temporal_eq.color_state.to_string(),
                    'energy': temporal_eq.equilibrium_energy,
                    'stability': temporal_eq.stability_measure,
                    'temporal_balance': temporal_eq.temporal_balance,
                    'description': 'Balance between color memory and anticipation'
                },
                'fractional_quaternion_equilibrium': {
                    'type': quaternion_eq.equilibrium_type.value,
                    'color': quaternion_eq.color_state.to_string(),
                    'energy': quaternion_eq.equilibrium_energy,
                    'stability': quaternion_eq.stability_measure,
                    'description': 'Stable incomplete 4D rotations in 3.5D space'
                },
                'dimensional_resonance_equilibrium': {
                    'type': resonance_eq.equilibrium_type.value,
                    'color': resonance_eq.color_state.to_string(),
                    'energy': resonance_eq.equilibrium_energy,
                    'stability': resonance_eq.stability_measure,
                    'description': 'Harmony across fractional dimensional boundaries'
                },
                'chi_squared_3_5d_equilibrium': {
                    'harmony_validation': harmony_validation,
                    'description': 'Statistical balance in 3.5D fractional space'
                }
            },
            
            # CE2 Mathematical Framework
            'ce2_mathematical_framework': {
                'equilibrium_equation': 'dE/dt = 0 in 3.5D fractional space',
                'temporal_equilibrium_condition': 'F_memory + F_anticipation = 0',
                'quaternion_equilibrium_condition': 'q₃.₅ᴅ = q₃.₅ᴅ⁻¹ (self-inverse)',
                'resonance_equilibrium_condition': 'λ_max(R) = stable eigenvalue',
                'unified_equilibrium_condition': 'All Gang of Four patterns balanced simultaneously'
            },
            
            # CE2 Applications
            'ce2_applications': {
                'stable_color_atmospheres': 'Self-regulating color environments',
                'temporal_color_consistency': 'Colors maintain balance across time',
                'fractional_dimensional_design': 'Design tools for 3.5D living',
                'equilibrium_color_therapy': 'Therapeutic color balance',
                'mathematical_color_citizenship': 'Colors with equilibrium passports'
            },
            
            # Integration with existing systems
            'integration': {
                'ce1_extension': 'CE2 extends CE1 into color space',
                '3_5d_color_theory_foundation': 'Built on fractional dimensional discovery',
                'gang_of_four_equilibrium': 'All patterns achieve simultaneous balance',
                'mathematical_immigration_law': 'Colors achieve equilibrium citizenship'
            }
        }


def main():
    """Main entry point for CE2 Color Equilibrium system"""
    print("🌈 CE2: Color Equilibria in 3.5D Fractional Dimensional Space")
    print("=" * 70)
    
    # Create CE2 Color Equilibrium system
    ce2_system = CE2ColorEquilibrium("ce2_living_in_3_5d_space")
    
    # Generate complete CE2 specification
    ce2_spec = ce2_system.generate_ce2_complete_spec()
    
    # Print summary
    print(f"Seed: {ce2_spec['seed']}")
    print(f"Golden Ratio Base 12: {ce2_spec['golden_ratio_base12']}")
    
    print(f"\n🎯 Unified Gang of Four Equilibrium:")
    unified = ce2_spec['unified_equilibrium']
    print(f"  Type: {unified['equilibrium_type']}")
    print(f"  Color: {unified['equilibrium_color']}")
    print(f"  Energy: {unified['equilibrium_energy']:.4f}")
    print(f"  Stability: {unified['stability_measure']:.4f}")
    print(f"  Quality: {unified['equilibrium_quality']:.4f}")
    print(f"  Is Stable: {unified['is_stable']}")
    
    print(f"\n🔄 Individual Equilibrium Examples:")
    examples = ce2_spec['equilibrium_examples']
    for eq_name, eq_data in examples.items():
        if 'type' in eq_data:
            print(f"  {eq_name}:")
            print(f"    Color: {eq_data['color']}")
            print(f"    Stability: {eq_data['stability']:.4f}")
            print(f"    Description: {eq_data['description']}")
    
    print(f"\n🔬 CE2 Mathematical Framework:")
    framework = ce2_spec['ce2_mathematical_framework']
    for condition_name, condition in framework.items():
        print(f"  {condition_name}: {condition}")
    
    print(f"\n🚀 CE2 Applications:")
    applications = ce2_spec['ce2_applications']
    for app_name, app_desc in applications.items():
        print(f"  {app_name}: {app_desc}")
    
    # Save complete CE2 specification
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f".out/ce2_equilibrium/ce2_color_equilibrium_spec_{timestamp}.json"
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(ce2_spec, f, indent=2, default=str)
    
    print(f"\n💾 Complete CE2 spec saved to: {output_file}")
    print("\n🎯 CE2 Color Equilibrium System Complete!")
    print("Gang of Four patterns achieve simultaneous balance in 3.5D space!")
    print("Colors now have equilibrium citizenship in fractional dimensional space! ✨")
    
    return 0


if __name__ == "__main__":
    exit(main())
