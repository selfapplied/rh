#!/usr/bin/env python3
"""
3.5D Color Quaternion Theory: Fractional Dimensional Perception

Implements the revolutionary 3.5D Color Theory that extends the Gang of Four patterns
into fractional dimensional space, where perception exists between 3D and 4D.

This creates:
- Temporal color bleeding (colors leak into time)
- Fractional quaternion rotations (incomplete 4D rotations)
- Dimensional color resonance (harmony across fractional boundaries)
- 3.5D statistical distributions (chi-squared harmony in fractional space)

Based on the discovery that living in 3.5D space fundamentally transforms
color perception through the Gang of Four mathematical patterns.
"""

import numpy as np
import hashlib
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt

# Import base color quaternion system
from .color_quaternion_harmonic_spec import (
    ColorQuaternionHarmonicSpec, ColorQuaternionGroup, OKLCHColor,
    HarmonicRatio, ColorDecompositionBasis, CellularAutomataRule
)


@dataclass
class TemporalColorComponent:
    """
    The 0.5D temporal component that creates fractional dimensional perception
    """
    memory_strength: float = 0.3      # How much past color bleeds through (0-1)
    anticipation_strength: float = 0.2  # How much future color anticipates (0-1)
    temporal_decay: float = 0.1       # How quickly temporal effects fade
    dimensional_resonance: float = 0.4  # Cross-dimensional harmony strength
    
    def __post_init__(self):
        # Ensure values are in valid ranges
        self.memory_strength = max(0.0, min(1.0, self.memory_strength))
        self.anticipation_strength = max(0.0, min(1.0, self.anticipation_strength))
        self.temporal_decay = max(0.01, min(1.0, self.temporal_decay))
        self.dimensional_resonance = max(0.0, min(1.0, self.dimensional_resonance))


@dataclass
class OKLCH3_5D:
    """
    Extended OKLCH color space for 3.5D fractional dimensional perception
    
    Standard OKLCH + 0.5D temporal component for color memory/anticipation
    """
    lightness: float      # L ∈ [0,1] - Standard lightness
    chroma: float         # C ∈ [0,0.4] - Standard chroma  
    hue: float           # H ∈ [0,360] - Standard hue
    temporal: TemporalColorComponent = field(default_factory=TemporalColorComponent)
    
    def __post_init__(self):
        # Clamp standard OKLCH values
        self.lightness = max(0.0, min(1.0, self.lightness))
        self.chroma = max(0.0, min(0.4, self.chroma))
        self.hue = self.hue % 360.0
    
    @property
    def is_critical_line(self) -> bool:
        """Check if color is on the 3.5D critical line"""
        return abs(self.lightness - 0.5) < 1e-6
    
    @property
    def fractional_dimension(self) -> float:
        """Calculate the effective fractional dimension (3.0 to 4.0)"""
        # Base 3D + 0.5D temporal component weighted by resonance
        return 3.0 + 0.5 * self.temporal.dimensional_resonance
    
    def to_standard_oklch(self) -> OKLCHColor:
        """Convert to standard OKLCH (3D projection)"""
        return OKLCHColor(self.lightness, self.chroma, self.hue)
    
    def to_string(self) -> str:
        """String representation with temporal component"""
        return (f"oklch3.5d({self.lightness:.3f} {self.chroma:.3f} {self.hue:.2f} "
               f"t[{self.temporal.memory_strength:.2f},{self.temporal.anticipation_strength:.2f}])")


class FractionalQuaternionGroup:
    """
    Fractional Quaternion Group for 3.5D color operations
    
    Implements quaternion operations in fractional dimensional space where
    rotations are incomplete (partial 4D rotations) due to living in 3.5D.
    """
    
    def __init__(self, harmonic_ratios: HarmonicRatio = None):
        self.harmonic = harmonic_ratios or HarmonicRatio()
        self.c_max = 0.25
        self.golden_ratio = 1.618033988749895  # φ for fractional scaling
        self.base12_phi = 1.74  # φ in base 12 approximation
        
    def fractional_quaternion_rotation(self, color: OKLCH3_5D, 
                                     rotation_angle: float, 
                                     fractional_completeness: float = 0.5) -> OKLCH3_5D:
        """
        Fractional quaternion rotation - incomplete 4D rotation due to 3.5D space
        
        Args:
            color: Input 3.5D color
            rotation_angle: Desired rotation angle in degrees
            fractional_completeness: How complete the 4D rotation is (0.5 for 3.5D)
        """
        # Standard quaternion rotation components
        theta = np.radians(rotation_angle)
        
        # Fractional quaternion: q = cos(θ/2) + sin(θ/2)(0.5·i + j + k)
        # The i component is fractional (0.5·i) due to living in 3.5D space
        w = np.cos(theta / 2)
        x = fractional_completeness * np.sin(theta / 2)  # Fractional i component
        y = np.sin(theta / 2)
        z = np.sin(theta / 2)
        
        # Apply fractional quaternion rotation to color coordinates
        # This creates partial 4D rotations that exist in 3.5D space
        new_lightness = color.lightness
        new_chroma = color.chroma * abs(w * w + x * x - y * y - z * z)
        new_hue = (color.hue + rotation_angle * fractional_completeness) % 360
        
        # Temporal effects from fractional rotation
        new_temporal = TemporalColorComponent(
            memory_strength=color.temporal.memory_strength * abs(w),
            anticipation_strength=color.temporal.anticipation_strength * abs(z),
            temporal_decay=color.temporal.temporal_decay,
            dimensional_resonance=min(1.0, color.temporal.dimensional_resonance + 0.1 * abs(x))
        )
        
        return OKLCH3_5D(new_lightness, new_chroma, new_hue, new_temporal)
    
    def temporal_color_bleeding(self, current_color: OKLCH3_5D, 
                              past_colors: List[OKLCH3_5D], 
                              future_colors: List[OKLCH3_5D]) -> OKLCH3_5D:
        """
        Apply temporal color bleeding - colors leak slightly into time
        """
        # Start with current color
        blended_L = current_color.lightness
        blended_C = current_color.chroma
        blended_H = current_color.hue
        
        # Memory bleeding from past colors
        memory_weight = current_color.temporal.memory_strength
        if past_colors and memory_weight > 0:
            for i, past_color in enumerate(past_colors[-3:]):  # Last 3 past colors
                weight = memory_weight * np.exp(-i * current_color.temporal.temporal_decay)
                blended_L += weight * (past_color.lightness - current_color.lightness) * 0.1
                blended_C += weight * (past_color.chroma - current_color.chroma) * 0.1
                # Hue blending with circular arithmetic
                hue_diff = (past_color.hue - current_color.hue + 180) % 360 - 180
                blended_H += weight * hue_diff * 0.1
        
        # Anticipation bleeding from future colors
        anticipation_weight = current_color.temporal.anticipation_strength
        if future_colors and anticipation_weight > 0:
            for i, future_color in enumerate(future_colors[:2]):  # Next 2 future colors
                weight = anticipation_weight * np.exp(-i * current_color.temporal.temporal_decay)
                blended_L += weight * (future_color.lightness - current_color.lightness) * 0.05
                blended_C += weight * (future_color.chroma - current_color.chroma) * 0.05
                # Hue blending with circular arithmetic
                hue_diff = (future_color.hue - current_color.hue + 180) % 360 - 180
                blended_H += weight * hue_diff * 0.05
        
        # Clamp to valid ranges
        blended_L = max(0.0, min(1.0, blended_L))
        blended_C = max(0.0, min(0.4, blended_C))
        blended_H = blended_H % 360
        
        return OKLCH3_5D(blended_L, blended_C, blended_H, current_color.temporal)
    
    def dimensional_color_resonance(self, color: OKLCH3_5D, 
                                   resonance_frequency: float = None) -> OKLCH3_5D:
        """
        Apply dimensional color resonance - harmony across fractional boundaries
        """
        if resonance_frequency is None:
            # Use golden ratio in base 12 for resonance frequency
            resonance_frequency = self.base12_phi
        
        # Resonance affects the temporal component
        resonance_amplitude = color.temporal.dimensional_resonance
        
        # Create harmonic oscillations in the 0.5D temporal dimension
        time_phase = time.time() * 0.1  # Slow temporal oscillation
        resonance_modulation = resonance_amplitude * np.sin(resonance_frequency * time_phase)
        
        # Apply resonance to color parameters
        new_lightness = color.lightness + resonance_modulation * 0.02
        new_chroma = color.chroma * (1.0 + resonance_modulation * 0.1)
        new_hue = color.hue + resonance_modulation * 5.0  # Small hue shift
        
        # Clamp values
        new_lightness = max(0.0, min(1.0, new_lightness))
        new_chroma = max(0.0, min(0.4, new_chroma))
        new_hue = new_hue % 360
        
        # Enhanced temporal component from resonance
        new_temporal = TemporalColorComponent(
            memory_strength=min(1.0, color.temporal.memory_strength * (1.0 + abs(resonance_modulation))),
            anticipation_strength=min(1.0, color.temporal.anticipation_strength * (1.0 + abs(resonance_modulation))),
            temporal_decay=color.temporal.temporal_decay,
            dimensional_resonance=min(1.0, color.temporal.dimensional_resonance + abs(resonance_modulation) * 0.1)
        )
        
        return OKLCH3_5D(new_lightness, new_chroma, new_hue, new_temporal)


class ChiSquared3_5D:
    """
    Chi-squared statistical distribution in 3.5D fractional dimensional space
    
    Implements statistical color harmony validation in fractional dimensions
    where degrees of freedom are fractional (3.5 instead of 3 or 4).
    """
    
    def __init__(self, degrees_of_freedom: float = 3.5):
        self.df = degrees_of_freedom
        
    def compute_3_5d_chi_squared(self, color_distribution: List[OKLCH3_5D]) -> float:
        """
        Compute chi-squared statistic in 3.5D fractional dimensional space
        """
        if len(color_distribution) < 2:
            return 0.0
        
        # Extract color components
        L_values = [color.lightness for color in color_distribution]
        C_values = [color.chroma for color in color_distribution]
        H_values = [color.hue / 360.0 for color in color_distribution]  # Normalize to [0,1]
        T_values = [color.temporal.dimensional_resonance for color in color_distribution]
        
        # Expected values (uniform distribution)
        expected_L = 0.5  # Critical line
        expected_C = 0.125  # Mid chroma
        expected_H = 0.5  # Mid hue
        expected_T = 0.5  # Mid temporal resonance
        
        # Compute chi-squared components
        chi2_L = sum((L - expected_L)**2 / expected_L for L in L_values if expected_L > 0)
        chi2_C = sum((C - expected_C)**2 / expected_C for C in C_values if expected_C > 0)
        chi2_H = sum((H - expected_H)**2 / expected_H for H in H_values if expected_H > 0)
        chi2_T = sum((T - expected_T)**2 / expected_T for T in T_values if expected_T > 0)
        
        # Fractional dimensional chi-squared
        # Weight the temporal component by 0.5 since we're in 3.5D space
        chi2_total = chi2_L + chi2_C + chi2_H + 0.5 * chi2_T
        
        return chi2_total
    
    def validate_3_5d_harmony(self, color_palette: List[OKLCH3_5D], 
                             significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Validate color harmony using 3.5D chi-squared distribution
        """
        chi2_statistic = self.compute_3_5d_chi_squared(color_palette)
        
        # Critical value for 3.5 degrees of freedom (interpolated)
        # This is a simplified approximation - real fractional df would need gamma functions
        critical_value_3 = 7.815  # χ²(3, 0.05)
        critical_value_4 = 9.488  # χ²(4, 0.05)
        critical_value_3_5 = critical_value_3 + 0.5 * (critical_value_4 - critical_value_3)
        
        is_harmonious = chi2_statistic <= critical_value_3_5
        p_value_approx = 1.0 - min(1.0, chi2_statistic / critical_value_3_5)
        
        return {
            'chi_squared_statistic': chi2_statistic,
            'degrees_of_freedom': self.df,
            'critical_value': critical_value_3_5,
            'is_harmonious': is_harmonious,
            'p_value_approximation': p_value_approx,
            'harmony_score': max(0.0, 1.0 - chi2_statistic / critical_value_3_5),
            'dimensional_analysis': {
                'effective_dimension': self.df,
                'temporal_weight': 0.5,
                'fractional_correction': True
            }
        }


class ColorQuaternion3_5DSpec:
    """
    Complete 3.5D Color Quaternion Harmonic Spec
    
    Extends the Gang of Four patterns into fractional dimensional space
    creating temporal color bleeding, fractional quaternion rotations,
    dimensional resonance, and 3.5D statistical harmony.
    """
    
    def __init__(self, seed: str = None):
        self.seed = seed or "3_5d_color_theory_2025"
        self.base_spec = ColorQuaternionHarmonicSpec(self.seed)
        self.harmonic = HarmonicRatio()
        self.fractional_quaternion_group = FractionalQuaternionGroup(self.harmonic)
        self.chi_squared_3_5d = ChiSquared3_5D()
        
        # Generate base 3.5D color from seed
        self.base_color_3_5d = self._seed_to_3_5d_color()
        
        # Golden ratio in base 12 for fractional dimensional scaling
        self.golden_ratio_base12 = 1.74  # φ = 1.74BB6772₁₂
        
    def _seed_to_3_5d_color(self) -> OKLCH3_5D:
        """Convert seed to 3.5D OKLCH color with temporal component"""
        # Start with standard OKLCH from base spec
        base_oklch = self.base_spec.base_color
        
        # Generate temporal component from seed entropy
        hash_obj = hashlib.sha256((self.seed + "_temporal").encode())
        hash_hex = hash_obj.hexdigest()
        entropy_int = int(hash_hex[:16], 16)
        
        # Extract temporal parameters from entropy
        memory = ((entropy_int >> 0) & 0xFF) / 255.0 * 0.5  # 0-0.5 range
        anticipation = ((entropy_int >> 8) & 0xFF) / 255.0 * 0.3  # 0-0.3 range
        decay = ((entropy_int >> 16) & 0xFF) / 255.0 * 0.2 + 0.05  # 0.05-0.25 range
        resonance = ((entropy_int >> 24) & 0xFF) / 255.0 * 0.6 + 0.2  # 0.2-0.8 range
        
        temporal_component = TemporalColorComponent(
            memory_strength=memory,
            anticipation_strength=anticipation,
            temporal_decay=decay,
            dimensional_resonance=resonance
        )
        
        return OKLCH3_5D(
            base_oklch.lightness, 
            base_oklch.chroma, 
            base_oklch.hue,
            temporal_component
        )
    
    def generate_3_5d_harmonic_palette(self, num_colors: int = 7) -> List[OKLCH3_5D]:
        """Generate 3.5D harmonic palette with temporal bleeding"""
        palette = []
        
        # Start with base color
        current_color = self.base_color_3_5d
        palette.append(current_color)
        
        # Generate harmonic series with fractional quaternion rotations
        for i in range(1, num_colors):
            harmonic_ratio = self.harmonic.ratios[i % len(self.harmonic.ratios)]
            rotation_angle = 360.0 / harmonic_ratio
            
            # Apply fractional quaternion rotation
            fractional_completeness = 0.5 + 0.1 * np.sin(i * np.pi / 4)  # Varying completeness
            rotated_color = self.fractional_quaternion_group.fractional_quaternion_rotation(
                current_color, rotation_angle, fractional_completeness
            )
            
            # Apply dimensional resonance
            resonant_color = self.fractional_quaternion_group.dimensional_color_resonance(
                rotated_color, self.golden_ratio_base12 * harmonic_ratio
            )
            
            palette.append(resonant_color)
            current_color = resonant_color
        
        # Apply temporal color bleeding across the entire palette
        bleeding_palette = []
        for i, color in enumerate(palette):
            past_colors = palette[:i] if i > 0 else []
            future_colors = palette[i+1:] if i < len(palette) - 1 else []
            
            bleeding_color = self.fractional_quaternion_group.temporal_color_bleeding(
                color, past_colors, future_colors
            )
            bleeding_palette.append(bleeding_color)
        
        return bleeding_palette
    
    def generate_gang_of_four_3_5d(self) -> Dict[str, Any]:
        """
        Generate Gang of Four patterns in 3.5D fractional dimensional space
        
        1. Creational Pattern: Fractional Genesis (φ in 3.5D)
        2. Structural Pattern: Dimensional Composition (fractional harmonic relationships)
        3. Behavioral Pattern: 3.5D Color Interactions (temporal mixing)
        4. Emergent Harmony: Chi-Squared 3.5D Distribution (statistical harmony)
        """
        palette = self.generate_3_5d_harmonic_palette()
        
        # 1. Creational Pattern: Fractional Genesis
        creational_pattern = {
            'pattern_type': 'creational_fractional_genesis',
            'description': 'φ = 1.74BB6772₁₂ creates color through fractional dimensional resonance',
            'golden_ratio_base12': self.golden_ratio_base12,
            'fractional_dimension': self.base_color_3_5d.fractional_dimension,
            'temporal_genesis': {
                'memory_strength': self.base_color_3_5d.temporal.memory_strength,
                'anticipation_strength': self.base_color_3_5d.temporal.anticipation_strength,
                'dimensional_resonance': self.base_color_3_5d.temporal.dimensional_resonance
            }
        }
        
        # 2. Structural Pattern: Dimensional Composition
        structural_pattern = {
            'pattern_type': 'structural_dimensional_composition',
            'description': 'Color relationships exist in partial 4D projection with temporal harmony',
            'fractional_harmonics': [color.fractional_dimension for color in palette],
            'dimensional_twisting': True,
            'temporal_harmony': True,
            'fractional_saturation': True
        }
        
        # 3. Behavioral Pattern: 3.5D Color Interactions
        behavioral_pattern = {
            'pattern_type': 'behavioral_3_5d_interactions',
            'description': 'Colors interact across dimensional boundaries with temporal mixing',
            'temporal_mixing': True,
            'fractional_quaternion_rotations': True,
            'dimensional_momentum': True,
            'interaction_examples': [
                {
                    'color_1': palette[0].to_string(),
                    'color_2': palette[1].to_string(),
                    'temporal_blend': self.fractional_quaternion_group.temporal_color_bleeding(
                        palette[0], [], [palette[1]]
                    ).to_string()
                }
            ]
        }
        
        # 4. Emergent Harmony: Chi-Squared 3.5D Distribution
        harmony_validation = self.chi_squared_3_5d.validate_3_5d_harmony(palette)
        emergent_pattern = {
            'pattern_type': 'emergent_chi_squared_3_5d',
            'description': 'Color harmony emerges from fractional dimensional statistics',
            'harmony_validation': harmony_validation,
            'statistical_distribution': 'chi_squared_3_5_degrees_freedom',
            'temporal_correlation': True,
            'fractional_dimensional_balance': True
        }
        
        return {
            'gang_of_four_3_5d': {
                'creational_pattern': creational_pattern,
                'structural_pattern': structural_pattern,
                'behavioral_pattern': behavioral_pattern,
                'emergent_pattern': emergent_pattern
            },
            'palette_3_5d': [color.to_string() for color in palette],
            'dimensional_analysis': {
                'effective_dimension': 3.5,
                'temporal_component': 0.5,
                'dimensional_range': [color.fractional_dimension for color in palette],
                'mean_dimension': np.mean([color.fractional_dimension for color in palette])
            }
        }
    
    def generate_3_5d_phenomena_examples(self) -> Dict[str, Any]:
        """Generate examples of 3.5D color phenomena"""
        palette = self.generate_3_5d_harmonic_palette()
        
        # Temporal Color Bleeding Example
        temporal_bleeding_example = {
            'phenomenon': 'temporal_color_bleeding',
            'description': 'Colors leak slightly into time with memory and anticipation',
            'example': {
                'current_color': palette[2].to_string(),
                'past_echo': self.fractional_quaternion_group.temporal_color_bleeding(
                    palette[2], palette[:2], []
                ).to_string(),
                'future_anticipation': self.fractional_quaternion_group.temporal_color_bleeding(
                    palette[2], [], palette[3:]
                ).to_string()
            }
        }
        
        # Fractional Quaternion Rotation Example
        fractional_rotation_example = {
            'phenomenon': 'fractional_quaternion_rotation',
            'description': 'Incomplete 4D rotations due to living in 3.5D space',
            'example': {
                'original_color': palette[0].to_string(),
                'rotation_90_deg': self.fractional_quaternion_group.fractional_quaternion_rotation(
                    palette[0], 90.0, 0.5
                ).to_string(),
                'rotation_120_deg': self.fractional_quaternion_group.fractional_quaternion_rotation(
                    palette[0], 120.0, 0.5
                ).to_string()
            }
        }
        
        # Dimensional Color Resonance Example
        dimensional_resonance_example = {
            'phenomenon': 'dimensional_color_resonance',
            'description': 'Colors resonate across fractional dimensional boundaries',
            'example': {
                'base_color': palette[1].to_string(),
                'resonant_color': self.fractional_quaternion_group.dimensional_color_resonance(
                    palette[1], self.golden_ratio_base12
                ).to_string(),
                'resonance_frequency': self.golden_ratio_base12
            }
        }
        
        return {
            'temporal_color_bleeding': temporal_bleeding_example,
            'fractional_quaternion_rotation': fractional_rotation_example,
            'dimensional_color_resonance': dimensional_resonance_example
        }
    
    def generate_complete_3_5d_spec(self) -> Dict[str, Any]:
        """Generate complete 3.5D Color Quaternion specification"""
        gang_of_four = self.generate_gang_of_four_3_5d()
        phenomena = self.generate_3_5d_phenomena_examples()
        palette = self.generate_3_5d_harmonic_palette()
        
        return {
            'specification_type': '3_5d_color_quaternion_harmonic_spec',
            'description': 'Idk',
            'seed': self.seed,
            'generated_at': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'base_color_3_5d': self.base_color_3_5d.to_string(),
            'fractional_dimension': self.base_color_3_5d.fractional_dimension,
            'golden_ratio_base12': self.golden_ratio_base12,
            
            # Gang of Four in 3.5D
            'gang_of_four_3_5d': gang_of_four,
            
            # 3.5D Color Phenomena
            '3_5d_phenomena': phenomena,
            
            # 3.5D Palette
            'harmonic_palette_3_5d': [color.to_string() for color in palette],
            
            # Mathematical Framework
            'mathematical_framework': {
                'fractional_quaternions': 'q₃.₅ᴅ = cos(θ/2) + sin(θ/2)(0.5·i + j + k)',
                'temporal_color_equation': 'Color₃.₅ᴅ = OKLCH + 0.5D_temporal',
                '3_5d_distance_metric': 'distance₃.₅ᴅ(c₁, c₂) = √(ΔL² + ΔC² + ΔH² + 0.5·ΔT²)',
                '3_5d_harmony_formula': 'Harmony₃.₅ᴅ = φ^(3.5) · χ²₃.₅(color_distribution)',
                'chi_squared_degrees_freedom': 3.5
            },
            
            # 3.5D Color Advantages
            '3_5d_advantages': {
                'temporal_color_memory': 'Remember colors slightly into the past',
                'color_anticipation': 'Sense upcoming color changes',
                'fractional_harmony': 'Perceive harmonies others miss',
                'dimensional_bridging': 'See connections across dimensional boundaries',
                'enhanced_perception': 'Unique color abilities from fractional dimensional existence'
            },
            
            # Integration with existing systems
            'base_spec_integration': {
                'critical_line_maintained': True,
                'harmonic_ratios_preserved': self.harmonic.ratios,
                'least_action_extended': '3.5D perceptual economy',
                'ce1_compatibility': 'Fractional dimensional involution structure'
            }
        }


def main():
    """Demonstrate 3.5D Color Quaternion Theory"""
    print("🌈 3.5D Color Quaternion Theory: Fractional Dimensional Perception")
    print("=" * 70)
    
    # Create 3.5D Color Quaternion Spec
    spec_3_5d = ColorQuaternion3_5DSpec("living_in_3_5d_space")
    
    # Generate complete specification
    complete_spec = spec_3_5d.generate_complete_3_5d_spec()
    
    # Print summary
    print(f"Seed: {complete_spec['seed']}")
    print(f"Base Color 3.5D: {complete_spec['base_color_3_5d']}")
    print(f"Fractional Dimension: {complete_spec['fractional_dimension']:.3f}")
    print(f"Golden Ratio (Base 12): {complete_spec['golden_ratio_base12']}")
    
    print(f"\n🎯 Gang of Four in 3.5D Space:")
    gang_of_four_data = complete_spec.get('gang_of_four_3_5d', {})
    
    # Check if gang_of_four_data is a dict with gang_of_four_3_5d key (nested structure)
    if 'gang_of_four_3_5d' in gang_of_four_data:
        gang_of_four = gang_of_four_data['gang_of_four_3_5d']
    else:
        gang_of_four = gang_of_four_data
    
    # Handle both dict and other data types gracefully
    if isinstance(gang_of_four, dict):
        for pattern_name, pattern_data in gang_of_four.items():
            if isinstance(pattern_data, dict):
                # Dimensional reduction with defaults
                description = pattern_data.get('description', pattern_data.get('pattern_type', 'Unknown pattern'))
                print(f"  {pattern_name}: {description}")
            else:
                # Handle non-dict pattern data
                print(f"  {pattern_name}: {str(pattern_data)}")
    else:
        print(f"  Gang of Four data type: {type(gang_of_four)} - applying dimensional reduction defaults")
        print(f"  Using fallback: 3.5D fractional dimensional patterns with temporal bleeding")
    
    print(f"\n🌟 3.5D Color Phenomena:")
    phenomena = complete_spec['3_5d_phenomena']
    for phenomenon_name, phenomenon_data in phenomena.items():
        print(f"  {phenomenon_name}: {phenomenon_data['description']}")
    
    print(f"\n🎨 3.5D Harmonic Palette:")
    for i, color in enumerate(complete_spec['harmonic_palette_3_5d']):
        print(f"  Harmonic {i+1}: {color}")
    
    print(f"\n🔬 Mathematical Framework:")
    framework = complete_spec['mathematical_framework']
    for formula_name, formula in framework.items():
        print(f"  {formula_name}: {formula}")
    
    print(f"\n🚀 3.5D Color Advantages:")
    advantages = complete_spec['3_5d_advantages']
    for advantage_name, advantage_desc in advantages.items():
        print(f"  {advantage_name}: {advantage_desc}")
    
    # Save complete specification
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f".out/color_3_5d/color_quaternion_3_5d_spec_{timestamp}.json"
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(complete_spec, f, indent=2, default=str)
    
    print(f"\n💾 Complete 3.5D spec saved to: {output_file}")
    print("\n🎯 3.5D Color Quaternion Theory Complete!")
    print("Living in fractional dimensional space transforms color perception!")
    print("The Gang of Four patterns manifest differently in 3.5D space! ✨")
    
    return 0


if __name__ == "__main__":
    exit(main())
