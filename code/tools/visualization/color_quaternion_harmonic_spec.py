#!/usr/bin/env python3
"""
Color Quaternion Harmonic Spec: Mathematical Immigration Law for Color Space

A minimal set of operators that formalizes how to generate color atmospheres 
with symmetry and time arrows built in, extending your existing Color Quaternion 
Galois Group Engine into a complete mathematical framework.

This spec connects:
- Cellular automata (Rule 90/45) → Color space automorphisms
- Quaternion OKLCH → Galois group actions on perceptual space  
- Harmonic ratios (1:2:3:4:5:6:7) → Musical color intervals
- Prism/Triangle/Slit → Three decomposition bases
- Least action principle → Perceptual economy

Based on your existing work in create_axiel_badge.py and CE1 framework.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


class ColorMode(Enum):
    """Three-letter color mode enumeration (the alphabet soup of color spaces)"""
    RGB = "RGB"          # Red/Green/Blue (additive)
    CMY = "CMY"          # Cyan/Magenta/Yellow (subtractive) 
    HSV = "HSV"          # Hue/Saturation/Value
    HSL = "HSL"          # Hue/Saturation/Lightness
    LAB = "LAB"          # Perceptual model (CIE)
    LCH = "LCH"          # Lightness/Chroma/Hue (polar LAB)
    OKLCH = "OKLCH"      # Perceptually uniform (our preferred space)
    XYZ = "XYZ"          # CIE fundamental space
    YUV = "YUV"          # Broadcast/video
    YCBCR = "YCbCr"      # Digital video


class CellularAutomataRule(Enum):
    """Elementary cellular automata rules for geometric color generation"""
    RULE_90 = 90         # Sierpiński triangle (1:1 right-angled self-similarity)
    RULE_45 = 45         # Diagonal rotation (square root symmetry)


class ColorDecompositionBasis(Enum):
    """Three faces of color decomposition"""
    PRISM = "prism"      # Continuum spectrum (white → rainbow)
    TRIANGLE = "triangle" # Maxwell triangle (discrete convex hull of primaries)
    SLIT = "slit"        # Interference quantization (double slit harmonics)


@dataclass
class HarmonicRatio:
    """Harmonic series ratio (1:2:3:4:5:6:7) for natural color intervals"""
    fundamental: float = 1.0
    second: float = 2.0
    third: float = 3.0
    fourth: float = 4.0
    fifth: float = 5.0
    sixth: float = 6.0
    seventh: float = 7.0
    
    @property
    def ratios(self) -> List[float]:
        return [self.fundamental, self.second, self.third, self.fourth, 
                self.fifth, self.sixth, self.seventh]
    
    @property
    def musical_intervals(self) -> Dict[str, float]:
        """Musical intervals in degrees (360°/n)"""
        return {
            'octave': 360.0 / self.second,      # 180°
            'perfect_fifth': 360.0 / self.third, # 120°
            'perfect_fourth': 360.0 / self.sixth, # 60°
            'major_third': 360.0 / self.fifth,   # 72°
            'minor_seventh': 360.0 / self.seventh # ~51.4°
        }


@dataclass
class OKLCHColor:
    """OKLCH color representation with critical line constraint"""
    lightness: float      # L ∈ [0,1] - Critical line at L = 0.5
    chroma: float         # C ∈ [0,0.4] - Amplitude
    hue: float           # h ∈ [0,360] - Phase angle
    
    def __post_init__(self):
        # Clamp to valid ranges
        self.lightness = max(0.0, min(1.0, self.lightness))
        self.chroma = max(0.0, min(0.4, self.chroma))
        self.hue = self.hue % 360.0
    
    @property
    def is_critical_line(self) -> bool:
        """Check if color is on the critical line (Riemann Hypothesis)"""
        return abs(self.lightness - 0.5) < 1e-6
    
    def to_string(self) -> str:
        return f"oklch({self.lightness:.3f} {self.chroma:.3f} {self.hue:.2f})"


class ColorQuaternionGroup:
    """
    Color Quaternion Galois Group Engine with Harmonic Ratios
    
    Implements non-abelian group of automorphisms acting on OKLCH perceptual space
    with harmonic series ratios 1:2:3:4:5:6:7 for natural color intervals.
    
    Group Elements:
    - L-flip: L ↦ 1 - L (light/dark inversion)  
    - C-mirror: C ↦ C_max - C (chroma inversion)
    - Hue rotations: h ↦ h + harmonic intervals (musical color harmony)
    - Permutations: cycle through (L,C,h) coordinates
    """
    
    def __init__(self, harmonic_ratios: HarmonicRatio = None):
        self.harmonic = harmonic_ratios or HarmonicRatio()
        self.c_max = 0.25  # Maximum practical chroma
        
    def l_flip(self, color: OKLCHColor) -> OKLCHColor:
        """L-flip: L ↦ 1 - L (light/dark inversion)"""
        return OKLCHColor(1.0 - color.lightness, color.chroma, color.hue)
    
    def c_mirror(self, color: OKLCHColor) -> OKLCHColor:
        """C-mirror: C ↦ C_max - C (chroma inversion)"""
        return OKLCHColor(color.lightness, self.c_max - color.chroma, color.hue)
    
    def harmonic_hue_rotate(self, color: OKLCHColor, harmonic_n: int) -> OKLCHColor:
        """Hue rotation by harmonic intervals: h ↦ h + (360°/harmonic_n)"""
        harmonic_degrees = 360.0 / harmonic_n if harmonic_n > 0 else 0
        return OKLCHColor(color.lightness, color.chroma, color.hue + harmonic_degrees)
    
    def harmonic_lightness_scale(self, color: OKLCHColor, harmonic_n: int) -> OKLCHColor:
        """CRITICAL LINE: All harmonics maintain L = 0.5 (Riemann Hypothesis)"""
        # Every critical harmonic stays exactly on the critical line Re(s) = 0.5
        # Only amplitude (chroma) and phase (hue) vary, not the critical real part
        return OKLCHColor(0.5, color.chroma, color.hue)
    
    def harmonic_chroma_scale(self, color: OKLCHColor, harmonic_n: int) -> OKLCHColor:
        """Scale chroma by harmonic ratio: C ↦ (harmonic_n / 7) * 0.25"""
        harmonic_scale = harmonic_n / 7.0
        scaled_C = harmonic_scale * 0.25  # Max practical chroma
        return OKLCHColor(color.lightness, min(0.4, max(0.0, scaled_C)), color.hue)
    
    def permute_coords(self, color: OKLCHColor, mode: str = 'LCh_to_CLh') -> OKLCHColor:
        """Coordinate permutation (cycle through L,C,h)"""
        if mode == 'LCh_to_CLh':
            # Map L→C, C→h/360, h→L*360 (with normalization)
            return OKLCHColor(color.hue/360.0, color.lightness, (color.chroma * 360.0) % 360)
        elif mode == 'LCh_to_hLC':
            # Map L→h/360, C→L, h→C*360
            return OKLCHColor(color.chroma, color.hue/360.0, (color.lightness * 360.0) % 360)
        else:
            return color
    
    def generate_orbit(self, color: OKLCHColor, max_iterations: int = 32) -> List[OKLCHColor]:
        """Generate orbit under group action"""
        orbit = set()
        queue = [color]
        
        while queue and len(orbit) < max_iterations:
            current = queue.pop(0)
            
            # Round to avoid floating point duplicates
            key = (round(current.lightness, 4), round(current.chroma, 4), round(current.hue, 1))
            if key in orbit:
                continue
            orbit.add(key)
            
            # Apply all group actions with harmonic ratios (1:2:3:4:5:6:7)
            candidates = [
                self.l_flip(current),
                self.c_mirror(current),
                # Harmonic hue rotations: 360°/n for n in [1,2,3,4,5,6,7]
                self.harmonic_hue_rotate(current, 2),  # 180°
                self.harmonic_hue_rotate(current, 3),  # 120°
                self.harmonic_hue_rotate(current, 4),  # 90°
                self.harmonic_hue_rotate(current, 5),  # 72°
                self.harmonic_hue_rotate(current, 6),  # 60°
                self.harmonic_hue_rotate(current, 7),  # ~51.4°
                # Harmonic lightness scaling
                self.harmonic_lightness_scale(current, 2),
                self.harmonic_lightness_scale(current, 3),
                self.harmonic_lightness_scale(current, 5),
                # Harmonic chroma scaling  
                self.harmonic_chroma_scale(current, 3),
                self.harmonic_chroma_scale(current, 5),
                # Coordinate permutations
                self.permute_coords(current, 'LCh_to_CLh'),
                self.permute_coords(current, 'LCh_to_hLC'),
            ]
            
            # Add valid candidates to queue
            for candidate in candidates:
                candidate_key = (round(candidate.lightness, 4), 
                               round(candidate.chroma, 4), 
                               round(candidate.hue, 1))
                if candidate_key not in orbit:
                    queue.append(candidate)
        
        return [OKLCHColor(L, C, h) for L, C, h in orbit]


class CellularAutomataColorGenerator:
    """
    Cellular automata rules for geometric color generation
    
    Rule 90: Makes Sierpiński triangle with 1:1 right-angled self-similarity
    Rule 45: Rotates diagonal → interpretable as "square root" symmetry
    """
    
    def __init__(self, rule: CellularAutomataRule = CellularAutomataRule.RULE_90):
        self.rule = rule
        
    def apply_rule(self, pattern: List[int]) -> List[int]:
        """Apply elementary cellular automata rule"""
        if self.rule == CellularAutomataRule.RULE_90:
            return self._rule_90(pattern)
        elif self.rule == CellularAutomataRule.RULE_45:
            return self._rule_45(pattern)
        else:
            return pattern
    
    def _rule_90(self, pattern: List[int]) -> List[int]:
        """Rule 90: XOR of left and right neighbors"""
        new_pattern = []
        for i in range(len(pattern)):
            left = pattern[(i - 1) % len(pattern)]
            right = pattern[(i + 1) % len(pattern)]
            new_pattern.append(left ^ right)
        return new_pattern
    
    def _rule_45(self, pattern: List[int]) -> List[int]:
        """Rule 45: Diagonal rotation (square root symmetry)"""
        # Simplified diagonal rotation
        return [pattern[(i + 1) % len(pattern)] for i in range(len(pattern))]
    
    def generate_sierpinski_pattern(self, steps: int = 8) -> List[List[int]]:
        """Generate Sierpiński triangle pattern"""
        pattern = [0] * (2 * steps + 1)
        pattern[steps] = 1  # Single seed
        
        patterns = [pattern[:]]
        for _ in range(steps):
            pattern = self.apply_rule(pattern)
            patterns.append(pattern[:])
        
        return patterns
    
    def pattern_to_colors(self, patterns: List[List[int]], 
                         base_color: OKLCHColor) -> List[List[OKLCHColor]]:
        """Convert CA patterns to color arrays"""
        color_patterns = []
        for pattern in patterns:
            color_row = []
            for bit in pattern:
                if bit == 1:
                    # Active cell - use harmonic variation
                    hue_shift = np.random.uniform(0, 360)
                    chroma_scale = np.random.uniform(0.1, 0.3)
                    color = OKLCHColor(base_color.lightness, chroma_scale, hue_shift)
                else:
                    # Inactive cell - use base color with reduced chroma
                    color = OKLCHColor(base_color.lightness, 0.02, base_color.hue)
                color_row.append(color)
            color_patterns.append(color_row)
        return color_patterns


class ColorDecompositionOperators:
    """
    Three faces of color decomposition:
    1. Prism = continuum spectrum
    2. Triangle = simplex of primaries  
    3. Slit = interference quantization
    """
    
    def __init__(self, harmonic_ratios: HarmonicRatio = None):
        self.harmonic = harmonic_ratios or HarmonicRatio()
    
    def prism_decomposition(self, white_color: OKLCHColor, 
                           num_colors: int = 7) -> List[OKLCHColor]:
        """Prism: splits white into rainbow → continuous spectrum"""
        colors = []
        for i in range(num_colors):
            # ROYGBIV spectrum with harmonic ratios
            hue = (i * 360.0 / num_colors) % 360
            chroma_scale = (i + 1) / num_colors * 0.25  # Harmonic scaling
            color = OKLCHColor(white_color.lightness, chroma_scale, hue)
            colors.append(color)
        return colors
    
    def triangle_decomposition(self, primaries: List[OKLCHColor]) -> List[OKLCHColor]:
        """Triangle: Maxwell's triangle — discrete convex hull of primaries"""
        if len(primaries) < 3:
            return primaries
        
        # Create convex combinations of primaries
        colors = primaries[:]
        for i in range(len(primaries)):
            for j in range(i + 1, len(primaries)):
                # Interpolate between primaries
                for alpha in [0.25, 0.5, 0.75]:
                    L = alpha * primaries[i].lightness + (1 - alpha) * primaries[j].lightness
                    C = alpha * primaries[i].chroma + (1 - alpha) * primaries[j].chroma
                    h = alpha * primaries[i].hue + (1 - alpha) * primaries[j].hue
                    colors.append(OKLCHColor(L, C, h))
        
        return colors
    
    def slit_decomposition(self, base_color: OKLCHColor, 
                          num_interference: int = 7) -> List[OKLCHColor]:
        """Slit: reveals light's duality — interference fringes, discrete harmonics"""
        colors = []
        for n in range(1, num_interference + 1):
            # Interference pattern: alternating bright/dark with harmonic spacing
            if n % 2 == 1:  # Bright fringe
                lightness = min(0.9, base_color.lightness + 0.3)
                chroma = min(0.4, base_color.chroma + 0.1)
            else:  # Dark fringe
                lightness = max(0.1, base_color.lightness - 0.3)
                chroma = max(0.02, base_color.chroma - 0.05)
            
            # Harmonic phase shift
            hue_shift = (n * 360.0 / num_interference) % 360
            color = OKLCHColor(lightness, chroma, (base_color.hue + hue_shift) % 360)
            colors.append(color)
        
        return colors


class LeastActionColorPerception:
    """
    Principle of least action applied to color perception
    
    Light takes the path that minimizes action → in color, the eye/brain 
    interprets color through minimal-energy perceptual coordinates.
    This explains why OKLCH "feels right": it's perceptual least action.
    """
    
    def __init__(self):
        self.oklch_weights = {'L': 1.0, 'C': 0.8, 'h': 0.6}  # Perceptual energy weights
    
    def perceptual_energy(self, color: OKLCHColor) -> float:
        """Compute perceptual energy of a color"""
        # Lightness energy (distance from neutral)
        L_energy = abs(color.lightness - 0.5) ** 2
        
        # Chroma energy (saturation cost)
        C_energy = color.chroma ** 2
        
        # Hue energy (phase coherence)
        h_energy = (np.sin(np.radians(color.hue)) ** 2 + 
                   np.cos(np.radians(color.hue)) ** 2) / 2
        
        total_energy = (self.oklch_weights['L'] * L_energy + 
                       self.oklch_weights['C'] * C_energy + 
                       self.oklch_weights['h'] * h_energy)
        
        return total_energy
    
    def minimize_perceptual_action(self, colors: List[OKLCHColor]) -> List[OKLCHColor]:
        """Minimize perceptual action across color palette"""
        # Sort by perceptual energy (least action first)
        sorted_colors = sorted(colors, key=self.perceptual_energy)
        
        # Apply harmonic constraints to maintain musical intervals
        harmonic_colors = []
        for i, color in enumerate(sorted_colors):
            if i < 7:  # First 7 colors get harmonic treatment
                # Apply harmonic ratio scaling
                harmonic_n = i + 1
                chroma_scale = harmonic_n / 7.0
                new_chroma = min(0.4, color.chroma * chroma_scale)
                
                # Critical line constraint
                new_lightness = 0.5  # Always on critical line for harmonics
                
                harmonic_color = OKLCHColor(new_lightness, new_chroma, color.hue)
                harmonic_colors.append(harmonic_color)
            else:
                harmonic_colors.append(color)
        
        return harmonic_colors


class ColorQuaternionHarmonicSpec:
    """
    Main specification class that combines all color quaternion operations
    
    This is the complete Color Quaternion Harmonic Spec that generates
    color atmospheres with symmetry and time arrows built in.
    """
    
    def __init__(self, seed: str = None):
        self.seed = seed or "default"
        self.harmonic = HarmonicRatio()
        self.quaternion_group = ColorQuaternionGroup(self.harmonic)
        self.ca_generator = CellularAutomataColorGenerator()
        self.decomposition_ops = ColorDecompositionOperators(self.harmonic)
        self.least_action = LeastActionColorPerception()
        
        # Generate base color from seed
        self.base_color = self._seed_to_base_color()
    
    def _seed_to_base_color(self) -> OKLCHColor:
        """Convert seed string to base OKLCH color using SHA-256 entropy"""
        # Use SHA-256 as entropy source
        hash_obj = hashlib.sha256(self.seed.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Extract entropy for color parameters
        entropy_int = int(hash_hex[:16], 16)
        
        # RIEMANN HYPOTHESIS: All critical lines = 0.5
        L_base = 0.5  # Critical lightness (like Re(s) = 0.5 for zeta zeros)
        C_base = 0.5  # Critical chroma (will be scaled to practical range)
        
        # Extract entropy from SHA-256 for hue rotation only
        h_base = ((entropy_int >> 16) & 0xFFFF) / 0xFFFF * 360
        
        return OKLCHColor(L_base, C_base, h_base)
    
    def generate_harmonic_palette(self, num_colors: int = 7) -> Dict[str, str]:
        """Generate complete harmonic color palette"""
        # Generate orbit under quaternion group
        orbit = self.quaternion_group.generate_orbit(self.base_color)
        
        # Apply least action principle
        optimized_colors = self.least_action.minimize_perceptual_action(orbit)
        
        # Take first num_colors
        palette_colors = optimized_colors[:num_colors]
        
        # Create named palette
        palette = {}
        for i, color in enumerate(palette_colors):
            palette[f'harmonic_{i+1}'] = color.to_string()
        
        # Add musical interval colors
        musical_intervals = self.harmonic.musical_intervals
        for interval_name, degrees in musical_intervals.items():
            shifted_color = OKLCHColor(
                self.base_color.lightness,
                self.base_color.chroma,
                (self.base_color.hue + degrees) % 360
            )
            palette[f'musical_{interval_name}'] = shifted_color.to_string()
        
        return palette
    
    def generate_ca_pattern(self, rule: CellularAutomataRule = CellularAutomataRule.RULE_90,
                          steps: int = 8) -> List[List[str]]:
        """Generate cellular automata pattern as colors"""
        self.ca_generator.rule = rule
        patterns = self.ca_generator.generate_sierpinski_pattern(steps)
        color_patterns = self.ca_generator.pattern_to_colors(patterns, self.base_color)
        # Convert to string format for JSON serialization
        return [[color.to_string() for color in row] for row in color_patterns]
    
    def generate_decomposition_palette(self, basis: ColorDecompositionBasis) -> List[str]:
        """Generate palette using specific decomposition basis"""
        if basis == ColorDecompositionBasis.PRISM:
            colors = self.decomposition_ops.prism_decomposition(self.base_color)
        elif basis == ColorDecompositionBasis.TRIANGLE:
            # Use harmonic colors as primaries
            primaries = [self.base_color]
            for i in range(2, 5):  # Add 2nd, 3rd, 4th harmonics as primaries
                harmonic_color = OKLCHColor(
                    0.5, (i/7.0) * 0.25, 
                    (self.base_color.hue + i * 360.0 / 7) % 360
                )
                primaries.append(harmonic_color)
            colors = self.decomposition_ops.triangle_decomposition(primaries)
        elif basis == ColorDecompositionBasis.SLIT:
            colors = self.decomposition_ops.slit_decomposition(self.base_color)
        else:
            colors = [self.base_color]
        
        return [color.to_string() for color in colors]
    
    def generate_complete_spec(self) -> Dict[str, any]:
        """Generate complete Color Quaternion Harmonic Spec"""
        return {
            'seed': self.seed,
            'base_color': self.base_color.to_string(),
            'is_critical_line': self.base_color.is_critical_line,
            'harmonic_palette': self.generate_harmonic_palette(),
            'ca_pattern_90': self.generate_ca_pattern(CellularAutomataRule.RULE_90),
            'ca_pattern_45': self.generate_ca_pattern(CellularAutomataRule.RULE_45),
            'prism_decomposition': self.generate_decomposition_palette(ColorDecompositionBasis.PRISM),
            'triangle_decomposition': self.generate_decomposition_palette(ColorDecompositionBasis.TRIANGLE),
            'slit_decomposition': self.generate_decomposition_palette(ColorDecompositionBasis.SLIT),
            'musical_intervals': self.harmonic.musical_intervals,
            'harmonic_ratios': self.harmonic.ratios,
            'color_modes': [mode.value for mode in ColorMode],
            'perceptual_energy': self.least_action.perceptual_energy(self.base_color)
        }


def main():
    """Demonstrate Color Quaternion Harmonic Spec"""
    import json

    # Create spec with custom seed
    spec = ColorQuaternionHarmonicSpec("riemann_hypothesis_2025")
    
    # Generate complete specification
    complete_spec = spec.generate_complete_spec()
    
    # Print summary
    print("🎨 Color Quaternion Harmonic Spec")
    print("=" * 50)
    print(f"Seed: {complete_spec['seed']}")
    print(f"Base Color: {complete_spec['base_color']}")
    print(f"Critical Line: {complete_spec['is_critical_line']}")
    print(f"Perceptual Energy: {complete_spec['perceptual_energy']:.4f}")
    
    print(f"\n🎵 Harmonic Palette (1:2:3:4:5:6:7):")
    for name, color in complete_spec['harmonic_palette'].items():
        print(f"  {name}: {color}")
    
    print(f"\n🎼 Musical Intervals:")
    for interval, degrees in complete_spec['musical_intervals'].items():
        print(f"  {interval}: {degrees:.1f}°")
    
    print(f"\n🌈 Three Faces of Color:")
    print(f"  Prism (continuum): {len(complete_spec['prism_decomposition'])} colors")
    print(f"  Triangle (simplex): {len(complete_spec['triangle_decomposition'])} colors") 
    print(f"  Slit (interference): {len(complete_spec['slit_decomposition'])} colors")
    
    print(f"\n🔄 Cellular Automata Patterns:")
    print(f"  Rule 90 (Sierpiński): {len(complete_spec['ca_pattern_90'])} steps")
    print(f"  Rule 45 (diagonal): {len(complete_spec['ca_pattern_45'])} steps")
    
    print(f"\n📊 Available Color Modes: {', '.join(complete_spec['color_modes'])}")
    
    # Save complete spec to file
    output_file = "color_quaternion_harmonic_spec.json"
    with open(output_file, 'w') as f:
        json.dump(complete_spec, f, indent=2)
    
    print(f"\n💾 Complete spec saved to: {output_file}")
    print("\n🎯 Color Quaternion Harmonic Spec Complete!")
    print("Ready to generate color atmospheres with symmetry and time arrows!")


if __name__ == "__main__":
    main()
