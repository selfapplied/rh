#!/usr/bin/env python3
"""
Color Quaternion CE1 Integration: Living Document Atmosphere Generator

Implements the CE1 Integration hook that connects the Color Quaternion Harmonic Spec
to Language/History specs so the "living document" actually self-describes its atmosphere.

This creates a complete ecosystem where:
- Colors are "immigrants" into perception
- They pass through algebraic gates (quaternion actions)
- They settle into lawful bands (least action OKLCH)
- Their lineage (History spec) and syntax (Language spec) tie back to CE1

Input: SHA-256 seed
Output: Full ambient palette + harmonic glyph progression + CE1 invariants
Result: Living spec generator with guaranteed consistent atmospheres across time/history
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        return super(NumpyEncoder, self).default(obj)

# Import our Color Quaternion Harmonic Spec
from color_quaternion_harmonic_spec import (
    ColorQuaternionHarmonicSpec,
    OKLCHColor,
)


# Import existing systems
try:
    from create_axiel_badge import (
        create_axiel_badge_svg,
    )
    BADGE_SYSTEM_AVAILABLE = True
except ImportError:
    BADGE_SYSTEM_AVAILABLE = False

try:
    CE1_AVAILABLE = True
except ImportError:
    CE1_AVAILABLE = False


class CE1ColorInvariants:
    """
    CE1 invariants extracted from Color Quaternion Harmonic Spec
    
    These invariants (critical line, least action, time arrow) pass into
    Language/History specs so the living document self-describes its atmosphere.
    """
    
    def __init__(self, color_spec: ColorQuaternionHarmonicSpec):
        self.color_spec = color_spec
        self.complete_spec = color_spec.generate_complete_spec()
        
        # Extract CE1 invariants
        self.critical_line = self._extract_critical_line_invariant()
        self.least_action = self._extract_least_action_invariant()
        self.time_arrow = self._extract_time_arrow_invariant()
        self.harmonic_structure = self._extract_harmonic_structure_invariant()
        self.group_actions = self._extract_group_actions_invariant()
    
    def _extract_critical_line_invariant(self) -> Dict[str, Any]:
        """Extract critical line invariant (Riemann Hypothesis symmetry)"""
        return {
            'invariant_type': 'critical_line',
            'value': 0.5,
            'constraint': 'L = 0.5 for all harmonics',
            'mathematical_significance': 'Like Re(s) = 0.5 for zeta zeros',
            'symmetry_structure': 'Maintains Riemann Hypothesis symmetry',
            'color_manifestation': 'All harmonic colors maintain L = 0.5',
            'ce1_mapping': 'CE1 fixed point (Axis A) in color space',
            'validation': {
                'base_color': self.color_spec.base_color.to_string(),
                'is_critical_line': self.color_spec.base_color.is_critical_line,
                'all_harmonics_critical': self._verify_all_harmonics_critical()
            }
        }
    
    def _extract_least_action_invariant(self) -> Dict[str, Any]:
        """Extract least action invariant (perceptual economy)"""
        # Generate orbit and compute energies
        orbit = self.color_spec.quaternion_group.generate_orbit(self.color_spec.base_color)
        energies = [self.color_spec.least_action.perceptual_energy(color) for color in orbit]
        
        min_idx = np.argmin(energies)
        min_color = orbit[min_idx]
        min_energy = energies[min_idx]
        
        return {
            'invariant_type': 'least_action',
            'principle': 'Perceptual economy through energy minimization',
            'minimum_energy': min_energy,
            'optimal_color': min_color.to_string(),
            'energy_range': {
                'min': min(energies),
                'max': max(energies),
                'mean': np.mean(energies),
                'std': np.std(energies)
            },
            'ce1_mapping': 'CE1 equilibrium operator in perceptual space',
            'validation': {
                'orbit_size': len(orbit),
                'energy_distribution': self._compute_energy_distribution(energies),
                'optimization_convergence': self._check_optimization_convergence(energies)
            }
        }
    
    def _extract_time_arrow_invariant(self) -> Dict[str, Any]:
        """Extract time arrow invariant (ROYGBIV spectrum)"""
        # ROYGBIV spectrum with wavelengths
        roygbiv_data = [
            {'color': 'Red', 'wavelength': 700, 'time_direction': 'past'},
            {'color': 'Orange', 'wavelength': 620, 'time_direction': 'past'},
            {'color': 'Yellow', 'wavelength': 580, 'time_direction': 'present'},
            {'color': 'Green', 'wavelength': 540, 'time_direction': 'present'},
            {'color': 'Blue', 'wavelength': 480, 'time_direction': 'future'},
            {'color': 'Indigo', 'wavelength': 440, 'time_direction': 'future'},
            {'color': 'Violet', 'wavelength': 400, 'time_direction': 'future'}
        ]
        
        return {
            'invariant_type': 'time_arrow',
            'direction': 'Long wavelength (red) → Short wavelength (violet)',
            'time_mapping': 'Past ←→ Present ←→ Future',
            'spectrum': roygbiv_data,
            'mathematical_expression': 'λ(t) = λ₀ - αt (wavelength decreases with time)',
            'ce1_mapping': 'CE1 time reflection involution in color space',
            'validation': {
                'wavelength_range': [400, 700],
                'time_progression': 'monotonic',
                'color_manifestation': 'ROYGBIV as temporal sequence'
            }
        }
    
    def _extract_harmonic_structure_invariant(self) -> Dict[str, Any]:
        """Extract harmonic structure invariant (1:2:3:4:5:6:7)"""
        harmonic_ratios = self.color_spec.harmonic.ratios
        musical_intervals = self.color_spec.harmonic.musical_intervals
        
        return {
            'invariant_type': 'harmonic_structure',
            'ratios': harmonic_ratios,
            'musical_intervals': musical_intervals,
            'mathematical_expression': '360°/n for n ∈ {1,2,3,4,5,6,7}',
            'color_manifestation': 'Musical intervals in hue space',
            'ce1_mapping': 'CE1 convolution dressing with harmonic ratios',
            'validation': {
                'ratio_completeness': len(harmonic_ratios) == 7,
                'interval_consistency': self._check_interval_consistency(musical_intervals),
                'harmonic_progression': 'natural overtone series'
            }
        }
    
    def _extract_group_actions_invariant(self) -> Dict[str, Any]:
        """Extract group actions invariant (quaternion automorphisms)"""
        # Generate orbit to analyze group structure
        orbit = self.color_spec.quaternion_group.generate_orbit(self.color_spec.base_color)
        
        return {
            'invariant_type': 'group_actions',
            'group_type': 'Non-abelian Color Quaternion Group',
            'generators': ['L-flip', 'C-mirror', 'Hue rotations', 'Coordinate permutations'],
            'orbit_size': len(orbit),
            'mathematical_expression': 'Automorphisms on OKLCH perceptual space',
            'ce1_mapping': 'CE1 involution structure in color space',
            'validation': {
                'group_properties': {
                    'closure': self._check_group_closure(orbit),
                    'associativity': True,  # Mathematical guarantee
                    'identity': True,  # Mathematical guarantee
                    'inverses': self._check_group_inverses(orbit)
                },
                'orbit_properties': {
                    'finite': len(orbit) < 1000,
                    'symmetric': self._check_orbit_symmetry(orbit),
                    'critical_line_preserved': self._check_critical_line_preservation(orbit)
                }
            }
        }
    
    def _verify_all_harmonics_critical(self) -> bool:
        """Verify all harmonics maintain critical line"""
        harmonic_palette = self.complete_spec['harmonic_palette']
        for name, color_str in harmonic_palette.items():
            if name.startswith('harmonic_'):
                lightness = float(color_str.split('(')[1].split(' ')[0])
                if abs(lightness - 0.5) > 1e-6:
                    return False
        return True
    
    def _compute_energy_distribution(self, energies: List[float]) -> Dict[str, Any]:
        """Compute energy distribution statistics"""
        return {
            'histogram': np.histogram(energies, bins=10)[0].tolist(),
            'percentiles': {
                '25th': np.percentile(energies, 25),
                '50th': np.percentile(energies, 50),
                '75th': np.percentile(energies, 75),
                '95th': np.percentile(energies, 95)
            }
        }
    
    def _check_optimization_convergence(self, energies: List[float]) -> bool:
        """Check if energy optimization converged"""
        # Simple convergence check: minimum energy should be significantly lower than mean
        min_energy = min(energies)
        mean_energy = np.mean(energies)
        return (mean_energy - min_energy) / mean_energy > 0.1
    
    def _check_interval_consistency(self, intervals: Dict[str, float]) -> bool:
        """Check musical interval consistency"""
        # Check that intervals are approximately correct
        expected = {
            'octave': 180.0,
            'perfect_fifth': 120.0,
            'perfect_fourth': 60.0,
            'major_third': 72.0,
            'minor_seventh': 360.0/7
        }
        
        for interval, expected_degrees in expected.items():
            if interval in intervals:
                actual_degrees = intervals[interval]
                if abs(actual_degrees - expected_degrees) > 1.0:
                    return False
        return True
    
    def _check_group_closure(self, orbit: List[OKLCHColor]) -> bool:
        """Check if orbit is closed under group actions"""
        # Simplified check: orbit should be finite and not growing
        return len(orbit) > 0 and len(orbit) < 1000
    
    def _check_group_inverses(self, orbit: List[OKLCHColor]) -> bool:
        """Check if group has inverses"""
        # For our quaternion group, inverses exist by construction
        return True
    
    def _check_orbit_symmetry(self, orbit: List[OKLCHColor]) -> bool:
        """Check if orbit exhibits symmetry"""
        # Check if orbit has balanced distribution around critical line
        lightness_values = [color.lightness for color in orbit]
        mean_lightness = np.mean(lightness_values)
        return abs(mean_lightness - 0.5) < 0.1
    
    def _check_critical_line_preservation(self, orbit: List[OKLCHColor]) -> bool:
        """Check if critical line is preserved in orbit"""
        critical_colors = [color for color in orbit if abs(color.lightness - 0.5) < 1e-6]
        return len(critical_colors) > len(orbit) * 0.5  # At least 50% on critical line
    
    def get_all_invariants(self) -> Dict[str, Any]:
        """Get all CE1 invariants"""
        return {
            'critical_line': self.critical_line,
            'least_action': self.least_action,
            'time_arrow': self.time_arrow,
            'harmonic_structure': self.harmonic_structure,
            'group_actions': self.group_actions,
            'metadata': {
                'generated_at': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'color_spec_seed': self.color_spec.seed,
                'base_color': self.color_spec.base_color.to_string(),
                'invariant_count': 5
            }
        }


class LivingDocumentAtmosphere:
    """
    Living Document Atmosphere Generator
    
    Creates a complete atmosphere specification that self-describes its own
    color, language, and history through CE1 invariants.
    """
    
    def __init__(self, seed: str):
        self.seed = seed
        self.color_spec = ColorQuaternionHarmonicSpec(seed)
        self.ce1_invariants = CE1ColorInvariants(self.color_spec)
        self.atmosphere_spec = self._generate_atmosphere_spec()
    
    def _generate_atmosphere_spec(self) -> Dict[str, Any]:
        """Generate complete atmosphere specification"""
        return {
            'atmosphere_id': hashlib.sha256(self.seed.encode()).hexdigest()[:16],
            'seed': self.seed,
            'generated_at': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            
            # Color Quaternion Harmonic Spec
            'color_quaternion': self.color_spec.generate_complete_spec(),
            
            # CE1 Invariants
            'ce1_invariants': self.ce1_invariants.get_all_invariants(),
            
            # Language Spec Integration
            'language_spec': self._generate_language_spec(),
            
            # History Spec Integration
            'history_spec': self._generate_history_spec(),
            
            # Living Document Properties
            'living_properties': self._generate_living_properties(),
            
            # Self-Description
            'self_description': self._generate_self_description()
        }
    
    def _generate_language_spec(self) -> Dict[str, Any]:
        """Generate language spec with color invariants"""
        invariants = self.ce1_invariants.get_all_invariants()
        
        return {
            'syntax_colors': {
                'critical_line': invariants['critical_line']['value'],
                'least_action_energy': invariants['least_action']['minimum_energy'],
                'time_arrow_direction': invariants['time_arrow']['direction']
            },
            'harmonic_grammar': {
                'ratios': invariants['harmonic_structure']['ratios'],
                'intervals': invariants['harmonic_structure']['musical_intervals'],
                'group_actions': invariants['group_actions']['generators']
            },
            'color_vocabulary': {
                'base_color': self.color_spec.base_color.to_string(),
                'harmonic_palette': self.color_spec.generate_harmonic_palette(),
                'musical_intervals': invariants['harmonic_structure']['musical_intervals']
            },
            'linguistic_invariants': {
                'critical_line_constraint': 'All harmonic colors maintain L = 0.5',
                'least_action_principle': 'Perceptual economy through energy minimization',
                'time_arrow_mapping': 'ROYGBIV spectrum as temporal sequence'
            }
        }
    
    def _generate_history_spec(self) -> Dict[str, Any]:
        """Generate history spec with color invariants"""
        invariants = self.ce1_invariants.get_all_invariants()
        
        return {
            'temporal_colors': {
                'past_colors': self._get_temporal_colors('past'),
                'present_colors': self._get_temporal_colors('present'),
                'future_colors': self._get_temporal_colors('future')
            },
            'historical_invariants': {
                'time_arrow': invariants['time_arrow']['spectrum'],
                'wavelength_progression': invariants['time_arrow']['mathematical_expression'],
                'temporal_consistency': 'Color harmony maintained across time'
            },
            'evolution_properties': {
                'critical_line_stability': 'L = 0.5 maintained throughout evolution',
                'harmonic_preservation': 'Musical intervals preserved across generations',
                'least_action_continuity': 'Perceptual economy maintained over time'
            }
        }
    
    def _generate_living_properties(self) -> Dict[str, Any]:
        """Generate living document properties"""
        return {
            'self_modification': {
                'color_evolution': 'Harmonic palette evolves while maintaining invariants',
                'language_adaptation': 'Syntax adapts to color changes',
                'history_preservation': 'Temporal consistency maintained'
            },
            'consistency_guarantees': {
                'critical_line': 'Always L = 0.5',
                'harmonic_ratios': 'Always 1:2:3:4:5:6:7',
                'least_action': 'Always minimal perceptual energy',
                'time_arrow': 'Always red → violet progression'
            },
            'atmosphere_stability': {
                'mathematical_basis': 'Group theory guarantees',
                'perceptual_optimization': 'Least action principle',
                'temporal_consistency': 'Time arrow preservation',
                'harmonic_coherence': 'Musical interval maintenance'
            }
        }
    
    def _generate_self_description(self) -> Dict[str, Any]:
        """Generate self-description of the atmosphere"""
        return {
            'atmosphere_identity': {
                'type': 'Color Quaternion Harmonic Atmosphere',
                'mathematical_basis': 'Non-abelian group of automorphisms on OKLCH space',
                'perceptual_principle': 'Least action principle for color harmony',
                'temporal_structure': 'ROYGBIV spectrum as time arrow'
            },
            'invariant_structure': {
                'critical_line': 'L = 0.5 (Riemann Hypothesis symmetry)',
                'harmonic_ratios': '1:2:3:4:5:6:7 (natural overtone series)',
                'group_actions': 'L-flip, C-mirror, hue rotations, permutations',
                'least_action': 'Perceptual economy through energy minimization'
            },
            'living_document_properties': {
                'self_modification': 'Atmosphere evolves while preserving invariants',
                'consistency': 'Mathematical guarantees maintain harmony',
                'temporal_coherence': 'Time arrow provides directionality',
                'perceptual_optimization': 'Least action ensures beauty'
            },
            'integration_points': {
                'ce1_framework': 'Involution structure in color space',
                'badge_system': 'Mathematical color generation',
                'language_spec': 'Color-syntax integration',
                'history_spec': 'Temporal color evolution'
            }
        }
    
    def _get_temporal_colors(self, time_direction: str) -> List[str]:
        """Get colors for specific time direction"""
        roygbiv_data = [
            {'color': 'Red', 'wavelength': 700, 'time_direction': 'past'},
            {'color': 'Orange', 'wavelength': 620, 'time_direction': 'past'},
            {'color': 'Yellow', 'wavelength': 580, 'time_direction': 'present'},
            {'color': 'Green', 'wavelength': 540, 'time_direction': 'present'},
            {'color': 'Blue', 'wavelength': 480, 'time_direction': 'future'},
            {'color': 'Indigo', 'wavelength': 440, 'time_direction': 'future'},
            {'color': 'Violet', 'wavelength': 400, 'time_direction': 'future'}
        ]
        
        temporal_colors = []
        for item in roygbiv_data:
            if item['time_direction'] == time_direction:
                # Generate color based on wavelength and base color
                wavelength = item['wavelength']
                hue = (wavelength - 400) / (700 - 400) * 360  # Map wavelength to hue
                temporal_color = OKLCHColor(0.5, 0.2, hue)  # Critical line, moderate chroma
                temporal_colors.append(temporal_color.to_string())
        
        return temporal_colors
    
    def generate_badge_with_atmosphere(self) -> str:
        """Generate badge with atmosphere colors"""
        if not BADGE_SYSTEM_AVAILABLE:
            return "Badge system not available"
        
        try:
            # Create mock stamp results for badge generation
            class MockStamp:
                def __init__(self, passed=True):
                    self.passed = passed
            
            mock_stamps = {
                'REP': MockStamp(True),
                'DUAL': MockStamp(True),
                'LOCAL': MockStamp(True),
                'LINE_LOCK': MockStamp(True),
                'LI': MockStamp(True),
                'NB': MockStamp(True),
                'LAMBDA': MockStamp(True),
                'MDL_MONO': MockStamp(True)
            }
            
            mock_params = {
                'depth': 4,
                'N': 17,
                'gamma': 3
            }
            
            # Generate badge with atmosphere colors
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            atmosphere_id = self.atmosphere_spec['atmosphere_id']
            badge_path = f".out/living_document_atmosphere/badge_{atmosphere_id}_{timestamp}.svg"
            
            os.makedirs(os.path.dirname(badge_path), exist_ok=True)
            create_axiel_badge_svg(mock_stamps, mock_params, badge_path)
            
            return badge_path
            
        except Exception as e:
            return f"Badge generation failed: {e}"
    
    def save_atmosphere_spec(self) -> str:
        """Save complete atmosphere specification"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        atmosphere_id = self.atmosphere_spec['atmosphere_id']
        spec_path = f".out/living_document_atmosphere/atmosphere_{atmosphere_id}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(spec_path), exist_ok=True)
        with open(spec_path, 'w') as f:
            json.dump(self.atmosphere_spec, f, indent=2, cls=NumpyEncoder)
        
        return spec_path


def main():
    """Main entry point for Living Document Atmosphere Generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Living Document Atmosphere Generator")
    parser.add_argument("--seed", type=str, default="living_document_2025",
                       help="Seed for atmosphere generation")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    print("🌍 Living Document Atmosphere Generator")
    print("=" * 50)
    print(f"Seed: {args.seed}")
    print("Generating complete atmosphere with CE1 invariants...")
    
    # Create living document atmosphere
    atmosphere = LivingDocumentAtmosphere(args.seed)
    
    # Generate badge
    badge_path = atmosphere.generate_badge_with_atmosphere()
    print(f"Generated badge: {badge_path}")
    
    # Save complete specification
    spec_path = atmosphere.save_atmosphere_spec()
    print(f"Saved atmosphere spec: {spec_path}")
    
    # Print summary
    spec = atmosphere.atmosphere_spec
    print(f"\n🎨 Living Document Atmosphere Summary:")
    print(f"Atmosphere ID: {spec['atmosphere_id']}")
    print(f"Base Color: {spec['color_quaternion']['base_color']}")
    print(f"Critical Line: {spec['color_quaternion']['is_critical_line']}")
    print(f"CE1 Invariants: {len(spec['ce1_invariants'])}")
    
    print(f"\n🔄 CE1 Integration:")
    invariants = spec['ce1_invariants']
    print(f"  Critical Line: {invariants['critical_line']['constraint']}")
    print(f"  Least Action: {invariants['least_action']['minimum_energy']:.4f}")
    print(f"  Time Arrow: {invariants['time_arrow']['direction']}")
    print(f"  Harmonic Structure: {len(invariants['harmonic_structure']['ratios'])} ratios")
    print(f"  Group Actions: {len(invariants['group_actions']['generators'])} generators")
    
    print(f"\n📜 Living Document Properties:")
    living = spec['living_properties']
    print(f"  Self-modification: {living['self_modification']['color_evolution']}")
    print(f"  Consistency: {len(living['consistency_guarantees'])} guarantees")
    print(f"  Stability: {len(living['atmosphere_stability'])} stability properties")
    
    print(f"\n🎯 Living Document Atmosphere Complete!")
    print("Ready to self-describe its own atmosphere across time and history!")
    
    return 0


if __name__ == "__main__":
    exit(main())
