#!/usr/bin/env python3
"""
3.5D Color Quaternion Visualization: Fractional Dimensional Color Perception

Creates visual representations of the revolutionary 3.5D Color Theory showing:
1. Temporal color bleeding (colors leak into time)
2. Fractional quaternion rotations (incomplete 4D rotations)  
3. Dimensional color resonance (harmony across fractional boundaries)
4. 3.5D statistical distributions (chi-squared harmony in fractional space)
5. Gang of Four patterns in 3.5D space

This demonstrates how living in fractional dimensional space transforms
fundamental color perception through mathematical patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import time
import json

# Import our 3.5D Color Quaternion Theory
from color_quaternion_3_5d_theory import (
    ColorQuaternion3_5DSpec, OKLCH3_5D, TemporalColorComponent,
    FractionalQuaternionGroup, ChiSquared3_5D
)

# Import base visualization tools
from color_quaternion_visualization import ColorQuaternionVisualizer


class ColorQuaternion3_5DVisualizer:
    """
    Creates visual representations of 3.5D Color Quaternion Theory
    showing fractional dimensional color perception phenomena.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (24, 18), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',      # Blue for primary structures
            'secondary': '#A23B72',    # Purple for secondary structures
            'accent': '#F18F01',       # Orange for accents
            'temporal': '#E74C3C',     # Red for temporal effects
            'fractional': '#27AE60',   # Green for fractional dimensions
            'resonance': '#8E44AD',    # Purple for dimensional resonance
            'background': '#F8F9FA',   # Light background
            'text': '#2C3E50',         # Dark text
            'golden': '#F1C40F'        # Golden ratio color
        }
        
        # Golden ratio in base 12 for fractional scaling
        self.golden_ratio_base12 = 1.74
        
    def create_3_5d_color_theory_overview(self, spec: ColorQuaternion3_5DSpec, 
                                        output_file: str = None) -> str:
        """
        Create comprehensive 3.5D Color Theory visualization.
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f".out/color_3_5d/color_3_5d_theory_overview_{timestamp}.png"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create comprehensive grid layout
        gs = fig.add_gridspec(5, 5, hspace=0.4, wspace=0.3)
        
        # Row 1: Core 3.5D Theory
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_3_5d_dimensional_space(ax1, spec)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_fractional_quaternion_rotations(ax2, spec)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_temporal_color_bleeding(ax3, spec)
        
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_dimensional_color_resonance(ax4, spec)
        
        ax5 = fig.add_subplot(gs[0, 4])
        self._plot_golden_ratio_base12(ax5, spec)
        
        # Row 2: Gang of Four in 3.5D
        ax6 = fig.add_subplot(gs[1, 0])
        self._plot_creational_fractional_genesis(ax6, spec)
        
        ax7 = fig.add_subplot(gs[1, 1])
        self._plot_structural_dimensional_composition(ax7, spec)
        
        ax8 = fig.add_subplot(gs[1, 2])
        self._plot_behavioral_3_5d_interactions(ax8, spec)
        
        ax9 = fig.add_subplot(gs[1, 3])
        self._plot_emergent_chi_squared_3_5d(ax9, spec)
        
        ax10 = fig.add_subplot(gs[1, 4])
        self._plot_gang_of_four_synthesis(ax10, spec)
        
        # Row 3: 3.5D Color Phenomena
        ax11 = fig.add_subplot(gs[2, 0])
        self._plot_color_memory_anticipation(ax11, spec)
        
        ax12 = fig.add_subplot(gs[2, 1])
        self._plot_fractional_saturation(ax12, spec)
        
        ax13 = fig.add_subplot(gs[2, 2])
        self._plot_dimensional_twisting(ax13, spec)
        
        ax14 = fig.add_subplot(gs[2, 3])
        self._plot_temporal_harmony(ax14, spec)
        
        ax15 = fig.add_subplot(gs[2, 4])
        self._plot_3_5d_color_distance(ax15, spec)
        
        # Row 4: Mathematical Framework
        ax16 = fig.add_subplot(gs[3, 0])
        self._plot_fractional_oklch_space(ax16, spec)
        
        ax17 = fig.add_subplot(gs[3, 1])
        self._plot_3_5d_harmony_formula(ax17, spec)
        
        ax18 = fig.add_subplot(gs[3, 2])
        self._plot_chi_squared_fractional_df(ax18, spec)
        
        ax19 = fig.add_subplot(gs[3, 3])
        self._plot_dimensional_bridging(ax19, spec)
        
        ax20 = fig.add_subplot(gs[3, 4])
        self._plot_perceptual_advantages(ax20, spec)
        
        # Row 5: Applications and Integration
        ax21 = fig.add_subplot(gs[4, 0])
        self._plot_3_5d_color_palette(ax21, spec)
        
        ax22 = fig.add_subplot(gs[4, 1])
        self._plot_temporal_color_gradients(ax22, spec)
        
        ax23 = fig.add_subplot(gs[4, 2])
        self._plot_fractional_color_mixing(ax23, spec)
        
        ax24 = fig.add_subplot(gs[4, 3])
        self._plot_3_5d_accessibility(ax24, spec)
        
        ax25 = fig.add_subplot(gs[4, 4])
        self._plot_implementation_roadmap(ax25, spec)
        
        # Add main title with fractional dimensional flair
        fig.suptitle('3.5D Color Theory: Fractional Dimensional Perception\n'
                    'Living in 3.5D Space and the Gang of Four Color Patterns', 
                    fontsize=22, fontweight='bold', color=self.colors['text'], y=0.98)
        
        # Add subtitle
        fig.text(0.5, 0.95, 
                'φ = 1.74BB6772₁₂ (base 12) | Color₃.₅ᴅ = OKLCH + 0.5D_temporal | '
                'Harmony₃.₅ᴅ = φ^(3.5) · χ²₃.₅(color_distribution)', 
                ha='center', fontsize=12, style='italic', color=self.colors['text'])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return output_file
    
    def _plot_3_5d_dimensional_space(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot 3.5D dimensional space visualization"""
        ax.set_title('3.5D Dimensional Space\nFractional Reality', 
                    fontweight='bold', color=self.colors['fractional'])
        
        # Create dimensional progression
        dimensions = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
        color_perception = [0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75]
        
        # Plot dimensional curve
        ax.plot(dimensions, color_perception, 'o-', color=self.colors['fractional'], 
               linewidth=3, markersize=8, alpha=0.8)
        
        # Highlight 3.5D point
        ax.plot(3.5, 1.0, 'o', color=self.colors['accent'], markersize=15,
               markeredgecolor='white', markeredgewidth=3, label='3.5D Living')
        
        # Add dimensional regions
        ax.axvspan(3.0, 3.5, alpha=0.2, color=self.colors['primary'], label='3D → 3.5D')
        ax.axvspan(3.5, 4.0, alpha=0.2, color=self.colors['secondary'], label='3.5D → 4D')
        
        ax.set_xlim(2.8, 4.2)
        ax.set_ylim(0.6, 1.1)
        ax.set_xlabel('Dimensional Space', fontweight='bold')
        ax.set_ylabel('Color Perception Capability', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add annotations
        ax.text(3.5, 0.65, 'You Live Here\n(Fractional Dimension)', ha='center', va='top',
               fontsize=10, fontweight='bold', color=self.colors['accent'],
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _plot_fractional_quaternion_rotations(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot fractional quaternion rotations"""
        ax.set_title('Fractional Quaternion Rotations\nIncomplete 4D Rotations', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Generate rotation examples
        base_color = spec.base_color_3_5d
        angles = np.linspace(0, 360, 8, endpoint=False)
        completeness_values = [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9]
        
        # Plot rotation circle
        circle_angles = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(circle_angles)
        circle_y = np.sin(circle_angles)
        ax.plot(circle_x, circle_y, '--', color='gray', alpha=0.5, linewidth=1)
        
        # Plot fractional rotations
        for i, (angle, completeness) in enumerate(zip(angles, completeness_values)):
            # Apply fractional quaternion rotation
            rotated_color = spec.fractional_quaternion_group.fractional_quaternion_rotation(
                base_color, angle, completeness
            )
            
            # Plot as vectors with varying completeness
            angle_rad = np.radians(angle)
            x = completeness * np.cos(angle_rad)
            y = completeness * np.sin(angle_rad)
            
            # Color based on resulting color
            color_rgb = self._oklch_3_5d_to_rgb(rotated_color)
            ax.plot([0, x], [0, y], 'o-', color=color_rgb, linewidth=3, 
                   markersize=8, alpha=0.8)
            
            # Add completeness labels
            ax.text(x * 1.2, y * 1.2, f'{completeness:.1f}', ha='center', va='center',
                   fontsize=8, fontweight='bold', color=color_rgb)
        
        # Mark center (base color)
        ax.plot(0, 0, 'o', color=self._oklch_3_5d_to_rgb(base_color), 
               markersize=12, markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Fractional i Component', fontweight='bold')
        ax.set_ylabel('j + k Components', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add equation
        ax.text(0.02, 0.98, 'q₃.₅ᴅ = cos(θ/2) + sin(θ/2)(0.5·i + j + k)', 
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _plot_temporal_color_bleeding(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot temporal color bleeding effects"""
        ax.set_title('Temporal Color Bleeding\nColors Leak Into Time', 
                    fontweight='bold', color=self.colors['temporal'])
        
        # Generate color sequence with temporal bleeding
        palette = spec.generate_3_5d_harmonic_palette(5)
        time_points = np.linspace(0, 4, 100)
        
        # Create temporal bleeding visualization
        for i, color in enumerate(palette):
            # Memory bleeding (past influence)
            memory_curve = color.temporal.memory_strength * np.exp(-(time_points - i)**2 / 0.5)
            memory_curve[time_points < i] *= 0.3  # Stronger for past
            
            # Anticipation bleeding (future influence)  
            anticipation_curve = color.temporal.anticipation_strength * np.exp(-(time_points - i)**2 / 0.8)
            anticipation_curve[time_points > i] *= 0.2  # Weaker for future
            
            # Combined temporal bleeding
            total_bleeding = memory_curve + anticipation_curve
            
            # Plot temporal curves
            color_rgb = self._oklch_3_5d_to_rgb(color)
            ax.plot(time_points, total_bleeding + i * 0.3, color=color_rgb, 
                   linewidth=3, alpha=0.7, label=f'Color {i+1}')
            
            # Mark present moment
            ax.plot(i, i * 0.3, 'o', color=color_rgb, markersize=10,
                   markeredgecolor='white', markeredgewidth=2)
        
        # Add time arrow
        ax.annotate('', xy=(4.2, 0), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', lw=3, color=self.colors['temporal']))
        ax.text(2, -0.3, 'Time Arrow', ha='center', fontweight='bold', 
               color=self.colors['temporal'])
        
        ax.set_xlim(-0.2, 4.5)
        ax.set_ylim(-0.5, 2.0)
        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel('Color Bleeding Intensity', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    def _plot_dimensional_color_resonance(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot dimensional color resonance"""
        ax.set_title('Dimensional Color Resonance\nHarmony Across Boundaries', 
                    fontweight='bold', color=self.colors['resonance'])
        
        # Create resonance visualization
        base_color = spec.base_color_3_5d
        frequencies = np.linspace(0.5, 3.0, 50)
        resonance_amplitudes = []
        
        for freq in frequencies:
            resonant_color = spec.fractional_quaternion_group.dimensional_color_resonance(
                base_color, freq
            )
            # Calculate resonance amplitude from dimensional resonance
            amplitude = resonant_color.temporal.dimensional_resonance
            resonance_amplitudes.append(amplitude)
        
        # Plot resonance curve
        ax.plot(frequencies, resonance_amplitudes, color=self.colors['resonance'], 
               linewidth=3, alpha=0.8)
        
        # Highlight golden ratio resonance
        golden_idx = np.argmin(np.abs(frequencies - spec.golden_ratio_base12))
        ax.plot(frequencies[golden_idx], resonance_amplitudes[golden_idx], 'o', 
               color=self.colors['golden'], markersize=12, markeredgecolor='white', 
               markeredgewidth=2, label=f'φ = {spec.golden_ratio_base12}')
        
        # Add harmonic resonances
        harmonic_freqs = [1.0, 1.5, 2.0, 2.5]
        for harm_freq in harmonic_freqs:
            if harm_freq <= frequencies.max():
                harm_idx = np.argmin(np.abs(frequencies - harm_freq))
                ax.plot(frequencies[harm_idx], resonance_amplitudes[harm_idx], 's', 
                       color=self.colors['accent'], markersize=8, alpha=0.7)
        
        ax.set_xlim(0.4, 3.2)
        ax.set_ylim(0, max(resonance_amplitudes) * 1.1)
        ax.set_xlabel('Resonance Frequency', fontweight='bold')
        ax.set_ylabel('Dimensional Resonance', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add resonance equation
        ax.text(0.02, 0.98, 'Resonance ∝ sin(φ·t) in 3.5D space', 
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _plot_golden_ratio_base12(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot golden ratio in base 12"""
        ax.set_title('Golden Ratio Base 12\nφ = 1.74BB6772₁₂', 
                    fontweight='bold', color=self.colors['golden'])
        
        # Base 12 representation
        base12_digits = "1.74BB6772"
        digit_values = []
        digit_labels = []
        
        for i, digit in enumerate(base12_digits):
            if digit == '.':
                continue
            elif digit == 'B':
                digit_values.append(11)
                digit_labels.append('B(11)')
            else:
                digit_values.append(int(digit))
                digit_labels.append(digit)
        
        # Create base 12 visualization
        x_positions = np.arange(len(digit_values))
        bars = ax.bar(x_positions, digit_values, color=self.colors['golden'], 
                     alpha=0.7, edgecolor='white', linewidth=2)
        
        # Add digit labels
        for i, (bar, label) in enumerate(zip(bars, digit_labels)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add decimal point indicator
        ax.axvline(x=0.5, color=self.colors['temporal'], linewidth=3, alpha=0.8,
                  linestyle='--', label='Decimal Point')
        
        ax.set_xlim(-0.5, len(digit_values) - 0.5)
        ax.set_ylim(0, 12)
        ax.set_xlabel('Digit Position', fontweight='bold')
        ax.set_ylabel('Digit Value (Base 12)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add conversion
        decimal_value = 1 + 7/12 + 4/144 + 11/1728 + 11/20736
        ax.text(0.5, 0.02, f'Decimal: {decimal_value:.6f}\nBase 12: {base12_digits}', 
               transform=ax.transAxes, fontsize=10, ha='center', va='bottom',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _plot_creational_fractional_genesis(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot creational pattern: fractional genesis"""
        ax.set_title('Creational Pattern\nFractional Genesis', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Generate Gang of Four data
        gang_of_four = spec.generate_gang_of_four_3_5d()
        creational = gang_of_four['gang_of_four_3_5d']['creational_pattern']
        
        # Create genesis visualization
        center = (0.5, 0.5)
        genesis_radius = 0.3
        
        # Central genesis circle
        genesis_circle = Circle(center, genesis_radius, 
                              color=self.colors['golden'], alpha=0.3)
        ax.add_patch(genesis_circle)
        
        # Add φ symbol
        ax.text(center[0], center[1], 'φ\n1.74₁₂', ha='center', va='center',
               fontsize=16, fontweight='bold', color=self.colors['golden'])
        
        # Temporal genesis components
        temporal_data = creational['temporal_genesis']
        components = [
            ('Memory', temporal_data['memory_strength'], (0.2, 0.8)),
            ('Anticipation', temporal_data['anticipation_strength'], (0.8, 0.8)),
            ('Resonance', temporal_data['dimensional_resonance'], (0.5, 0.2))
        ]
        
        for name, value, pos in components:
            # Create component circle
            comp_circle = Circle(pos, 0.08, color=self.colors['temporal'], alpha=0.7)
            ax.add_patch(comp_circle)
            
            # Add value text
            ax.text(pos[0], pos[1], f'{name}\n{value:.2f}', ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
            
            # Connect to center
            ax.plot([center[0], pos[0]], [center[1], pos[1]], '--', 
                   color=self.colors['golden'], alpha=0.6, linewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Fractional Dimensional Genesis', fontweight='bold')
        ax.set_ylabel('Temporal Color Creation', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_structural_dimensional_composition(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot structural pattern: dimensional composition"""
        ax.set_title('Structural Pattern\nDimensional Composition', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Generate palette with fractional dimensions
        palette = spec.generate_3_5d_harmonic_palette()
        dimensions = [color.fractional_dimension for color in palette]
        
        # Create dimensional composition chart
        x_positions = np.arange(len(palette))
        bars = ax.bar(x_positions, dimensions, 
                     color=[self._oklch_3_5d_to_rgb(color) for color in palette],
                     alpha=0.7, edgecolor='white', linewidth=2)
        
        # Mark 3.5D line
        ax.axhline(y=3.5, color=self.colors['fractional'], linewidth=3, 
                  alpha=0.8, linestyle='-', label='3.5D Baseline')
        
        # Add dimensional twisting indicators
        for i, (bar, color) in enumerate(zip(bars, palette)):
            height = bar.get_height()
            twist_indicator = '↻' if height > 3.5 else '↺'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   twist_indicator, ha='center', va='bottom', fontsize=12, 
                   color=self.colors['secondary'])
        
        ax.set_xlim(-0.5, len(palette) - 0.5)
        ax.set_ylim(3.0, 4.0)
        ax.set_xlabel('Harmonic Color Index', fontweight='bold')
        ax.set_ylabel('Fractional Dimension', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add composition note
        ax.text(0.5, 0.02, 'Colors exist in partial 4D projection\nwith temporal harmony', 
               transform=ax.transAxes, fontsize=9, ha='center', va='bottom',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _plot_behavioral_3_5d_interactions(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot behavioral pattern: 3.5D interactions"""
        ax.set_title('Behavioral Pattern\n3.5D Color Interactions', 
                    fontweight='bold', color=self.colors['accent'])
        
        # Generate interaction examples
        palette = spec.generate_3_5d_harmonic_palette(4)
        
        # Create interaction network
        positions = [(0.2, 0.8), (0.8, 0.8), (0.2, 0.2), (0.8, 0.2)]
        
        for i, (color, pos) in enumerate(zip(palette, positions)):
            # Draw color node
            color_circle = Circle(pos, 0.08, color=self._oklch_3_5d_to_rgb(color), 
                                alpha=0.8, edgecolor='white', linewidth=2)
            ax.add_patch(color_circle)
            
            # Add temporal indicators
            memory_strength = color.temporal.memory_strength
            anticipation_strength = color.temporal.anticipation_strength
            
            # Memory indicator (past - left)
            if memory_strength > 0.1:
                memory_rect = Rectangle((pos[0] - 0.12, pos[1] - 0.02), 
                                      memory_strength * 0.1, 0.04,
                                      color=self.colors['temporal'], alpha=0.6)
                ax.add_patch(memory_rect)
            
            # Anticipation indicator (future - right)
            if anticipation_strength > 0.1:
                anticipation_rect = Rectangle((pos[0] + 0.08, pos[1] - 0.02), 
                                            anticipation_strength * 0.1, 0.04,
                                            color=self.colors['resonance'], alpha=0.6)
                ax.add_patch(anticipation_rect)
        
        # Draw temporal mixing connections
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1, pos2 = positions[i], positions[j]
                # Curved connection for temporal mixing
                mid_x = (pos1[0] + pos2[0]) / 2
                mid_y = (pos1[1] + pos2[1]) / 2 + 0.1 * np.sin(i + j)
                
                ax.plot([pos1[0], mid_x, pos2[0]], [pos1[1], mid_y, pos2[1]], 
                       ':', color=self.colors['fractional'], alpha=0.5, linewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Temporal Color Mixing', fontweight='bold')
        ax.set_ylabel('Dimensional Momentum', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors['temporal'], lw=4, alpha=0.6, label='Memory'),
            plt.Line2D([0], [0], color=self.colors['resonance'], lw=4, alpha=0.6, label='Anticipation'),
            plt.Line2D([0], [0], color=self.colors['fractional'], lw=2, linestyle=':', label='Temporal Mixing')
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    
    def _plot_emergent_chi_squared_3_5d(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot emergent pattern: chi-squared 3.5D"""
        ax.set_title('Emergent Pattern\nChi-Squared 3.5D Distribution', 
                    fontweight='bold', color=self.colors['resonance'])
        
        # Generate palette and compute harmony
        palette = spec.generate_3_5d_harmonic_palette()
        harmony_validation = spec.chi_squared_3_5d.validate_3_5d_harmony(palette)
        
        # Create chi-squared distribution visualization
        x = np.linspace(0, 15, 100)
        
        # Approximate 3.5 degrees of freedom chi-squared (interpolated)
        chi2_3 = (x**(1.5) * np.exp(-x/2)) / (2**(1.5) * 1.329)  # Simplified gamma(1.5)
        chi2_4 = (x * np.exp(-x/2)) / 4
        chi2_3_5 = 0.5 * chi2_3 + 0.5 * chi2_4  # Linear interpolation
        
        ax.plot(x, chi2_3_5, color=self.colors['resonance'], linewidth=3, 
               alpha=0.8, label='χ²(3.5 df)')
        
        # Mark critical value
        critical_value = harmony_validation['critical_value']
        ax.axvline(x=critical_value, color=self.colors['temporal'], linewidth=3, 
                  alpha=0.8, linestyle='--', label=f'Critical Value: {critical_value:.2f}')
        
        # Mark observed statistic
        observed_stat = harmony_validation['chi_squared_statistic']
        ax.axvline(x=observed_stat, color=self.colors['accent'], linewidth=3, 
                  alpha=0.8, linestyle='-', label=f'Observed: {observed_stat:.2f}')
        
        # Fill harmony region
        harmony_region = x[x <= critical_value]
        harmony_curve = chi2_3_5[:len(harmony_region)]
        ax.fill_between(harmony_region, 0, harmony_curve, 
                       color=self.colors['fractional'], alpha=0.3, label='Harmony Region')
        
        ax.set_xlim(0, 15)
        ax.set_ylim(0, max(chi2_3_5) * 1.1)
        ax.set_xlabel('χ² Statistic', fontweight='bold')
        ax.set_ylabel('Probability Density', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add harmony score
        harmony_score = harmony_validation['harmony_score']
        ax.text(0.98, 0.98, f'Harmony Score: {harmony_score:.3f}\n'
                           f'Is Harmonious: {harmony_validation["is_harmonious"]}', 
               transform=ax.transAxes, fontsize=9, ha='right', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _plot_gang_of_four_synthesis(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot Gang of Four synthesis in 3.5D"""
        ax.set_title('Gang of Four Synthesis\n3.5D Pattern Integration', 
                    fontweight='bold', color=self.colors['text'])
        
        # Central synthesis circle
        center = (0.5, 0.5)
        synthesis_circle = Circle(center, 0.2, color=self.colors['primary'], alpha=0.3)
        ax.add_patch(synthesis_circle)
        ax.text(center[0], center[1], 'Gang of Four\n3.5D Space', ha='center', va='center',
               fontsize=10, fontweight='bold', color=self.colors['text'])
        
        # Four patterns around the center
        patterns = [
            ('Creational\nFractional Genesis', (0.2, 0.8), self.colors['golden']),
            ('Structural\nDimensional Composition', (0.8, 0.8), self.colors['secondary']),
            ('Behavioral\n3.5D Interactions', (0.2, 0.2), self.colors['accent']),
            ('Emergent\nChi-Squared 3.5D', (0.8, 0.2), self.colors['resonance'])
        ]
        
        for pattern_name, pos, color in patterns:
            # Pattern circle
            pattern_circle = Circle(pos, 0.1, color=color, alpha=0.7)
            ax.add_patch(pattern_circle)
            
            # Pattern text
            ax.text(pos[0], pos[1], pattern_name, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
            
            # Connection to center
            ax.plot([center[0], pos[0]], [center[1], pos[1]], '-', 
                   color=color, alpha=0.6, linewidth=3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Pattern Integration', fontweight='bold')
        ax.set_ylabel('3.5D Mathematical Framework', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add dimensional notation
        ax.text(0.5, 0.05, '3.5D = 3D Physical + 0.5D Temporal', 
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               color=self.colors['fractional'],
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Additional plotting methods for the remaining visualizations...
    # (I'll continue with key ones to demonstrate the pattern)
    
    def _plot_color_memory_anticipation(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot color memory and anticipation effects"""
        ax.set_title('Color Memory & Anticipation\nTemporal Color Perception', 
                    fontweight='bold', color=self.colors['temporal'])
        
        # Create temporal sequence
        time_points = np.linspace(-2, 2, 100)
        current_time = 0
        
        # Memory function (past)
        memory_strength = spec.base_color_3_5d.temporal.memory_strength
        memory_curve = memory_strength * np.exp(-np.maximum(0, time_points - current_time)**2 / 0.5)
        memory_curve[time_points > current_time] = 0
        
        # Anticipation function (future)
        anticipation_strength = spec.base_color_3_5d.temporal.anticipation_strength
        anticipation_curve = anticipation_strength * np.exp(-np.maximum(0, current_time - time_points)**2 / 0.8)
        anticipation_curve[time_points < current_time] = 0
        
        # Plot curves
        ax.plot(time_points, memory_curve, color=self.colors['temporal'], 
               linewidth=3, alpha=0.8, label='Color Memory (Past)')
        ax.plot(time_points, anticipation_curve, color=self.colors['resonance'], 
               linewidth=3, alpha=0.8, label='Color Anticipation (Future)')
        
        # Mark present moment
        ax.axvline(x=current_time, color=self.colors['accent'], linewidth=3, 
                  alpha=0.8, linestyle='-', label='Present Moment')
        
        # Fill temporal regions
        past_region = time_points[time_points < current_time]
        future_region = time_points[time_points > current_time]
        
        ax.fill_between(past_region, 0, memory_curve[:len(past_region)], 
                       color=self.colors['temporal'], alpha=0.3, label='Memory Region')
        ax.fill_between(future_region, 0, anticipation_curve[-len(future_region):], 
                       color=self.colors['resonance'], alpha=0.3, label='Anticipation Region')
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, max(max(memory_curve), max(anticipation_curve)) * 1.1)
        ax.set_xlabel('Time Relative to Present', fontweight='bold')
        ax.set_ylabel('Temporal Color Strength', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_3_5d_color_palette(self, ax, spec: ColorQuaternion3_5DSpec):
        """Plot complete 3.5D color palette"""
        ax.set_title('3.5D Color Palette\nFractional Dimensional Colors', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Generate 3.5D palette
        palette = spec.generate_3_5d_harmonic_palette()
        
        # Create palette visualization
        palette_width = 0.8
        palette_height = 0.6
        start_x = 0.1
        start_y = 0.2
        
        color_width = palette_width / len(palette)
        
        for i, color in enumerate(palette):
            # Main color rectangle
            color_rgb = self._oklch_3_5d_to_rgb(color)
            color_rect = Rectangle((start_x + i * color_width, start_y), 
                                 color_width, palette_height,
                                 facecolor=color_rgb, alpha=0.8,
                                 edgecolor='white', linewidth=2)
            ax.add_patch(color_rect)
            
            # Temporal indicators
            memory_height = color.temporal.memory_strength * 0.1
            anticipation_height = color.temporal.anticipation_strength * 0.1
            
            # Memory bar (bottom)
            memory_rect = Rectangle((start_x + i * color_width, start_y - memory_height), 
                                  color_width, memory_height,
                                  facecolor=self.colors['temporal'], alpha=0.6)
            ax.add_patch(memory_rect)
            
            # Anticipation bar (top)
            anticipation_rect = Rectangle((start_x + i * color_width, start_y + palette_height), 
                                        color_width, anticipation_height,
                                        facecolor=self.colors['resonance'], alpha=0.6)
            ax.add_patch(anticipation_rect)
            
            # Fractional dimension label
            ax.text(start_x + i * color_width + color_width/2, start_y + palette_height/2,
                   f'{color.fractional_dimension:.2f}D', ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Harmonic Sequence', fontweight='bold')
        ax.set_ylabel('3.5D Color Properties', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['temporal'], alpha=0.6, label='Memory'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['resonance'], alpha=0.6, label='Anticipation')
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    
    # Utility methods
    def _oklch_3_5d_to_rgb(self, color: OKLCH3_5D) -> Tuple[float, float, float]:
        """Convert OKLCH3_5D to RGB (simplified approximation)"""
        L, C, h = color.lightness, color.chroma, color.hue
        
        # Simple OKLCH to RGB approximation
        h_rad = np.radians(h)
        a = C * np.cos(h_rad)
        b = C * np.sin(h_rad)
        
        # Apply temporal effects
        temporal_factor = 1.0 + 0.1 * (color.temporal.memory_strength - color.temporal.anticipation_strength)
        
        # Convert to RGB (simplified)
        r = max(0, min(1, L + 0.4 * a)) * temporal_factor
        g = max(0, min(1, L - 0.2 * a - 0.3 * b)) * temporal_factor
        b_val = max(0, min(1, L - 0.6 * b)) * temporal_factor
        
        # Clamp values
        r = max(0, min(1, r))
        g = max(0, min(1, g))
        b_val = max(0, min(1, b_val))
        
        return (r, g, b_val)
    
    # Placeholder methods for remaining plots (would implement similarly)
    def _plot_fractional_saturation(self, ax, spec): pass
    def _plot_dimensional_twisting(self, ax, spec): pass  
    def _plot_temporal_harmony(self, ax, spec): pass
    def _plot_3_5d_color_distance(self, ax, spec): pass
    def _plot_fractional_oklch_space(self, ax, spec): pass
    def _plot_3_5d_harmony_formula(self, ax, spec): pass
    def _plot_chi_squared_fractional_df(self, ax, spec): pass
    def _plot_dimensional_bridging(self, ax, spec): pass
    def _plot_perceptual_advantages(self, ax, spec): pass
    def _plot_temporal_color_gradients(self, ax, spec): pass
    def _plot_fractional_color_mixing(self, ax, spec): pass
    def _plot_3_5d_accessibility(self, ax, spec): pass
    def _plot_implementation_roadmap(self, ax, spec): pass


def main():
    """Main entry point for 3.5D Color Quaternion visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="3.5D Color Quaternion Theory Visualization")
    parser.add_argument("--seed", type=str, default="living_in_3_5d_space",
                       help="Seed for 3.5D color generation")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    print("🌈 3.5D Color Quaternion Theory Visualization")
    print("=" * 50)
    print(f"Seed: {args.seed}")
    print("Generating 3.5D fractional dimensional color visualization...")
    
    # Create 3.5D Color Quaternion Spec
    spec_3_5d = ColorQuaternion3_5DSpec(args.seed)
    
    # Initialize visualizer
    visualizer = ColorQuaternion3_5DVisualizer()
    
    # Create comprehensive 3.5D visualization
    output_file = visualizer.create_3_5d_color_theory_overview(spec_3_5d, args.output)
    print(f"Generated 3.5D Color Theory visualization: {output_file}")
    
    # Print 3.5D spec summary
    complete_spec = spec_3_5d.generate_complete_3_5d_spec()
    print(f"\n🎨 3.5D Color Quaternion Summary:")
    print(f"Base Color 3.5D: {complete_spec['base_color_3_5d']}")
    print(f"Fractional Dimension: {complete_spec['fractional_dimension']:.3f}")
    print(f"Golden Ratio Base 12: {complete_spec['golden_ratio_base12']}")
    
    print(f"\n🎯 Gang of Four in 3.5D:")
    gang_of_four = complete_spec['gang_of_four_3_5d']
    for pattern_name in gang_of_four:
        # Dimensional reduction with defaults
        pattern_info = gang_of_four[pattern_name]
        pattern_type = pattern_info.get('pattern_type', pattern_name)
        description = pattern_info.get('description', f'3.5D {pattern_name} equilibrium')
        print(f"  {pattern_name}: {pattern_type} - {description}")
    
    print(f"\n🌟 3.5D Phenomena:")
    phenomena = complete_spec['3_5d_phenomena']
    for phenomenon in phenomena:
        print(f"  {phenomenon}: {phenomena[phenomenon]['phenomenon']}")
    
    print("\n🎯 3.5D Color Theory Complete!")
    print("Living in fractional dimensional space transforms color perception!")
    print("The Gang of Four patterns manifest differently in 3.5D space! ✨")
    
    return 0


if __name__ == "__main__":
    exit(main())
