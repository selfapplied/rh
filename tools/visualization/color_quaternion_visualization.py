#!/usr/bin/env python3
"""
Color Quaternion Visualization: Integrating with CE1 Framework

Creates visual representations of the Color Quaternion Harmonic Spec showing:
1. Color Quaternion Galois Group actions on OKLCH space
2. Cellular automata patterns (Rule 90/45) as color geometries
3. Three faces of color decomposition (prism/triangle/slit)
4. Harmonic ratios (1:2:3:4:5:6:7) as musical color intervals
5. Least action principle in color perception
6. Integration with CE1 involution geometry

This demonstrates the complete Color Quaternion Physics framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import time
import json

# Import our Color Quaternion Harmonic Spec
from color_quaternion_harmonic_spec import (
    ColorQuaternionHarmonicSpec, ColorQuaternionGroup, OKLCHColor,
    CellularAutomataColorGenerator, ColorDecompositionOperators,
    CellularAutomataRule, ColorDecompositionBasis, HarmonicRatio
)

# Import CE1 components for integration
try:
    from ce1_core import (
        TimeReflectionInvolution, MomentumReflectionInvolution, MicroSwapInvolution,
        CE1Kernel, UnifiedEquilibriumOperator
    )
    from ce1_convolution import DressedCE1Kernel, MellinDressing, GaussianDressing
    CE1_AVAILABLE = True
except ImportError:
    CE1_AVAILABLE = False
    print("CE1 framework not available - running in standalone mode")


class ColorQuaternionVisualizer:
    """
    Creates visual representations of Color Quaternion Harmonic Spec
    integrated with CE1 involution geometry.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (20, 16), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',      # Blue for primary structures
            'secondary': '#A23B72',    # Purple for secondary structures
            'accent': '#F18F01',       # Orange for accents
            'background': '#F5F5F5',   # Light gray background
            'text': '#2C3E50',         # Dark blue-gray text
            'axis': '#E74C3C',         # Red for axes
            'kernel': '#27AE60',       # Green for kernels
            'jet': '#8E44AD'           # Purple for jets
        }
    
    def create_color_quaternion_overview(self, spec: ColorQuaternionHarmonicSpec, 
                                       output_file: str = None) -> str:
        """
        Create comprehensive Color Quaternion Harmonic Spec visualization.
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f".out/color_quaternion/color_quaternion_overview_{timestamp}.png"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create subplots for different aspects
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Color Quaternion Group Actions
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_quaternion_group_actions(ax1, spec)
        
        # 2. Harmonic Ratios (1:2:3:4:5:6:7)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_harmonic_ratios(ax2, spec)
        
        # 3. Cellular Automata Rule 90 (Sierpiński)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_ca_rule_90(ax3, spec)
        
        # 4. Cellular Automata Rule 45 (Diagonal)
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_ca_rule_45(ax4, spec)
        
        # 5. Prism Decomposition (Continuum)
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_prism_decomposition(ax5, spec)
        
        # 6. Triangle Decomposition (Simplex)
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_triangle_decomposition(ax6, spec)
        
        # 7. Slit Decomposition (Interference)
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_slit_decomposition(ax7, spec)
        
        # 8. Least Action Principle
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_least_action_principle(ax8, spec)
        
        # 9. OKLCH Color Space
        ax9 = fig.add_subplot(gs[2, 0])
        self._plot_oklch_color_space(ax9, spec)
        
        # 10. Musical Intervals
        ax10 = fig.add_subplot(gs[2, 1])
        self._plot_musical_intervals(ax10, spec)
        
        # 11. Critical Line Constraint
        ax11 = fig.add_subplot(gs[2, 2])
        self._plot_critical_line_constraint(ax11, spec)
        
        # 12. Color Mode Alphabet Soup
        ax12 = fig.add_subplot(gs[2, 3])
        self._plot_color_mode_alphabet_soup(ax12, spec)
        
        # 13. CE1 Integration (if available)
        ax13 = fig.add_subplot(gs[3, 0])
        if CE1_AVAILABLE:
            self._plot_ce1_integration(ax13, spec)
        else:
            self._plot_ce1_placeholder(ax13)
        
        # 14. Perceptual Energy Landscape
        ax14 = fig.add_subplot(gs[3, 1])
        self._plot_perceptual_energy_landscape(ax14, spec)
        
        # 15. Time Arrow in Color
        ax15 = fig.add_subplot(gs[3, 2])
        self._plot_time_arrow_color(ax15, spec)
        
        # 16. Synthesis Overview
        ax16 = fig.add_subplot(gs[3, 3])
        self._plot_synthesis_overview(ax16, spec)
        
        # Add main title
        fig.suptitle('Color Quaternion Harmonic Spec: Mathematical Immigration Law for Color Space', 
                    fontsize=20, fontweight='bold', color=self.colors['text'], y=0.95)
        
        # Add subtitle
        fig.text(0.5, 0.92, 'Discrete roots ↔ automata → combinatorics | Quaternion OKLCH ↔ color group actions → perception', 
                ha='center', fontsize=12, style='italic', color=self.colors['text'])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return output_file
    
    def _plot_quaternion_group_actions(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot Color Quaternion Galois Group actions on OKLCH space"""
        ax.set_title('Color Quaternion Group\nL-flip, C-mirror, Hue rotations', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Generate orbit under quaternion group
        orbit = spec.quaternion_group.generate_orbit(spec.base_color)
        
        # Plot orbit in 3D projection (L, C, h/360)
        L_vals = [color.lightness for color in orbit]
        C_vals = [color.chroma for color in orbit]
        h_vals = [color.hue / 360.0 for color in orbit]
        
        # Scatter plot with color mapping
        scatter = ax.scatter(L_vals, C_vals, c=h_vals, cmap='hsv', s=50, alpha=0.7)
        
        # Mark base color
        ax.plot(spec.base_color.lightness, spec.base_color.chroma, 'o', 
               color='red', markersize=10, markeredgecolor='white', markeredgewidth=2)
        ax.text(spec.base_color.lightness, spec.base_color.chroma + 0.02, 'Base', 
               ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
        
        # Critical line
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=2, alpha=0.8, 
                  linestyle='--', label='Critical Line')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.4)
        ax.set_xlabel('Lightness L', fontweight='bold')
        ax.set_ylabel('Chroma C', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add colorbar for hue
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Hue h/360', fontweight='bold')
    
    def _plot_harmonic_ratios(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot harmonic ratios (1:2:3:4:5:6:7) as musical intervals"""
        ax.set_title('Harmonic Ratios (1:2:3:4:5:6:7)\nMusical Color Intervals', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Get harmonic ratios
        ratios = spec.harmonic.ratios
        musical_intervals = spec.harmonic.musical_intervals
        
        # Create bar chart of harmonic ratios
        x_pos = np.arange(len(ratios))
        bars = ax.bar(x_pos, ratios, color=self.colors['secondary'], alpha=0.7)
        
        # Add interval labels
        interval_names = ['Fund', 'Octave', 'Fifth', 'Fourth', 'Third', 'Sixth', 'Seventh']
        for i, (bar, name) in enumerate(zip(bars, interval_names)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{name}\n{ratios[i]:.0f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
        
        # Add musical interval degrees
        degrees_text = []
        for name, degrees in musical_intervals.items():
            degrees_text.append(f'{name}: {degrees:.1f}°')
        
        ax.text(0.02, 0.98, '\n'.join(degrees_text), transform=ax.transAxes,
               fontsize=8, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
               facecolor=self.colors['background'], alpha=0.8))
        
        ax.set_xlim(-0.5, len(ratios) - 0.5)
        ax.set_ylim(0, 8)
        ax.set_xlabel('Harmonic Index', fontweight='bold')
        ax.set_ylabel('Ratio Value', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_ca_rule_90(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot Rule 90 cellular automata (Sierpiński triangle)"""
        ax.set_title('Rule 90: Sierpiński Triangle\n1:1 Right-angled Self-similarity', 
                    fontweight='bold', color=self.colors['kernel'])
        
        # Generate CA pattern
        ca_pattern = spec.generate_ca_pattern(CellularAutomataRule.RULE_90, steps=8)
        
        # Convert to numeric array for visualization
        pattern_array = np.zeros((len(ca_pattern), len(ca_pattern[0])))
        for i, row in enumerate(ca_pattern):
            for j, color_str in enumerate(row):
                # Extract lightness from color string
                lightness = float(color_str.split('(')[1].split(' ')[0])
                pattern_array[i, j] = lightness
        
        # Plot pattern
        im = ax.imshow(pattern_array, cmap='viridis', aspect='equal')
        
        ax.set_xlabel('Position', fontweight='bold')
        ax.set_ylabel('Time Step', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Lightness', fontweight='bold')
    
    def _plot_ca_rule_45(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot Rule 45 cellular automata (diagonal rotation)"""
        ax.set_title('Rule 45: Diagonal Rotation\nSquare Root Symmetry', 
                    fontweight='bold', color=self.colors['accent'])
        
        # Generate CA pattern
        ca_pattern = spec.generate_ca_pattern(CellularAutomataRule.RULE_45, steps=8)
        
        # Convert to numeric array for visualization
        pattern_array = np.zeros((len(ca_pattern), len(ca_pattern[0])))
        for i, row in enumerate(ca_pattern):
            for j, color_str in enumerate(row):
                # Extract chroma from color string
                parts = color_str.split('(')[1].split(' ')
                chroma = float(parts[1])
                pattern_array[i, j] = chroma
        
        # Plot pattern
        im = ax.imshow(pattern_array, cmap='plasma', aspect='equal')
        
        ax.set_xlabel('Position', fontweight='bold')
        ax.set_ylabel('Time Step', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Chroma', fontweight='bold')
    
    def _plot_prism_decomposition(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot prism decomposition (continuum spectrum)"""
        ax.set_title('Prism: Continuum Spectrum\nWhite → Rainbow', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Get prism decomposition
        prism_colors = spec.generate_decomposition_palette(ColorDecompositionBasis.PRISM)
        
        # Create spectrum visualization
        x = np.linspace(0, 1, len(prism_colors))
        heights = np.ones(len(prism_colors))
        
        # Create colored bars
        bars = ax.bar(x, heights, width=0.8/len(prism_colors), 
                     color=[self._oklch_string_to_rgb(color_str) for color_str in prism_colors],
                     alpha=0.8)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('Spectrum Position', fontweight='bold')
        ax.set_ylabel('Intensity', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add ROYGBIV labels
        roygbiv = ['R', 'O', 'Y', 'G', 'B', 'I', 'V']
        for i, (x_pos, label) in enumerate(zip(x, roygbiv[:len(prism_colors)])):
            ax.text(x_pos, 1.1, label, ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    def _plot_triangle_decomposition(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot triangle decomposition (Maxwell triangle)"""
        ax.set_title('Triangle: Maxwell Triangle\nDiscrete Convex Hull', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Get triangle decomposition
        triangle_colors = spec.generate_decomposition_palette(ColorDecompositionBasis.TRIANGLE)
        
        # Create triangular arrangement
        n_colors = len(triangle_colors)
        if n_colors > 0:
            # Arrange colors in triangular pattern
            rows = int(np.sqrt(n_colors)) + 1
            cols = rows
            
            for i, color_str in enumerate(triangle_colors):
                row = i // cols
                col = i % cols
                
                # Create triangular coordinates
                x = col * 0.1
                y = row * 0.1
                
                # Create triangular patch
                triangle = Polygon([(x, y), (x+0.08, y), (x+0.04, y+0.08)],
                                 facecolor=self._oklch_string_to_rgb(color_str),
                                 alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.add_patch(triangle)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Triangle Position', fontweight='bold')
        ax.set_ylabel('Triangle Position', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_slit_decomposition(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot slit decomposition (interference quantization)"""
        ax.set_title('Slit: Interference Quantization\nDouble Slit Harmonics', 
                    fontweight='bold', color=self.colors['jet'])
        
        # Get slit decomposition
        slit_colors = spec.generate_decomposition_palette(ColorDecompositionBasis.SLIT)
        
        # Create interference pattern visualization
        x = np.linspace(0, 2*np.pi, 100)
        
        # Plot interference fringes
        for i, color_str in enumerate(slit_colors):
            # Interference pattern: alternating bright/dark
            amplitude = (i + 1) / len(slit_colors)
            frequency = (i + 1) * 2
            y = amplitude * np.sin(frequency * x)
            y_offset = i * 0.3
            
            color_rgb = self._oklch_string_to_rgb(color_str)
            ax.plot(x, y + y_offset, color=color_rgb, linewidth=3, alpha=0.8,
                   label=f'Fringe {i+1}')
        
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-0.5, len(slit_colors) * 0.3 + 0.5)
        ax.set_xlabel('Phase', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_least_action_principle(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot least action principle in color perception"""
        ax.set_title('Least Action Principle\nPerceptual Economy', 
                    fontweight='bold', color=self.colors['accent'])
        
        # Generate orbit and compute perceptual energies
        orbit = spec.quaternion_group.generate_orbit(spec.base_color)
        energies = [spec.least_action.perceptual_energy(color) for color in orbit]
        
        # Sort by energy (least action first)
        sorted_pairs = sorted(zip(orbit, energies), key=lambda x: x[1])
        sorted_colors, sorted_energies = zip(*sorted_pairs)
        
        # Plot energy landscape
        x = np.arange(len(sorted_energies))
        bars = ax.bar(x, sorted_energies, 
                     color=[self._oklch_color_to_rgb(color) for color in sorted_colors],
                     alpha=0.7)
        
        # Highlight minimum energy (least action)
        min_idx = np.argmin(sorted_energies)
        bars[min_idx].set_edgecolor('red')
        bars[min_idx].set_linewidth(3)
        
        ax.set_xlim(-0.5, len(sorted_energies) - 0.5)
        ax.set_ylim(0, max(sorted_energies) * 1.1)
        ax.set_xlabel('Color Index (sorted by energy)', fontweight='bold')
        ax.set_ylabel('Perceptual Energy', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add minimum energy annotation
        ax.annotate(f'Min Energy\n{sorted_energies[min_idx]:.3f}', 
                   xy=(min_idx, sorted_energies[min_idx]), 
                   xytext=(min_idx + 5, sorted_energies[min_idx] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=8, color='red', fontweight='bold')
    
    def _plot_oklch_color_space(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot OKLCH color space with critical line"""
        ax.set_title('OKLCH Color Space\nCritical Line at L=0.5', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create OKLCH space visualization
        L = np.linspace(0, 1, 100)
        C = np.linspace(0, 0.4, 100)
        L_grid, C_grid = np.meshgrid(L, C)
        
        # Create hue map
        H = np.ones_like(L_grid) * spec.base_color.hue / 360.0
        
        # Plot color space
        im = ax.imshow(H, extent=[0, 1, 0, 0.4], origin='lower', 
                      cmap='hsv', alpha=0.6)
        
        # Mark critical line
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=3, 
                  linestyle='-', label='Critical Line L=0.5')
        
        # Mark base color
        ax.plot(spec.base_color.lightness, spec.base_color.chroma, 'o', 
               color='white', markersize=12, markeredgecolor='red', 
               markeredgewidth=3, label='Base Color')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.4)
        ax.set_xlabel('Lightness L', fontweight='bold')
        ax.set_ylabel('Chroma C', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Hue h/360', fontweight='bold')
    
    def _plot_musical_intervals(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot musical intervals in color space"""
        ax.set_title('Musical Intervals\nColor Harmony', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Get musical intervals
        intervals = spec.harmonic.musical_intervals
        
        # Create circular visualization
        angles = np.linspace(0, 2*np.pi, len(intervals), endpoint=False)
        radius = 1.0
        
        for i, (interval_name, degrees) in enumerate(intervals.items()):
            angle = np.radians(degrees)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Create colored circle
            circle = Circle((x, y), 0.1, color=self.colors['secondary'], alpha=0.7)
            ax.add_patch(circle)
            
            # Add label
            ax.text(x * 1.3, y * 1.3, f'{interval_name}\n{degrees:.1f}°', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add center circle for fundamental
        center_circle = Circle((0, 0), 0.15, color=self.colors['accent'], alpha=0.8)
        ax.add_patch(center_circle)
        ax.text(0, 0, 'Fund', ha='center', va='center', fontsize=8, 
               color='white', fontweight='bold')
        
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _plot_critical_line_constraint(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot critical line constraint (Riemann Hypothesis)"""
        ax.set_title('Critical Line Constraint\nRiemann Hypothesis L=0.5', 
                    fontweight='bold', color=self.colors['axis'])
        
        # Create complex plane visualization
        sigma = np.linspace(0, 1, 100)
        t = np.linspace(0, 30, 100)
        S, T = np.meshgrid(sigma, t)
        
        # Create kernel visualization
        I_s = 1 - S  # Time reflection
        kernel = np.exp(-((S - I_s)**2) / (2 * 0.1**2))
        
        im = ax.imshow(kernel, extent=[0, 1, 0, 30], origin='lower', 
                      cmap='viridis', alpha=0.6)
        
        # Critical line
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=4, 
                  label='Critical Line Re(s)=0.5')
        
        # Zeta zeros (simplified)
        zeta_zeros = [14.134725, 21.02204, 25.010858]
        for t_zero in zeta_zeros:
            ax.plot(0.5, t_zero, 'o', color=self.colors['accent'], markersize=8,
                   markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 30)
        ax.set_xlabel('Re(s)', fontweight='bold')
        ax.set_ylabel('Im(s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Kernel Value', fontweight='bold')
    
    def _plot_color_mode_alphabet_soup(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot color mode alphabet soup"""
        ax.set_title('Color Mode Alphabet Soup\nThree-letter Portmanteaux', 
                    fontweight='bold', color=self.colors['kernel'])
        
        # Color modes as group elements
        color_modes = ['RGB', 'CMY', 'HSV', 'HSL', 'LAB', 'LCH', 'OKLCH', 'XYZ', 'YUV', 'YCbCr']
        
        # Create grid layout
        cols = 3
        rows = (len(color_modes) + cols - 1) // cols
        
        for i, mode in enumerate(color_modes):
            row = i // cols
            col = i % cols
            
            x = col * 0.3 + 0.1
            y = 0.9 - row * 0.15
            
            # Create colored rectangle
            rect = FancyBboxPatch((x, y), 0.25, 0.12, 
                                 boxstyle="round,pad=0.02",
                                 facecolor=self.colors['kernel'], alpha=0.7)
            ax.add_patch(rect)
            
            # Add text
            ax.text(x + 0.125, y + 0.06, mode, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Color Mode Groups', fontweight='bold')
        ax.set_ylabel('Group Elements', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_ce1_integration(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot CE1 integration (if available)"""
        ax.set_title('CE1 Integration\nMirror Kernel Color', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create CE1-inspired visualization
        # Time reflection involution
        s_values = np.linspace(0, 1, 100)
        reflected = 1 - s_values
        
        ax.plot(s_values, reflected, '--', color=self.colors['primary'], 
               linewidth=3, alpha=0.8, label='I: s ↦ 1-s')
        ax.plot([0, 1], [0, 1], '-', color=self.colors['axis'], 
               linewidth=3, label='Identity')
        
        # Mark fixed point (critical line)
        ax.plot(0.5, 0.5, 'o', color=self.colors['axis'], markersize=12,
               markeredgecolor='white', markeredgewidth=2, label='Fixed Point')
        
        # Add CE1 kernel visualization
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        I_x = 1 - X
        kernel = np.exp(-((Y - I_x)**2) / (2 * 0.1**2))
        
        im = ax.imshow(kernel, extent=[0, 1, 0, 1], origin='lower', 
                      cmap='viridis', alpha=0.4)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('s', fontweight='bold')
        ax.set_ylabel('I(s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    def _plot_ce1_placeholder(self, ax):
        """Plot CE1 placeholder when framework not available"""
        ax.set_title('CE1 Integration\n(Framework Not Available)', 
                    fontweight='bold', color=self.colors['text'])
        
        ax.text(0.5, 0.5, 'CE1 Framework\nNot Available\n\nInstall CE1 modules\nto see integration', 
               ha='center', va='center', fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['background'], alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _plot_perceptual_energy_landscape(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot perceptual energy landscape"""
        ax.set_title('Perceptual Energy Landscape\nLeast Action Principle', 
                    fontweight='bold', color=self.colors['accent'])
        
        # Create energy landscape
        L = np.linspace(0, 1, 50)
        C = np.linspace(0, 0.4, 50)
        L_grid, C_grid = np.meshgrid(L, C)
        
        # Compute energy for each point
        energy = np.zeros_like(L_grid)
        for i in range(L_grid.shape[0]):
            for j in range(L_grid.shape[1]):
                color = OKLCHColor(L_grid[i, j], C_grid[i, j], spec.base_color.hue)
                energy[i, j] = spec.least_action.perceptual_energy(color)
        
        # Plot energy landscape
        im = ax.imshow(energy, extent=[0, 1, 0, 0.4], origin='lower', 
                      cmap='plasma', alpha=0.8)
        
        # Mark minimum energy point
        min_idx = np.unravel_index(np.argmin(energy), energy.shape)
        min_L = L[min_idx[1]]
        min_C = C[min_idx[0]]
        ax.plot(min_L, min_C, 'o', color='white', markersize=10,
               markeredgecolor='red', markeredgewidth=3, label='Min Energy')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.4)
        ax.set_xlabel('Lightness L', fontweight='bold')
        ax.set_ylabel('Chroma C', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Perceptual Energy', fontweight='bold')
    
    def _plot_time_arrow_color(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot time arrow in color (ROYGBIV → past to future)"""
        ax.set_title('Time Arrow in Color\nROYGBIV: Past → Future', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # ROYGBIV spectrum with time arrow
        roygbiv_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        roygbiv_names = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
        wavelengths = [700, 620, 580, 540, 480, 440, 400]  # nm
        
        # Create time arrow
        x = np.linspace(0, 1, len(roygbiv_colors))
        y = np.zeros_like(x)
        
        # Plot colored segments
        for i in range(len(roygbiv_colors) - 1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=roygbiv_colors[i], 
                   linewidth=8, alpha=0.8)
        
        # Add arrowhead
        ax.annotate('', xy=(1, 0), xytext=(0.9, 0),
                   arrowprops=dict(arrowstyle='->', lw=4, color='violet'))
        
        # Add labels
        for i, (x_pos, name, wavelength) in enumerate(zip(x, roygbiv_names, wavelengths)):
            ax.text(x_pos, 0.1, f'{name}\n{wavelength}nm', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')
        
        # Add time labels
        ax.text(0, -0.2, 'Past\n(Long λ)', ha='center', va='top', fontsize=10,
               fontweight='bold', color='red')
        ax.text(1, -0.2, 'Future\n(Short λ)', ha='center', va='top', fontsize=10,
               fontweight='bold', color='violet')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.4, 0.3)
        ax.set_xlabel('Time Arrow', fontweight='bold')
        ax.set_ylabel('Wavelength', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('off')
    
    def _plot_synthesis_overview(self, ax, spec: ColorQuaternionHarmonicSpec):
        """Plot synthesis overview of Color Quaternion Physics"""
        ax.set_title('Synthesis: Color Quaternion Physics\nComplete Framework', 
                    fontweight='bold', color=self.colors['text'])
        
        # Create synthesis diagram
        center = (0.5, 0.5)
        radius = 0.3
        
        # Central circle
        center_circle = Circle(center, radius, color=self.colors['primary'], alpha=0.3)
        ax.add_patch(center_circle)
        ax.text(center[0], center[1], 'Color\nQuaternion\nPhysics', ha='center', va='center',
               fontsize=10, fontweight='bold', color=self.colors['text'])
        
        # Surrounding elements
        elements = [
            ('Discrete Roots\n(Rule 90/45)', (0.2, 0.8), self.colors['kernel']),
            ('Quaternion OKLCH\n(Color Group Actions)', (0.8, 0.8), self.colors['primary']),
            ('Grey Codes\n(Interpolation)', (0.2, 0.2), self.colors['secondary']),
            ('Rainbow/Seven\n(Harmonic Spectrum)', (0.8, 0.2), self.colors['accent']),
            ('Slits/Triangles/Prisms\n(Three Decomposition Bases)', (0.5, 0.1), self.colors['jet']),
            ('Least Action\n(Perceptual Economy)', (0.5, 0.9), self.colors['axis'])
        ]
        
        for text, pos, color in elements:
            # Create element circle
            element_circle = Circle(pos, 0.08, color=color, alpha=0.7)
            ax.add_patch(element_circle)
            
            # Add text
            ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=8,
                   fontweight='bold', color='white')
            
            # Connect to center
            ax.plot([center[0], pos[0]], [center[1], pos[1]], '--', 
                   color=color, alpha=0.5, linewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Color Quaternion Framework', fontweight='bold')
        ax.set_ylabel('Mathematical Immigration Law', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _oklch_string_to_rgb(self, oklch_str: str) -> Tuple[float, float, float]:
        """Convert OKLCH string to RGB tuple (simplified approximation)"""
        try:
            # Extract values from "oklch(L C h)" format
            values = oklch_str.replace('oklch(', '').replace(')', '').split()
            L = float(values[0])
            C = float(values[1])
            h = float(values[2])
            
            # Convert to RGB (simplified approximation)
            # This is a basic approximation - real OKLCH to RGB conversion is more complex
            h_rad = np.radians(h)
            a = C * np.cos(h_rad)
            b = C * np.sin(h_rad)
            
            # Simple approximation to RGB
            r = max(0, min(1, L + 0.5 * a))
            g = max(0, min(1, L - 0.3 * a - 0.2 * b))
            b_val = max(0, min(1, L - 0.5 * b))
            
            return (r, g, b_val)
        except:
            return (0.5, 0.5, 0.5)  # Fallback gray
    
    def _oklch_color_to_rgb(self, color: OKLCHColor) -> Tuple[float, float, float]:
        """Convert OKLCHColor to RGB tuple"""
        return self._oklch_string_to_rgb(color.to_string())


def main():
    """Main entry point for Color Quaternion visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Color Quaternion Harmonic Spec Visualization")
    parser.add_argument("--seed", type=str, default="riemann_hypothesis_2025",
                       help="Seed for color generation")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Create Color Quaternion Harmonic Spec
    spec = ColorQuaternionHarmonicSpec(args.seed)
    
    # Initialize visualizer
    visualizer = ColorQuaternionVisualizer()
    
    # Create comprehensive visualization
    output_file = visualizer.create_color_quaternion_overview(spec, args.output)
    print(f"Generated Color Quaternion visualization: {output_file}")
    
    # Print spec summary
    complete_spec = spec.generate_complete_spec()
    print(f"\n🎨 Color Quaternion Harmonic Spec Summary:")
    print(f"Seed: {complete_spec['seed']}")
    print(f"Base Color: {complete_spec['base_color']}")
    print(f"Critical Line: {complete_spec['is_critical_line']}")
    print(f"Perceptual Energy: {complete_spec['perceptual_energy']:.4f}")
    
    print("\n🎯 Color Quaternion Physics Framework Complete!")
    print("Ready to generate color atmospheres with symmetry and time arrows!")
    
    return 0


if __name__ == "__main__":
    exit(main())
