#!/usr/bin/env python3
"""
CE1 Visualization: Involution Geometry Rendering

Creates visual representations of the CE1 framework showing:
1. Involution transformations and their fixed points (axes)
2. Kernel structure K(x,y) = δ(y - I·x)
3. Convolution dressing and spectrum
4. Jet expansion and rank drop patterns
5. Domain-specific geometries (ζ, chemical, dynamical)

This demonstrates the "balance-geometry" that emerges from involution symmetry.
"""

from __future__ import annotations

import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# Import CE1 components
from matplotlib.patches import Circle, FancyBboxPatch


class CE1Visualizer:
    """
    Creates visual representations of CE1 involution geometry.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12), dpi: int = 150):
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
    
    def create_involution_geometry(self, output_file: str = None) -> str:
        """
        Create comprehensive involution geometry visualization.
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f".out/ce1_visualization/involution_geometry_{timestamp}.png"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create subplots for different aspects
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Time Reflection Involution (Riemann ζ)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_time_reflection(ax1)
        
        # 2. Momentum Reflection Involution (Dynamical Systems)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_momentum_reflection(ax2)
        
        # 3. Microswap Involution (Chemical Systems)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_microswap_involution(ax3)
        
        # 4. CE1 Kernel Structure
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_ce1_kernel(ax4)
        
        # 5. Convolution Dressing
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_convolution_dressing(ax5)
        
        # 6. Jet Expansion
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_jet_expansion(ax6)
        
        # 7. Spectrum Analysis
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_spectrum_analysis(ax7)
        
        # 8. Rank Drop Patterns
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_rank_drop_patterns(ax8)
        
        # 9. Balance Geometry Overview
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_balance_geometry(ax9)
        
        # Add main title
        fig.suptitle('CE1 Framework: Involution Geometry and Balance-Geometry', 
                    fontsize=20, fontweight='bold', color=self.colors['text'], y=0.95)
        
        # Add subtitle
        fig.text(0.5, 0.92, 'Mirror Kernel K(x,y) = δ(y - I·x) generates equilibrium geometry through involution symmetry', 
                ha='center', fontsize=12, style='italic', color=self.colors['text'])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return output_file
    
    def _plot_time_reflection(self, ax):
        """Plot time reflection involution I: s ↦ 1-s"""
        ax.set_title('Time Reflection: I: s ↦ 1-s\n(Riemann ζ)', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create complex plane
        s_values = np.linspace(0, 1, 100)
        reflected = 1 - s_values
        
        # Plot the involution
        ax.plot(s_values, reflected, '--', color=self.colors['primary'], linewidth=2, alpha=0.7, label='I(s) = 1-s')
        ax.plot([0, 1], [0, 1], '-', color=self.colors['axis'], linewidth=3, label='Identity')
        
        # Mark fixed point (critical line)
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=2, alpha=0.8, label='Axis A: s = 1/2')
        ax.axhline(y=0.5, color=self.colors['axis'], linewidth=2, alpha=0.8)
        ax.plot(0.5, 0.5, 'o', color=self.colors['axis'], markersize=8, label='Fixed Point')
        
        # Add some zeta zeros
        zeta_zeros = [14.134725, 21.02204, 25.010858]
        for i, t in enumerate(zeta_zeros):
            ax.plot(0.5, 0.5, 's', color=self.colors['accent'], markersize=6, alpha=0.8)
            ax.text(0.5, 0.5, f'ζ₀', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Re(s)', fontweight='bold')
        ax.set_ylabel('I(s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    def _plot_momentum_reflection(self, ax):
        """Plot momentum reflection involution I: (q,p) ↦ (q,-p)"""
        ax.set_title('Momentum Reflection: I: (q,p) ↦ (q,-p)\n(Dynamical Systems)', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Create phase space
        q = np.linspace(-2, 2, 50)
        p = np.linspace(-2, 2, 50)
        Q, P = np.meshgrid(q, p)
        
        # Plot momentum reflection
        ax.plot(q, -p, '--', color=self.colors['secondary'], linewidth=2, alpha=0.7, label='I: p ↦ -p')
        ax.plot([-2, 2], [0, 0], '-', color=self.colors['axis'], linewidth=3, label='Axis A: p = 0')
        
        # Add some phase space trajectories
        t = np.linspace(0, 4*np.pi, 100)
        for i in range(3):
            q_traj = np.cos(t + i*np.pi/3)
            p_traj = np.sin(t + i*np.pi/3)
            ax.plot(q_traj, p_traj, '-', color=self.colors['accent'], alpha=0.6, linewidth=1)
            ax.plot(q_traj, -p_traj, '--', color=self.colors['accent'], alpha=0.6, linewidth=1)
        
        # Mark critical points
        ax.plot(0, 0, 'o', color=self.colors['axis'], markersize=8, label='Critical Point')
        ax.plot(1, 0, 's', color=self.colors['accent'], markersize=6, label='Equilibrium')
        ax.plot(-1, 0, 's', color=self.colors['accent'], markersize=6)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('Position q', fontweight='bold')
        ax.set_ylabel('Momentum p', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    def _plot_microswap_involution(self, ax):
        """Plot microswap involution for chemical systems"""
        ax.set_title('Microswap Involution\n(Chemical Kinetics)', 
                    fontweight='bold', color=self.colors['jet'])
        
        # Create concentration space
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # Plot microswap (simplified as coordinate swap)
        ax.plot(y, x, '--', color=self.colors['jet'], linewidth=2, alpha=0.7, label='I: (x,y) ↦ (y,x)')
        ax.plot([0, 1], [0, 1], '-', color=self.colors['axis'], linewidth=3, label='Axis A: x = y')
        
        # Add reaction network
        # A ⇄ B ⇄ C
        ax.plot(0.2, 0.3, 'o', color=self.colors['primary'], markersize=10, label='Species A')
        ax.plot(0.5, 0.5, 'o', color=self.colors['secondary'], markersize=10, label='Species B')
        ax.plot(0.8, 0.7, 'o', color=self.colors['accent'], markersize=10, label='Species C')
        
        # Add reaction arrows
        ax.annotate('', xy=(0.4, 0.4), xytext=(0.3, 0.3), 
                   arrowprops=dict(arrowstyle='<->', color=self.colors['primary'], lw=2))
        ax.annotate('', xy=(0.6, 0.6), xytext=(0.5, 0.5), 
                   arrowprops=dict(arrowstyle='<->', color=self.colors['secondary'], lw=2))
        
        # Add equilibrium line
        eq_line = np.linspace(0, 1, 100)
        ax.plot(eq_line, eq_line, '-', color=self.colors['axis'], linewidth=2, alpha=0.8, label='Equilibrium Line')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Concentration x', fontweight='bold')
        ax.set_ylabel('Concentration y', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    def _plot_ce1_kernel(self, ax):
        """Plot CE1 kernel structure K(x,y) = δ(y - I·x)"""
        ax.set_title('CE1 Kernel: K(x,y) = δ(y - I·x)', 
                    fontweight='bold', color=self.colors['kernel'])
        
        # Create grid
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # Create kernel visualization (Gaussian approximation)
        sigma = 0.05
        I_x = 1 - X  # Time reflection
        kernel = np.exp(-((Y - I_x)**2) / (2 * sigma**2))
        
        # Plot kernel
        im = ax.imshow(kernel, extent=[0, 1, 0, 1], origin='lower', 
                      cmap='viridis', alpha=0.8)
        
        # Add diagonal line (identity)
        ax.plot([0, 1], [0, 1], '--', color='white', linewidth=2, alpha=0.8, label='Identity')
        
        # Add involution line
        ax.plot(x, 1-x, '--', color=self.colors['accent'], linewidth=2, alpha=0.8, label='I: s ↦ 1-s')
        
        # Add fixed point
        ax.plot(0.5, 0.5, 'o', color=self.colors['axis'], markersize=8, label='Fixed Point')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x', fontweight='bold')
        ax.set_ylabel('y', fontweight='bold')
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_convolution_dressing(self, ax):
        """Plot convolution dressing K_dressed = G * δ∘I"""
        ax.set_title('Convolution Dressing: K_dressed = G * δ∘I', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create dressing functions
        x = np.linspace(0, 1, 100)
        
        # Gaussian dressing
        gaussian = np.exp(-((x - 0.5)**2) / (2 * 0.1**2))
        ax.plot(x, gaussian, '-', color=self.colors['primary'], linewidth=2, label='Gaussian G')
        
        # Mellin dressing (simplified)
        mellin = np.power(np.abs(x - 0.5) + 0.1, -0.5)
        mellin = mellin / np.max(mellin)  # Normalize
        ax.plot(x, mellin, '-', color=self.colors['secondary'], linewidth=2, label='Mellin G')
        
        # Wavelet dressing (simplified)
        wavelet = np.exp(-((x - 0.5)**2) / (2 * 0.05**2)) * np.cos(10 * (x - 0.5))
        ax.plot(x, wavelet, '-', color=self.colors['accent'], linewidth=2, label='Wavelet G')
        
        # Add axis
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=2, alpha=0.8, label='Axis A')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('s', fontweight='bold')
        ax.set_ylabel('G(s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_jet_expansion(self, ax):
        """Plot jet expansion and normal forms"""
        ax.set_title('Jet Expansion: Order Detection', 
                    fontweight='bold', color=self.colors['jet'])
        
        # Create jet orders
        orders = [0, 1, 2, 3, 4, 5]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['kernel'], self.colors['jet'], self.colors['axis']]
        
        x = np.linspace(-1, 1, 100)
        
        for i, order in enumerate(orders):
            if order == 0:
                y = np.ones_like(x) * 0.1
            elif order == 1:
                y = x
            elif order == 2:
                y = x**2
            elif order == 3:
                y = x**3
            elif order == 4:
                y = x**4
            else:
                y = x**5
            
            # Normalize
            y = y / (np.max(np.abs(y)) + 1e-12)
            y = y + i * 0.3  # Offset for visibility
            
            ax.plot(x, y, '-', color=colors[i], linewidth=2, label=f'Order {order}')
        
        # Add normal form labels
        normal_forms = ['Regular', 'Fold', 'Cusp', 'Swallowtail', 'Butterfly', 'Higher']
        for i, form in enumerate(normal_forms):
            ax.text(1.1, i * 0.3, form, fontsize=8, color=colors[i], fontweight='bold')
        
        ax.set_xlim(-1.2, 1.5)
        ax.set_ylim(-0.2, 1.8)
        ax.set_xlabel('Direction v', fontweight='bold')
        ax.set_ylabel('Jet Order', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_spectrum_analysis(self, ax):
        """Plot spectrum analysis and eigenmodes"""
        ax.set_title('Spectrum Analysis: Eigenmodes', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create synthetic spectrum
        n_modes = 20
        eigenvals = np.linspace(1, 0.1, n_modes) + 0.05 * np.random.randn(n_modes)
        eigenvals = np.sort(eigenvals)[::-1]
        
        # Plot spectrum
        modes = np.arange(1, n_modes + 1)
        ax.bar(modes, eigenvals, color=self.colors['primary'], alpha=0.7, label='Eigenvalues')
        
        # Highlight spectral gap
        gap_idx = 3
        ax.bar(modes[:gap_idx], eigenvals[:gap_idx], color=self.colors['accent'], alpha=0.8, label='Spectral Gap')
        
        # Add spectral gap line
        ax.axhline(y=eigenvals[gap_idx], color=self.colors['axis'], linewidth=2, linestyle='--', 
                  label=f'Gap = {eigenvals[0] - eigenvals[gap_idx]:.3f}')
        
        ax.set_xlim(0, n_modes + 1)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('Mode Index', fontweight='bold')
        ax.set_ylabel('Eigenvalue', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_rank_drop_patterns(self, ax):
        """Plot rank drop patterns and geometric structures"""
        ax.set_title('Rank Drop: Geometric Structures', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Create rank drop visualization
        ranks = [4, 3, 2, 1, 0]
        structures = ['Points', 'Curves', 'Surfaces', '3D Manifolds', 'Hyperplanes']
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['kernel'], self.colors['jet']]
        
        for i, (rank, structure) in enumerate(zip(ranks, structures)):
            # Create geometric representation
            if rank == 4:  # Points
                ax.plot(0.5, 0.5, 'o', color=colors[i], markersize=15, label=structure)
            elif rank == 3:  # Curves
                t = np.linspace(0, 2*np.pi, 100)
                x = 0.5 + 0.1 * np.cos(t)
                y = 0.5 + 0.1 * np.sin(t)
                ax.plot(x, y, '-', color=colors[i], linewidth=3, label=structure)
            elif rank == 2:  # Surfaces
                circle = Circle((0.5, 0.5), 0.15, color=colors[i], alpha=0.6, label=structure)
                ax.add_patch(circle)
            elif rank == 1:  # 3D Manifolds
                rect = FancyBboxPatch((0.35, 0.35), 0.3, 0.3, boxstyle="round,pad=0.02", 
                                    color=colors[i], alpha=0.6, label=structure)
                ax.add_patch(rect)
            else:  # Hyperplanes
                ax.axhline(y=0.5, xmin=0.2, xmax=0.8, color=colors[i], linewidth=5, label=structure)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Dimension', fontweight='bold')
        ax.set_ylabel('Geometric Structure', fontweight='bold')
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_aspect('equal')
    
    def _plot_balance_geometry(self, ax):
        """Plot balance geometry overview"""
        ax.set_title('Balance-Geometry Overview', 
                    fontweight='bold', color=self.colors['text'])
        
        # Create balance geometry visualization
        # Central involution
        center = (0.5, 0.5)
        ax.plot(center[0], center[1], 'o', color=self.colors['axis'], markersize=15, label='Involution I')
        
        # Symmetric structures
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for i, angle in enumerate(angles):
            x = center[0] + 0.3 * np.cos(angle)
            y = center[1] + 0.3 * np.sin(angle)
            
            # Plot symmetric points
            ax.plot(x, y, 'o', color=self.colors['primary'], markersize=8, alpha=0.8)
            
            # Connect to center
            ax.plot([center[0], x], [center[1], y], '--', color=self.colors['primary'], alpha=0.5)
            
            # Add reflected point
            x_ref = center[0] + 0.3 * np.cos(angle + np.pi)
            y_ref = center[1] + 0.3 * np.sin(angle + np.pi)
            ax.plot(x_ref, y_ref, 's', color=self.colors['secondary'], markersize=6, alpha=0.8)
            ax.plot([center[0], x_ref], [center[1], y_ref], '--', color=self.colors['secondary'], alpha=0.5)
        
        # Add axis
        ax.axhline(y=0.5, color=self.colors['axis'], linewidth=2, alpha=0.8, label='Axis A')
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=2, alpha=0.8)
        
        # Add title text
        ax.text(0.5, 0.1, 'Equilibrium through\nInvolution Symmetry', 
               ha='center', va='center', fontsize=10, fontweight='bold', 
               color=self.colors['text'], bbox=dict(boxstyle="round,pad=0.3", 
               facecolor=self.colors['background'], alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Balance-Geometry', fontweight='bold')
        ax.set_ylabel('Involution Structure', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    def create_zeta_landscape(self, output_file: str = None) -> str:
        """
        Create a detailed visualization of the zeta landscape with CE1 structure.
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f".out/ce1_visualization/zeta_landscape_{timestamp}.png"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        # 1. Critical line and involution
        self._plot_critical_line(ax1)
        
        # 2. Zeta zeros and CE1 structure
        self._plot_zeta_zeros_ce1(ax2)
        
        # 3. Convolution dressing in zeta context
        self._plot_zeta_convolution(ax3)
        
        # 4. Jet expansion for zeta zeros
        self._plot_zeta_jets(ax4)
        
        fig.suptitle('CE1 Framework: Riemann Zeta Landscape', 
                    fontsize=20, fontweight='bold', color=self.colors['text'])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return output_file
    
    def _plot_critical_line(self, ax):
        """Plot critical line and time reflection involution"""
        ax.set_title('Critical Line: Axis A = {Re(s) = 1/2}', 
                    fontweight='bold', color=self.colors['axis'])
        
        # Critical line
        np.linspace(0, 30, 1000)
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=4, label='Critical Line')
        
        # Time reflection involution
        s_values = np.linspace(0, 1, 100)
        for t_val in [10, 20, 30]:
            ax.plot(s_values, t_val * np.ones_like(s_values), '--', 
                   color=self.colors['primary'], alpha=0.6, linewidth=1)
            ax.plot(1 - s_values, t_val * np.ones_like(s_values), '--', 
                   color=self.colors['secondary'], alpha=0.6, linewidth=1)
        
        # Mark some zeta zeros
        zeta_zeros = [14.134725, 21.02204, 25.010858, 30.424876, 32.935062]
        for t_zero in zeta_zeros:
            ax.plot(0.5, t_zero, 'o', color=self.colors['accent'], markersize=8, 
                   markeredgecolor='white', markeredgewidth=2)
            ax.text(0.52, t_zero, f'ζ₀', fontsize=8, color=self.colors['accent'], fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 35)
        ax.set_xlabel('Re(s)', fontweight='bold')
        ax.set_ylabel('Im(s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    def _plot_zeta_zeros_ce1(self, ax):
        """Plot zeta zeros with CE1 structure overlay"""
        ax.set_title('Zeta Zeros: CE1 Structure', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create complex plane
        sigma = np.linspace(0, 1, 50)
        t = np.linspace(0, 30, 50)
        S, T = np.meshgrid(sigma, t)
        
        # CE1 kernel visualization
        I_s = 1 - S  # Time reflection
        kernel = np.exp(-((S - I_s)**2) / (2 * 0.1**2))
        
        im = ax.imshow(kernel, extent=[0, 1, 0, 30], origin='lower', 
                      cmap='viridis', alpha=0.6)
        
        # Critical line
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=3, label='Critical Line')
        
        # Zeta zeros
        zeta_zeros = [14.134725, 21.02204, 25.010858, 30.424876, 32.935062]
        for t_zero in zeta_zeros:
            ax.plot(0.5, t_zero, 'o', color=self.colors['accent'], markersize=10, 
                   markeredgecolor='white', markeredgewidth=2)
        
        # Add involution lines
        ax.plot([0, 1], [14.134725, 14.134725], '--', color=self.colors['primary'], linewidth=2, alpha=0.8)
        ax.plot([0, 1], [21.02204, 21.02204], '--', color=self.colors['secondary'], linewidth=2, alpha=0.8)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 35)
        ax.set_xlabel('Re(s)', fontweight='bold')
        ax.set_ylabel('Im(s)', fontweight='bold')
        ax.legend(fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_zeta_convolution(self, ax):
        """Plot convolution dressing in zeta context"""
        ax.set_title('Zeta Convolution: Mellin Dressing', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Mellin factor
        s = np.linspace(0.1, 0.9, 100)
        mellin_factor = np.power(np.abs(s - 0.5) + 0.1, -0.5)
        mellin_factor = mellin_factor / np.max(mellin_factor)
        
        ax.plot(s, mellin_factor, '-', color=self.colors['secondary'], linewidth=3, label='Mellin Factor')
        
        # Completed zeta factor
        completed_factor = np.power(np.abs(s - 0.5) + 0.1, -0.3)
        completed_factor = completed_factor / np.max(completed_factor)
        ax.plot(s, completed_factor, '-', color=self.colors['primary'], linewidth=3, label='Completed ζ')
        
        # Critical line
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=3, label='Critical Line')
        
        # Functional equation
        ax.plot(s, 1-s, '--', color=self.colors['accent'], linewidth=2, alpha=0.8, label='I: s ↦ 1-s')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('s', fontweight='bold')
        ax.set_ylabel('Factor Value', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    def _plot_zeta_jets(self, ax):
        """Plot jet expansion for zeta zeros"""
        ax.set_title('Zeta Jets: Order at Zeros', 
                    fontweight='bold', color=self.colors['jet'])
        
        # Zeta zeros with jet orders
        zeta_zeros = [14.134725, 21.02204, 25.010858, 30.424876, 32.935062]
        jet_orders = [5, 5, 5, 5, 5]  # Typical orders for simple zeros
        
        # Plot zeros with jet order visualization
        for i, (t_zero, order) in enumerate(zip(zeta_zeros, jet_orders)):
            # Create jet expansion visualization
            x = np.linspace(-0.1, 0.1, 50)
            y = np.power(x, order) + 0.1 * np.random.randn(50)  # Add noise
            y = y / (np.max(np.abs(y)) + 1e-12)
            
            ax.plot(x + 0.5, y + t_zero, '-', color=self.colors['jet'], linewidth=2, alpha=0.8)
            ax.plot(0.5, t_zero, 'o', color=self.colors['accent'], markersize=8, 
                   markeredgecolor='white', markeredgewidth=2)
            ax.text(0.52, t_zero, f'Order {order}', fontsize=8, color=self.colors['jet'], fontweight='bold')
        
        ax.set_xlim(0.3, 0.7)
        ax.set_ylim(10, 35)
        ax.set_xlabel('Re(s)', fontweight='bold')
        ax.set_ylabel('Im(s)', fontweight='bold')
        ax.grid(True, alpha=0.3)


def main():
    """Main entry point for CE1 visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CE1 Framework Visualization")
    parser.add_argument("--type", choices=["geometry", "zeta", "both"], default="both",
                       help="Type of visualization to create")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = CE1Visualizer()
    
    if args.type in ["geometry", "both"]:
        # Create involution geometry visualization
        geometry_file = visualizer.create_involution_geometry(args.output)
        print(f"Generated involution geometry: {geometry_file}")
    
    if args.type in ["zeta", "both"]:
        # Create zeta landscape visualization
        zeta_file = visualizer.create_zeta_landscape(args.output)
        print(f"Generated zeta landscape: {zeta_file}")
    
    print("CE1 visualization complete!")
    return 0


if __name__ == "__main__":
    exit(main())
