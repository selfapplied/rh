#!/usr/bin/env python3
"""
CE1 Simple Visualization: Focused Involution Geometry

Creates a clean, focused visualization of the CE1 involution geometry
showing the core concept: K(x,y) = δ(y - I·x) and balance-geometry.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np


def create_ce1_involution_diagram(output_file: str = None) -> str:
    """
    Create a clean diagram showing CE1 involution geometry.
    """
    if output_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f".out/ce1_visualization/ce1_involution_{timestamp}.png"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create figure with custom styling
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
    fig.patch.set_facecolor('#FAFAFA')
    
    # Colors
    colors = {
        'primary': '#2E86AB',      # Blue
        'secondary': '#A23B72',    # Purple  
        'accent': '#F18F01',       # Orange
        'axis': '#E74C3C',         # Red
        'kernel': '#27AE60',       # Green
        'text': '#2C3E50'          # Dark blue-gray
    }
    
    # Main title
    ax.text(0.5, 0.95, 'CE1 Framework: Involution Geometry', 
           ha='center', va='center', fontsize=24, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    
    ax.text(0.5, 0.90, 'K(x,y) = δ(y - I·x) generates balance-geometry through involution symmetry', 
           ha='center', va='center', fontsize=14, style='italic', 
           color=colors['text'], transform=ax.transAxes)
    
    # 1. Time Reflection Involution (Riemann ζ)
    ax.text(0.15, 0.80, 'Time Reflection: I: s ↦ 1-s', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['primary'], transform=ax.transAxes)
    
    # Create complex plane for time reflection
    s_values = np.linspace(0, 1, 100)
    reflected = 1 - s_values
    
    # Plot in transformed coordinates
    x_offset = 0.15
    y_offset = 0.65
    scale = 0.08
    
    ax.plot(x_offset + scale * s_values, y_offset + scale * reflected, 
           '--', color=colors['primary'], linewidth=3, alpha=0.8, label='I(s) = 1-s')
    ax.plot(x_offset + scale * s_values, y_offset + scale * s_values, 
           '-', color=colors['axis'], linewidth=3, label='Identity')
    
    # Mark fixed point (critical line)
    ax.plot(x_offset + scale * 0.5, y_offset + scale * 0.5, 
           'o', color=colors['axis'], markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.text(x_offset + scale * 0.5, y_offset + scale * 0.5, 'A', 
           ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Add zeta zero
    ax.plot(x_offset + scale * 0.5, y_offset + scale * 0.5, 
           's', color=colors['accent'], markersize=8, markeredgecolor='white', markeredgewidth=1)
    ax.text(x_offset + scale * 0.5 + 0.01, y_offset + scale * 0.5 + 0.01, 'ζ₀', 
           ha='left', va='bottom', fontsize=8, color=colors['accent'], fontweight='bold')
    
    ax.text(x_offset, y_offset - 0.05, 'Critical Line\nRe(s) = 1/2', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # 2. Momentum Reflection Involution (Dynamical Systems)
    ax.text(0.5, 0.80, 'Momentum Reflection: I: (q,p) ↦ (q,-p)', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['secondary'], transform=ax.transAxes)
    
    # Create phase space
    x_offset = 0.5
    y_offset = 0.65
    scale = 0.08
    
    # Plot momentum reflection
    q = np.linspace(-1, 1, 50)
    p = np.linspace(-1, 1, 50)
    
    # Show reflection
    ax.plot(x_offset + scale * q, y_offset + scale * p, 
           '--', color=colors['secondary'], linewidth=3, alpha=0.8, label='I: p ↦ -p')
    ax.plot(x_offset + scale * q, y_offset + scale * (-p), 
           '--', color=colors['secondary'], linewidth=3, alpha=0.8)
    
    # Axis (p = 0)
    ax.plot(x_offset + scale * q, y_offset + scale * np.zeros_like(q), 
           '-', color=colors['axis'], linewidth=3, label='Axis A: p = 0')
    
    # Critical point
    ax.plot(x_offset, y_offset, 'o', color=colors['axis'], markersize=12, 
           markeredgecolor='white', markeredgewidth=2)
    ax.text(x_offset, y_offset, 'A', ha='center', va='center', 
           fontsize=10, color='white', fontweight='bold')
    
    ax.text(x_offset, y_offset - 0.05, 'Configuration Space\np = 0', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # 3. Microswap Involution (Chemical Systems)
    ax.text(0.85, 0.80, 'Microswap: I: (x,y) ↦ (y,x)', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['kernel'], transform=ax.transAxes)
    
    # Create concentration space
    x_offset = 0.85
    y_offset = 0.65
    scale = 0.08
    
    # Plot microswap
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    
    ax.plot(x_offset + scale * x, y_offset + scale * y, 
           '--', color=colors['kernel'], linewidth=3, alpha=0.8, label='I: (x,y) ↦ (y,x)')
    ax.plot(x_offset + scale * y, y_offset + scale * x, 
           '--', color=colors['kernel'], linewidth=3, alpha=0.8)
    
    # Axis (x = y)
    ax.plot(x_offset + scale * x, y_offset + scale * x, 
           '-', color=colors['axis'], linewidth=3, label='Axis A: x = y')
    
    # Equilibrium point
    ax.plot(x_offset + scale * 0.5, y_offset + scale * 0.5, 
           'o', color=colors['axis'], markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.text(x_offset + scale * 0.5, y_offset + scale * 0.5, 'A', 
           ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    ax.text(x_offset, y_offset - 0.05, 'Log-Toric\nx = y', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # 4. CE1 Kernel Structure
    ax.text(0.5, 0.50, 'CE1 Kernel: K(x,y) = δ(y - I·x)', 
           ha='center', va='center', fontsize=18, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    
    # Create kernel visualization
    x_offset = 0.5
    y_offset = 0.35
    scale = 0.12
    
    # Create grid
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y)
    
    # CE1 kernel (Gaussian approximation)
    I_x = 1 - X  # Time reflection
    kernel = np.exp(-((Y - I_x)**2) / (2 * 0.1**2))
    
    # Plot kernel
    im = ax.imshow(kernel, extent=[x_offset-scale/2, x_offset+scale/2, 
                                  y_offset-scale/2, y_offset+scale/2], 
                  origin='lower', cmap='viridis', alpha=0.8)
    
    # Add diagonal (identity)
    ax.plot([x_offset-scale/2, x_offset+scale/2], [y_offset-scale/2, y_offset+scale/2], 
           '--', color='white', linewidth=2, alpha=0.8)
    
    # Add involution line
    ax.plot([x_offset-scale/2, x_offset+scale/2], [y_offset+scale/2, y_offset-scale/2], 
           '--', color=colors['accent'], linewidth=2, alpha=0.8)
    
    # Fixed point
    ax.plot(x_offset, y_offset, 'o', color=colors['axis'], markersize=8, 
           markeredgecolor='white', markeredgewidth=2)
    
    ax.text(x_offset, y_offset - 0.08, 'Involution Kernel\nSymmetry → Geometry', 
           ha='center', va='top', fontsize=12, color=colors['text'], transform=ax.transAxes)
    
    # 5. Balance Geometry
    ax.text(0.5, 0.20, 'Balance-Geometry: Equilibrium through Involution Symmetry', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    
    # Create balance geometry visualization
    center_x = 0.5
    center_y = 0.10
    radius = 0.06
    
    # Central involution
    ax.plot(center_x, center_y, 'o', color=colors['axis'], markersize=15, 
           markeredgecolor='white', markeredgewidth=2)
    ax.text(center_x, center_y, 'I', ha='center', va='center', 
           fontsize=12, color='white', fontweight='bold')
    
    # Symmetric structures
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for i, angle in enumerate(angles):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Plot symmetric points
        ax.plot(x, y, 'o', color=colors['primary'], markersize=8, alpha=0.8)
        
        # Connect to center
        ax.plot([center_x, x], [center_y, y], '--', color=colors['primary'], alpha=0.6)
        
        # Add reflected point
        x_ref = center_x + radius * np.cos(angle + np.pi)
        y_ref = center_y + radius * np.sin(angle + np.pi)
        ax.plot(x_ref, y_ref, 's', color=colors['secondary'], markersize=6, alpha=0.8)
        ax.plot([center_x, x_ref], [center_y, y_ref], '--', color=colors['secondary'], alpha=0.6)
    
    # Add axis
    ax.axhline(y=center_y, xmin=0.2, xmax=0.8, color=colors['axis'], linewidth=2, alpha=0.8)
    ax.axvline(x=center_x, ymin=0.02, ymax=0.18, color=colors['axis'], linewidth=2, alpha=0.8)
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add footer
    ax.text(0.5, 0.02, 'CE1 Framework v0.1 - Mirror Kernel System for Universal Equilibrium Operators', 
           ha='center', va='center', fontsize=10, style='italic', 
           color=colors['text'], transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
               facecolor='#FAFAFA', edgecolor='none')
    plt.close()
    
    return output_file


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CE1 Simple Visualization")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Create visualization
    output_file = create_ce1_involution_diagram(args.output)
    print(f"Generated CE1 involution diagram: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
