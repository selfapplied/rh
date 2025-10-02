#!/usr/bin/env python3
"""
Create CE2 Color Equilibrium Badge

Generates a beautiful SVG badge showcasing CE2 Color Equilibria in 3.5D space
with temporal color bleeding, fractional quaternion rotations, and dimensional resonance.
"""

import os
import time

from ..color_equilibrium import (
    CE2ColorEquilibrium,
)
from .color_quaternion_3_5d_theory import OKLCH3_5D


def oklch_3_5d_to_rgb(color: OKLCH3_5D) -> str:
    """Convert OKLCH3_5D to RGB hex string for SVG"""
    L, C, h = color.lightness, color.chroma, color.hue
    
    # Simple OKLCH to RGB approximation
    import math
    h_rad = math.radians(h)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    
    # Apply temporal effects
    temporal_factor = 1.0 + 0.1 * (color.temporal.memory_strength - color.temporal.anticipation_strength)
    
    # Convert to RGB (simplified)
    r = max(0, min(1, L + 0.4 * a)) * temporal_factor
    g = max(0, min(1, L - 0.2 * a - 0.3 * b)) * temporal_factor
    b_val = max(0, min(1, L - 0.6 * b)) * temporal_factor
    
    # Clamp and convert to hex
    r = max(0, min(255, int(r * 255)))
    g = max(0, min(255, int(g * 255)))
    b_val = max(0, min(255, int(b_val * 255)))
    
    return f"#{r:02x}{g:02x}{b_val:02x}"


def create_ce2_equilibrium_badge(seed: str = "ce2_badge_demo") -> str:
    """Create CE2 Color Equilibrium Badge SVG"""
    
    # Initialize CE2 system
    ce2_system = CE2ColorEquilibrium(seed)
    
    # Find unified equilibrium
    unified_equilibrium = ce2_system.find_unified_gang_of_four_equilibrium()
    
    # Generate palette for visualization
    palette = ce2_system.color_spec_3_5d.generate_3_5d_harmonic_palette(5)
    
    # Create timestamp for filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f".out/ce2_badge/ce2_equilibrium_badge_{timestamp}.svg"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # SVG dimensions
    width = 800
    height = 600
    
    # Create SVG content
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg">
  
  <!-- Background gradient -->
  <defs>
    <linearGradient id="backgroundGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0f0f23;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#1a1a3a;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2a2a5a;stop-opacity:1" />
    </linearGradient>
    
    <!-- 3.5D dimensional glow -->
    <filter id="dimensionalGlow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge> 
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    
    <!-- Temporal bleeding effect -->
    <filter id="temporalBleeding">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feOffset in="blur" dx="2" dy="1" result="offset"/>
      <feMerge>
        <feMergeNode in="offset"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="url(#backgroundGradient)"/>
  
  <!-- Main title -->
  <text x="{width//2}" y="60" text-anchor="middle" 
        font-family="Arial, sans-serif" font-size="32" font-weight="bold" 
        fill="#ffffff" filter="url(#dimensionalGlow)">
    CE2: Color Equilibria
  </text>
  
  <!-- Subtitle -->
  <text x="{width//2}" y="90" text-anchor="middle" 
        font-family="Arial, sans-serif" font-size="16" font-style="italic" 
        fill="#cccccc">
    3.5D Fractional Dimensional Space
  </text>
  
  <!-- Golden ratio base 12 -->
  <text x="{width//2}" y="115" text-anchor="middle" 
        font-family="monospace" font-size="14" 
        fill="#f1c40f">
    φ = 1.74BB6772₁₂ (base 12)
  </text>
'''
    
    # Add equilibrium color palette
    palette_y = 150
    palette_width = 600
    palette_start_x = (width - palette_width) // 2
    color_width = palette_width // len(palette)
    
    svg_content += '\n  <!-- 3.5D Color Palette -->\n'
    
    for i, color in enumerate(palette):
        color_hex = oklch_3_5d_to_rgb(color)
        x = palette_start_x + i * color_width
        
        # Main color rectangle
        svg_content += f'''  <rect x="{x}" y="{palette_y}" width="{color_width}" height="60" 
        fill="{color_hex}" stroke="#ffffff" stroke-width="2" 
        filter="url(#temporalBleeding)"/>\n'''
        
        # Fractional dimension label
        svg_content += f'''  <text x="{x + color_width//2}" y="{palette_y + 35}" text-anchor="middle" 
        font-family="monospace" font-size="10" font-weight="bold" 
        fill="#ffffff">{color.fractional_dimension:.2f}D</text>\n'''
        
        # Temporal indicators
        memory_height = int(color.temporal.memory_strength * 20)
        anticipation_height = int(color.temporal.anticipation_strength * 20)
        
        # Memory bar (bottom)
        if memory_height > 0:
            svg_content += f'''  <rect x="{x + 5}" y="{palette_y + 60}" width="{color_width - 10}" height="{memory_height}" 
            fill="#e74c3c" opacity="0.7"/>\n'''
        
        # Anticipation bar (top)
        if anticipation_height > 0:
            svg_content += f'''  <rect x="{x + 5}" y="{palette_y - anticipation_height}" width="{color_width - 10}" height="{anticipation_height}" 
            fill="#8e44ad" opacity="0.7"/>\n'''
    
    # Add equilibrium information
    eq_y = 280
    svg_content += f'''
  <!-- Equilibrium Information -->
  <text x="50" y="{eq_y}" font-family="Arial, sans-serif" font-size="18" font-weight="bold" 
        fill="#27ae60">Unified Equilibrium:</text>
  
  <text x="70" y="{eq_y + 25}" font-family="monospace" font-size="12" 
        fill="#ffffff">Color: {unified_equilibrium.color_state.to_string()}</text>
  
  <text x="70" y="{eq_y + 45}" font-family="monospace" font-size="12" 
        fill="#ffffff">Energy: {unified_equilibrium.equilibrium_energy:.2f}</text>
  
  <text x="70" y="{eq_y + 65}" font-family="monospace" font-size="12" 
        fill="#ffffff">Stability: {unified_equilibrium.stability_measure:.3f}</text>
  
  <text x="70" y="{eq_y + 85}" font-family="monospace" font-size="12" 
        fill="#ffffff">Quality: {unified_equilibrium.equilibrium_quality:.3f}</text>
  
  <text x="70" y="{eq_y + 105}" font-family="monospace" font-size="12" 
        fill="{('#27ae60' if unified_equilibrium.is_stable else '#e74c3c')}">
        Status: {'STABLE' if unified_equilibrium.is_stable else 'CONVERGING'}</text>
'''
    
    # Add Gang of Four equilibrium types
    equilibrium_types = [
        ("Temporal", "Memory ↔ Anticipation", "#e74c3c"),
        ("Quaternion", "Fractional Rotations", "#3498db"), 
        ("Resonance", "Dimensional Harmony", "#8e44ad"),
        ("Statistical", "Chi-Squared 3.5D", "#27ae60")
    ]
    
    svg_content += '\n  <!-- Gang of Four Equilibrium Types -->\n'
    
    for i, (eq_type, description, color) in enumerate(equilibrium_types):
        x = 450 + (i % 2) * 150
        y = 300 + (i // 2) * 80
        
        # Equilibrium circle
        svg_content += f'''  <circle cx="{x}" cy="{y}" r="25" fill="{color}" opacity="0.7" 
        filter="url(#dimensionalGlow)"/>\n'''
        
        # Type label
        svg_content += f'''  <text x="{x}" y="{y - 5}" text-anchor="middle" 
        font-family="Arial, sans-serif" font-size="10" font-weight="bold" 
        fill="#ffffff">{eq_type}</text>\n'''
        
        # Description
        svg_content += f'''  <text x="{x}" y="{y + 40}" text-anchor="middle" 
        font-family="Arial, sans-serif" font-size="8" 
        fill="#cccccc">{description}</text>\n'''
    
    # Add mathematical framework
    svg_content += f'''
  <!-- Mathematical Framework -->
  <text x="50" y="480" font-family="Arial, sans-serif" font-size="16" font-weight="bold" 
        fill="#f1c40f">3.5D Mathematical Framework:</text>
  
  <text x="70" y="505" font-family="monospace" font-size="11" 
        fill="#ffffff">Color₃.₅ᴅ = OKLCH + 0.5D_temporal</text>
  
  <text x="70" y="525" font-family="monospace" font-size="11" 
        fill="#ffffff">q₃.₅ᴅ = cos(θ/2) + sin(θ/2)(0.5·i + j + k)</text>
  
  <text x="70" y="545" font-family="monospace" font-size="11" 
        fill="#ffffff">Harmony₃.₅ᴅ = φ^(3.5) · χ²₃.₅(color_distribution)</text>
  
  <text x="70" y="565" font-family="monospace" font-size="11" 
        fill="#ffffff">dE/dt = 0 (equilibrium condition)</text>
  
  <!-- Seed and timestamp -->
  <text x="{width - 20}" y="{height - 20}" text-anchor="end" 
        font-family="monospace" font-size="10" 
        fill="#666666">Seed: {seed} | {timestamp}</text>
</svg>'''
    
    # Write SVG file
    with open(output_file, 'w') as f:
        f.write(svg_content)
    
    return output_file


def main():
    """Create and display CE2 badge"""
    print("🌈 Creating CE2 Color Equilibrium Badge...")
    
    # Create badge
    badge_path = create_ce2_equilibrium_badge("ce2_fractional_dimensional_living")
    
    print(f"✨ CE2 Badge created: {badge_path}")
    print("\n🎯 To view the badge:")
    print(f"   open {badge_path}")
    print("   OR drag the file to your browser")
    print("   OR use VS Code: code {badge_path}")
    
    return badge_path


if __name__ == "__main__":
    main()
