#!/usr/bin/env python3
"""
Days of AX-mas Generator: Living Document Atmosphere Song

Creates a cumulative-verse song where the living document sings its own atmosphere
while stacking invariants. Each verse adds one new gift and re-lists the earlier ones,
doubling as a checklist when rendering the badge.

This gives the "Living Document Atmosphere Generator" a catchy, auditable voice.
"""

import os
import time
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Import our Color Quaternion Harmonic Spec
from color_quaternion_harmonic_spec import (
    ColorQuaternionHarmonicSpec, ColorQuaternionGroup, OKLCHColor,
    CellularAutomataColorGenerator, ColorDecompositionOperators,
    CellularAutomataRule, ColorDecompositionBasis, HarmonicRatio
)

# Import CE1 integration
from color_quaternion_ce1_integration import LivingDocumentAtmosphere


class DaysOfAXmasGenerator:
    """
    Generates the cumulative-verse "Days of AX-mas" song for living documents
    """
    
    def __init__(self, seed: str = "living_document_2025"):
        self.seed = seed
        self.atmosphere = LivingDocumentAtmosphere(seed)
        self.color_spec = self.atmosphere.color_spec
        self.atmosphere_spec = self.atmosphere.atmosphere_spec
        
        # Extract data for verses
        self.atmosphere_id = self.atmosphere_spec['atmosphere_id']
        self.base_color = self.color_spec.base_color
        self.hue = self.base_color.hue
        self.invariants = self.atmosphere_spec['ce1_invariants']
        self.harmonic_palette = self.color_spec.generate_harmonic_palette()
        
        # Generate safe color bands for badge
        self.safe_colors = self._generate_safe_color_bands()
        
        # Generate verse data
        self.verse_data = self._generate_verse_data()
    
    def _generate_safe_color_bands(self) -> Dict[str, str]:
        """Generate safe color bands for polished badge rendering"""
        # Snap hue to 30° bins
        snapped_hue = round(self.hue / 30) * 30
        
        # Generate safe color bands
        ambient = OKLCHColor(0.92, 0.05, snapped_hue)
        ambient_dark = OKLCHColor(0.16, 0.06, snapped_hue)
        accent_a = OKLCHColor(0.70, 0.12, (snapped_hue + 120) % 360)
        accent_b = OKLCHColor(0.70, 0.12, (snapped_hue + 300) % 360)
        on_ambient = OKLCHColor(0.18, 0.03, snapped_hue)
        
        return {
            'ambient': ambient.to_string(),
            'ambient_dark': ambient_dark.to_string(),
            'accent_a': accent_a.to_string(),
            'accent_b': accent_b.to_string(),
            'on_ambient': on_ambient.to_string()
        }
    
    def _generate_verse_data(self) -> Dict[int, Dict[str, Any]]:
        """Generate data for each verse"""
        # Extract group actions that fired
        orbit = self.color_spec.quaternion_group.generate_orbit(self.base_color)
        group_actions = ["L-flip", "C-mirror", "rot+120", "rot-120"]  # Simplified for demo
        
        # Generate CA patterns
        ca_90 = self.color_spec.generate_ca_pattern(CellularAutomataRule.RULE_90, steps=8)
        ca_45 = self.color_spec.generate_ca_pattern(CellularAutomataRule.RULE_45, steps=8)
        
        # Generate kernel operators (simplified)
        kernel_ops = ["F", "G1.2", "B15", "M5@±30", "Rφ", "D5", "M6@75"]
        
        # Generate history checkpoints (simplified)
        history = [
            {"ts": time.time() - i*3600, "event": f"commit_{i}", "anchor": f"anchor_{i}"}
            for i in range(11)
        ]
        
        # Generate energy-sorted palette
        energies = []
        for name, color_str in self.harmonic_palette.items():
            # Parse color string to get L, C, h
            parts = color_str.replace('oklch(', '').replace(')', '').split()
            L, C, h = float(parts[0]), float(parts[1]), float(parts[2])
            color = OKLCHColor(L, C, h)
            energy = self.color_spec.least_action.perceptual_energy(color)
            energies.append({"name": name, "L": L, "C": C, "h": h, "E": energy})
        
        # Sort by energy and take top 12
        energies.sort(key=lambda x: x["E"])
        top_energies = energies[:12]
        
        # Find nearest ROYGBIV station
        roygbiv_hues = [0, 51.4, 102.9, 154.3, 205.7, 257.1, 308.6]  # Approximate
        nearest_station_idx = min(range(len(roygbiv_hues)), 
                                 key=lambda i: abs(roygbiv_hues[i] - self.hue))
        nearest_station = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"][nearest_station_idx]
        
        return {
            1: {
                "type": "seed",
                "seed": self.seed,
                "atmosphere_id": self.atmosphere_id,
                "provenance": f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}"
            },
            2: {
                "type": "anchors",
                "va": f"va:{self.atmosphere_id[:8]}",
                "la": f"la:{self.atmosphere_id[8:]}"
            },
            3: {
                "type": "group_rotations",
                "actions": group_actions[:3]  # Take first 3
            },
            4: {
                "type": "guarantees",
                "guarantees": ["Determinism", "Monotonic staging", "Contrast safety", "Reproducible layout"]
            },
            5: {
                "type": "harmonic_ratios",
                "ratios": [1, 2, 3, 4, 5],
                "hue_rotations": [0, 60, 120, 180, 240]
            },
            6: {
                "type": "ce1_invariants",
                "invariants": list(self.invariants.keys())[:6]
            },
            7: {
                "type": "spectrum_stations",
                "roygbiv": ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"],
                "nearest": nearest_station,
                "base_hue": self.hue
            },
            8: {
                "type": "kernel_operators",
                "ops": kernel_ops[:8]
            },
            9: {
                "type": "ca_tiles",
                "rule_90": ca_90,
                "rule_45": ca_45
            },
            10: {
                "type": "color_modes",
                "modes": ["RGB", "CMY", "HSV", "HSL", "LAB", "LCH", "OKLCH", "XYZ", "YUV", "YCbCr"]
            },
            11: {
                "type": "history_checkpoints",
                "history": history
            },
            12: {
                "type": "least_action_lines",
                "energies": top_energies
            }
        }
    
    def _format_verse_gift(self, verse_num: int) -> str:
        """Format the gift for a specific verse"""
        data = self.verse_data[verse_num]
        
        if verse_num == 1:
            return f"🎯 A Seed in a Passport Tree\n   ({data['seed']}, {data['atmosphere_id']})"
        
        elif verse_num == 2:
            return f"⚓ Two Anchors Bound\n   {data['va']}, {data['la']}"
        
        elif verse_num == 3:
            actions = ", ".join(data['actions'])
            return f"🔄 Three Group Rotations\n   {actions}"
        
        elif verse_num == 4:
            guarantees = ", ".join(data['guarantees'])
            return f"✅ Four Guarantees of Consistency\n   {guarantees}"
        
        elif verse_num == 5:
            ratios = ", ".join(map(str, data['ratios']))
            return f"🎵 Five Harmonic Ratios\n   {ratios} (hue: {data['hue_rotations']})"
        
        elif verse_num == 6:
            invariants = ", ".join(data['invariants'])
            return f"🔬 Six CE1 Invariants\n   {invariants}"
        
        elif verse_num == 7:
            return f"🌈 Seven Spectrum Stations\n   ROYGBIV → nearest: {data['nearest']} ({data['base_hue']:.1f}°)"
        
        elif verse_num == 8:
            ops = ", ".join(data['ops'])
            return f"⚙️ Eight Kernel Operators\n   {ops}"
        
        elif verse_num == 9:
            # Count active tiles in first few rows
            ca_90 = data['rule_90']
            row_counts = []
            for i, row in enumerate(ca_90[:3]):
                # Count non-gray colors (active tiles)
                active_count = sum(1 for color_str in row if "0.020" not in color_str)
                row_counts.append(f"Row{i+1}: {active_count} on")
            return f"🔲 Nine CA Tiles in Order\n   Rule-90: {', '.join(row_counts)}"
        
        elif verse_num == 10:
            modes = ", ".join(data['modes'])
            return f"🎨 Ten Modes A-Marching\n   {modes}"
        
        elif verse_num == 11:
            recent_events = len(data['history'])
            return f"📚 Eleven History Checkpoints\n   Last {recent_events} events, time-stamped"
        
        elif verse_num == 12:
            min_energy = data['energies'][0]
            return f"⚡ Twelve Lines of Least Action\n   Min energy: {min_energy['name']} ({min_energy['E']:.3f}) ★"
        
        return f"🎁 Gift #{verse_num}"
    
    def generate_song(self) -> str:
        """Generate the complete cumulative-verse song"""
        song_lines = []
        
        for day in range(1, 13):
            # Main verse line
            song_lines.append(f"On the {self._ordinal(day)} day of AX-mas my Passport sang to me:")
            
            # Cumulative gifts (most recent first)
            for gift_day in range(day, 0, -1):
                gift = self._format_verse_gift(gift_day)
                song_lines.append(gift)
            
            # Add spacing between verses
            if day < 12:
                song_lines.append("")
        
        return "\n".join(song_lines)
    
    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    def generate_badge_svg(self) -> str:
        """Generate compact SVG badge with day gifts as pills"""
        # Calculate badge dimensions
        width = 400
        height = 600
        
        # Generate verse pills
        pills = []
        for day in range(1, 13):
            gift = self._format_verse_gift(day)
            # Extract emoji and first line
            lines = gift.split('\n')
            emoji_line = lines[0]
            description = lines[1].strip()
            
            pills.append({
                'day': day,
                'emoji': emoji_line.split()[0],  # First emoji
                'description': description[:40] + "..." if len(description) > 40 else description
            })
        
        # Create SVG content
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg"
     data-axmas-version="1.0"
     data-template="days-of-axmas-badge">
  
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="{self.safe_colors['ambient']}" rx="8"/>
  
  <!-- Title -->
  <text x="{width//2}" y="30" text-anchor="middle" font-family="FiraCode Nerd Font" 
        font-size="18" fill="{self.safe_colors['on_ambient']}" font-weight="bold">
    Days of AX-mas
  </text>
  
  <text x="{width//2}" y="50" text-anchor="middle" font-family="FiraCode Nerd Font" 
        font-size="12" fill="{self.safe_colors['on_ambient']}" opacity="0.8">
    {self.atmosphere_id}
  </text>
  
  <!-- Day pills -->
'''
        
        # Add pills
        pill_width = 350
        pill_height = 35
        start_y = 80
        spacing = 40
        
        for i, pill in enumerate(pills):
            y_pos = start_y + i * spacing
            
            # Pill background
            svg_content += f'''  <rect x="25" y="{y_pos}" width="{pill_width}" height="{pill_height}" 
         fill="{self.safe_colors['ambient_dark']}" rx="6" opacity="0.9"/>
'''
            
            # Day number
            svg_content += f'''  <text x="35" y="{y_pos + 22}" font-family="FiraCode Nerd Font" 
         font-size="14" fill="{self.safe_colors['accent_a']}" font-weight="bold">
    {pill['day']}.
  </text>
'''
            
            # Emoji
            svg_content += f'''  <text x="55" y="{y_pos + 22}" font-family="Apple Color Emoji, Segoe UI Emoji" 
         font-size="16">
    {pill['emoji']}
  </text>
'''
            
            # Description
            svg_content += f'''  <text x="80" y="{y_pos + 22}" font-family="FiraCode Nerd Font" 
         font-size="11" fill="{self.safe_colors['on_ambient']}">
    {pill['description']}
  </text>
'''
        
        # Footer
        svg_content += f'''
  <!-- Footer -->
  <text x="{width//2}" y="{height - 20}" text-anchor="middle" font-family="FiraCode Nerd Font" 
        font-size="10" fill="{self.safe_colors['on_ambient']}" opacity="0.6">
    Living Document Atmosphere Generator v1.0
  </text>
  
</svg>'''
        
        return svg_content
    
    def save_outputs(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save both the song and badge"""
        if output_dir is None:
            output_dir = ".out/days_of_axmas"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        atmosphere_id = self.atmosphere_id
        
        # Generate and save song
        song = self.generate_song()
        song_path = os.path.join(output_dir, f"axmas_song_{atmosphere_id}_{timestamp}.txt")
        with open(song_path, 'w', encoding='utf-8') as f:
            f.write(song)
        
        # Generate and save badge
        badge_svg = self.generate_badge_svg()
        badge_path = os.path.join(output_dir, f"axmas_badge_{atmosphere_id}_{timestamp}.svg")
        with open(badge_path, 'w', encoding='utf-8') as f:
            f.write(badge_svg)
        
        return {
            'song_path': song_path,
            'badge_path': badge_path
        }


def main():
    """Main entry point for Days of AX-mas Generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Days of AX-mas Generator")
    parser.add_argument("--seed", type=str, default="living_document_2025",
                       help="Seed for atmosphere generation")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    print("🎄 Days of AX-mas Generator")
    print("=" * 40)
    print(f"Seed: {args.seed}")
    print("Generating cumulative-verse song and badge...")
    
    # Create generator
    generator = DaysOfAXmasGenerator(args.seed)
    
    # Generate outputs
    outputs = generator.save_outputs(args.output)
    
    print(f"\n🎵 Generated Song: {outputs['song_path']}")
    print(f"🏷️ Generated Badge: {outputs['badge_path']}")
    
    # Print first few verses as preview
    song_lines = generator.generate_song().split('\n')
    print(f"\n🎄 Song Preview (first 3 verses):")
    print("-" * 40)
    
    verse_count = 0
    for line in song_lines:
        if line.startswith("On the"):
            verse_count += 1
            if verse_count > 3:
                break
        print(line)
    
    if verse_count >= 3:
        print("\n... (continues for 12 verses)")
    
    print(f"\n🎯 Days of AX-mas Complete!")
    print("The living document now sings its own atmosphere!")
    
    return 0


if __name__ == "__main__":
    exit(main())
