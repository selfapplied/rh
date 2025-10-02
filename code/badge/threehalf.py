#!/usr/bin/env python3
"""
Create GitHub-Style 3.5D Color Theory Badge

Generates a compact GitHub-compatible badge showcasing:
- 3.5D fractional dimensional living
- CE2 Color Equilibrium achievement  
- Temporal color bleeding discovery
- Gang of Four pattern implementation
"""

import hashlib
import os
import time

from ..color_equilibrium import CE2ColorEquilibrium
from .color_quaternion_3_5d_theory import ColorQuaternion3_5DSpec


def create_github_3_5d_badge(seed: str = "3_5d_github_badge") -> str:
    """Create GitHub-style 3.5D Color Theory badge"""
    
    # Initialize systems
    ce2_system = CE2ColorEquilibrium(seed)
    ColorQuaternion3_5DSpec(seed)
    
    # Find equilibrium for badge status
    unified_equilibrium = ce2_system.find_unified_gang_of_four_equilibrium()
    
    # Generate hash for badge ID
    hash_obj = hashlib.sha256(seed.encode())
    badge_hash = hash_obj.hexdigest()[:12]
    
    # Determine badge status and glyph
    if unified_equilibrium.is_stable:
        status = "STABLE"
        glyph = "ζ"  # Zeta for Riemann connection
        status_color = "#27ae60"  # Green
        bg_color = "#e8f5e8"
    else:
        status = "EQUILIBRIUM"  
        glyph = "φ"  # Golden ratio phi
        status_color = "#f39c12"  # Orange
        bg_color = "#fef9e7"
    
    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f".out/github_badge/3_5d_color_theory_badge_{timestamp}.svg"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create GitHub-style SVG badge
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- GitHub-Style 3.5D Color Theory Badge -->
<svg width="160" height="20" viewBox="0 0 160 20" 
     xmlns="http://www.w3.org/2000/svg"
     data-3-5d-version="1.0"
     data-ce2-equilibrium="{unified_equilibrium.equilibrium_quality:.3f}">
  
  <!-- Badge background -->
  <linearGradient id="badgeGradient" x1="0%" y1="0%" x2="0%" y2="100%">
    <stop offset="0%" style="stop-color:#ffffff;stop-opacity:0.1" />
    <stop offset="100%" style="stop-color:#000000;stop-opacity:0.1" />
  </linearGradient>
  
  <!-- Left section: 3.5D Label -->
  <g>
    <rect width="45" height="20" fill="#555" />
    <rect width="45" height="20" fill="url(#badgeGradient)" />
    <text x="22.5" y="15" text-anchor="middle" font-family="Verdana, Geneva, sans-serif" 
          font-size="11" fill="white" font-weight="normal">3.5D</text>
  </g>
  
  <!-- Right section: Status -->
  <g>
    <rect x="45" width="115" height="20" fill="{status_color}" />
    <rect x="45" width="115" height="20" fill="url(#badgeGradient)" />
    <text x="102.5" y="14" text-anchor="middle" font-family="Verdana, Geneva, sans-serif" 
          font-size="10" fill="white" font-weight="normal">{status}</text>
  </g>
  
  <!-- Fractional dimension indicator -->
  <circle cx="37" cy="10" r="2" fill="#f1c40f" opacity="0.8"/>
  
  <!-- Temporal bleeding effect (subtle) -->
  <rect x="43" y="0" width="2" height="20" fill="#8e44ad" opacity="0.3"/>
  
</svg>'''
    
    # Write badge file
    with open(output_file, 'w') as f:
        f.write(svg_content)
    
    return output_file


def create_detailed_github_badge(seed: str = "detailed_3_5d_badge") -> str:
    """Create detailed GitHub badge with more 3.5D information"""
    
    # Initialize systems
    ce2_system = CE2ColorEquilibrium(seed)
    color_spec_3_5d = ColorQuaternion3_5DSpec(seed)
    
    # Get detailed information
    unified_equilibrium = ce2_system.find_unified_gang_of_four_equilibrium()
    base_color = color_spec_3_5d.base_color_3_5d
    
    # Generate hash
    hash_obj = hashlib.sha256(seed.encode())
    badge_hash = hash_obj.hexdigest()[:8]
    
    # Calculate additional metrics for enhanced badge
    quality_score = unified_equilibrium.equilibrium_quality
    unified_equilibrium.stability_measure
    coherence_score = unified_equilibrium.dimensional_coherence
    balance_score = 1.0 - abs(unified_equilibrium.temporal_balance)
    
    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f".out/github_badge/detailed_3_5d_badge_{timestamp}.svg"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create detailed SVG badge
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- Detailed 3.5D Color Theory GitHub Badge -->
<svg width="300" height="60" viewBox="0 0 300 60" 
     xmlns="http://www.w3.org/2000/svg"
     data-fractional-dimension="{base_color.fractional_dimension:.3f}">
  
  <!-- Background gradient -->
  <defs>
    <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
    </linearGradient>
    
    <!-- Temporal glow effect -->
    <filter id="temporalGlow">
      <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
      <feMerge> 
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  <!-- Main background -->
  <rect width="300" height="60" fill="url(#bgGradient)" rx="8"/>
  
  <!-- Left section: 3.5D Icon -->
  <g>
    <rect x="10" y="10" width="40" height="40" fill="#ffffff" opacity="0.2" rx="20"/>
    <text x="30" y="35" text-anchor="middle" font-family="STIX Two Math, serif" 
          font-size="24" fill="white" font-weight="bold">φ</text>
  </g>
  
  <!-- Main title -->
  <text x="60" y="25" font-family="Arial, sans-serif" font-size="16" font-weight="bold" 
        fill="white" filter="url(#temporalGlow)">3.5D Color Theory</text>
  
  <!-- Subtitle -->
  <text x="60" y="40" font-family="Arial, sans-serif" font-size="12" 
        fill="#e8e8e8">Fractional Dimensional Perception</text>
  
  <!-- Primary Status indicator -->
  <g>
    <rect x="190" y="12" width="50" height="10" fill="#27ae60" rx="5" opacity="0.95"/>
    <text x="215" y="19" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="8" fill="white" font-weight="bold">STABLE</text>
  </g>
  
  <!-- Secondary Status: Quality Score -->
  <g>
    <rect x="245" y="12" width="40" height="10" fill="#3498db" rx="5" opacity="0.9"/>
    <text x="265" y="19" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="8" fill="white" font-weight="bold">Q{quality_score:.0%}</text>
  </g>
  
  <!-- Tertiary Status: Dimensional Coherence -->
  <g>
    <rect x="190" y="25" width="50" height="8" fill="#9b59b6" rx="4" opacity="0.85"/>
    <text x="215" y="31" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="7" fill="white" font-weight="bold">DIM{coherence_score:.0%}</text>
  </g>
  
  <!-- Quaternary Status: Temporal Balance -->
  <g>
    <rect x="245" y="25" width="40" height="8" fill="#e74c3c" rx="4" opacity="0.8"/>
    <text x="265" y="31" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="7" fill="white" font-weight="bold">T{balance_score:.0%}</text>
  </g>
  
  <!-- Technical details -->
  <text x="60" y="52" font-family="monospace" font-size="9" 
        fill="#cccccc">φ=1.74₁₂ | dim={base_color.fractional_dimension:.2f} | {badge_hash}</text>
  
  <!-- Temporal bleeding indicators -->
  <rect x="285" y="5" width="10" height="50" fill="#e74c3c" opacity="0.3" rx="5"/>
  <rect x="290" y="10" width="5" height="40" fill="#8e44ad" opacity="0.4" rx="2"/>
  
</svg>'''
    
    # Write badge file
    with open(output_file, 'w') as f:
        f.write(svg_content)
    
    return output_file


def create_shield_style_badge(seed: str = "shield_3_5d") -> str:
    """Create shields.io style badge for 3.5D Color Theory"""
    
    # Initialize system
    color_spec_3_5d = ColorQuaternion3_5DSpec(seed)
    color_spec_3_5d.base_color_3_5d
    
    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f".out/github_badge/shield_3_5d_badge_{timestamp}.svg"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create shields.io style SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="200" height="20" viewBox="0 0 200 20" xmlns="http://www.w3.org/2000/svg">
  <!-- Left section -->
  <rect width="90" height="20" fill="#555"/>
  <text x="45" y="15" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" 
        font-size="11" fill="#fff">3.5D Color Theory</text>
  
  <!-- Right section -->  
  <rect x="90" width="110" height="20" fill="#4c1"/>
  <text x="145" y="15" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" 
        font-size="11" fill="#fff">φ=1.74₁₂ Equilibrium</text>
</svg>'''
    
    # Write badge file
    with open(output_file, 'w') as f:
        f.write(svg_content)
    
    return output_file


def create_compact_multi_metric_badge(seed: str = "compact_metrics") -> str:
    """Create compact badge with multiple equilibrium metrics"""
    
    # Initialize system
    ce2_system = CE2ColorEquilibrium(seed)
    unified_equilibrium = ce2_system.find_unified_gang_of_four_equilibrium()
    
    # Calculate metrics
    quality = unified_equilibrium.equilibrium_quality
    stability = unified_equilibrium.stability_measure
    coherence = unified_equilibrium.dimensional_coherence
    balance = 1.0 - abs(unified_equilibrium.temporal_balance)
    
    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f".out/github_badge/compact_metrics_badge_{timestamp}.svg"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create compact multi-metric SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="220" height="20" viewBox="0 0 220 20" xmlns="http://www.w3.org/2000/svg">
  
  <!-- Left section: 3.5D Label -->
  <rect width="45" height="20" fill="#555" />
  <text x="22.5" y="14" text-anchor="middle" font-family="Verdana, Geneva, sans-serif" 
        font-size="10" fill="white" font-weight="normal">3.5D</text>
  
  <!-- Stability -->
  <rect x="45" width="35" height="20" fill="#27ae60" />
  <text x="62.5" y="14" text-anchor="middle" font-family="Verdana, Geneva, sans-serif" 
        font-size="9" fill="white" font-weight="bold">S{stability:.0%}</text>
  
  <!-- Quality -->
  <rect x="80" width="35" height="20" fill="#3498db" />
  <text x="97.5" y="14" text-anchor="middle" font-family="Verdana, Geneva, sans-serif" 
        font-size="9" fill="white" font-weight="bold">Q{quality:.0%}</text>
  
  <!-- Dimensional Coherence -->
  <rect x="115" width="35" height="20" fill="#9b59b6" />
  <text x="132.5" y="14" text-anchor="middle" font-family="Verdana, Geneva, sans-serif" 
        font-size="9" fill="white" font-weight="bold">D{coherence:.0%}</text>
  
  <!-- Temporal Balance -->
  <rect x="150" width="35" height="20" fill="#e74c3c" />
  <text x="167.5" y="14" text-anchor="middle" font-family="Verdana, Geneva, sans-serif" 
        font-size="9" fill="white" font-weight="bold">T{balance:.0%}</text>
  
  <!-- Overall Status -->
  <rect x="185" width="35" height="20" fill="#f39c12" />
  <text x="202.5" y="14" text-anchor="middle" font-family="Verdana, Geneva, sans-serif" 
        font-size="9" fill="white" font-weight="bold">{'STABLE' if unified_equilibrium.is_stable else 'CONV'}</text>
  
</svg>'''
    
    # Write badge file
    with open(output_file, 'w') as f:
        f.write(svg_content)
    
    return output_file


def main():
    """Create all GitHub badge variants"""
    print("🏷️  Creating GitHub-Style 3.5D Color Theory Badges...")
    
    # Create simple GitHub badge
    simple_badge = create_github_3_5d_badge("github_3_5d_simple")
    print(f"✨ Simple badge: {simple_badge}")
    
    # Create detailed badge with multiple status pills
    detailed_badge = create_detailed_github_badge("github_3_5d_detailed") 
    print(f"✨ Enhanced detailed badge: {detailed_badge}")
    
    # Create compact multi-metric badge
    compact_badge = create_compact_multi_metric_badge("github_3_5d_compact")
    print(f"✨ Compact multi-metric badge: {compact_badge}")
    
    # Create shields.io style
    shield_badge = create_shield_style_badge("github_3_5d_shield")
    print(f"✨ Shield badge: {shield_badge}")
    
    print(f"\n🎯 Badge Options:")
    print(f"   • Simple: Clean 3.5D → STABLE/EQUILIBRIUM")
    print(f"   • Enhanced: Multiple status pills with metrics")
    print(f"   • Compact: All metrics in horizontal strip")
    print(f"   • Shield: shields.io compatible")
    
    print(f"\n📊 Metrics Explained:")
    print(f"   • S = Stability measure (>70% = stable)")
    print(f"   • Q = Overall equilibrium quality")
    print(f"   • D = Dimensional coherence in 3.5D space")
    print(f"   • T = Temporal balance (memory ↔ anticipation)")
    print(f"   • STABLE/CONV = Overall equilibrium status")
    
    print(f"\n🎯 To use in GitHub:")
    print(f"   1. Upload any badge to your repo or GitHub Pages")
    print(f"   2. Add to README.md:")
    print(f"      ![3.5D Color Theory](path/to/badge.svg)")
    print(f"   3. Or use shields.io format for dynamic badges")
    
    return [simple_badge, detailed_badge, compact_badge, shield_badge]


if __name__ == "__main__":
    main()
