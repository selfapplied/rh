#!/usr/bin/env python3
"""
Create Axiel Badge: Dense GitHub-style badge with dynamic glyph and actual hash.

Creates tight, compact badges like GitHub status badges:
- Left: Dynamic glyph based on passing invariants (8/8 = ζ, 7/8 = ∇, etc.)
- Right: "Axiel Passport" + actual computed hash
- Dense, clean, minimal
"""

import argparse
import hashlib
import os
import time

from tools.unlock import ProofUnlockStamper, create_proof_unlock_params


def get_badge_glyph(passed_count: int, total_count: int) -> str:
    """Get badge glyph based on passing invariants count."""
    
    # Glyph progression based on mathematical completion
    glyph_map = {
        8: "ζ",  # Complete (zeta function)
        7: "∇",  # Near complete (gradient)
        6: "≈",  # Mostly (approximation)
        5: "⊕",  # Partial (direct sum)
        4: "∫",  # Foundation (integration)
        3: "∂",  # Basic (partial derivative)
        2: "⌊⌋", # Minimal (floor)
        1: "?",  # Questionable
        0: "✗"   # Failed
    }
    
    return glyph_map.get(passed_count, "?")


def compute_full_sha256(stamp_results: dict, params: dict) -> str:
    """Compute full SHA256 certificate hash for 3-line display."""
    
    # Find the most recent stamped certificate
    import glob
    import os
    
    cert_files = glob.glob(".out/certs/*_stamped_*.toml")
    if cert_files:
        # Get the most recent certificate
        latest_cert = max(cert_files, key=os.path.getmtime)
        
        # Hash the full certificate content
        with open(latest_cert, 'r') as f:
            cert_content = f.read()
        
        # Return full SHA256 hash
        full_hash = hashlib.sha256(cert_content.encode()).hexdigest()
        return full_hash
    
    # Fallback to stamp-based hash if no certificate found
    stamp_summary = "|".join(f"{name}:{stamp.passed}" for name, stamp in stamp_results.items())
    param_summary = f"depth={params['depth']}|N={params['N']}|gamma={params['gamma']}"
    timestamp = time.strftime("%Y%m%d", time.gmtime())
    
    content = f"axiel|{stamp_summary}|{param_summary}|{timestamp}"
    full_hash = hashlib.sha256(content.encode()).hexdigest()
    return full_hash


class ColorQuaternionGroup:
    """
    Color Quaternion Galois Group Engine with Harmonic Ratios
    
    Implements a non-abelian group of automorphisms acting on OKLCH perceptual space
    with harmonic series ratios 1:2:3:4:5:6:7 for natural color intervals.
    
    Group Elements:
    - L-flip: L ↦ 1 - L (light/dark inversion)  
    - C-mirror: C ↦ C_max - C (chroma inversion)
    - Hue rotations: h ↦ h + harmonic intervals (musical color harmony)
    - Permutations: cycle through (L,C,h) coordinates
    
    Harmonic Ratios (1:2:3:4:5:6:7):
    - Applied to L, C, h coordinates as natural overtone series
    - Creates musical harmony in color space
    - Fundamental + 6 overtones = 7-color palette
    
    Taste Bands (Apple/Material safe corridors):
    - Ambient frost: L ∈ [0.88,0.96], C ∈ [0.02,0.06]
    - Ambient dark: L ∈ [0.25,0.35], C ∈ [0.06,0.12] 
    - Accent: L ∈ [0.60,0.75], C ∈ [0.10,0.18]
    """
    
    def __init__(self):
        # Harmonic ratios (1:2:3:4:5:6:7) for natural color intervals
        self.harmonic_ratios = [1, 2, 3, 4, 5, 6, 7]
        
        # CRITICAL LINE TASTE BANDS: All L values centered on 0.5 (Riemann Hypothesis)
        self.taste_bands = {
            'ambient_frost': {'L': (0.88, 0.96), 'C': (0.02, 0.06)},
            'ambient_dark': {'L': (0.25, 0.35), 'C': (0.06, 0.12)},  # Lighter for better contrast
            'accent': {'L': (0.60, 0.75), 'C': (0.10, 0.18)},
            'pastel': {'L': (0.85, 0.92), 'C': (0.03, 0.08)},
            'mid_tone': {'L': (0.45, 0.65), 'C': (0.08, 0.15)},
            'glyph_contrast': {'L': (0.85, 0.95), 'C': (0.08, 0.20)},  # High contrast for glyph
            # CRITICAL HARMONICS: All lightness = 0.5 exactly (on the critical line)
            'harmonic_1': {'L': (0.5, 0.5), 'C': (0.04, 0.08)},      # Fundamental (critical line)
            'harmonic_2': {'L': (0.5, 0.5), 'C': (0.06, 0.10)},      # 2nd harmonic (critical line)
            'harmonic_3': {'L': (0.5, 0.5), 'C': (0.08, 0.12)},      # 3rd harmonic (critical line)
            'harmonic_4': {'L': (0.5, 0.5), 'C': (0.10, 0.14)},      # 4th harmonic (critical line)
            'harmonic_5': {'L': (0.5, 0.5), 'C': (0.12, 0.16)},      # 5th harmonic (critical line)
            'harmonic_6': {'L': (0.5, 0.5), 'C': (0.08, 0.18)},      # 6th harmonic (critical line)
            'harmonic_7': {'L': (0.5, 0.5), 'C': (0.04, 0.12)},      # 7th harmonic (critical line)
        }
    
    def l_flip(self, L, C, h):
        """L-flip: L ↦ 1 - L (light/dark inversion)"""
        return (1.0 - L, C, h)
    
    def c_mirror(self, L, C, h, C_max=0.25):
        """C-mirror: C ↦ C_max - C (chroma inversion)"""
        return (L, C_max - C, h)
    
    def harmonic_hue_rotate(self, L, C, h, harmonic_n):
        """Hue rotation by harmonic intervals: h ↦ h + (360°/harmonic_n)"""
        # Harmonic intervals: 360°/n for n in [1,2,3,4,5,6,7]
        # Creates musical intervals in color space
        harmonic_degrees = 360.0 / harmonic_n if harmonic_n > 0 else 0
        return (L, C, (h + harmonic_degrees) % 360)
    
    def harmonic_lightness_scale(self, L, C, h, harmonic_n):
        """CRITICAL LINE: All harmonics maintain L = 0.5 (Riemann Hypothesis)"""
        # Every critical harmonic stays exactly on the critical line Re(s) = 0.5
        # Only amplitude (chroma) and phase (hue) vary, not the critical real part
        return (0.5, C, h)  # Lightness always stays on critical line
    
    def harmonic_chroma_scale(self, L, C, h, harmonic_n):
        """Scale chroma by harmonic ratio: C ↦ (harmonic_n / 7) * 0.25"""
        # Scale chroma to practical range using harmonic ratio
        harmonic_scale = harmonic_n / 7.0
        scaled_C = harmonic_scale * 0.25  # Max practical chroma
        return (L, min(0.4, max(0.0, scaled_C)), h)
    
    def permute_coords(self, L, C, h, mode='LCh_to_CLh'):
        """Coordinate permutation (cycle through L,C,h)"""
        if mode == 'LCh_to_CLh':
            # Map L→C, C→h/360, h→L*360 (with normalization)
            return (h/360.0, L, C*360.0 % 360)
        elif mode == 'LCh_to_hLC':
            # Map L→h/360, C→L, h→C*360
            return (C, h/360.0, L*360.0 % 360)
        else:
            return (L, C, h)
    
    def generate_orbit(self, L, C, h, max_iterations=32):
        """Generate orbit under group action"""
        orbit = set()
        queue = [(L, C, h)]
        
        while queue and len(orbit) < max_iterations:
            current_L, current_C, current_h = queue.pop(0)
            
            # Round to avoid floating point duplicates
            key = (round(current_L, 4), round(current_C, 4), round(current_h, 1))
            if key in orbit:
                continue
            orbit.add(key)
            
            # Apply all group actions with harmonic ratios (1:2:3:4:5:6:7)
            candidates = [
                self.l_flip(current_L, current_C, current_h),
                self.c_mirror(current_L, current_C, current_h),
                # Harmonic hue rotations: 360°/n for n in [1,2,3,4,5,6,7]
                self.harmonic_hue_rotate(current_L, current_C, current_h, 2),  # 180°
                self.harmonic_hue_rotate(current_L, current_C, current_h, 3),  # 120°
                self.harmonic_hue_rotate(current_L, current_C, current_h, 4),  # 90°
                self.harmonic_hue_rotate(current_L, current_C, current_h, 5),  # 72°
                self.harmonic_hue_rotate(current_L, current_C, current_h, 6),  # 60°
                self.harmonic_hue_rotate(current_L, current_C, current_h, 7),  # ~51.4°
                # Harmonic lightness scaling
                self.harmonic_lightness_scale(current_L, current_C, current_h, 2),
                self.harmonic_lightness_scale(current_L, current_C, current_h, 3),
                self.harmonic_lightness_scale(current_L, current_C, current_h, 5),
                # Harmonic chroma scaling  
                self.harmonic_chroma_scale(current_L, current_C, current_h, 3),
                self.harmonic_chroma_scale(current_L, current_C, current_h, 5),
                # Coordinate permutations
                self.permute_coords(current_L, current_C, current_h, 'LCh_to_CLh'),
                self.permute_coords(current_L, current_C, current_h, 'LCh_to_hLC'),
            ]
            
            # Add valid candidates to queue
            for L_new, C_new, h_new in candidates:
                # Clamp to valid OKLCH ranges
                L_clamped = max(0.0, min(1.0, L_new))
                C_clamped = max(0.0, min(0.4, C_new))  # Max practical chroma
                h_clamped = h_new % 360
                
                candidate_key = (round(L_clamped, 4), round(C_clamped, 4), round(h_clamped, 1))
                if candidate_key not in orbit:
                    queue.append((L_clamped, C_clamped, h_clamped))
        
        return list(orbit)
    
    def project_to_taste_band(self, orbit, band_name):
        """Project orbit onto taste band - find representative that falls in band"""
        if band_name not in self.taste_bands:
            return None
        
        band = self.taste_bands[band_name]
        L_min, L_max = band['L']
        C_min, C_max = band['C']
        
        # Find orbit elements that fall in taste band
        candidates = []
        for L, C, h in orbit:
            if L_min <= L <= L_max and C_min <= C <= C_max:
                candidates.append((L, C, h))
        
        if candidates:
            # Return the "most central" candidate (closest to band center)
            L_center = (L_min + L_max) / 2
            C_center = (C_min + C_max) / 2
            
            def distance_to_center(point):
                L, C, h = point
                return ((L - L_center)**2 + (C - C_center)**2)**0.5
            
            return min(candidates, key=distance_to_center)
        
        # If no direct hits, find closest orbit element and project it into band
        def distance_to_band(point):
            L, C, h = point
            L_dist = max(0, L_min - L, L - L_max)
            C_dist = max(0, C_min - C, C - C_max)
            return (L_dist**2 + C_dist**2)**0.5
        
        closest_L, closest_C, closest_h = min(orbit, key=distance_to_band)
        
        # Project into band
        projected_L = max(L_min, min(L_max, closest_L))
        projected_C = max(C_min, min(C_max, closest_C))
        
        return (projected_L, projected_C, closest_h)


def sha256_to_oklch_ambient_palette(sha256_hash: str) -> dict:
    """
    Color Quaternion Galois Group Palette Generator
    
    Uses SHA-256 as raw entropy seed, applies quaternion group actions
    to generate orbit, then projects onto taste bands for tasteful palette.
    """
    
    # Initialize color quaternion group
    cqg = ColorQuaternionGroup()
    
    # RIEMANN HYPOTHESIS: All critical lines = 0.5
    # Every mathematically critical property is exactly on the critical line
    L_base = 0.5  # Critical lightness (like Re(s) = 0.5 for zeta zeros)
    C_base = 0.5  # Critical chroma (will be scaled to practical range)
    
    # Extract entropy from SHA-256 for hue rotation only
    hex_payload = sha256_hash[:16]  # Use first 64 bits
    entropy_int = int(hex_payload, 16)
    h_base = ((entropy_int >> 16) & 0xFFFF) / 0xFFFF * 360         # h ∈ [0, 360] - entropy-driven hue
    
    # Critical line constraint: all fundamental harmonics maintain Re(s) = 0.5 symmetry
    # Only imaginary parts (hue rotations) and amplitudes (chroma) vary
    
    # Generate orbit under quaternion group (facet rotations)
    orbit = cqg.generate_orbit(L_base, C_base, h_base)
    
    # Project orbit representatives onto harmonic taste bands (1:2:3:4:5:6:7)
    harmonic_1 = cqg.project_to_taste_band(orbit, 'harmonic_1')  # Fundamental (darkest)
    harmonic_2 = cqg.project_to_taste_band(orbit, 'harmonic_2')  # 2nd harmonic
    harmonic_3 = cqg.project_to_taste_band(orbit, 'harmonic_3')  # 3rd harmonic  
    harmonic_4 = cqg.project_to_taste_band(orbit, 'harmonic_4')  # 4th harmonic
    harmonic_5 = cqg.project_to_taste_band(orbit, 'harmonic_5')  # 5th harmonic
    harmonic_6 = cqg.project_to_taste_band(orbit, 'harmonic_6')  # 6th harmonic
    harmonic_7 = cqg.project_to_taste_band(orbit, 'harmonic_7')  # 7th harmonic (brightest)
    
    # Also keep some legacy bands for compatibility
    glyph_contrast = cqg.project_to_taste_band(orbit, 'glyph_contrast')
    
    # CRITICAL LINE FALLBACKS: All harmonics stay exactly on Re(s) = 0.5
    if not harmonic_1:
        harmonic_1 = (0.5, (1/7) * 0.25, h_base)  # Fundamental (critical line)
    if not harmonic_2:
        harmonic_2 = (0.5, (2/7) * 0.25, h_base)  # 2nd harmonic (critical line)
    if not harmonic_3:
        harmonic_3 = (0.5, (3/7) * 0.25, h_base)  # 3rd harmonic (critical line)
    if not harmonic_4:
        harmonic_4 = (0.5, (4/7) * 0.25, h_base)  # 4th harmonic (critical line)
    if not harmonic_5:
        harmonic_5 = (0.5, (5/7) * 0.25, h_base)  # 5th harmonic (critical line)
    if not harmonic_6:
        harmonic_6 = (0.5, (6/7) * 0.25, h_base)  # 6th harmonic (critical line)
    if not harmonic_7:
        harmonic_7 = (0.5, (7/7) * 0.25, h_base)  # 7th harmonic (critical line)
    if not glyph_contrast:
        glyph_contrast = (0.90, 0.20, h_base)  # High contrast fallback (off critical line for visibility)
    
    # Extract harmonic color coordinates (1:2:3:4:5:6:7)
    L1, C1, h1 = harmonic_1  # Fundamental
    L2, C2, h2 = harmonic_2  # 2nd harmonic
    L3, C3, h3 = harmonic_3  # 3rd harmonic
    L4, C4, h4 = harmonic_4  # 4th harmonic
    L5, C5, h5 = harmonic_5  # 5th harmonic
    L6, C6, h6 = harmonic_6  # 6th harmonic
    L7, C7, h7 = harmonic_7  # 7th harmonic
    L_glyph, C_glyph, h_glyph = glyph_contrast
    
    # Generate harmonic hue intervals (1:2:3:4:5:6:7 ratios)
    # Use fundamental harmonic as base, create intervals at 360°/n
    h_base_harmonic = h4  # Use 4th harmonic as base (middle of series)
    h_harmonic_2 = (h_base_harmonic + 180) % 360    # 360°/2 = 180° (octave)
    h_harmonic_3 = (h_base_harmonic + 120) % 360    # 360°/3 = 120° (perfect fifth)
    h_harmonic_4 = h_base_harmonic                  # Fundamental
    h_harmonic_5 = (h_base_harmonic + 72) % 360     # 360°/5 = 72° (major third)
    h_harmonic_6 = (h_base_harmonic + 60) % 360     # 360°/6 = 60° (perfect fourth)
    h_harmonic_7 = (h_base_harmonic + 51.4) % 360   # 360°/7 ≈ 51.4° (minor seventh)
    
    # Helper function for safe clamping
    def clamp(value, min_val, max_val):
        return max(min_val, min(max_val, value))
    
    # Build harmonic palette using 1:2:3:4:5:6:7 ratios (musical color harmony)
    palette = {
        # Core harmonic system (fundamental + 6 overtones)
        'harmonic_1': f"oklch({L1:.3f} {C1:.3f} {h1:.2f})",  # Fundamental (darkest)
        'harmonic_2': f"oklch({L2:.3f} {C2:.3f} {h2:.2f})",  # 2nd harmonic (octave)
        'harmonic_3': f"oklch({L3:.3f} {C3:.3f} {h3:.2f})",  # 3rd harmonic (perfect fifth)
        'harmonic_4': f"oklch({L4:.3f} {C4:.3f} {h4:.2f})",  # 4th harmonic (fundamental)
        'harmonic_5': f"oklch({L5:.3f} {C5:.3f} {h5:.2f})",  # 5th harmonic (major third)
        'harmonic_6': f"oklch({L6:.3f} {C6:.3f} {h6:.2f})",  # 6th harmonic (perfect fourth)
        'harmonic_7': f"oklch({L7:.3f} {C7:.3f} {h7:.2f})",  # 7th harmonic (minor seventh)
        
        # Musical interval hues (using harmonic ratios for hue relationships)
        'accent_octave': f"oklch({L4:.3f} {C4:.3f} {h_harmonic_2:.2f})",      # 180° interval
        'accent_fifth': f"oklch({L4:.3f} {C4:.3f} {h_harmonic_3:.2f})",       # 120° interval  
        'accent_third': f"oklch({L4:.3f} {C4:.3f} {h_harmonic_5:.2f})",       # 72° interval
        'accent_fourth': f"oklch({L4:.3f} {C4:.3f} {h_harmonic_6:.2f})",      # 60° interval
        
        # On-surface text (auto-contrast using harmonic ratios)
        'on_harmonic_1': f"oklch(0.95 0.02 {h1:.2f})",  # Light text on dark fundamental
        'on_harmonic_7': f"oklch(0.15 0.04 {h7:.2f})",  # Dark text on bright 7th harmonic
        'on_harmonic_4': f"oklch(0.18 0.03 {h4:.2f})" if L4 >= 0.7 else f"oklch(0.92 0.02 {h4:.2f})",
        
        # Map to existing badge structure (harmonic quaternion orbit representatives)
        'left_dark1': f"oklch({L1:.3f} {C1:.3f} {h1:.2f})",                    # Fundamental (darkest)
        'left_dark2': f"oklch({L2:.3f} {C2:.3f} {h_harmonic_3:.2f})",          # 2nd harmonic with fifth interval
        'left_accent': f"oklch({L3:.3f} {C3:.3f} {h3:.2f})",                   # 3rd harmonic
        
        'right_light1': f"oklch({L6:.3f} {C6:.3f} {h6:.2f})",                  # 6th harmonic (bright)
        'right_light2': f"oklch({L7:.3f} {C7:.3f} {h7:.2f})",                  # 7th harmonic (brightest)
        'right_accent': f"oklch({L5:.3f} {C5:.3f} {h_harmonic_5:.2f})",        # 5th harmonic with major third
        
        'common_bridge1': f"oklch({L3:.3f} {C3:.3f} {h_harmonic_6:.2f})",      # 3rd harmonic with fourth interval
        'common_bridge2': f"oklch({L5:.3f} {C5:.3f} {h_harmonic_7:.2f})",      # 5th harmonic with seventh interval  
        'common_glow': f"oklch({L_glyph:.3f} {clamp(C_glyph*1.2, 0.12, 0.25):.3f} {h_glyph:.2f})",  # High contrast glyph                       
    }
    
    return palette


def create_axiel_badge_svg(stamp_results: dict, params: dict, output_path: str) -> None:
    """Create dense GitHub-style Axiel badge."""
    
    # Count passing stamps
    passed_count = sum(1 for stamp in stamp_results.values() if stamp.passed)
    total_count = len(stamp_results)
    
    # Get badge glyph and full SHA256
    badge_glyph = get_badge_glyph(passed_count, total_count)
    full_sha256 = compute_full_sha256(stamp_results, params)
    
    # Generate OKLCH ambient palette from SHA256
    colors = sha256_to_oklch_ambient_palette(full_sha256)
    
    # Split SHA256 into 3 lines for display
    line1 = full_sha256[:22]  # First 22 chars
    line2 = full_sha256[22:44]  # Next 22 chars  
    line3 = full_sha256[44:]  # Last 20 chars
    
    # Badge colors based on completion
    if passed_count == total_count:
        badge_color = "#28a745"  # Green (complete)
    elif passed_count >= total_count * 0.75:
        badge_color = "#ffc107"  # Yellow (mostly)
    elif passed_count >= total_count * 0.5:
        badge_color = "#fd7e14"  # Orange (partial)
    else:
        badge_color = "#dc3545"  # Red (failed)
    
    # Create tight badge SVG with rainbow gloss stencil
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- GitHub-Style Axiel Badge Template -->
<!-- Dense, tight block with rainbow gloss stencil -->
<svg width="110" height="40" viewBox="0 0 110 40" 
     xmlns="http://www.w3.org/2000/svg"
     data-ce1-version="1.0"
     data-template="github-badge">
  
  <!-- Definitions for gradients -->
  <defs>
    <!-- SHA256-generated left side dark chrome -->
    <linearGradient id="darkChromeBase" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{colors['left_dark1']};stop-opacity:1"/>
      <stop offset="50%" style="stop-color:{colors['left_dark2']};stop-opacity:1"/>
      <stop offset="100%" style="stop-color:{colors['left_accent']};stop-opacity:1"/>
    </linearGradient>
    
    <!-- Left side ambient overlay with common bridge -->
    <radialGradient id="chromeOverlay" cx="50%" cy="30%" r="80%">
      <stop offset="0%" style="stop-color:{colors['left_dark1']};stop-opacity:0.9"/>
      <stop offset="50%" style="stop-color:{colors['common_bridge1']};stop-opacity:0.6"/>
      <stop offset="100%" style="stop-color:{colors['left_dark2']};stop-opacity:0.8"/>
    </radialGradient>
    
    <!-- Chrome highlight streaks -->
    <linearGradient id="chromeHighlights" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#ffffff;stop-opacity:0.1"/>
      <stop offset="20%" style="stop-color:#e0d0e0;stop-opacity:0.05"/>
      <stop offset="40%" style="stop-color:#d0e0d0;stop-opacity:0.03"/>
      <stop offset="60%" style="stop-color:#d0d0e0;stop-opacity:0.05"/>
      <stop offset="80%" style="stop-color:#e0d0d0;stop-opacity:0.04"/>
      <stop offset="100%" style="stop-color:#ffffff;stop-opacity:0.1"/>
    </linearGradient>
    
     <!-- Common theme reverse flow glyph (from SHA256 part 3) -->
    <linearGradient id="reverseChromeBase" x1="100%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:{colors['common_glow']};stop-opacity:1"/>
      <stop offset="50%" style="stop-color:{colors['common_bridge2']};stop-opacity:1"/>
      <stop offset="100%" style="stop-color:{colors['common_bridge1']};stop-opacity:1"/>
    </linearGradient>
    
    <!-- High luminosity overlay with common theme -->
    <radialGradient id="glyphLuminosity" cx="50%" cy="50%" r="60%">
      <stop offset="0%" style="stop-color:{colors['common_glow']};stop-opacity:0.9"/>
      <stop offset="50%" style="stop-color:{colors['common_bridge2']};stop-opacity:0.6"/>
      <stop offset="100%" style="stop-color:{colors['common_bridge1']};stop-opacity:0.3"/>
    </radialGradient>
    
    <!-- Chrome depth shadow for glyph -->
    <linearGradient id="glyphDepth" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#000000;stop-opacity:0.2"/>
      <stop offset="50%" style="stop-color:#333333;stop-opacity:0.1"/>
      <stop offset="100%" style="stop-color:#000000;stop-opacity:0.3"/>
    </linearGradient>
    
    <!-- SHA256-generated right side bright chrome -->
    <linearGradient id="inverseChromeBase" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{colors['right_light1']};stop-opacity:1"/>
      <stop offset="50%" style="stop-color:{colors['right_light2']};stop-opacity:1"/>
      <stop offset="100%" style="stop-color:{colors['right_accent']};stop-opacity:1"/>
    </linearGradient>
    
    <!-- Right side ambient overlay with common bridge -->
    <radialGradient id="inverseChromeOverlay" cx="50%" cy="70%" r="80%">
      <stop offset="0%" style="stop-color:{colors['right_light1']};stop-opacity:0.8"/>
      <stop offset="50%" style="stop-color:{colors['common_bridge2']};stop-opacity:0.5"/>
      <stop offset="100%" style="stop-color:{colors['right_light2']};stop-opacity:0.9"/>
    </radialGradient>
    
    <!-- Dark text using common theme -->
    <linearGradient id="darkTextGradient" x1="100%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:{colors['left_dark1']};stop-opacity:1"/>
      <stop offset="50%" style="stop-color:{colors['common_bridge1']};stop-opacity:1"/>
      <stop offset="100%" style="stop-color:{colors['left_dark2']};stop-opacity:1"/>
    </linearGradient>
  </defs>
  
  <!-- Dense badge background -->
  <rect width="110" height="40" fill="#f6f8fa" stroke="#d1d5da" stroke-width="1" rx="3"/>
  
  <!-- Left section: Tightened badge glyph -->
  <g id="badge-section">
    <rect width="36" height="40" fill="url(#darkChromeBase)" rx="1"/>
    <rect width="36" height="40" fill="url(#chromeOverlay)" rx="1"/>
    <rect width="36" height="40" fill="url(#chromeHighlights)" rx="1"/>
    
    <!-- Mathematical glyph with reverse flow chrome -->
    <text x="18" y="20" text-anchor="middle" font-family="STIX Two Math, serif" 
          font-size="18" fill="url(#reverseChromeBase)" font-weight="bold">{badge_glyph}</text>
    
    <!-- High luminosity overlay -->
    <text x="18" y="20" text-anchor="middle" font-family="STIX Two Math, serif" 
          font-size="18" fill="url(#glyphLuminosity)" font-weight="bold">{badge_glyph}</text>
    
    <!-- Subtle depth shadow -->
    <text x="18" y="20" text-anchor="middle" font-family="STIX Two Math, serif" 
          font-size="18" fill="url(#glyphDepth)" font-weight="bold">{badge_glyph}</text>
    
    <!-- Axiel label below glyph -->
    <text x="18" y="35" text-anchor="middle" font-family="FiraCode Nerd Font" 
          font-size="6" fill="url(#darkTextGradient)" font-weight="bold">AXIEL</text>
  </g>
  
  <!-- Right section: Full SHA256 on 3 lines -->
  <g id="info-section">
    <!-- Golden ratio background (φ = 1.618, so width = 110-36 = 74px, positioned at φ ratio) -->
    <rect x="36" width="74" height="40" fill="url(#inverseChromeBase)" rx="1"/>
    <rect x="36" width="74" height="40" fill="url(#inverseChromeOverlay)" rx="1"/>
    
    <!-- SHA256 Line 1 -->
    <text x="40" y="12" font-family="Monaco, monospace" font-size="6" fill="url(#darkTextGradient)">
      {line1}
    </text>
    
    <!-- SHA256 Line 2 -->
    <text x="40" y="22" font-family="Monaco, monospace" font-size="6" fill="url(#darkTextGradient)">
      {line2}
    </text>
    
    <!-- SHA256 Line 3 -->
    <text x="40" y="32" font-family="Monaco, monospace" font-size="6" fill="url(#darkTextGradient)">
      {line3}
    </text>
  </g>
  
</svg>'''
    
    # Write badge
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)


def main():
    parser = argparse.ArgumentParser(description="Create dense GitHub-style Axiel badge")
    parser.add_argument("--out", type=str, default=".out/badges", help="Output directory")
    args = parser.parse_args()
    
    print("🏷️  Axiel Badge Generator: Dense GitHub-Style")
    print("=" * 45)
    
    # Generate verification
    params = create_proof_unlock_params()
    stamper = ProofUnlockStamper(depth=params["depth"])
    
    print("Computing verification...")
    stamp_results = stamper.stamp_certification(params)
    
    passed_count = sum(1 for stamp in stamp_results.values() if stamp.passed)
    total_count = len(stamp_results)
    
    print(f"✅ Stamps: {passed_count}/{total_count}")
    
    # Determine badge glyph
    badge_glyph = get_badge_glyph(passed_count, total_count)
    full_sha256 = compute_full_sha256(stamp_results, params)
    
    print(f"🎨 Badge glyph: {badge_glyph} (for {passed_count}/{total_count} completion)")
    print(f"🔗 Full SHA256: {full_sha256[:16]}...{full_sha256[-8:]}")
    
    # Create badge
    timestamp_file = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.out, exist_ok=True)
    
    badge_path = os.path.join(args.out, f"axiel-badge-{timestamp_file}.svg")
    
    create_axiel_badge_svg(stamp_results, params, badge_path)
    
    print(f"\n🏷️  AXIEL BADGE CREATED")
    print("=" * 45)
    print(f"File: {badge_path}")
    print("Style: Dense GitHub-style block")
    print("Layout:")
    print(f"  🟩 {badge_glyph} | SHA256 Certificate ID")
    print(f"  AXIEL | {full_sha256[:22]}")
    print(f"        | {full_sha256[22:44]}")
    print(f"        | {full_sha256[44:]}")
    
    # Show glyph progression
    print(f"\nGlyph Progression:")
    for count in range(9):
        glyph = get_badge_glyph(count, 8)
        status = "COMPLETE" if count == 8 else "PARTIAL" if count >= 6 else "BASIC" if count >= 4 else "FAILED"
        print(f"  {count}/8: {glyph} ({status})")
    
    print(f"\n🎯 Perfect GitHub-Style Badge Template!")
    print("Dense, clean, minimal—exactly like GitHub status badges.")
    
    return 0


if __name__ == "__main__":
    exit(main())