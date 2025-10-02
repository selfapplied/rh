#!/usr/bin/env python3
"""
RIE: Riemann Intelligence Engine

A beautiful CLI for your mathematical framework that unifies:
- 3.5D Color Theory
- CE2 Color Equilibrium  
- Gang of Four Patterns
- Mathematical Badges & Certificates
- Smart templating with intelligent defaults
- Historical timeline like git/filesystem hybrid

Usage Examples:
  rie --github --badge cert.svg          # GitHub badge from template
  rie --ce2 --equilibrium analysis.json  # CE2 equilibrium analysis
  rie --3-5d --palette colors.svg        # 3.5D color palette
  rie ls                                  # Historical timeline tree
  rie status                              # Current system state
  rie cert --type passport               # Generate mathematical passport
"""

import hashlib
import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

# Import our mathematical systems
from color_equilibrium import CE2ColorEquilibrium
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.tree import Tree
from visualization.color_quaternion_3_5d_theory import ColorQuaternion3_5DSpec
from visualization.create_ce2_badge import create_ce2_equilibrium_badge
from visualization.create_github_3_5d_badge import (
    create_detailed_github_badge,
    create_github_3_5d_badge,
)


# Initialize Typer app and Rich console
app = typer.Typer(
    name="rie",
    help="🌈 Riemann Intelligence Engine - Your 3.5D Mathematical Framework CLI",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()


class BadgeType(str, Enum):
    """Badge types for generation"""
    GITHUB = "github"
    CE2 = "ce2" 
    PASSPORT = "passport"
    CERTIFICATE = "certificate"
    DETAILED = "detailed"
    COMPACT = "compact"
    SHIELD = "shield"


class OutputFormat(str, Enum):
    """Output format options"""
    SVG = "svg"
    JSON = "json"
    PNG = "png"
    PDF = "pdf"


class RieContext:
    """Smart context manager for RIE operations"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.output_dir = self.project_root / ".out"
        self.history_file = self.output_dir / "rie_history.json"
        self.context_cache = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Load history
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load command history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self):
        """Save command history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def add_to_history(self, command: str, args: Dict[str, Any], output_file: str, metadata: Dict[str, Any] = None):
        """Add command to history"""
        entry = {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'command': command,
            'args': args,
            'output_file': output_file,
            'metadata': metadata or {},
            'hash': hashlib.sha256(f"{command}{time.time()}".encode()).hexdigest()[:8]
        }
        self.history.append(entry)
        self._save_history()
    
    def get_pearl_filename(self, template: str, extension: str, context: Dict[str, Any] = None) -> str:
        """Generate pearl-worthy filename - respecting your anchored naming preferences"""
        context = context or {}
        
        # Clean base name - avoid unfortunate combinations
        base_name = self._get_clean_base(template)
        
        # Detect density family for this base in current timeframe
        family_nonce = self._get_family_nonce(base_name, context)
        
        # Pure boundary conditioning - no separators!
        filename = f"{base_name}{family_nonce}.{extension}"
        
        return filename
    
    def _get_clean_base(self, template: str) -> str:
        """Get clean base name, avoiding unfortunate word fragments"""
        if not template or template == "cert":
            return "cert"
        elif "passport" in template.lower():
            return "pass"
        elif "analysis" in template.lower() or "equilibrium" in template.lower():
            return "equi"  # Clean, no accidents
        elif "palette" in template.lower():
            return "pal"
        else:
            # Keep it short but meaningful
            base = Path(template).stem[:4]
            # Avoid known problematic fragments
            if base.startswith("anal"):
                return "equi"
            return base
    
    def _get_family_nonce(self, base_name: str, context: Dict[str, Any]) -> str:
        """Get family dialect nonce based on density in current workspace"""
        # Check existing files with this base
        pattern = f"{base_name}*.svg"
        existing_files = list(self.output_dir.rglob(pattern))
        
        current_hour = time.strftime("%H")
        time.strftime("%d")
        
        # Family promotion based on density
        if len(existing_files) == 0:
            return ""  # First one gets clean name
        elif len(existing_files) < 3:
            return f"b{len(existing_files) + 1}"  # batch family: b1, b2, b3
        elif len(existing_files) < 10:
            return f"h{current_hour}"  # hour family: h09, h14, h16
        else:
            # Dense workspace - use hour + letter
            hour_pattern = f"{base_name}h{current_hour}*.svg"
            hour_files = list(self.output_dir.rglob(hour_pattern))
            letter = chr(ord('a') + len(hour_files))
            return f"h{current_hour}{letter}"  # h14a, h14b, h14c
    
    def get_output_path(self, category: str, filename: str) -> Path:
        """Get output path with smart directory energy distribution"""
        
        # Check if we should suggest lifecycle directories
        base_name = Path(filename).stem.rstrip('0123456789abcdefghijklmnopqrstuvwxyz')
        
        # Count similar files in category
        similar_files = list((self.output_dir / category).glob(f"{base_name}*"))
        
        # Suggest energy distribution through directories
        if len(similar_files) >= 3:
            # Multiple similar files - suggest grouping
            if time.strftime("%H") >= "09" and time.strftime("%H") <= "17":
                # Working hours - suggest today/
                output_path = self.output_dir / category / "today"
            else:
                # Off hours - suggest the base name as directory
                output_path = self.output_dir / category / base_name
        else:
            # Few files - keep in main category
            output_path = self.output_dir / category
            
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / filename


# Global context
rie_context = RieContext()


@app.command()
def badge(
    template: str = typer.Argument("cert", help="Template (cert/passport/eq)"),
    # Single letter flags - save those fingers!
    g: bool = typer.Option(False, "-g", help="GitHub style"),
    c: bool = typer.Option(False, "-c", help="CE2 style"), 
    p: bool = typer.Option(False, "-p", help="Passport"),
    d: bool = typer.Option(False, "-d", help="Detailed"),
    s: Optional[str] = typer.Option(None, "-s", help="Seed"),
    o: Optional[str] = typer.Option(None, "-o", help="Output")
):
    """
    🏷️ Generate badges (save your fingers!)
    
    Examples:
      rie b cert -g -d        # GitHub detailed badge  
      rie b -c -s test        # CE2 badge with seed
      rie badge passport -p   # Passport badge
    """
    
    with Progress() as progress:
        task = progress.add_task("🎨 Generating badge...", total=100)
        
        # Determine badge type from single-letter flags
        if g:  # GitHub
            badge_category = "github_badge"
            if d:  # Detailed
                generator_func = create_detailed_github_badge
            else:
                generator_func = create_github_3_5d_badge
        elif c:  # CE2
            badge_category = "ce2_badge"
            generator_func = create_ce2_equilibrium_badge
        elif p:  # Passport
            badge_category = "passport_badge"
            generator_func = create_github_3_5d_badge  # Placeholder
        else:
            # Smart default based on template
            if "cert" in template.lower():
                badge_category = "github_badge"
                generator_func = create_github_3_5d_badge
            else:
                badge_category = "badge"
                generator_func = create_github_3_5d_badge
        
        progress.update(task, advance=30)
        
        # Generate smart seed if not provided
        if not s:
            s = f"{template}_{badge_category}"  # Much shorter seed
        
        progress.update(task, advance=20)
        
        # Generate badge
        try:
            output_file = generator_func(s)
            progress.update(task, advance=30)
            
            # Generate pearl-worthy filename!
            old_path = Path(output_file)
            pearl_name = rie_context.get_pearl_filename(template, "svg", {"type": badge_category})
            new_path = old_path.parent / pearl_name
            
            os.rename(output_file, new_path)
            output_file = str(new_path)
            progress.update(task, advance=10)
            
            # Handle custom output filename
            if o:
                custom_path = rie_context.get_output_path(badge_category, o)
                os.rename(output_file, custom_path)
                output_file = str(custom_path)
            
            progress.update(task, advance=10)
            
            # Add to history
            rie_context.add_to_history(
                command="badge",
                args={
                    "template": template,
                    "badge_type": badge_category,
                    "seed": s,
                    "github": g,
                    "ce2": c,
                    "passport": p,
                    "detailed": d
                },
                output_file=output_file,
                metadata={
                    "generator": generator_func.__name__,
                    "file_size": os.path.getsize(output_file)
                }
            )
            
        except Exception as e:
            console.print(f"[red]❌ Error generating badge: {e}[/red]")
            raise typer.Exit(1)
    
    # Visual success - metadata through color/shape, not text length!
    file_name = Path(output_file).name
    
    # Color-code by type (visual metadata!)
    type_color = "green" if g else "blue" if c else "magenta" if p else "cyan"
    type_emoji = "🏷️" if g else "⚖️" if c else "🏆" if p else "✨"
    
    console.print(f"{type_emoji} [{type_color}]●[/{type_color}] [dim]{file_name}[/dim]")
    
    # Only show seed if it was custom (reduce noise!)
    if s and not s.startswith("rie_"):
        console.print(f"   [yellow]↳[/yellow] [dim]{s}[/dim]")


@app.command()
def analyze(
    analysis_file: str = typer.Argument("analysis.json", help="Output analysis filename"),
    seed: Optional[str] = typer.Option(None, "--seed", "-s", help="Seed for equilibrium"),
    iterations: int = typer.Option(100, "--iterations", "-i", help="Max iterations"),
    tolerance: float = typer.Option(1e-6, "--tolerance", "-t", help="Convergence tolerance"),
    save_history: bool = typer.Option(True, "--history", help="Save iteration history"),
    output_format: OutputFormat = typer.Option(OutputFormat.JSON, "--format", "-f", help="Output format")
):
    """
    ⚖️ Analyze CE2 Color Equilibrium in 3.5D space
    
    Examples:
      rie equilibrium analysis.json --seed "my_system"
      rie equilibrium --iterations 200 --tolerance 1e-8
    """
    
    with Progress() as progress:
        task = progress.add_task("⚖️ Computing equilibrium...", total=100)
        
        # Initialize CE2 system
        if not seed:
            seed = f"rie_equilibrium_{time.time()}"
        
        ce2_system = CE2ColorEquilibrium(seed)
        progress.update(task, advance=20)
        
        # Find equilibrium
        console.print("🔄 Finding unified Gang of Four equilibrium...")
        unified_equilibrium = ce2_system.find_unified_gang_of_four_equilibrium(
            max_iterations=iterations,
            tolerance=tolerance
        )
        progress.update(task, advance=60)
        
        # Generate complete spec
        complete_spec = ce2_system.generate_ce2_complete_spec()
        progress.update(task, advance=15)
        
        # Save results
        output_path = rie_context.get_output_path("equilibrium", analysis_file)
        
        with open(output_path, 'w') as f:
            json.dump(complete_spec, f, indent=2, default=str)
        
        progress.update(task, advance=5)
        
        # Add to history
        rie_context.add_to_history(
            command="equilibrium",
            args={
                "seed": seed,
                "iterations": iterations,
                "tolerance": tolerance,
                "format": output_format.value
            },
            output_file=str(output_path),
            metadata={
                "is_stable": unified_equilibrium.is_stable,
                "equilibrium_quality": unified_equilibrium.equilibrium_quality,
                "stability_measure": unified_equilibrium.stability_measure
            }
        )
    
    # Display results
    unified = complete_spec['unified_equilibrium']
    
    table = Table(title="🎯 CE2 Equilibrium Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Status", style="green")
    
    table.add_row("Stability", f"{unified['stability_measure']:.3f}", "✅ STABLE" if unified['is_stable'] else "🔄 CONVERGING")
    table.add_row("Quality", f"{unified['equilibrium_quality']:.3f}", "🌟 EXCELLENT" if unified['equilibrium_quality'] > 0.8 else "✨ GOOD")
    table.add_row("Energy", f"{unified['equilibrium_energy']:.2f}", "⚡ BALANCED")
    table.add_row("Coherence", f"{unified['dimensional_coherence']:.3f}", "🔮 3.5D")
    
    console.print(table)
    console.print(f"\n📁 Analysis saved to: [cyan]{output_path}[/cyan]")


@app.command()
def palette(
    output_file: str = typer.Argument("colors.svg", help="Output palette filename"),
    seed: Optional[str] = typer.Option(None, "--seed", "-s", help="Seed for palette generation"),
    colors: int = typer.Option(7, "--colors", "-n", help="Number of colors in palette"),
    three_five_d: bool = typer.Option(True, "--3-5d", help="Generate 3.5D fractional palette"),
    format: OutputFormat = typer.Option(OutputFormat.SVG, "--format", "-f", help="Output format"),
    show_temporal: bool = typer.Option(True, "--temporal", help="Show temporal bleeding effects"),
    show_metrics: bool = typer.Option(True, "--metrics", help="Show dimensional metrics")
):
    """
    🎨 Generate 3.5D color palettes with temporal bleeding
    
    Examples:
      rie palette colors.svg --colors 5 --seed "sunset"
      rie palette --3-5d --temporal --metrics
    """
    
    with Progress() as progress:
        task = progress.add_task("🎨 Generating palette...", total=100)
        
        # Initialize 3.5D color system
        if not seed:
            seed = f"rie_palette_{time.time()}"
        
        color_spec_3_5d = ColorQuaternion3_5DSpec(seed)
        progress.update(task, advance=30)
        
        # Generate palette
        palette = color_spec_3_5d.generate_3_5d_harmonic_palette(colors)
        progress.update(task, advance=40)
        
        # Create visualization (placeholder - would implement full palette SVG generator)
        output_path = rie_context.get_output_path("palette", output_file)
        
        # Generate palette data
        palette_data = {
            'seed': seed,
            'colors': [color.to_string() for color in palette],
            'fractional_dimensions': [color.fractional_dimension for color in palette],
            'temporal_components': [
                {
                    'memory': color.temporal.memory_strength,
                    'anticipation': color.temporal.anticipation_strength,
                    'resonance': color.temporal.dimensional_resonance
                } for color in palette
            ]
        }
        
        # Save palette data
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(palette_data, f, indent=2, default=str)
        
        progress.update(task, advance=30)
        
        # Add to history
        rie_context.add_to_history(
            command="palette",
            args={
                "seed": seed,
                "colors": colors,
                "3_5d": three_five_d,
                "temporal": show_temporal,
                "metrics": show_metrics
            },
            output_file=str(output_path.with_suffix('.json')),
            metadata={
                "palette_size": len(palette),
                "avg_dimension": sum(color.fractional_dimension for color in palette) / len(palette)
            }
        )
    
    # Display palette
    console.print(f"\n🎨 [bold]3.5D Color Palette Generated[/bold]")
    console.print(f"🌱 Seed: [yellow]{seed}[/yellow]")
    console.print(f"🎯 Colors: [cyan]{len(palette)}[/cyan]")
    
    for i, color in enumerate(palette):
        console.print(f"  {i+1}. [magenta]{color.to_string()}[/magenta] (dim: {color.fractional_dimension:.2f})")
    
    console.print(f"\n📁 Data saved to: [cyan]{output_path.with_suffix('.json')}[/cyan]")


@app.command("ls")
def list_history():
    """
    📜 Show historical timeline tree of RIE operations (like git log + ls)
    
    Displays your mathematical operations as a beautiful timeline tree
    """
    
    if not rie_context.history:
        console.print("[yellow]📭 No history yet! Run some RIE commands to build your timeline.[/yellow]")
        return
    
    # Create timeline tree
    tree = Tree("🌳 [bold cyan]RIE Timeline[/bold cyan] (Mathematical Operations History)")
    
    # Group by date
    history_by_date = {}
    for entry in reversed(rie_context.history[-20:]):  # Last 20 entries
        date = entry['timestamp'][:10]  # YYYY-MM-DD
        if date not in history_by_date:
            history_by_date[date] = []
        history_by_date[date].append(entry)
    
    for date, entries in history_by_date.items():
        date_branch = tree.add(f"📅 [bold blue]{date}[/bold blue]")
        
        for entry in entries:
            time_str = entry['timestamp'][11:16]  # HH:MM (shorter!)
            command = entry['command']
            output_file = Path(entry['output_file']).stem  # No extension clutter
            hash_short = entry['hash'][:4]  # Even shorter hash
            
            # Visual metadata through emoji + color, not text!
            if command == "badge":
                icon = "🏷️"
                color = "green"
            elif command == "equilibrium":
                icon = "⚖️" 
                color = "magenta"
                # Visual stability through color intensity!
                if 'metadata' in entry and entry.get('metadata', {}).get('is_stable'):
                    color = "bright_green"  # Stable = brighter
            elif command == "palette":
                icon = "🎨"
                color = "yellow"
            else:
                icon = "🔧"
                color = "cyan"
            
            # Minimal, visual entry - metadata in color/position, not text!
            date_branch.add(
                f"{icon}[{color}]●[/{color}] [dim]{time_str} {output_file}[/dim] [bright_black]{hash_short}[/bright_black]"
            )
    
    console.print(tree)
    
    # Summary stats
    total_commands = len(rie_context.history)
    command_types = {}
    for entry in rie_context.history:
        cmd = entry['command']
        command_types[cmd] = command_types.get(cmd, 0) + 1
    
    console.print(f"\n📊 [bold]Summary[/bold]: {total_commands} total operations")
    for cmd, count in command_types.items():
        console.print(f"  {cmd}: {count}")


@app.command()
def status():
    """
    🔍 Show current RIE system status and recent activity
    """
    
    console.print(Panel(
        "[bold cyan]🌈 Riemann Intelligence Engine[/bold cyan]\n"
        "[dim]3.5D Fractional Dimensional Mathematical Framework[/dim]",
        title="RIE Status",
        border_style="cyan"
    ))
    
    # System info
    table = Table(title="🔧 System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="yellow")
    
    table.add_row("3.5D Color Theory", "✅ Active", "v1.0")
    table.add_row("CE2 Equilibrium", "✅ Active", "v1.0")
    table.add_row("Gang of Four", "✅ Implemented", "v1.0")
    table.add_row("Badge Generation", "✅ Ready", "v1.0")
    table.add_row("Smart Templates", "✅ Ready", "v1.0")
    
    console.print(table)
    
    # Recent activity
    if rie_context.history:
        recent = rie_context.history[-5:]  # Last 5 operations
        console.print(f"\n🕒 [bold]Recent Activity[/bold] (last {len(recent)} operations)")
        
        for entry in reversed(recent):
            time_str = entry['timestamp'][11:19]
            command = entry['command']
            output = Path(entry['output_file']).name
            console.print(f"  [dim]{time_str}[/dim] [cyan]{command}[/cyan] → [blue]{output}[/blue]")
    else:
        console.print("\n[yellow]📭 No recent activity[/yellow]")
    
    # Output directory info
    if rie_context.output_dir.exists():
        total_files = len(list(rie_context.output_dir.rglob("*")))
        console.print(f"\n📁 Output directory: [cyan]{rie_context.output_dir}[/cyan] ({total_files} files)")


@app.command()
def rename(
    pattern: str = typer.Argument(..., help="File pattern to rename (e.g. 'cert*')"),
    target: Optional[str] = typer.Option(None, "-t", help="Target pattern or directory"),
    suggest: bool = typer.Option(False, "--suggest", "-s", help="Suggest energy distribution"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Batch rename similar files")
):
    """
    🔄 Rename files with energy distribution through directories
    
    Examples:
      rie rename "cert*" --suggest           # Suggest better organization
      rie rename "cert*" -t "today/"        # Move to today/ directory  
      rie rename "certb*" -t "batch/cert"   # Distribute energy to batch/cert*
    """
    
    # Find matching files
    matching_files = []
    for category_dir in rie_context.output_dir.iterdir():
        if category_dir.is_dir():
            matches = list(category_dir.glob(pattern))
            matching_files.extend(matches)
    
    if not matching_files:
        console.print(f"[yellow]No files matching '{pattern}' found[/yellow]")
        return
    
    console.print(f"Found {len(matching_files)} files matching '{pattern}':")
    
    if suggest:
        # Suggest energy distribution
        console.print("\n💡 [bold]Energy Distribution Suggestions:[/bold]")
        
        # Group by base name
        base_groups = {}
        for file in matching_files:
            base = file.stem.rstrip('0123456789abcdefghijklmnopqrstuvwxyz')
            if base not in base_groups:
                base_groups[base] = []
            base_groups[base].append(file)
        
        for base, files in base_groups.items():
            if len(files) >= 3:
                console.print(f"  📁 {base}/ directory for {len(files)} files")
                for file in files:
                    new_name = file.name.replace(base, "").lstrip("b0123456789")
                    if not new_name:
                        new_name = f"{len([f for f in files if f <= file])}.svg"
                    console.print(f"    {file.name} → {base}/{new_name}")
            else:
                console.print(f"  ✅ {base} files are fine as-is ({len(files)} files)")
        
        # Suggest lifecycle directories
        current_hour = int(time.strftime("%H"))
        if 9 <= current_hour <= 17:
            console.print(f"\n⏰ Working hours - consider [cyan]today/[/cyan] for active work")
        else:
            console.print(f"\n🌙 Off hours - consider [cyan]{base}/[/cyan] directories for grouping")
    
    else:
        # Show current files
        for file in matching_files:
            console.print(f"  📄 {file}")


@app.command()
def cert(
    cert_type: str = typer.Argument("passport", help="Certificate type (passport, badge, equilibrium)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output filename"),
    seed: Optional[str] = typer.Option(None, "--seed", "-s", help="Seed for generation"),
    format: OutputFormat = typer.Option(OutputFormat.SVG, "--format", "-f", help="Output format")
):
    """
    🏆 Generate mathematical certificates and passports
    
    Examples:
      rie cert passport --output my_passport.svg
      rie cert equilibrium --seed "stable_system"
    """
    
    console.print(f"🏆 Generating [cyan]{cert_type}[/cyan] certificate...")
    
    # Delegate to appropriate generator
    if cert_type == "passport":
        # Would implement passport generator
        console.print("[yellow]🚧 Passport generation coming soon![/yellow]")
    elif cert_type == "badge":
        # Use badge command
        from typer.testing import CliRunner
        runner = CliRunner()
        runner.invoke(app, ["badge", "--github", "--detailed"])
    else:
        console.print(f"[red]❌ Unknown certificate type: {cert_type}[/red]")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    🎯 Contextual RIE - Intelligent without defaults
    
    Run 'rie' alone for contextual intelligence:
    - After renaming: Shows grammar rule updates
    - After generating: Shows family promotions  
    - Multiple runs: Alternative interpretations
    - Eventually: Drops into REPL
    """
    if ctx.invoked_subcommand is None:
        # No command given - contextual intelligence mode!
        handle_contextual_rie()


def handle_contextual_rie():
    """Handle contextual RIE based on recent activity"""
    
    # Check recent activity context
    recent_activity = rie_context.history[-5:] if rie_context.history else []
    
    if not recent_activity:
        # First time - welcome
        console.print(Panel(
            "[bold cyan]🌈 Welcome to RIE![/bold cyan]\n\n"
            "Try: [yellow]rie badge cert -g[/yellow] or [yellow]rie --help[/yellow]",
            title="Riemann Intelligence Engine",
            border_style="cyan"
        ))
        return
    
    # Check what happened recently
    recent_activity[-1]['command']
    last_timestamp = recent_activity[-1]['timestamp']
    
    # Time since last command  
    import datetime
    try:
        last_time = datetime.datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
        now = datetime.datetime.now(datetime.timezone.utc)
        minutes_ago = (now - last_time).total_seconds() / 60
    except:
        minutes_ago = 999  # Fallback to old activity
    
    if minutes_ago < 5:  # More generous window
        # Very recent activity - show what just happened
        show_grammar_updates(recent_activity)
    elif minutes_ago < 10:
        # Recent activity - show alternative interpretations
        show_alternative_interpretations(recent_activity)
    else:
        # Older activity - suggest cleanup or new work
        suggest_next_actions(recent_activity)


def show_grammar_updates(recent_activity: List[Dict[str, Any]]):
    """Show grammar rule updates from recent activity"""
    console.print("📝 [bold]Grammar Rules Updated:[/bold]")
    
    # Analyze recent naming patterns
    for entry in recent_activity[-3:]:
        output_file = Path(entry['output_file'])
        command = entry['command']
        
        # Show what the naming algorithm learned
        if 'badge' in command:
            console.print(f"├─ learned: [cyan]{output_file.stem}[/cyan] pattern")
            
            # Check for family promotion
            if any(char.isdigit() for char in output_file.stem):
                if 'b' in output_file.stem:
                    console.print(f"├─ promoted: batch family [yellow]b{output_file.stem[-1]}[/yellow]")
                elif 'h' in output_file.stem:
                    console.print(f"├─ promoted: hour family [blue]h{output_file.stem[-2:]}[/blue]")
            
            # Check for clean base extraction
            base = output_file.stem.rstrip('0123456789abcdefghijklmnopqrstuvwxyz')
            if base in ['cert', 'pass', 'equi']:
                console.print(f"├─ clean base: [green]{base}[/green] (avoided accidents)")
        
        elif 'equilibrium' in command or 'analyze' in command:
            console.print(f"├─ analysis pattern: [magenta]{output_file.stem}[/magenta]")
    
    console.print(f"└─ [dim]Run 'rie' again for alternative interpretations[/dim]")


def show_alternative_interpretations(recent_activity: List[Dict[str, Any]]):
    """Show alternative interpretations of recent naming"""
    console.print("🤔 [bold]Alternative Interpretation:[/bold]")
    
    # Find recent files that could be organized differently
    recent_files = [Path(entry['output_file']) for entry in recent_activity[-5:]]
    
    # Group by base name
    base_groups = {}
    for file in recent_files:
        base = file.stem.rstrip('0123456789abcdefghijklmnopqrstuvwxyz')
        if base not in base_groups:
            base_groups[base] = []
        base_groups[base].append(file)
    
    for base, files in base_groups.items():
        if len(files) >= 2:
            console.print(f"Maybe [cyan]{base}[/cyan] files want their own directory?")
            console.print(f"├─ current: {', '.join([f.name for f in files])}")
            console.print(f"└─ alternative: [green]{base}/[/green] → {', '.join([f.stem.replace(base, '') or '1' for f in files])}")
    
    console.print(f"\n[dim]Run 'rie' again to enter REPL mode[/dim]")


def suggest_next_actions(recent_activity: List[Dict[str, Any]]):
    """Suggest next actions based on older activity"""
    console.print("💡 [bold]Suggestions:[/bold]")
    
    # Check for cleanup opportunities
    output_dirs = list(rie_context.output_dir.iterdir())
    
    for dir_path in output_dirs:
        if dir_path.is_dir():
            files = list(dir_path.glob("*"))
            if len(files) > 10:
                console.print(f"📁 [yellow]{dir_path.name}/[/yellow] has {len(files)} files - consider organizing")
    
    # Suggest based on time of day
    current_hour = int(time.strftime("%H"))
    if 9 <= current_hour <= 17:
        console.print("⏰ Working hours - try [cyan]rie badge cert -g[/cyan] for quick iteration")
    else:
        console.print("🌙 Off hours - good time for [cyan]rie analyze[/cyan] or cleanup")
    
    console.print(f"\n[dim]Or run 'rie --help' for all commands[/dim]")


if __name__ == "__main__":
    app()
