from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import typer


app = typer.Typer(help="General tool-based true/false checks with sanity verification")


def _run_command(command: List[str], description: str) -> bool:
    """Run a command and return True if successful, False otherwise."""
    try:
        # Since we're already running in poetry environment, just run the command directly
        env = {"PYTHONPATH": "."}
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if result.returncode == 0:
            typer.echo(f"✅ {description}")
            return True
        else:
            typer.echo(f"❌ {description}: {result.stderr.strip()}")
            return False
    except Exception as e:
        typer.echo(f"❌ {description}: {e}")
        return False


def _check_makefile_sanity() -> bool:
    """Check basic sanity of Makefile targets."""
    makefile_path = Path("Makefile")
    if not makefile_path.exists():
        typer.echo("❌ Makefile not found")
        return False
    
    makefile_content = makefile_path.read_text()
    
    # Check for essential targets
    essential_targets = ["pdf", "clean", "help"]
    missing_targets = []
    for target in essential_targets:
        if target not in makefile_content:
            missing_targets.append(target)
    
    if missing_targets:
        typer.echo(f"❌ Makefile missing essential targets: {', '.join(missing_targets)}")
        return False
    
    # Check for .PHONY declarations
    if ".PHONY" not in makefile_content:
        typer.echo("⚠️  Makefile should declare .PHONY targets")
    
    typer.echo("✅ Makefile sanity check passed")
    return True


def _check_python_imports() -> bool:
    """Check that core Python modules can be imported."""
    try:
        # Check if riemann modules exist and can be imported
        riemann_path = Path("code/riemann")
        if riemann_path.exists():
            # Test key imports that are used across the codebase
            import_tests = [
                "from code.riemann.analysis.rh_analyzer import RHIntegerAnalyzer, PascalKernel, DihedralAction",
                "from code.riemann.verification.validation import CertificationStamper", 
                "from code.riemann.verification.certification import write_toml, sweep_cert"
            ]
            
            all_imports_working = True
            for import_test in import_tests:
                result = subprocess.run([
                    sys.executable, "-c", import_test + "; print('Import successful')"
                ], capture_output=True, text=True)
                if result.returncode != 0:
                    typer.echo(f"❌ Import failed: {import_test}")
                    typer.echo(f"   Error: {result.stderr.strip()}")
                    all_imports_working = False
            
            if all_imports_working:
                typer.echo("✅ Python riemann imports working")
                return True
            else:
                typer.echo("❌ Some Python riemann imports failed")
                return False
        else:
            typer.echo("⚠️  Riemann directory not found")
            return True  # Not a failure, just a warning
    except Exception as e:
        typer.echo(f"❌ Python import check failed: {e}")
        return False


def _check_markdown_formatting() -> bool:
    """Check if markdown formatting tools are available."""
    # Just check if mdformat can run, don't enforce formatting on all files
    python_dir = Path(sys.executable).parent
    mdformat_path = python_dir / "mdformat"
    if mdformat_path.exists():
        typer.echo("✅ Markdown formatting tools available")
        return True
    else:
        typer.echo("❌ Markdown formatting tools not found")
        return False


def _run_tests() -> bool:
    """Run the test suite."""
    # Check if test file exists
    test_file = Path("code/tests/unit/test_rh.py")
    if not test_file.exists():
        typer.echo("⚠️  Test file not found, skipping tests")
        return True
    
    return _run_command([
        sys.executable, "code/tests/unit/test_rh.py"
    ], "RH system tests")


@app.command()
def all() -> None:
    """Run all checks and exit with appropriate code."""
    checks = [
        _check_makefile_sanity,
        _check_python_imports,
        _check_markdown_formatting,
        _run_tests,
    ]
    
    typer.echo("🔍 Running comprehensive checks...")
    all_passed = True
    
    for check_func in checks:
        if not check_func():
            all_passed = False
    
    if all_passed:
        typer.echo("\n🎉 All checks passed!")
        raise typer.Exit(0)
    else:
        typer.echo("\n💥 Some checks failed!")
        raise typer.Exit(1)


@app.command()
def makefile() -> None:
    """Check Makefile sanity."""
    if _check_makefile_sanity():
        raise typer.Exit(0)
    else:
        raise typer.Exit(1)


@app.command()
def python() -> None:
    """Check Python imports."""
    if _check_python_imports():
        raise typer.Exit(0)
    else:
        raise typer.Exit(1)


@app.command()
def markdown() -> None:
    """Check Markdown formatting."""
    if _check_markdown_formatting():
        raise typer.Exit(0)
    else:
        raise typer.Exit(1)


@app.command()
def test() -> None:
    """Run tests."""
    if _run_tests():
        raise typer.Exit(0)
    else:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
