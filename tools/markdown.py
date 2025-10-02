from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import typer


app = typer.Typer(help="Utilities to format Markdown, manage TOCs, and check links (GitHub-friendly)")


def _run(command: List[str]) -> int:
    # Use the current Python executable's directory to find tools
    python_dir = Path(sys.executable).parent
    full_command = [str(python_dir / "mdformat") if cmd == "mdformat" else cmd for cmd in command]
    completed = subprocess.run(full_command)
    return completed.returncode


def _iter_markdown_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix.lower() in {".md", ".markdown"}:
            yield root
        elif root.is_dir():
            for path in root.rglob("*.md"):
                if "/.git/" in str(path):
                    continue
                yield path


def _github_slug(text: str) -> str:
    # Matches GitHub slugification reasonably well for headings
    lowered = text.strip().lower()
    # Remove punctuation except hyphens and spaces
    lowered = re.sub(r"[\t\n\r`~!@#$%^&*()=+[{]}\\|;:'\",<.>/?]", "", lowered)
    # Collapse whitespace to single hyphens
    lowered = re.sub(r"\s+", "-", lowered)
    # Collapse multiple hyphens
    lowered = re.sub(r"-+", "-", lowered)
    return lowered


@app.command()
def format(
    paths: List[Path] = typer.Argument(..., help="Files or directories to format"),
    check: bool = typer.Option(False, "--check", help="Don't write changes; exit nonzero if changes would be made"),
) -> None:
    """Format Markdown with mdformat (GFM)."""
    # Check if mdformat is available in the current environment
    python_dir = Path(sys.executable).parent
    mdformat_path = python_dir / "mdformat"
    if not mdformat_path.exists():
        typer.echo("mdformat not found. Install with: poetry add -G dev mdformat mdformat-gfm mdformat-toc mdformat-frontmatter", err=True)
        raise typer.Exit(127)

    argv: List[str] = ["mdformat"]
    if check:
        argv += ["--check"]
    argv += [str(p) for p in paths]
    raise typer.Exit(_run(argv))


@app.command()
def toc(
    paths: List[Path] = typer.Argument(..., help="Files or directories where TOC markers are ensured/updated"),
    insert: bool = typer.Option(True, "--insert/--no-insert", help="Insert mdformat-toc markers if missing"),
) -> None:
    """Ensure mdformat-toc markers exist, then update TOCs via mdformat."""
    files = list(_iter_markdown_files(paths))
    if insert:
        for f in files:
            text = f.read_text(encoding="utf-8")
            if "mdformat-toc" not in text:
                lines = text.splitlines()
                # Find first ATX heading to place TOC after
                insert_idx: Optional[int] = None
                for i, line in enumerate(lines):
                    if line.lstrip().startswith("#"):
                        insert_idx = i + 1
                        break
                toc_block = [
                    "",
                    "<!-- mdformat-toc start --><details>\n<summary>Table of contents</summary>",
                    "",
                    "<!-- mdformat-toc end --></details>",
                    "",
                ]
                if insert_idx is None:
                    new_lines = toc_block + lines
                else:
                    new_lines = lines[:insert_idx] + toc_block + lines[insert_idx:]
                f.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")

    # Update TOC via mdformat (requires mdformat-toc plugin)
    python_dir = Path(sys.executable).parent
    mdformat_path = python_dir / "mdformat"
    if not mdformat_path.exists():
        typer.echo("mdformat not found. Install with: poetry add -G dev mdformat mdformat-gfm mdformat-toc mdformat-frontmatter", err=True)
        raise typer.Exit(127)
    argv: List[str] = ["mdformat"] + [str(p) for p in paths]
    raise typer.Exit(_run(argv))


@app.command("check-links")
def check_links(
    globs: List[str] = typer.Argument(..., help="Glob(s) of markdown files to scan"),
    exclude_mail: bool = typer.Option(True, "--exclude-mail/--no-exclude-mail", help="Exclude mailto: links"),
    offline: bool = typer.Option(False, "--offline", help="Only check local links"),
) -> None:
    """Run lychee to check links."""
    if shutil.which("lychee") is None:
        typer.echo("lychee not found. Install via: brew install lychee", err=True)
        raise typer.Exit(127)
    argv: List[str] = ["lychee", "--no-progress"]
    if exclude_mail:
        argv.append("--exclude-mail")
    if offline:
        argv.append("--offline")
    argv += globs
    raise typer.Exit(_run(argv))


@app.command("convert-wikilinks")
def convert_wikilinks(
    paths: List[Path] = typer.Argument(..., help="Files or directories to convert [[wikilinks]] to relative links"),
) -> None:
    """Convert [[path#Heading]] to [Heading](path#github-slug) in-place."""
    wikilink_pattern = re.compile(r"\[\[([^\]]+)\]\]")
    for f in _iter_markdown_files(paths):
        text = f.read_text(encoding="utf-8")
        def _replace(match: re.Match[str]) -> str:
            target = match.group(1)
            if "|" in target:
                target, label = target.split("|", 1)
            else:
                label = None
            if "#" in target:
                path_part, heading = target.split("#", 1)
            else:
                path_part, heading = target, None
            display = label or (heading if heading else Path(path_part).stem)
            slug = f"#{_github_slug(heading)}" if heading else ""
            return f"[{display}]({path_part}{slug})"

        new_text = wikilink_pattern.sub(_replace, text)
        if new_text != text:
            f.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    app()


