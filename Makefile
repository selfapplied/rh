# Makefile for Riemann Hypothesis Equilibrium Geometry Paper

.PHONY: pdf open clean help check markdown riemann cert install fix test prove

# Default target - show available options
.DEFAULT_GOAL := help

# Install dependencies
install:
	@echo "Installing dependencies..."
	poetry install

# Master rule - activate environment and run the actual target
pdf open clean riemann cert check markdown help fix:
	@echo "Activating Poetry environment and running: $@"
	@$(SHELL) -c "source $$(poetry env info --path)/bin/activate && $(MAKE) _$(MAKECMDGOALS)"

# Actual implementations (prefixed with _)

# Build the complete PDF with visualizations
_pdf: _riemann_hypothesis_equilibrium_geometry.pdf

# Build and open the PDF
_open: _pdf
	open riemann_hypothesis_equilibrium_geometry.pdf

# Clean build artifacts
_clean:
	rm -rf .out
	rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

# Build the PDF using latexmk (includes visualization generation)
_riemann_hypothesis_equilibrium_geometry.pdf: riemann_hypothesis_equilibrium_geometry.tex
	@echo "Generating CE1 visualizations..."
	@mkdir -p docs/readme
	python code/riemann/ce1.py --type both || echo "Note: Some visualizations may require dependencies"
	@echo "Copying latest visualizations to docs/readme/ directory..."
	@ls -t .out/ce1_visualization/ce1_involution_*.png 2>/dev/null | head -1 | xargs -I {} cp {} docs/readme/ce1_involution.png 2>/dev/null || echo "No new ce1_involution images"
	@ls -t .out/ce1_visualization/involution_geometry_*.png 2>/dev/null | head -1 | xargs -I {} cp {} docs/readme/involution_geometry.png 2>/dev/null || echo "No new involution_geometry images"
	@ls -t .out/ce1_visualization/zeta_landscape_*.png 2>/dev/null | head -1 | xargs -I {} cp {} docs/readme/zeta_landscape.png 2>/dev/null || echo "No new zeta_landscape images"
	@echo "Generating prism visualizations..."
	python code/tools/prism.py --facet 8 --out .out/prism_visualization/ || echo "Note: Prism visualizations may require dependencies"
	@ls -t .out/prism_visualization/*-poly-curved.png 2>/dev/null | head -1 | xargs -I {} cp {} docs/readme/prism_curved.png 2>/dev/null || echo "No new prism_curved images"
	@ls -t .out/prism_visualization/*-poly-surface.png 2>/dev/null | head -1 | xargs -I {} cp {} docs/readme/prism_surface.png 2>/dev/null || echo "No new prism_surface images"
	@echo "Visualizations generated and copied successfully"
	@echo "Building Riemann Hypothesis Equilibrium Geometry paper..."
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -shell-escape -output-directory=.out riemann_hypothesis_equilibrium_geometry.tex
	@echo "Copying PDF to repo root..."
	cp .out/riemann_hypothesis_equilibrium_geometry.pdf .
	@echo "Paper built successfully: riemann_hypothesis_equilibrium_geometry.pdf"

# Run riemann analysis
_riemann:
	@echo "Running riemann analysis..."
	python -m code.riemann.rieman

# Run certification system
_cert:
	@echo "Running certification system..."
	python -m tools.certification --help

# General check system (replaces md-check and test)
_check:
	@echo "Running comprehensive checks..."
	@echo "Checking import organization (no changes)..."
	isort --check-only --diff code/ tools/
	@echo "Checking for unused imports and variables (no changes)..."
	autoflake --check --recursive code/ tools/
	@echo "Running comprehensive checks..."
	python -m tools.check all

# Fix common import and code issues automatically
_fix:
	@echo "Fixing common import and code issues..."
	@echo "Fixing broken imports using advanced tool..."
	python .working/fix_imports.py code/ tools/
	@echo "Organizing imports (applying changes)..."
	isort code/ tools/
	@echo "Removing unused imports and variables (applying changes)..."
	autoflake --in-place --recursive code/ tools/
	@echo "Fixes applied successfully!"

# Markdown utilities (renamed from md, absorbs link generation)
_markdown:
	@echo "Running comprehensive markdown processing..."
	python -m tools.markdown convert-wikilinks .
	python -m tools.markdown toc .
	python -m tools.markdown format .

# Help target
_help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies with Poetry"
	@echo "  pdf        - Build the complete PDF (with visualizations)"
	@echo "  check      - Run comprehensive checks (tests, formatting, sanity)"
	@echo "  markdown   - Process markdown (format, TOCs, links)"
	@echo "  riemann    - Run riemann analysis"
	@echo "  cert       - Run certification system"
	@echo "  open       - Build and open the PDF"
	@echo "  clean      - Clean build artifacts"
	@echo "  fix        - Fix common import and code issues"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Environment is automatically activated for all commands!"