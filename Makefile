# Makefile for Riemann Hypothesis Equilibrium Geometry Paper

# Main targets
.PHONY: pdf diagram open clean help

# Default target - show available options
.DEFAULT_GOAL := help

# Build the complete PDF
pdf: riemann_hypothesis_equilibrium_geometry.pdf

# Regenerate visualizations
diagram:
	@echo "Generating CE1 visualizations..."
	@mkdir -p readme
	@python3 ce1_simple_visualization.py || echo "Note: Some visualizations may require dependencies"
	@echo "Copying latest visualizations to readme/ directory..."
	@ls -t .out/ce1_visualization/ce1_involution_*.png 2>/dev/null | head -1 | xargs -I {} cp {} readme/ce1_involution.png 2>/dev/null || echo "No new ce1_involution images"
	@ls -t .out/ce1_visualization/involution_geometry_*.png 2>/dev/null | head -1 | xargs -I {} cp {} readme/involution_geometry.png 2>/dev/null || echo "No new involution_geometry images"
	@ls -t .out/ce1_visualization/zeta_landscape_*.png 2>/dev/null | head -1 | xargs -I {} cp {} readme/zeta_landscape.png 2>/dev/null || echo "No new zeta_landscape images"
	@echo "Visualizations generated and copied successfully"

# Build and open the PDF
open: pdf
	open riemann_hypothesis_equilibrium_geometry.pdf

# Clean build artifacts
clean:
	rm -rf .out
	rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

# Build the PDF using latexmk
riemann_hypothesis_equilibrium_geometry.pdf: riemann_hypothesis_equilibrium_geometry.tex
	@echo "Building Riemann Hypothesis Equilibrium Geometry paper..."
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -shell-escape -output-directory=.out riemann_hypothesis_equilibrium_geometry.tex
	@echo "Copying PDF to repo root..."
	cp .out/riemann_hypothesis_equilibrium_geometry.pdf .
	@echo "Paper built successfully: riemann_hypothesis_equilibrium_geometry.pdf"

# Help target
help:
	@echo "Available targets:"
	@echo "  pdf     - Build the complete PDF"
	@echo "  diagram - Regenerate CE1 visualizations"
	@echo "  open    - Build and open the PDF"
	@echo "  clean   - Clean build artifacts"
	@echo "  help    - Show this help message"
