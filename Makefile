# Makefile for Riemann Hypothesis Equilibrium Geometry Paper

# Main targets
.PHONY: pdf diagram open clean

# Build the complete PDF
pdf: riemann_hypothesis_equilibrium_geometry.pdf

# Regenerate any diagrams (placeholder for future diagram generation)
diagram:
	@echo "Generating diagrams..."
	@echo "Diagrams generated successfully"

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
	@echo "  diagram - Regenerate diagrams (placeholder)"
	@echo "  open    - Build and open the PDF"
	@echo "  clean   - Clean build artifacts"
	@echo "  help    - Show this help message"
