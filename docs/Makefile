# Minimal Sphinx Makefile
SPHINXBUILD = sphinx-build
SOURCEDIR = source
BUILDDIR = build

.PHONY: help clean html

help:
	@echo "Makefile for Sphinx documentation"
	@echo "  make html    # build HTML docs"
	@echo "  make clean   # remove build directory"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
