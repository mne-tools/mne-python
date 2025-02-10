# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -nWT --keep-going
SPHINXBUILD   = sphinx-build
MPROF         = SG_STAMP_STARTS=true mprof run -E --python sphinx

# Internal variables.
ALLSPHINXOPTS   = -d _build/doctrees $(SPHINXOPTS) .

.PHONY: help clean html html-noplot html-pattern linkcheck linkcheck-grep doctest

# make with no arguments will build the first target by default, i.e., build standalone HTML files
first_target: html-noplot

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html             to make standalone HTML files"
	@echo "  html-memory      to make standalone HTML files while monitoring memory usage"
	@echo "  html-pattern     to make standalone HTML files for a specific filename pattern"
	@echo "  html-front       to make standalone HTML files with only the frontpage examples"
	@echo "  html-noplot      to make standalone HTML files without plotting"
	@echo "  clean            to clean HTML files"
	@echo "  linkcheck        to check all external links for integrity"
	@echo "  linkcheck-grep   to grep the linkcheck result"
	@echo "  doctest          to run all doctests embedded in the documentation (if enabled)"
	@echo "  view             to view the built HTML"

clean:
	-rm -rf _build sg_execution_times.rst auto_examples auto_tutorials generated *.stc *.fif *.nii.gz

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html."

html-memory:
	$(MPROF) -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html."

html-pattern:
	$(SPHINXBUILD) -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html"

html-pattern-memory:
	$(MPROF) -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html"

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html."

html-front:
	@PATTERN="\(30_mne_dspm_loreta.py\|50_decoding.py\|30_strf.py\|20_cluster_1samp_spatiotemporal.py\|20_visualize_evoked.py\)" make html-pattern

# Aliases for old methods
html_dev-pattern-memory: html-pattern-memory
html_dev-pattern: html-pattern
html_dev-noplot: html-noplot
html_dev-front: html-front

linkcheck:
	@$(SPHINXBUILD) -b linkcheck -D nitpicky=0 -q -D plot_gallery=0 -D exclude_patterns="cited.rst,whats_new.rst,configure_git.rst,_includes,changes/devel" -d _build/doctrees . _build/linkcheck

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) _build/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in _build/doctest/output.txt."

view:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/_build/html/sg_execution_times.html')"

show: view

serve:
	python -m http.server -d _build/html
