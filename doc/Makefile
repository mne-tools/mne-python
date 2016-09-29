# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d _build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean html dirhtml pickle json htmlhelp qthelp latex changes linkcheck doctest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html_stable      to make standalone HTML files (stable version)"
	@echo "  html_dev         to make standalone HTML files (dev version)"
	@echo "  html_dev-pattern to make standalone HTML files for one example dir (dev version)"
	@echo "  *-noplot     	  to make standalone HTML files without plotting"
	@echo "  dirhtml          to make HTML files named index.html in directories"
	@echo "  pickle           to make pickle files"
	@echo "  json             to make JSON files"
	@echo "  htmlhelp         to make HTML files and a HTML help project"
	@echo "  qthelp           to make HTML files and a qthelp project"
	@echo "  latex            to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes          to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck        to check all external links for integrity"
	@echo "  doctest          to run all doctests embedded in the documentation (if enabled)"

clean:
	-rm -rf _build/*
	-rm -rf auto_examples
	-rm -rf auto_tutorials
	-rm -rf generated
	-rm -rf *.stc
	-rm -rf *.fif
	-rm -rf *.nii.gz

html_stable:
	$(SPHINXBUILD) -D raise_gallery=1 -b html $(ALLSPHINXOPTS) _build/html_stable
	@echo
	@echo "Build finished. The HTML pages are in _build/html_stable."

html_dev:
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D raise_gallery=1 -D abort_on_example_error=1 -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html"

html_dev-pattern:
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D plot_gallery=1 -D raise_gallery=1 -D abort_on_example_error=1 -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html"

html_dev-noslow:
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D plot_gallery=fast -b html $(ALLSPHINXOPTS) _build/html

html_dev-noplot:
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) _build/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in _build/dirhtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) _build/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) _build/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) _build/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in _build/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) _build/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in _build/qthelp, like this:"
	@echo "# qcollectiongenerator _build/qthelp/MNE.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile _build/qthelp/MNE.qhc"

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) _build/latex
	@echo
	@echo "Build finished; the LaTeX files are in _build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) _build/changes
	@echo
	@echo "The overview file is in _build/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) _build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in _build/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) _build/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in _build/doctest/output.txt."

view:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/_build/html/index.html')"

show: view
