# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
PYTESTS ?= py.test
CODESPELL_SKIPS ?= "doc/_build,doc/auto_*,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,install_mne_c.rst,plot_*.rst,*.rst.txt,c_EULA.rst*,*.html,gdf_encodes.txt,*.svg,references.bib,*.css,*.edf,*.bdf,*.vhdr"
CODESPELL_DIRS ?= mne/ doc/ tutorials/ examples/
all: clean test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build dist

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

wheel:
	$(PYTHON) -m build -w

sample_data:
	@python -c "import mne; mne.datasets.sample.data_path(verbose=True);"

testing_data:
	@python -c "import mne; mne.datasets.testing.data_path(verbose=True);"

test-no-network: in
	sudo unshare -n -- sh -c 'MNE_SKIP_NETWORK_TESTS=1 py.test mne'

test-no-testing-data: in
	@MNE_SKIP_TESTING_DATASET_TESTS=true \
	$(PYTESTS) mne

test-doc: sample_data testing_data
	$(PYTESTS) --tb=short --cov=mne --cov-report=xml --cov-branch --doctest-modules --doctest-ignore-import-errors --doctest-glob='*.rst' ./doc/ --ignore=./doc/auto_examples --ignore=./doc/auto_tutorials --ignore=./doc/_build --ignore=./doc/conf.py --ignore=doc/sphinxext --fulltrace

pre-commit:
	@pre-commit run -a --show-diff-on-failure

# Aliases for stuff we used to support or users might think of
ruff: pre-commit
flake: pre-commit
pep: pre-commit

codespell:  # running manually
	@codespell --builtin clear,rare,informal,names,usage -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt --uri-ignore-words-list=bu $(CODESPELL_DIRS)

check-readme: clean wheel
	twine check dist/*

nesting:
	@echo "Running import nesting tests"
	@$(PYTESTS) mne/tests/test_import_nesting.py
