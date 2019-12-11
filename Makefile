# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= py.test
CTAGS ?= ctags
CODESPELL_SKIPS ?= "doc/auto_*,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,install_mne_c.rst,plot_*.rst,*.rst.txt,c_EULA.rst*,*.html,gdf_encodes.txt,*.svg"
CODESPELL_DIRS ?= mne/ doc/ tutorials/ examples/
all: clean inplace test test-doc

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

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

wheel:
	$(PYTHON) setup.py sdist bdist_wheel

wheel_quiet:
	$(PYTHON) setup.py -q sdist bdist_wheel

sample_data:
	@python -c "import mne; mne.datasets.sample.data_path(verbose=True);"

testing_data:
	@python -c "import mne; mne.datasets.testing.data_path(verbose=True);"

pytest: test

test: in
	rm -f .coverage
	$(PYTESTS) -m 'not ultraslowtest' mne

test-verbose: in
	rm -f .coverage
	$(PYTESTS) -m 'not ultraslowtest' mne --verbose

test-fast: in
	rm -f .coverage
	$(PYTESTS) -m 'not slowtest' mne

test-full: in
	rm -f .coverage
	$(PYTESTS) mne

test-no-network: in
	sudo unshare -n -- sh -c 'MNE_SKIP_NETWORK_TESTS=1 py.test mne'

test-no-testing-data: in
	@MNE_SKIP_TESTING_DATASET_TESTS=true \
	$(PYTESTS) mne

test-no-sample-with-coverage: in testing_data
	rm -rf coverage .coverage
	$(PYTESTS) --cov=mne --cov-report html:coverage

test-doc: sample_data testing_data
	$(PYTESTS) --doctest-modules --doctest-ignore-import-errors --doctest-glob='*.rst' ./doc/ --ignore=./doc/auto_examples --ignore=./doc/auto_tutorials --ignore=./doc/_build --fulltrace

test-coverage: testing_data
	rm -rf coverage .coverage
	$(PYTESTS) --cov=mne --cov-report html:coverage
# whats the difference with test-no-sample-with-coverage?

test-mem: in testing_data
	ulimit -v 1097152 && $(PYTESTS) mne

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

upload-pipy:
	python setup.py sdist bdist_egg register upload

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count mne examples tutorials setup.py; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle mne

docstring:
	@echo "Running docstring tests"
	@$(PYTESTS) --doctest-modules mne/tests/test_docstring_parameters.py

check-manifest:
	check-manifest --ignore .circleci*,doc,logo,mne/io/*/tests/data*,mne/io/tests/data,mne/preprocessing/tests/data,.DS_Store

check-readme: clean wheel_quiet
	twine check dist/*

nesting:
	@echo "Running import nesting tests"
	@$(PYTESTS) mne/tests/test_import_nesting.py

pep:
	@$(MAKE) -k flake pydocstyle docstring codespell-error check-manifest nesting check-readme

manpages:
	@echo "I: generating manpages"
	set -e; mkdir -p _build/manpages && \
	cd bin && for f in mne*; do \
			descr=$$(grep -h -e "^ *'''" -e 'DESCRIP =' $$f -h | sed -e "s,.*' *\([^'][^']*\)'.*,\1,g" | head -n 1); \
	PYTHONPATH=../ \
			help2man -n "$$descr" --no-discard-stderr --no-info --version-string "$(uver)" ./$$f \
			>| ../_build/manpages/$$f.1; \
	done

build-doc-dev:
	cd doc; make clean
	cd doc; DISPLAY=:1.0 xvfb-run -n 1 -s "-screen 0 1280x1024x24 -noreset -ac +extension GLX +render" make html_dev

build-doc-stable:
	cd doc; make clean
	cd doc; DISPLAY=:1.0 xvfb-run -n 1 -s "-screen 0 1280x1024x24 -noreset -ac +extension GLX +render" make html_stable

docstyle: pydocstyle
