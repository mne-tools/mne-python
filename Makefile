# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags
CODESPELL_SKIPS ?= "*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb"
CODESPELL_DIRS ?= mne/ doc/ tutorials/ examples/
all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

sample_data:
	@python -c "import mne; mne.datasets.sample.data_path(verbose=True);"

testing_data:
	@python -c "import mne; mne.datasets.testing.data_path(verbose=True);"

test: in
	rm -f .coverage
	$(NOSETESTS) -a '!ultra_slow_test' mne

test-verbose: in
	rm -f .coverage
	$(NOSETESTS) -a '!ultra_slow_test' mne --verbose

test-fast: in
	rm -f .coverage
	$(NOSETESTS) -a '!slow_test' mne

test-full: in
	rm -f .coverage
	$(NOSETESTS) mne

test-no-network: in
	sudo unshare -n -- sh -c 'MNE_SKIP_NETWORK_TESTS=1 nosetests mne'

test-no-testing-data: in
	@MNE_SKIP_TESTING_DATASET_TESTS=true \
	$(NOSETESTS) mne

test-no-sample-with-coverage: in testing_data
	rm -rf coverage .coverage
	$(NOSETESTS) --with-coverage --cover-package=mne --cover-html --cover-html-dir=coverage

test-doc: sample_data testing_data
	$(NOSETESTS) --with-doctest --doctest-tests --doctest-extension=rst doc/

test-coverage: testing_data
	rm -rf coverage .coverage
	$(NOSETESTS) --with-coverage --cover-package=mne --cover-html --cover-html-dir=coverage

test-profile: testing_data
	$(NOSETESTS) --with-profile --profile-stats-file stats.pf mne
	hotshot2dot stats.pf | dot -Tpng -o profile.png

test-mem: in testing_data
	ulimit -v 1097152 && $(NOSETESTS)

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
		flake8 --count mne examples tutorials; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

codespell:  # running manually
	@codespell.py -w -i 3 -q 3 -S $(CODESPELL_SKIPS) -D ./dictionary.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell.py -i 0 -q 7 -S $(CODESPELL_SKIPS) -D ./dictionary.txt $(CODESPELL_DIRS)

pydocstyle:
	@pydocstyle

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

docstyle:
	@pydocstyle
