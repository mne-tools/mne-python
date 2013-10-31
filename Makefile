# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

sample_data: $(CURDIR)/examples/MNE-sample-data/MEG/sample/sample_audvis_raw.fif
	@echo "Target needs sample data"

$(CURDIR)/examples/MNE-sample-data/MEG/sample/sample_audvis_raw.fif:
	wget ftp://surfer.nmr.mgh.harvard.edu/pub/data/MNE-sample-data-processed.tar.gz
	tar xvzf MNE-sample-data-processed.tar.gz
	mv MNE-sample-data examples/
	ln -s ${PWD}/examples/MNE-sample-data ${PWD}/MNE-sample-data -f

test: in sample_data
	$(NOSETESTS) mne

test-no-sample: in
	@MNE_SKIP_SAMPLE_DATASET_TESTS=true \
	$(NOSETESTS) mne


test-no-sample-with-coverage: in
	rm -rf coverage .coverage
	@MNE_SKIP_SAMPLE_DATASET_TESTS=true \
	$(NOSETESTS) --with-coverage --cover-package=mne --cover-html --cover-html-dir=coverage

test-doc: sample_data
	$(NOSETESTS) --with-doctest --doctest-tests --doctest-extension=rst doc/ doc/source/

test-coverage: sample_data
	rm -rf coverage .coverage
	$(NOSETESTS) --with-coverage --cover-package=mne --cover-html --cover-html-dir=coverage

test-profile: sample_data
	$(NOSETESTS) --with-profile --profile-stats-file stats.pf mne
	hotshot2dot stats.pf | dot -Tpng -o profile.png

test-mem: in sample_data
	ulimit -v 1097152 && $(NOSETESTS)

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

upload-pipy:
	python setup.py sdist bdist_egg register upload

codespell:
	# The *.fif had to be there twice to be properly ignored (!)
	codespell.py -w -i 3 -S="*.fif,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.coverage,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii" ./dictionary.txt -r .

manpages:
	@echo "I: generating manpages"
	set -e; mkdir -p build/manpages && \
	cd bin && for f in *; do \
			descr=$$(grep -h -e "^ *'''" -e 'DESCRIP =' $$f -h | sed -e "s,.*' *\([^'][^']*\)'.*,\1,g" | head -n 1); \
	PYTHONPATH=../ \
			help2man -n "$$descr" --no-discard-stderr --no-info --version-string "$(uver)" ./$$f \
			>| ../build/manpages/$$f.1; \
	done
