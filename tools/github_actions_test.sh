#!/bin/bash -ef

# Remove numpydoc tests on older Python (builtin docstrings not as good)
if [ "${PYTHON_VERSION}" == "3.6" ]; then
	pip uninstall -yq numpydoc;
fi;

# Test run_tests_if_main
if [ "${DEPS}" == "minimal" ]; then
	pip uninstall -yq mne;
	pip install -e .;
	python mne/tests/test_evoked.py;
fi;

USE_DIRS="mne/"
if [ "${CI_OS_NAME}" != "osx" ]; then
  CONDITION="not ultraslowtest"
else
  CONDITION="not slowtest"
fi
echo 'pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml -vv ${USE_DIRS}'
pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml -vv ${USE_DIRS}

# run the minimal one with the testing data as well
if [ "${DEPS}" == "minimal" ]; then
	export MNE_SKIP_TESTING_DATASET_TESTS=false;
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
fi;
if [ "${DEPS}" == "minimal" ]; then
	pytest -m "${CONDITION}" --tb=short --cov=mne -vv ${USE_DIRS};
fi;
