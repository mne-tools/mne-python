#!/bin/bash -ef

if [ "${DEPS}" != "minimal" ]; then
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
	python -c "import mne; mne.datasets.misc.data_path(verbose=True)";
fi
