#!/bin/bash -ef

if [ "${MNE_CI_KIND}" != "minimal" ]; then
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
	python -c "import mne; mne.datasets.misc.data_path(verbose=True)";
fi
