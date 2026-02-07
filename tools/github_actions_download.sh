#!/bin/bash -ef

if [[ ${MNE_CI_KIND} == "old" ]]; then
    source .venv/bin/activate
fi

if [ "${DEPS}" != "minimal" ]; then
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
	python -c "import mne; mne.datasets.misc.data_path(verbose=True)";
fi
