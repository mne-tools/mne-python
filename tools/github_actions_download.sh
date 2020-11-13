#!/bin/bash -ef

if [ ! -z "$CONDA_ENV" ]; then
	CONDA_BASE=$(conda info --base)
	source $CONDA_BASE/etc/profile.d/conda.sh
	conda activate mne
fi

if [ "${DEPS}" != "minimal" ]; then
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
fi
