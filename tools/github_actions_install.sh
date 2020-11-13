#!/bin/bash -ef

if [ ! -z "$CONDA_ENV" ]; then
	CONDA_BASE=$(conda info --base)
	source $CONDA_BASE/etc/profile.d/conda.sh
	conda activate mne
fi

python setup.py build
python setup.py install
