#!/bin/bash -ef
# TODO: I think that DEPS is cruft. Its not set anywhere??
if [ "${DEPS}" != "minimal" ]; then
	run_python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
	run_python -c "import mne; mne.datasets.misc.data_path(verbose=True)";
fi
