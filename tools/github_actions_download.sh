#!/bin/bash -ef

# TODO: I think that DEPS is cruft. Its not set anywhere??
if [ "${MNE_CI_KIND}" != "minimal" ]; then
	${PREFIX} python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
	${PREFIX} python -c "import mne; mne.datasets.misc.data_path(verbose=True)";
fi