#!/bin/bash -ef

if [ "${MNE_CI_KIND}" != "minimal" ]; then
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
	python -c "import mne; mne.datasets.misc.data_path(verbose=True)";
	# Make read-only to make sure we don't modify its contents
	TESTING_PATH=$(python -c "import mne; print(mne.datasets.testing.data_path(verbose=False))")
	echo "Testing data path: $TESTING_PATH"
	chmod -R a-w "$TESTING_PATH"
fi
