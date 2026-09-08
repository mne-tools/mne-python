#!/bin/bash -ef

# Datasets the test suite needs. Downstream projects that reuse this script (e.g.
# MNE-BIDS) can override it to fetch only what they actually use; whatever is
# listed here must also be reflected in the actions/cache key of the caller.
MNE_CI_DATASETS="${MNE_CI_DATASETS:-testing misc}"

if [ "${MNE_CI_KIND}" != "minimal" ]; then
	# These are cached, so this only really downloads when a dataset version changes
	for DATASET in ${MNE_CI_DATASETS}; do
		python -uc "import mne; mne.datasets.${DATASET}.data_path(verbose=True)"
		# Make read-only to make sure we don't modify its contents
		DATASET_PATH=$(python -c "import mne; print(mne.datasets.${DATASET}.data_path(verbose=False))")
		echo "${DATASET} data path: $DATASET_PATH"
		chmod -R a-w "$DATASET_PATH"
	done
fi
