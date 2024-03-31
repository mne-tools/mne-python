#!/bin/bash -ef

set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STD_ARGS="--progress-bar off --upgrade"
INSTALL_ARGS="-e"
INSTALL_KIND="test_extra,hdf5"
if [ ! -z "$CONDA_ENV" ]; then
	echo "Uninstalling MNE for CONDA_ENV=${CONDA_ENV}"
	conda remove -c conda-forge --force -yq mne-base
	python -m pip uninstall -y mne
	if [[ "${RUNNER_OS}" != "Windows" ]]; then
		INSTALL_ARGS=""
	fi
elif [ ! -z "$CONDA_DEPENDENCIES" ]; then
	echo "Using Mamba to install CONDA_DEPENDENCIES=${CONDA_DEPENDENCIES}"
	mamba install -y $CONDA_DEPENDENCIES
	# for compat_minimal and compat_old, we don't want to --upgrade
	STD_ARGS="--progress-bar off"
	INSTALL_KIND="test"
else
	test "${MNE_CI_KIND}" == "pip-pre"
	STD_ARGS="$STD_ARGS --pre"
	${SCRIPT_DIR}/install_pre_requirements.sh
	INSTALL_KIND="test_extra"
fi
echo ""

echo "Installing test dependencies using pip"
python -m pip install $STD_ARGS $INSTALL_ARGS .[$INSTALL_KIND]
