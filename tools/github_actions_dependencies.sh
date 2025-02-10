#!/bin/bash -ef

set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STD_ARGS="--progress-bar off --upgrade"
INSTALL_ARGS="-e"
INSTALL_KIND="test_extra,hdf5"
if [ ! -z "$CONDA_ENV" ]; then
	echo "Uninstalling MNE for CONDA_ENV=${CONDA_ENV}"
	# This will fail if mne-base is not in the env (like in our minimial/old envs, so ||true them):
	conda remove -c conda-forge --force -yq mne-base || true
	python -m pip uninstall -y mne || true
	# If using bare environment.yml and not on windows, do a non-editable install
	if [[ "${RUNNER_OS}" != "Windows" ]] && [[ "${CONDA_ENV}" != "environment_"* ]]; then
		INSTALL_ARGS=""
	fi
	# If on minimal or old, just install testing deps
	if [[ "${CONDA_ENV}" == "environment_"* ]]; then
		INSTALL_KIND="test"
		STD_ARGS="--progress-bar off"
	fi
elif [[ "${MNE_CI_KIND}" == "pip" ]]; then
	# Only used for 3.13 at the moment, just get test deps plus a few extras
	# that we know are available
	INSTALL_ARGS="nibabel scikit-learn numpydoc PySide6 mne-qt-browser pandas h5io mffpy defusedxml numba"
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
