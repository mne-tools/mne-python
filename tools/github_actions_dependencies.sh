#!/bin/bash -ef

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STD_ARGS="--progress-bar off --upgrade"
INSTALL_ARGS="-e"
if [ ! -z "$CONDA_ENV" ]; then
	echo "Uninstalling MNE for CONDA_ENV=${CONDA_ENV}"
	# This will fail if mne-base is not in the env (like in our minimial/old envs, so ||true them):
	echo "::group::Uninstalling MNE"
	conda remove -c conda-forge --force -yq mne-base || true
	python -m pip uninstall -y mne || true
	echo "::endgroup::"
	# If using bare environment.yml and not on windows, do a non-editable install
	if [[ "${RUNNER_OS}" != "Windows" ]] && [[ "${CONDA_ENV}" != "environment_"* ]]; then
		INSTALL_ARGS=""
	fi
	# TODO: Until a PyVista release supports VTK 9.5+
	STD_ARGS="$STD_ARGS https://github.com/pyvista/pyvista/archive/refs/heads/main.zip"
	# If on minimal or old, just install testing deps
	if [[ "${CONDA_ENV}" == "environment_"* ]]; then
		GROUP="test"
		EXTRAS=""
		STD_ARGS="--progress-bar off"
	else
		GROUP="test_extra"
		EXTRAS="[hdf5]"
	fi
elif [[ "${MNE_CI_KIND}" == "pip" ]]; then
	GROUP="test_extra"
	EXTRAS="[full-pyside6]"
else
	test "${MNE_CI_KIND}" == "pip-pre"
	STD_ARGS="$STD_ARGS --pre"
	${SCRIPT_DIR}/install_pre_requirements.sh || exit 1
	GROUP="test_extra"
	EXTRAS=""
fi
echo ""

echo "::group::Installing test dependencies using pip"
set -x
python -m pip install $STD_ARGS $INSTALL_ARGS .$EXTRAS --group=$GROUP
set +x
echo "::endgroup::"
