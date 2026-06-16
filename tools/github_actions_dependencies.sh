#!/bin/bash -ef

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ONLY_BINARY_ARG="--only-binary=:all:"
STD_ARGS="--progress-bar off --upgrade"
INSTALL_ARGS="-e"
EXTRAS=""
GROUP="test_extra"
if [ ! -z "$CONDA_ENV" ]; then
	echo "Uninstalling MNE for CONDA_ENV=${CONDA_ENV}"
	echo "::group::Uninstalling MNE"
	conda remove -c conda-forge --force -y mne-base
	echo "::endgroup::"
	# If not on windows, do a non-editable install
	if [[ "${CI_OS_NAME}" != "windows"* ]]; then
		INSTALL_ARGS=""
	fi
	EXTRAS="[hdf5]"
elif [[ "${MNE_CI_KIND}" == "minimal" ]]; then
	GROUP="test"
	STD_ARGS="--progress-bar off ${MNE_QT_BACKEND}"
	echo "::group::Upgrading pip installation"
	python -m pip install --upgrade pip setuptools
	echo "::endgroup::"
elif [[ "${MNE_CI_KIND}" == "old" ]]; then
	GROUP=""  # group "test" already included when pylock file generated
	STD_ARGS="--progress-bar off"
	echo "::group::Syncing old environment dependencies from lockfile using uv"
	uv pip sync ${SCRIPT_DIR}/pylock.ci-old.toml
	uv pip install pip tomlkit ${MNE_QT_BACKEND}
	echo "::endgroup::"
elif [[ "${MNE_CI_KIND}" == "pip-ft" ]]; then
	# This one is free-threaded so can't have numba or PySide6/PyQt6 as of 2025/07/15
	EXTRAS="[hdf5]"
	INSTALL_KIND="test"
elif [[ "${MNE_CI_KIND}" == "pip" ]]; then
	EXTRAS="[full-pyside6]"
	python -m pip install --upgrade pip setuptools
else
	test "${MNE_CI_KIND}" == "pip-pre"
	python -m pip install $STD_ARGS pip setuptools
	STD_ARGS="$STD_ARGS --pre"
	${SCRIPT_DIR}/install_pre_requirements.sh
fi
echo ""

<<<<<<< HEAD
echo "Installing test dependencies using pip"
python -m pip install $STD_ARGS $INSTALL_ARGS .[$INSTALL_KIND]
if [[ "${MNE_CI_KIND}" == "pip" ]]; then
	python -m pip uninstall -yq pytest-qt
fi
=======
# Make sure we only pass non-empty groups argument
if [ -z "$GROUP" ]; then
	GROUP_ARG=""
else
	GROUP_ARG="--group=$GROUP"
fi

if [[ "${MNE_CI_KIND}" != "old" ]]; then
	echo "::group::Installing test dependencies using pip"
else
	echo "::group::Installing MNE in development mode using pip"
fi
set -x
python -m pip install $STD_ARGS $ONLY_BINARY_ARG $INSTALL_ARGS .$EXTRAS $GROUP_ARG
echo "::endgroup::"
>>>>>>> upstream/main
