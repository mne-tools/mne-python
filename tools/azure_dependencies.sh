#!/bin/bash

set -eo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STD_ARGS="--progress-bar off --upgrade --only-binary=:all:"
python -m pip install $STD_ARGS pip setuptools wheel
if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install $STD_ARGS --only-binary="numba,llvmlite,numpy,scipy,vtk,dipy,openmeeg" -e .[full-pyside6] --group=test "mne-qt-browser @ https://github.com/mne-tools/mne-qt-browser/archive/refs/heads/main.zip"
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	${SCRIPT_DIR}/install_pre_requirements.sh
	python -m pip install $STD_ARGS --pre -e . --group=test_extra
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
