#!/bin/bash

set -eo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STD_ARGS="--progress-bar off --upgrade"
python -m pip install $STD_ARGS pip setuptools wheel
if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install $STD_ARGS --only-binary="numba,llvmlite,numpy,scipy,vtk,dipy,openmeeg" -e .[test,full]
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	${SCRIPT_DIR}/install_pre_requirements.sh
	python -m pip install $STD_ARGS --pre -e .[test_extra]
	echo "##vso[task.setvariable variable=MNE_TEST_ALLOW_SKIP].*(Requires (spm|brainstorm) dataset|Requires MNE-C|CUDA not|Numba not| on Windows|MNE_FORCE_SERIAL|PySide6 causes segfaults).*"
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
