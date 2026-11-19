#!/bin/bash

set -eo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# uv resolves and installs the full Windows dependency set in a fraction of the
# time pip takes; UV_SYSTEM_PYTHON/UV_CACHE_DIR are set in azure-pipelines.yml.
python -m pip install --progress-bar off --upgrade "uv>=0.9"
# uv, unlike pip, honours --only-binary=:all: for the local editable project and
# for direct-URL source archives too, so both need an explicit exemption
STD_ARGS="--upgrade --only-binary=:all: --no-binary=mne --no-binary=mne-qt-browser"
if [ "${TEST_MODE}" == "pip" ]; then
	uv pip install $STD_ARGS -e .[full-pyside6] --group=test_extra "mne-qt-browser @ https://github.com/mne-tools/mne-qt-browser/archive/refs/heads/main.zip"
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	${SCRIPT_DIR}/install_pre_requirements.sh
	uv pip install $STD_ARGS --prerelease=allow -e . --group=test_extra
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
