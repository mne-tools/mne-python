#!/bin/bash
set -eo pipefail

# old and minimal use conda
if [[ "$MNE_CI_KIND" == "pip"* ]] || [[ "$MNE_CI_KIND" == "minimal" ]]; then
    echo "Setting pip env vars for $MNE_CI_KIND"
    if [[ "$MNE_CI_KIND" == "pip-pre" ]]; then
        # We should test an eager import somewhere, might as well be here
        echo "EAGER_IMPORT=true" | tee -a $GITHUB_ENV
        # Make sure nothing unexpected is skipped
        echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm) dataset|CUDA not|Numba not|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
    fi
    echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "old" ]]; then
    echo "MNE_IGNORE_WARNINGS_IN_TESTS=true" | tee -a $GITHUB_ENV
    echo "MNE_SKIP_NETWORK_TESTS=1" | tee -a $GITHUB_ENV
    echo "MNE_QT_BACKEND=PyQt5" | tee -a $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "conda" ]]; then
    echo "Setting conda env vars for $MNE_CI_KIND"
    echo "CONDA_ENV=environment.yml" | tee -a $GITHUB_ENV
    echo "MNE_LOGGING_LEVEL=warning" | tee -a $GITHUB_ENV
    echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm) dataset|CUDA not|PySide6 causes segfaults|Accelerate|Flakey verbose behavior).*" | tee -a $GITHUB_ENV
    # Our cache_dir test has problems when the path is too long, so prevent it from getting too long
    if [[ "$CI_OS_NAME" == "macos"* ]]; then
        echo "PYTEST_DEBUG_TEMPROOT=/tmp" | tee -a $GITHUB_ENV
    fi
    echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
else
    echo "✕ ERROR: Unrecognized MNE_CI_KIND=${MNE_CI_KIND}"
    exit 1
fi
if [[ "$CI_OS_NAME" == "windows"* ]]; then
    echo "MNE_IS_OSMESA=true" | tee -a $GITHUB_ENV
fi
