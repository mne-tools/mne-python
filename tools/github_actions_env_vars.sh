#!/bin/bash
set -eo pipefail -x

# old and minimal use conda
if [[ "$MNE_CI_KIND" == "pip"* ]]; then
    echo "Setting pip env vars for $MNE_CI_KIND"
    if [[ "$MNE_CI_KIND" == "pip-pre" ]]; then
        echo "MNE_QT_BACKEND=PyQt6" | tee -a $GITHUB_ENV
        # We should test an eager import somewhere, might as well be here
        echo "EAGER_IMPORT=true" | tee -a $GITHUB_ENV
        # Make sure nothing unexpected is skipped
        echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm) dataset|CUDA not|Numba not|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
    else
        echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
    fi
else  # conda-like
    echo "Setting conda env vars for $MNE_CI_KIND"
    if [[ "$MNE_CI_KIND" == "old" ]]; then
        echo "CONDA_ENV=tools/environment_old.yml" | tee -a $GITHUB_ENV
        echo "MNE_IGNORE_WARNINGS_IN_TESTS=true" | tee -a $GITHUB_ENV
        echo "MNE_SKIP_NETWORK_TESTS=1" | tee -a $GITHUB_ENV
        echo "MNE_QT_BACKEND=PyQt5" | tee -a $GITHUB_ENV
    elif [[ "$MNE_CI_KIND" == "minimal" ]]; then
        echo "CONDA_ENV=tools/environment_minimal.yml" | tee -a $GITHUB_ENV
        echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
    else  # conda, mamba (use warning level for completeness)
        echo "CONDA_ENV=environment.yml" | tee -a $GITHUB_ENV
        echo "MNE_LOGGING_LEVEL=warning" | tee -a $GITHUB_ENV
        echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
        # TODO: Also need "|unreliable on GitHub Actions conda" on macOS, but omit for now to make sure the failure actually shows up
        echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm) dataset|CUDA not|PySide6 causes segfaults|Accelerate|Flakey verbose behavior).*" | tee -a $GITHUB_ENV
    fi
fi
set +x
