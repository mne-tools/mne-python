#!/bin/bash
set -eo pipefail -x

# old and minimal use conda
if [[ "$MNE_CI_KIND" == "pip"* ]]; then
    echo "Setting pip env vars for $MNE_CI_KIND"
    echo "MNE_QT_BACKEND=PyQt6" >> $GITHUB_ENV
    # We should test an eager import somewhere, might as well be here
    echo "EAGER_IMPORT=true" >> $GITHUB_ENV
else  # conda-like
    echo "Setting conda env vars for $MNE_CI_KIND"
    if [[ "$MNE_CI_KIND" == "old" ]]; then
        echo "CONDA_ENV=tools/environment_old.yml" >> $GITHUB_ENV
        echo "MNE_IGNORE_WARNINGS_IN_TESTS=true" >> $GITHUB_ENV
        echo "MNE_SKIP_NETWORK_TESTS=1" >> $GITHUB_ENV
        echo "MNE_QT_BACKEND=PyQt5" >> $GITHUB_ENV
    elif [[ "$MNE_CI_KIND" == "minimal" ]]; then
        echo "CONDA_ENV=tools/environment_minimal.yml" >> $GITHUB_ENV
        echo "MNE_QT_BACKEND=PySide6" >> $GITHUB_ENV
    else  # conda, mamba (use warning level for completeness)
        echo "CONDA_ENV=environment.yml" >> $GITHUB_ENV
        echo "MNE_LOGGING_LEVEL=warning" >> $GITHUB_ENV
        echo "MNE_QT_BACKEND=PySide6" >> $GITHUB_ENV
    fi
fi
set +x
