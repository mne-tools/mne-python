#!/bin/bash
set -eo pipefail -x

# old and minimal use conda
if [[ "$MNE_CI_KIND" == "old" ]]; then
    echo "Setting conda env vars for old"
    echo "CONDA_DEPENDENCIES=numpy=1.21.2 scipy=1.7.1 matplotlib=3.5.0 pandas=1.3.2 scikit-learn=1.0" >> $GITHUB_ENV
    echo "MNE_IGNORE_WARNINGS_IN_TESTS=true" >> $GITHUB_ENV
    echo "MNE_SKIP_NETWORK_TESTS=1" >> $GITHUB_ENV
    echo "MNE_QT_BACKEND=PyQt5" >> $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "minimal" ]]; then
    echo "Setting conda env vars for minimal"
    echo "CONDA_DEPENDENCIES=numpy scipy matplotlib" >> $GITHUB_ENV
    echo "MNE_QT_BACKEND=PyQt5" >> $GITHUB_ENV
elif [[ "$MNE_QT_BACKEND" != "pip"* ]]; then  # conda, mamba (use warning level for completeness)
    echo "Setting conda env vars for $MNE_CI_KIND"
    echo "CONDA_ENV=environment.yml" >> $GITHUB_ENV
    echo "MNE_QT_BACKEND=PyQt5" >> $GITHUB_ENV
    echo "MNE_LOGGING_LEVEL=warning" >> $GITHUB_ENV
else  # pip-like
    echo "Setting pip env vars for $MNE_CI_KIND"
    echo "MNE_QT_BACKEND=PyQt6" >> $GITHUB_ENV
    # We should test an eager import somewhere, might as well be here
    echo "EAGER_IMPORT=true" >> $GITHUB_ENV
fi
set +x
