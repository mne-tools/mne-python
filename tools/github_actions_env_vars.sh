#!/bin/bash
set -eo pipefail -x

# old and minimal use conda
if [[ "$MNE_CI_KIND" == "old" ]]; then
    echo "Setting conda env vars for old"
    echo "CONDA_ACTIVATE_ENV=true" >> $GITHUB_ENV
    echo "CONDA_DEPENDENCIES=numpy=1.20.2 scipy=1.6.3 matplotlib=3.4 pandas=1.2.4 scikit-learn=0.24.2" >> $GITHUB_ENV
    echo "MNE_IGNORE_WARNINGS_IN_TESTS=true" >> $GITHUB_ENV
    echo "MNE_SKIP_NETWORK_TESTS=1" >> $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "minimal" ]]; then
    echo "Setting conda env vars for minimal"
    echo "CONDA_ACTIVATE_ENV=true" >> $GITHUB_ENV
    echo "CONDA_DEPENDENCIES=numpy scipy matplotlib" >> $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "notebook" ]]; then
    echo "CONDA_ENV=environment.yml" >> $GITHUB_ENV
    echo "CONDA_ACTIVATE_ENV=mne" >> $GITHUB_ENV
    # TODO: This should work but breaks stuff...
    # echo "MNE_3D_BACKEND=notebook" >> $GITHUB_ENV
elif [[ "$MNE_CI_KIND" != "pip"* ]]; then  # conda, mamba (use warning level for completeness)
    echo "Setting conda env vars for $MNE_CI_KIND"
    echo "CONDA_ENV=environment.yml" >> $GITHUB_ENV
    echo "CONDA_ACTIVATE_ENV=mne" >> $GITHUB_ENV
    echo "MNE_QT_BACKEND=PyQt5" >> $GITHUB_ENV
    echo "MNE_LOGGING_LEVEL=warning" >> $GITHUB_ENV
else  # pip-like
    echo "Setting pip env vars for $MNE_CI_KIND"
    echo "MNE_QT_BACKEND=PyQt6" >> $GITHUB_ENV
fi
set +x
