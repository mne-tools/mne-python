#!/bin/bash
set -eo pipefail -x

# old and minimal use conda
if [[ "$MNE_CI_KIND" == "old" ]]; then
    echo "Setting env vars for old"
    echo "CONDA_DEPENDENCIES=numpy=1.20.2 scipy=1.6.3 matplotlib=3.4 pandas=1.2.4 scikit-learn=0.24.2" >> $GITHUB_ENV
    echo "MNE_IGNORE_WARNINGS_IN_TESTS=true" >> $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "minimal" ]]; then
    echo "Setting env vars for minimal"
    echo "CONDA_DEPENDENCIES=numpy scipy matplotlib" >> $GITHUB_ENV
elif [[ "$MNE_CI_KIND" != "pip"* ]]; then
    echo "Setting env vars for $MNE_CI_KIND"
    echo "CONDA_ENV=environment.yml" >> $GITHUB_ENV
    echo "CONDA_ACTIVATE_ENV=mne" >> $GITHUB_ENV
else
    echo "Not setting env vars for $MNE_CI_KIND"
fi
set +x
