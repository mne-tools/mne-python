#!/bin/bash

set -eo pipefail

if [[ "${CI_OS_NAME}" == "ubuntu"* ]]; then
  CONDITION="not (ultraslowtest or pgtest)"
else  # macOS or Windows
  CONDITION="not (slowtest or pgtest)"
fi
if [ "${MNE_CI_KIND}" == "notebook" ]; then
  USE_DIRS=mne/viz/
else
  USE_DIRS="mne/"
fi
JUNIT_PATH="junit-results.xml"
if [ ! -z "$CONDA_ENV" ]; then  # use installed verison
  cd ..
  echo "Executing from $(pwd)"
  USE_DIRS="mne-python/${USE_DIRS}"
  JUNIT_PATH="mne-python/${JUNIT_PATH}"
fi
set -x
pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml --junit-xml=$JUNIT_PATH -vv ${USE_DIRS}
set +x
