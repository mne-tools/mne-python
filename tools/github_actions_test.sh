#!/bin/bash

set -eo pipefail

if [[ "${CI_OS_NAME}" == "ubuntu"* ]]; then
  if [[ "${MNE_CI_KIND}" == "pip-pre" ]]; then
    CONDITION="not (slowtest or pgtest)"
  else
    CONDITION="not (ultraslowtest or pgtest)"
  fi
else
  CONDITION="not (slowtest or pgtest)"
fi
if [ "${MNE_CI_KIND}" == "notebook" ]; then
  USE_DIRS=mne/viz/
else
  USE_DIRS="mne/"
fi
set -x
pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml -vv ${USE_DIRS}
set +x
