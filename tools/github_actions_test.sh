#!/bin/bash

set -eo pipefail

if [ "${CI_OS_NAME}" != "macos"* ]; then
  CONDITION="not (ultraslowtest or pgtest)"
else
  CONDITION="not (slowtest or pgtest)"
fi
if [ "${MNE_CI_KIND}" == "notebook" ]; then
  USE_DIRS=mne/viz/
else
  USE_DIRS="mne/"
fi
set -x
# TODO: Remove -x before merge!
pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml -vv ${USE_DIRS} -x
set +x
