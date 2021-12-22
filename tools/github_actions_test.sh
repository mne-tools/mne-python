#!/bin/bash -ef

USE_DIRS="mne/"
if [ "${CI_OS_NAME}" != "osx" ]; then
  CONDITION="not (ultraslowtest or pgtest)"
else
  CONDITION="not (slowtest or pgtest)"
fi
echo 'pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml -vv ${USE_DIRS}'
pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml -vv ${USE_DIRS}
