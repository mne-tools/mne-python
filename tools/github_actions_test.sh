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
if [[ ! -z "$CONDA_ENV" ]] && [[ "${RUNNER_OS}" != "Windows" ]]; then
  PROJ_PATH="$(pwd)"
  JUNIT_PATH="$PROJ_PATH/${JUNIT_PATH}"
  # Use the installed version after adding all (excluded) test files
  cd ..
  INSTALL_PATH=$(python -c "import mne, pathlib; print(str(pathlib.Path(mne.__file__).parents[1]))")
  echo "Copying tests from $(pwd)/mne-python/mne/ to ${INSTALL_PATH}/mne/"
  echo "::group::rsync"
  rsync -a --partial --progress --prune-empty-dirs --exclude="*.pyc" --include="**/" --include="**/tests/*" --include="**/tests/data/**" --exclude="**" ./mne-python/mne/ ${INSTALL_PATH}/mne/
  cd $INSTALL_PATH
  cp -av $PROJ_PATH/pyproject.toml .
  echo "::endgroup::"
  echo "Executing from $(pwd)"
fi

set -x
pytest -m "${CONDITION}" --cov=mne --cov-report xml --color=yes --continue-on-collection-errors --junit-xml=$JUNIT_PATH -vv ${USE_DIRS}
set +x
