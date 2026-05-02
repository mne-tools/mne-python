#!/bin/bash

set -eo pipefail

if [[ "${CI_OS_NAME}" == "ubuntu"* ]]; then
  CONDITION="not (ultraslowtest or pgtest)"
elif [[ "${CI_OS_NAME}" == "macos"* ]]; then
  # detect arch and run slowtest on arm64 only (pgtest is already ultraslow on macOS)
  if [[ "$(uname -m)" == "arm64" ]]; then
    CONDITION="not (ultraslowtest or pgtest)"
  else
    CONDITION="not (slowtest or pgtest)"
  fi
elif [[ "${CI_OS_NAME}" == "windows"* ]]; then
  CONDITION="not (slowtest or pgtest)"
else
  echo "✕ ERROR: Unrecognized CI_OS_NAME=${CI_OS_NAME}"
  exit 1
fi
if [ "${MNE_CI_KIND}" == "notebook" ]; then
  USE_DIRS=mne/viz/
else
  USE_DIRS="mne/"
fi
JUNIT_PATH="junit-results.xml"
if [[ ! -z "$CONDA_ENV" ]] && [[ "${CI_OS_NAME}" != "windows"* ]] && [[ "${MNE_CI_KIND}" != "minimal" ]] && [[ "${MNE_CI_KIND}" != "old" ]]; then
  PROJ_PATH="$(pwd)"
  JUNIT_PATH="$PROJ_PATH/${JUNIT_PATH}"
  # Use the installed version after adding all (excluded) test files
  cd ~  # so that "import mne" doesn't just import the checked-out data
  INSTALL_PATH=$(python -c "import mne, pathlib; print(str(pathlib.Path(mne.__file__).parents[1]))")
  echo "Copying tests from ${PROJ_PATH}/mne-python/mne/ to ${INSTALL_PATH}/mne/"
  echo "::group::rsync mne"
  set -x
  rsync -a --partial --progress --prune-empty-dirs --exclude="*.pyc" --include="*/" --include="tests/**" --include="**/tests/**" --exclude="**" ${PROJ_PATH}/mne/ ${INSTALL_PATH}/mne/
  echo "::endgroup::"
  echo "::group::rsync doc"
  mkdir -p ${INSTALL_PATH}/doc/
  rsync -a --partial --progress --prune-empty-dirs --include="api/" --include="api/*.rst" --exclude="*" ${PROJ_PATH}/doc/ ${INSTALL_PATH}/doc/
  test -f ${INSTALL_PATH}/doc/api/reading_raw_data.rst
  cd $INSTALL_PATH
  cp -av $PROJ_PATH/pyproject.toml .
  set +x
  echo "::endgroup::"
fi

set -x
pytest -m "${CONDITION}" --cov=mne --cov-report xml --color=yes --continue-on-collection-errors --junit-xml=$JUNIT_PATH -vv ${USE_DIRS}
echo "Exited with code $?"
