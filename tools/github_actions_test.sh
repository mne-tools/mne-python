#!/bin/bash

set -eo pipefail


if [[ "${CI_OS_NAME}" == "ubuntu"* ]]; then
  CONDITIONS=("not (ultraslowtest or pgtest)")
elif [[ "${CI_OS_NAME}" == "macos"* ]]; then
  # detect arch and run slowtest on arm64 only (pgtest is already ultraslow on macOS)
  if [[ "$(uname -m)" == "arm64" ]]; then
    # Split the PyVista/VTK ("pvtest") tests into their own xdist invocation.
    # Run all together, a worker that accumulates heavy state (loky pools,
    # leftover ipykernel/jupyter_client asyncio loops, tqdm monitors) creates
    # the scheduling jitter that trips an xdist loadscope dispatch deadlock at
    # end-of-run: workers idle in TestQueue.get, controller loops forever in
    # dsession.loop_once without dispatching the remaining scopes or SHUTDOWN
    # (see pytest-xdist#1313). Two shorter, lighter invocations avoid it while
    # keeping parallelism; pvtest alone under xdist does not hang.
    CONDITIONS=("not (ultraslowtest or pgtest or pvtest)" "pvtest and not ultraslowtest")
  else
    CONDITIONS=("not (slowtest or pgtest)")
  fi
elif [[ "${CI_OS_NAME}" == "windows"* ]]; then
  CONDITIONS=("not (slowtest or pgtest)")
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

# $COV_ARGS is set in github_actions_env_vars.sh (coverage only on Python >= 3.14)
# Run each marker expression as its own pytest invocation (usually just one; on
# macOS arm64 the pvtest tests are split out, see above). Later runs append
# coverage, each gets its own junit file, and we run them all before exiting
# with the first nonzero code so one failing split does not mask the other.
set +e  # capture each pytest's exit code manually rather than aborting on it
CODE=0
COV_APPEND=""
i=0
for CONDITION in "${CONDITIONS[@]}"; do
  if [ "${#CONDITIONS[@]}" -gt 1 ]; then
    THIS_JUNIT="${JUNIT_PATH%.xml}-${i}.xml"
  else
    THIS_JUNIT="$JUNIT_PATH"
  fi
  set -x
  pytest -m "${CONDITION}" -n "$PYTEST_XDIST_N" --dist loadscope --timeout=120 --timeout-method=thread -o faulthandler_timeout=110 ${COV_ARGS} ${COV_APPEND} --color=yes --continue-on-collection-errors --junit-xml="$THIS_JUNIT" -vv ${USE_DIRS}
  THIS_CODE=$?
  set +x
  echo "Exited with code $THIS_CODE for: ${CONDITION}"
  if [ "$THIS_CODE" -ne 0 ] && [ "$CODE" -eq 0 ]; then
    CODE=$THIS_CODE
  fi
  COV_APPEND="--cov-append"
  i=$((i + 1))
done
exit $CODE
