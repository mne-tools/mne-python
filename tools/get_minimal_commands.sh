#!/bin/bash

set -eo pipefail

if [ "${DEPS}" == "minimal" ]; then
	return 0 2>/dev/null || exit "0"
fi;

pushd ~ > /dev/null
export MNE_ROOT="${PWD}/minimal_cmds"
export PATH=${MNE_ROOT}/bin:$PATH
if [ "${GITHUB_ACTIONS}" == "true" ]; then
	echo "Setting MNE_ROOT for GHA"
	echo "MNE_ROOT=${MNE_ROOT}" | tee -a $GITHUB_ENV;
	echo "${MNE_ROOT}/bin" >> $GITHUB_PATH;
elif [ "${AZURE_CI}" == "true" ]; then
	echo "Setting MNE_ROOT for Azure"
	echo "##vso[task.setvariable variable=MNE_ROOT]${MNE_ROOT}"
	echo "##vso[task.setvariable variable=PATH]${PATH}";
elif [ "${CIRCLECI}" == "true" ]; then
	echo "Setting MNE_ROOT for CircleCI"
	echo "export MNE_ROOT=${MNE_ROOT}" >> "$BASH_ENV";
	echo "export PATH=${MNE_ROOT}/bin:\$PATH" >> "$BASH_ENV";
fi;
if [[ "${CI_OS_NAME}" != "macos"* ]]; then
	echo "Getting files for Linux..."
	if [ ! -d "${PWD}/minimal_cmds" ]; then
		curl -L https://osf.io/g7dzs/download?version=5 | tar xz
	else
		echo "Minimal commands already downloaded"
	fi;
	export LD_LIBRARY_PATH=${MNE_ROOT}/lib:$LD_LIBRARY_PATH
	export NEUROMAG2FT_ROOT="${PWD}/minimal_cmds/bin"
	export FREESURFER_HOME="${MNE_ROOT}"
	if [ "${GITHUB_ACTIONS}" == "true" ]; then
		echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" | tee -a "$GITHUB_ENV";
		echo "NEUROMAG2FT_ROOT=${NEUROMAG2FT_ROOT}" | tee -a "$GITHUB_ENV";
		echo "FREESURFER_HOME=${FREESURFER_HOME}" | tee -a "$GITHUB_ENV";
	fi;
	if [ "${AZURE_CI}" == "true" ]; then
		echo "##vso[task.setvariable variable=LD_LIBRARY_PATH]${LD_LIBRARY_PATH}"
		echo "##vso[task.setvariable variable=NEUROMAG2FT_ROOT]${NEUROMAG2FT_ROOT}"
		echo "##vso[task.setvariable variable=FREESURFER_HOME]${FREESURFER_HOME}"
	fi;
	if [ "${CIRCLECI}" == "true" ]; then
		echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> "$BASH_ENV";
		echo "export NEUROMAG2FT_ROOT=${NEUROMAG2FT_ROOT}" >> "$BASH_ENV";
		echo "export FREESURFER_HOME=${FREESURFER_HOME}" >> "$BASH_ENV";
	fi;
else
	echo "Getting files for macOS Intel..."
	if [ ! -d "${PWD}/minimal_cmds" ]; then
		curl -L https://osf.io/rjcz4/download?version=2 | tar xz
	else
		echo "Minimal commands already downloaded"
	fi;
	export DYLD_LIBRARY_PATH=${MNE_ROOT}/lib:$DYLD_LIBRARY_PATH
	if [ "${GITHUB_ACTIONS}" == "true" ]; then
		echo "Setting variables for GHA"
		echo "DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}" | tee -a "$GITHUB_ENV";
		set -x
		wget https://github.com/XQuartz/XQuartz/releases/download/XQuartz-2.7.11/XQuartz-2.7.11.dmg
		sudo hdiutil attach XQuartz-2.7.11.dmg
		sudo installer -package /Volumes/XQuartz-2.7.11/XQuartz.pkg -target /
		sudo ln -s /opt/X11 /usr/X11
	elif [ "${AZURE_CI}" == "true" ]; then
		echo "Setting variables for Azure"
		echo "##vso[task.setvariable variable=DYLD_LIBRARY_PATH]${DYLD_LIBRARY_PATH}"
	elif [ "${CIRCLECI}" == "true" ]; then
		echo "Setting variables for CircleCI"
		echo "export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}" >> "$BASH_ENV";
	fi;
fi
popd > /dev/null
set -x
which mne_process_raw
mne_process_raw --version
which mne_surf2bem
mne_surf2bem --version
which mri_average
mri_average --version
set +x
