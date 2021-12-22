#!/bin/bash -ef

if [ "${DEPS}" == "minimal" ]; then
	return 0 2>/dev/null || exit "0"
fi;

pushd ~ > /dev/null
export MNE_ROOT="${PWD}/minimal_cmds"
export PATH=${MNE_ROOT}/bin:$PATH
if [ "${GITHUB_ACTIONS}" == "true" ]; then
	echo "MNE_ROOT=${MNE_ROOT}" >> $GITHUB_ENV;
	echo "${MNE_ROOT}/bin" >> $GITHUB_PATH;
fi;
if [ "${AZURE_CI}" == "true" ]; then
	echo "##vso[task.setvariable variable=MNE_ROOT]${MNE_ROOT}"
	echo "##vso[task.setvariable variable=PATH]${PATH}";
fi;
if [ "${CIRCLECI}" == "true" ]; then
	echo "export MNE_ROOT=${MNE_ROOT}" >> "$BASH_ENV";
	echo "export PATH=${MNE_ROOT}/bin:$PATH" >> "$BASH_ENV";
fi;
if [ "${CI_OS_NAME}" != "osx" ]; then
	if [ ! -d "${PWD}/minimal_cmds" ]; then
		curl -L https://osf.io/g7dzs/download | tar xz
	fi;
	export LD_LIBRARY_PATH=${MNE_ROOT}/lib:$LD_LIBRARY_PATH
	export NEUROMAG2FT_ROOT="${PWD}/minimal_cmds/bin"
	export FREESURFER_HOME="${MNE_ROOT}"
	if [ "${GITHUB_ACTIONS}" == "true" ]; then
		echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> "$GITHUB_ENV";
		echo "NEUROMAG2FT_ROOT=${NEUROMAG2FT_ROOT}" >> "$GITHUB_ENV";
		echo "FREESURFER_HOME=${FREESURFER_HOME}" >> "$GITHUB_ENV";
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
	if [ ! -d "${PWD}/minimal_cmds" ]; then
		curl -L https://osf.io/rjcz4/download | tar xz
	fi;
	export DYLD_LIBRARY_PATH=${MNE_ROOT}/lib:$DYLD_LIBRARY_PATH
	if [ "${GITHUB_ACTIONS}" == "true" ]; then
		echo "DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}" >> "$GITHUB_ENV";
		wget https://github.com/XQuartz/XQuartz/releases/download/XQuartz-2.7.11/XQuartz-2.7.11.dmg
		sudo hdiutil attach XQuartz-2.7.11.dmg
		sudo installer -package /Volumes/XQuartz-2.7.11/XQuartz.pkg -target /
		sudo ln -s /opt/X11 /usr/X11
	fi;
	if [ "${AZURE_CI}" == "true" ]; then
		echo "##vso[task.setvariable variable=DYLD_LIBRARY_PATH]${DYLD_LIBRARY_PATH}"
	fi;
	if [ "${CIRCLECI}" == "true" ]; then
		echo "export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}" >> "$BASH_ENV";
	fi;
fi
popd > /dev/null
mne_process_raw --version
