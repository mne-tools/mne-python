#!/bin/bash -ef

if [ "${DEPS}" != "minimal" ]; then
	pushd ~ > /dev/null
	export MNE_ROOT="${PWD}/minimal_cmds"
	export PATH=${MNE_ROOT}/bin:$PATH
	echo "MNE_ROOT=${MNE_ROOT}" >> $GITHUB_ENV;
	echo "${MNE_ROOT}/bin" >> $GITHUB_PATH;
	if [ "${CI_OS_NAME}" != "osx" ]; then
		if [ ! -d "${PWD}/minimal_cmds" ]; then
			curl -L https://osf.io/g7dzs/download | tar xz
		fi;
		export LD_LIBRARY_PATH=${MNE_ROOT}/lib:$LD_LIBRARY_PATH
		export NEUROMAG2FT_ROOT="${PWD}/minimal_cmds/bin"
		export FREESURFER_HOME="${MNE_ROOT}"
		if [ "${GITHUB_ACTIONS}" == "true" ]; then
			echo "LD_LIBRARY_PATH=${DYLD_PATH}" >> $GITHUB_ENV;
			echo "NEUROMAG2FT_ROOT=${NEUROMAG2FT_ROOT}" >> $GITHUB_ENV;
			echo "FREESURFER_HOME=${FREESURFER_HOME}" >> $GITHUB_ENV;
		fi;
	else
		if [ ! -d "${PWD}/minimal_cmds" ]; then
			curl -L https://osf.io/rjcz4/download | tar xz
		fi;
		export DYLD_LIBRARY_PATH=${MNE_ROOT}/lib:$DYLD_LIBRARY_PATH
		if [ "${GITHUB_ACTIONS}" == "true" ]; then
			echo "DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}" >> $GITHUB_ENV;
		fi;
	fi
	popd > /dev/null
	mne_surf2bem --version
fi
