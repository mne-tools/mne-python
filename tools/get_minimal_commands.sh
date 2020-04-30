#!/bin/bash -ef

pushd ~ > /dev/null
export MNE_ROOT="${PWD}/minimal_cmds"
export PATH=${MNE_ROOT}/bin:$PATH
if [ "${TRAVIS_OS_NAME}" != "osx" ]; then
	if [ ! -d "${PWD}/minimal_cmds" ]; then
		curl https://staff.washington.edu/larsoner/minimal_cmds.tar.gz | tar xz
	fi;
	export LD_LIBRARY_PATH=${MNE_ROOT}/lib:$LD_LIBRARY_PATH
	export NEUROMAG2FT_ROOT="${PWD}/minimal_cmds/bin"
	export FREESURFER_HOME="${MNE_ROOT}"
else
	if [ ! -d "${PWD}/minimal_cmds" ]; then
		curl https://staff.washington.edu/larsoner/minimal_cmds_osx.tar.gz | tar xz
	fi;
	export DYLD_LIBRARY_PATH=${MNE_ROOT}/lib:$DYLD_LIBRARY_PATH
fi
popd > /dev/null
