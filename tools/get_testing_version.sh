#!/bin/bash -ef

# TESTING_VERSION=$(grep -oP "(?<=testing=')[0-9.]*(?=')" mne/datasets/utils.py)
TESTING_VERSION=$(grep -o "testing='[0-9.]\+'" mne/datasets/utils.py | cut -d \' -f 2)
if [ ! -z $GITHUB_ENV ]; then
	echo "TESTING_VERSION="$TESTING_VERSION >> $GITHUB_ENV
elif [ ! -z $Build.Repository.Name ]; then
	echo "##vso[task.setvariable variable=testing_version;isOutput=true]$TESTING_VERSION"
else
	echo $TESTING_VERSION
fi
