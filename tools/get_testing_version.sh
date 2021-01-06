#!/bin/bash -ef

TESTING_VERSION=`grep -o "testing='[0-9.]\+'" mne/datasets/utils.py | cut -d \' -f 2 | sed "s/\./-/g"`
if [ ! -z $GITHUB_ENV ]; then
	echo "TESTING_VERSION="$TESTING_VERSION >> $GITHUB_ENV
elif [ ! -z $AZURE_CI ]; then
	echo "##vso[task.setvariable variable=testing_version]$TESTING_VERSION"
else
	echo $TESTING_VERSION
fi
