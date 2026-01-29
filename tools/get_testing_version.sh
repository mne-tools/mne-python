#!/bin/bash -e

set -o pipefail

TESTING_VERSION=`grep -o "testing=\"[0-9.]\+\"" mne/datasets/config.py | cut -d \" -f 2 | sed "s/\./-/g"`
# This can be incremented to start fresh when the cache misbehaves, e.g.:
# TESTING_VERSION=${TESTING_VERSION}-1
if [ ! -z $GITHUB_ENV ]; then
	echo "TESTING_VERSION="$TESTING_VERSION | tee -a $GITHUB_ENV
elif [ ! -z $AZURE_CI ]; then
	echo "##vso[task.setvariable variable=testing_version]$TESTING_VERSION"
elif [ ! -z $CIRCLECI ]; then
	echo "$TESTING_VERSION" > testing_version.txt
else
	echo $TESTING_VERSION
fi
