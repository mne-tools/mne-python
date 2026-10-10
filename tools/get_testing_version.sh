#!/bin/bash -e

set -o pipefail

# Versions of the datasets that CI downloads for the test suite, in a form usable
# in cache keys. These can be suffixed (e.g. TESTING_VERSION=${TESTING_VERSION}-1)
# to start fresh when a cache misbehaves.
TESTING_VERSION=`grep -o "testing=\"[0-9.]\+\"" mne/datasets/config.py | cut -d \" -f 2 | sed "s/\./-/g"`
MISC_VERSION=`grep -o "misc=\"[0-9.]\+\"" mne/datasets/config.py | cut -d \" -f 2 | sed "s/\./-/g"`
if [ ! -z $GITHUB_ENV ]; then
	echo "TESTING_VERSION="$TESTING_VERSION | tee -a $GITHUB_ENV
	echo "MISC_VERSION="$MISC_VERSION | tee -a $GITHUB_ENV
elif [ ! -z $AZURE_CI ]; then
	echo "##vso[task.setvariable variable=testing_version]$TESTING_VERSION"
	echo "##vso[task.setvariable variable=misc_version]$MISC_VERSION"
elif [ ! -z $CIRCLECI ]; then
	echo "$TESTING_VERSION" > testing_version.txt
	echo "$MISC_VERSION" > misc_version.txt
else
	echo $TESTING_VERSION
	echo $MISC_VERSION
fi
