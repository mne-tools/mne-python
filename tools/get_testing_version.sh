#!/bin/bash -ef

# TESTING_VERSION=$(grep -oP "(?<=testing=')[0-9.]*(?=')" mne/datasets/utils.py)
TESTING_VERSION=$(grep -o "testing='[0-9.]\+'" mne/datasets/utils.py | cut -d \' -f 2)
if [ ! -z $GITHUB_ENV ]; then
	echo "TESTING_VERSION="$TESTING_VERSION >> $GITHUB_ENV
else
	echo $TESTING_VERSION
fi
