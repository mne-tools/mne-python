#!/bin/bash -ef

TESTING_VERSION=$(grep -oP "(?<=testing=')[0-9.]*(?=')" mne/datasets/utils.py)
if [ ! -z $GITHUB_ENV ]; then
	echo $TESTING_VERSION >> $GITHUB_ENV
else
	echo $TESTING_VERSION
fi
