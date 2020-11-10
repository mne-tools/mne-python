#!/bin/bash -ef

pip uninstall -yq mne
pip install --upgrade -r requirements_testing.txt
pip install nitime
