#!/bin/bash -ef
source ./github_actions_helpers.sh

which mne
mne sys_info -pd
run_python -c "import numpy; numpy.show_config()"
