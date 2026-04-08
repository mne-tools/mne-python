#!/bin/bash -ef
TOOLS_DIR=$(dirname "${BASH_SOURCE[0]}") 
source "$TOOLS_DIR/.github_actions_helpers.sh"

which mne
mne sys_info -pd
run_python -c "import numpy; numpy.show_config()"
