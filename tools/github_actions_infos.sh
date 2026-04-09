#!/bin/bash -ef

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/github_actions_helpers.sh"

which mne
mne sys_info -pd
run_python -c "import numpy; numpy.show_config()"
