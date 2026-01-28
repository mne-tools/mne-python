#!/bin/bash -ef

if [[ ${MNE_CI_KIND} == "old" ]]; then
    source .venv/bin/activate
fi

which mne
mne sys_info -pd
python -c "import numpy; numpy.show_config()"
