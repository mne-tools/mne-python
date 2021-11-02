#!/bin/bash -ef

which mne
printf '%.s─' $(seq 1 $(tput cols))
mne sys_info -pd
printf '%.s─' $(seq 1 $(tput cols))
python -c "import numpy; numpy.show_config()"
