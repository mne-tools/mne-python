#!/bin/bash -ef

which mne
mne sys_info -pd
python -c "import numpy; numpy.show_config()"
