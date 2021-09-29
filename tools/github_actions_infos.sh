#!/bin/bash -ef

mne sys_info -pd
python -c "import numpy; numpy.show_config()"
