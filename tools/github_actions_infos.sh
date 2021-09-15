#!/bin/bash -ef

mne sys_info
python -c "import numpy; numpy.show_config()"
