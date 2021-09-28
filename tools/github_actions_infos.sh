#!/bin/bash -ef

mne sys_info --show-paths
python -c "import numpy; numpy.show_config()"
