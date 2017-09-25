"""
==============
HF-SEF dataset
==============

This example looks at high frequency SEF responses
"""
# Author: Jussi Nurminen (jnu@iki.fi)
#
# License: BSD (3-clause)


import mne
import os
from mne.datasets import hf_sef

fname_evoked = os.path.join(hf_sef.data_path(),
                            'MEG/subject_a/sef2_right-ave.fif')

print(__doc__)

###############################################################################
# Read raw file
evoked = mne.Evoked(fname_evoked)

evoked.filter(l_freq=200, h_freq=400).plot()
