"""
.. _ex-hf-sef-data:

==============
HF-SEF dataset
==============

This example looks at high-frequency SEF responses.
"""
# Author: Jussi Nurminen (jnu@iki.fi)
#
# License: BSD (3-clause)


import mne
import os
from mne.datasets import hf_sef

fname_evoked = os.path.join(hf_sef.data_path(),
                            'MEG/subject_b/hf_sef_15min-ave.fif')

print(__doc__)

###############################################################################
# Read evoked data
evoked = mne.Evoked(fname_evoked)

###############################################################################
# Create a highpass filtered version
evoked_hp = evoked.copy()
evoked_hp.filter(l_freq=300, h_freq=None)

###############################################################################
# Compare high-pass filtered and unfiltered data on a single channel
ch = 'MEG0443'
pick = evoked.ch_names.index(ch)
edi = {'HF': evoked_hp, 'Regular': evoked}
mne.viz.plot_compare_evokeds(edi, picks=pick)
