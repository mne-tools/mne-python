"""
==========================
Reading a raw file segment
==========================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
from mne import fiff
from mne.datasets import sample
data_path = sample.data_path('.')

fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = fiff.setup_read_raw(fname)

exclude = ['MEG 2443', 'EEG 053'] # bad channels

meg_channels_idx = fiff.pick_types(raw['info'], meg=True, exclude=exclude)
meg_channels_idx = meg_channels_idx[:5] # take 5 first

start, stop = raw.time_to_index(100, 115) # 100 s to 115 s data segment
data, times = raw[meg_channels_idx, start:stop]
# data, times = raw[:, start:stop] # read all channels

raw.close()

###############################################################################
# Show MEG data
import pylab as pl
pl.close('all')
pl.plot(times, data.T)
pl.xlabel('time (s)')
pl.ylabel('MEG data (T)')
pl.show()
