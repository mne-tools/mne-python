"""
=============================
Reading and writing raw files
=============================

In this example we read a raw file. Plot a segment of MEG data
restricted to MEG channels. And save these data in a new
raw file.
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

from mne import fiff
from mne.datasets import sample
data_path = sample.data_path('.')

fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = fiff.Raw(fname)

# Set up pick list: MEG + STI 014 - bad channels
want_meg = True
want_eeg = False
want_stim = False
include = ['STI 014']
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053'] # bad channels + 2 more

picks = fiff.pick_types(raw.info, meg=want_meg, eeg=want_eeg,
                                   stim=want_stim, include=include,
                                   exclude=exclude)

some_picks = picks[:5] # take 5 first
start, stop = raw.time_to_index(100, 115) # 100 s to 115 s data segment
data, times = raw[some_picks, start:stop]

raw.save('sample_audvis_meg_raw.fif', picks=picks, first_samp=start,
            last_samp=stop-1) # save MEG data in FIF file

###############################################################################
# Show MEG data
import pylab as pl
pl.close('all')
pl.plot(times, data.T)
pl.xlabel('time (s)')
pl.ylabel('MEG data (T)')
pl.show()
