"""
=============================
Reading and writing raw files
=============================

In this example, we read a raw file. Plot a segment of MEG data
restricted to MEG channels. And save these data in a new
raw file.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(fname)

# Set up pick list: MEG + STI 014 - bad channels
want_meg = True
want_eeg = False
want_stim = False
include = ['STI 014']
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bad channels + 2 more

picks = mne.pick_types(raw.info, meg=want_meg, eeg=want_eeg, stim=want_stim,
                       include=include, exclude='bads')

some_picks = picks[:5]  # take 5 first
start, stop = raw.time_as_index([0, 15])  # read the first 15s of data
data, times = raw[some_picks, start:(stop + 1)]

# save 150s of MEG data in FIF file
raw.save('sample_audvis_meg_raw.fif', tmin=0, tmax=150, picks=picks,
         overwrite=True)

###############################################################################
# Show MEG data
raw.plot()
