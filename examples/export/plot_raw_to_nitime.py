"""
============================
Export Raw Objects to NiTime
============================

This script shows how to export raw files to the NiTime library
for further signal processing and data analysis.

"""

# Author: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import mne

from mne.fiff import Raw
from mne.datasets import sample

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

###############################################################################
# get raw data
raw = Raw(raw_fname)

# set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude=raw.info['bads'])

# pick times
start, stop = raw.time_to_index(100, 115)

# export to nitime using a copy of the data
raw_ts = raw.to_nitime(start=start, stop=stop, picks=picks, copy=True)
