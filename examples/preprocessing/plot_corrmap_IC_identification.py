"""
==========================================================
Identify similar ICs across multiple datasets via CORRMAP
==========================================================
After fitting ICA to multiple data sets, CORRMAP[1] 
automatically identifies similar ICs in all sets based 
on a manually selected template. These ICs can then be 
removed, or further investigated.

[1] Viola FC, et al. Semi-automatic identification of independent components 
representing EEG artifact. Clin Neurophysiol 2009, May; 120(5): 868-77.
"""

# Authors: Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import mne

from mne.io import Raw
from mne.preprocessing import ICA
from mne.preprocessing.ica import corrmap
from mne.datasets import sample

print(__doc__)

###############################################################################
# Setup paths and prepare epochs data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)
raw.filter(1, 30, method='iir')
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, ecg=False,
                       stim=False, exclude='bads')

events = mne.find_events(raw, stim_channel='STI 014')
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
reject = dict(eog=250e-6)
tmin, tmax = -0.5, 0.75

# subsetting the data set into 3 independent sets of epochs 
# in a real-world case, this would instead be multiple subjects/data sets, 
# not subsets of one data set

subsets = [mne.Epochs(raw, events[ranges[0]:ranges[1]], event_id, tmin, tmax, 
                      proj=False, picks=picks, baseline=(None, 0), 
                      preload=True, reject=None, verbose=False) 
           for ranges in [(0, 100), (101, 200), (201,300)]]

###############################################################################
# 1) Fit ICA

icas = [ICA(n_components=20, random_state=0).fit(epochs) 
        for epochs in subsets]

# 2) Use corrmap to identify the maps best corresponding 
#    to a pre-specified template across all subsets
#    (or, in the real world, multiple participant data sets)

template=(0,0)
corrmap(icas, template=template, label="blinks", inplace=True)

# 3) Zeroing the identified blink components for all data sets
#    results in individually cleaned data sets

cleaned_epochs = [ica.apply(epoch, exclude=ica.labels["blinks"], copy=True)
                  for i, (epoch, ica) in enumerate(zip(subsets, icas))]
