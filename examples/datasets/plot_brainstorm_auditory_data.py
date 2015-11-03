"""
====================================
Brainstorm auditory tutorial dataset
====================================

Here we compute the evoked from raw for the auditory Brainstorm
tutorial dataset. For comparison, see:
http://neuroimage.usc.edu/brainstorm/Tutorials/Auditory

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
Computational Intelligence and Neuroscience, vol. 2011, Article ID 879716,
13 pages, 2011. doi:10.1155/2011/879716
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets.brainstorm import bst_auditory
from mne.io import Raw

print(__doc__)

tmin, tmax = -0.1, 0.5
event_id = dict(standard=1, deviant=2)
reject = dict(mag=4e-12, eog=250e-6)

data_path = bst_auditory.data_path()

raw_fname = data_path + '/MEG/bst_auditory/S01_AEF_20131218_01_raw.fif'
raw = Raw(raw_fname, preload=True)
raw.plot()

# set EOG channel
raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})

# # show power line interference and remove it
# raw.plot_psd()
# raw.notch_filter(np.arange(60, 181, 60))

events = mne.find_events(raw, stim_channel='UPPT001')
# events = mne.find_events(raw, stim_channel='UADC001-4408')

raw.info['bads'] = ['MLO52-4408', 'MRT51-4408']

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Compute epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False)

# compute evoked
evoked_standard = epochs['standard'].average()
evoked_deviant = epochs['deviant'].average()

# plot the result
evoked_standard.plot()
evoked_deviant.plot()

# show topomaps
evoked_standard.plot_topomap(times=np.array([0.1]))
evoked_deviant.plot_topomap(times=np.array([0.1]))
