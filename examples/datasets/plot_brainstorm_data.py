"""
============================
Brainstorm tutorial datasets
============================

Here we compute the evoked from raw for the Brainstorm
tutorial dataset

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

import mne
from mne.datasets.brainstorm import bst_raw
from mne.io import Raw

print(__doc__)

tmin, tmax, event_id = -0.2, 0.5, 1
reject = dict(mag=4e-12)

data_path = bst_raw.data_path()

raw_fname = data_path + '/MEG/bst_raw/' + \
                        'subj001_somatosensory_20111109_01_AUX-f_raw.fif'
raw = Raw(raw_fname)

events = mne.find_events(raw, stim_channel='UPPT001')

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False)

# compute evoked
evoked = epochs.average()

# remove artifacts using SSP on baseline
evoked.add_proj(mne.compute_proj_evoked(evoked.crop(tmax=0, copy=True)))
evoked.apply_proj()

# plot the result
evoked.plot()
