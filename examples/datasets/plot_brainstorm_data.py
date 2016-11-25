"""
============================
Brainstorm tutorial datasets
============================

Here we compute the evoked from raw for the Brainstorm
tutorial dataset. For comparison, see [1]_ and:

    http://neuroimage.usc.edu/brainstorm/Tutorials/MedianNerveCtf

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets.brainstorm import bst_raw

print(__doc__)

tmin, tmax, event_id = -0.1, 0.3, 2  # take right-hand somato
reject = dict(mag=4e-12, eog=250e-6)

data_path = bst_raw.data_path()

raw_fname = data_path + '/MEG/bst_raw/' + \
                        'subj001_somatosensory_20111109_01_AUX-f_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.plot()

# set EOG channel
raw.set_channel_types({'EEG058': 'eog'})
raw.set_eeg_reference()

# show power line interference and remove it
raw.plot_psd(tmax=60.)
raw.notch_filter(np.arange(60, 181, 60))

events = mne.find_events(raw, stim_channel='UPPT001')

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Compute epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False)

# compute evoked
evoked = epochs.average()

# remove physiological artifacts (eyeblinks, heartbeats) using SSP on baseline
evoked.add_proj(mne.compute_proj_evoked(evoked.copy().crop(tmax=0)))
evoked.apply_proj()

# fix stim artifact
mne.preprocessing.fix_stim_artifact(evoked)

# correct delays due to hardware (stim artifact is at 4 ms)
evoked.shift_time(-0.004)

# plot the result
evoked.plot()

# show topomaps
evoked.plot_topomap(times=np.array([0.016, 0.030, 0.060, 0.070]))
