# -*- coding: utf-8 -*-
"""
.. _ex-brainstorm-raw:

=====================================
Brainstorm raw (median nerve) dataset
=====================================

Here we compute the evoked from raw for the Brainstorm
tutorial dataset. For comparison, see :footcite:`TadelEtAl2011` and:

    https://neuroimage.usc.edu/brainstorm/Tutorials/MedianNerveCtf
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD-3-Clause

# %%

import numpy as np

import mne
from mne.datasets.brainstorm import bst_raw
from mne.io import read_raw_ctf

print(__doc__)

tmin, tmax, event_id = -0.1, 0.3, 2  # take right-hand somato
reject = dict(mag=4e-12, eog=250e-6)

data_path = bst_raw.data_path()

raw_path = (data_path / 'MEG' / 'bst_raw' /
            'subj001_somatosensory_20111109_01_AUX-f.ds')
# Here we crop to half the length to save memory
raw = read_raw_ctf(raw_path).crop(0, 120).load_data()
raw.plot()

# set EOG channel
raw.set_channel_types({'EEG058': 'eog'})
raw.set_eeg_reference('average', projection=True)

# show power line interference and remove it
raw.plot_psd(tmax=60., average=False)
raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin')

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
evoked.plot(time_unit='s')

# show topomaps
evoked.plot_topomap(times=np.array([0.016, 0.030, 0.060, 0.070]),
                    time_unit='s')

# %%
# References
# ----------
# .. footbibliography::
