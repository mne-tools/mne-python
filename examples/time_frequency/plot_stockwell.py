"""
=======================================================
Time frequency with Stockwell transform in sensor space
=======================================================

This script shows how to compute induced power and inter-trial
phase-lock for a list of epochs read in a raw file given
a list of events. The Stockwell transform, a.k.a. S-Transform
is used as time-frequency representation.

"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne import io
from mne.time_frequency import tfr_stockwell
from mne.datasets import somato

###############################################################################
# Set parameters
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
event_id, tmin, tmax = 1, -1., 3.

# Setup for reading the raw data
raw = io.Raw(raw_fname)
baseline = (None, 0)
events = mne.find_events(raw, stim_channel='STI 014')

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True, stim=False)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6),
                    preload=True)

###############################################################################
# Calculate power and intertrial coherence

epochs = epochs.pick_channels([epochs.ch_names[82]])

power = tfr_stockwell(epochs[:60], fmin=6., fmax=30., decim=3, n_jobs=6)

power.plot([0], baseline=(-0.5, 0), mode=None)

# # Baseline correction can be applied to power or done in plots
# # To illustrate the baseline correction in plots the next line is commented
# # power.apply_baseline(baseline=(-0.5, 0), mode='logratio')
#
# # Inspect power
# power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
# power.plot([82], baseline=(-0.5, 0), mode='logratio')
#
# import matplotlib.pyplot as plt
# fig, axis = plt.subplots(1, 2, figsize=(7, 4))
# power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
#                    baseline=(-0.5, 0), mode='logratio', axes=axis[0],
#                    title='Alpha', vmin=-0.45, vmax=0.45)
# power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=13, fmax=25,
#                    baseline=(-0.5, 0), mode='logratio', axes=axis[1],
#                    title='Beta', vmin=-0.45, vmax=0.45)
# mne.viz.tight_layout()
#
# # Inspect ITC
# itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds')
