"""
===============================================
Create topographic ERF maps in delayed SSP mode
===============================================

This script shows how to apply SSP projectors delayed, that is,
at the evoked stage. This is particularly useful to support decisions
related to the trade-off between denoising and preserving signal.
In this example we demonstrate how to use topographic maps for delayed
SSP application.
"""
# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg-proj.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# delete EEG projections (we know it's the last one)
raw.del_proj(-1)
# add ECG projs for magnetometers
[raw.add_proj(p) for p in mne.read_proj(ecg_fname) if 'axial' in p['desc']]

# pick magnetometer channels
picks = mne.pick_types(raw.info, meg='mag', stim=False, eog=True,
                       include=[], exclude='bads')

# We will make of the proj `delayed` option to
# interactively select projections at the evoked stage.
# more information can be found in the example/plot_evoked_delayed_ssp.py
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12), proj='delayed')

evoked = epochs.average()  # average epochs and get an Evoked dataset.

###############################################################################
# Interactively select / deselect the SSP projection vectors

# set time instants in seconds (from 50 to 150ms in a step of 10ms)
times = np.arange(0.05, 0.15, 0.01)

evoked.plot_topomap(times, proj='interactive')
# Hint: the same works for evoked.plot and evoked.plot_topo
