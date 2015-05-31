"""
=============================================================
Demonstration for plotting and rejecting epochs interactively
=============================================================

This example shows how to visualize and reject bad epochs interactively.
Bad epochs can be selected by a left click on the top of the epoch.
The selected bad epochs are dropped on closing of the figure.
Scaling can be changed runtime by pressing page up and page down keys.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5

# Select events to extract epochs from.
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}

#   Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG and EOG channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                       include=include, exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(eeg=80e-6, eog=150e-6),
                    preload=True)

epochs.plot_concat(n_channels=20, n_epochs=10, block=True)
