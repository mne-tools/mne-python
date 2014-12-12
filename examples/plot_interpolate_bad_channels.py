"""
================================================
Interpolate bad channels using spherical splines
================================================

This example shows how to interpolate bad EEG channels using spherical
splines as described in [1].

Referneces
----------
[1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989) Spherical
    splines for scalp potential and current density mapping.
    Electroencephalography and Clinical Neurophysiology, Feb; 72(2):184-7.
"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne import io
from mne.datasets import sample
data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

#   Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

raw.info['bads'] = ['EEG 053', 'EEG 012']  # mark bad channels

# pick EEG channels and keep bads for interpolation
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True,
                       exclude=[])

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0),
                    reject=dict(eog=150e-6, eeg=80e-6),
                    preload=True)

# plot with bads
evoked_before = epochs.average()
evoked_before.plot(exclude=[])

# compute interpolation (also works with Raw and Epochs objects)
evoked_after = evoked_before.copy().interpolate_bads_eeg()

# plot interpolated (prevsious bads)
evoked_after.plot(exclude=[])
