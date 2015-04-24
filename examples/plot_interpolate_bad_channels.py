"""
=============================================
Interpolate bad channels for MEG/EEG channels
=============================================

This example shows how to interpolate bad MEG/EEG channels
    - Using spherical splines as described in [1] for EEG data.
    - Using field interpolation for MEG data.

The bad channels will still be marked as bad. Only the data in those channels
is removed.

References
----------
[1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989) Spherical
    splines for scalp potential and current density mapping.
    Electroencephalography and Clinical Neurophysiology, Feb; 72(2):184-7.
"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
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
event_id, tmin, tmax = 1, -0.2, 0.5

#   Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

# pick EEG channels and keep bads for interpolation
picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True,
                       exclude=[])

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=None, preload=False)

# plot with bads
evoked_before = epochs.average()
evoked_before.plot(exclude=[])

# compute interpolation (also works with Raw and Epochs objects)
evoked_after = evoked_before.interpolate_bads()

# plot interpolated (previous bads)
evoked_after.plot(exclude=[])
