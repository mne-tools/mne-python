"""
==============================================
Visualize stim channel over epochs as an image
==============================================

This will produce what is sometimes called an event related
potential / field (ERP/ERF) image based on the stimulus channel

"""
# Authors: Jean-Rémi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from matplotlib.colors import LogNorm
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

# Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=True, eog=False)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True)

# Transform STIM channels into binary code
n_ep, n_ch, n_t = epochs._data.shape
STI101 = np.zeros((n_ep, n_t))
for ch in range(0, n_ch):
    x = epochs._data[:, ch, :]
    M = np.max(x)
    if M > 0.0:
        STI101 += (x / M) * (2 ** ch)
# to increase diversity of trigger values in the example
STI101 *= 2 ** np.round(6.0 * np.random.rand(n_ep, n_t))
epochs._data[:, 0, :] = STI101

###############################################################################
# Show stimulus channel
mne.viz.plot_image_epochs(epochs, 0, colorbar=True, show=True,
                          norm=LogNorm(vmin=1.0, vmax=np.max(STI101)),
                          units=dict(stim=''),
                          scalings=dict(stim=np.max(STI101)))
