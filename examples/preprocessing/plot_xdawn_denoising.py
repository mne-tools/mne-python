"""
================
 XDAWN Denoising
================

Xdawn filters are trained from epochs, signal is projected in the sources
space and then projected back in the sensor space using only the first two
xdawn components. The process is similar to an ICA, but is
supervised in order to maximize the signal to signal + noise ratio of the
evoked response.
"""

# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)


import mne
from mne import io, compute_raw_data_covariance
from mne.datasets import sample
from mne.preprocessing.xdawn import Xdawn
from mne.viz import plot_image_epochs

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.3
event_id = dict(vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(1, 20, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       exclude='bads')
# Epoching
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True,
                    add_eeg_ref=False, verbose=False)

# Plot image epoch before xdawn
plot_image_epochs(epochs['vis_r'], picks=[230], vmin=-500, vmax=500)

# Estimates signal covariance
signal_cov = compute_raw_data_covariance(raw, picks=picks)

# Xdawn instance
xd = Xdawn(n_components=2, signal_cov=signal_cov)

# Fit xdawn
xd.fit(epochs)

# Denoise epochs
epochs_denoised = xd.apply(epochs)

# Plot image epoch after xdawn
plot_image_epochs(epochs_denoised['vis_r'],
                  picks=[230], vmin=-500, vmax=500)
