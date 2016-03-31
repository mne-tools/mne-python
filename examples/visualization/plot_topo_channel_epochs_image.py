"""
============================================================
Visualize channel over epochs as images in sensor topography
============================================================

This will produce what is sometimes called event related
potential / field (ERP/ERF) images.

One sensor topography plot is produced with the evoked field images from
the selected channels.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

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
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] = ['MEG 2443', 'EEG 053']
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=True, eog=True,
                       exclude='bads')

# Plot sensors in 3d
raw.plot_sensors(kind='3d', picks=picks)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))

###############################################################################
# Show event related fields images

layout = mne.find_layout(epochs.info, 'meg')  # use full layout

title = 'ERF images - MNE sample data'
mne.viz.plot_topo_image_epochs(epochs, layout, sigma=0.5, vmin=-200, vmax=200,
                               colorbar=True, title=title)
plt.show()
