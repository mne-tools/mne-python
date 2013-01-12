"""
============================================================
Visualize channel over epochs as images in sensor topography
============================================================

This will produce what is sometimes called event related
potential / field (ERP/ERF) images.

One sensor topography plot is produced with the evoked field images from
the selected channels.
"""
print __doc__

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pylab as pl

import mne
from mne import fiff
from mne.datasets import sample
from mne.layouts import read_layout
data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] = ['MEG 2443', 'EEG 053']
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=True, eog=True,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))

###############################################################################
# Show event related fields images

layout = read_layout('Vectorview-all')

title = 'ERF images - MNE sample data'
mne.viz.plot_topo_image_epochs(epochs, layout, sigma=0.5, vmin=-200, vmax=200,
                               colorbar=True, title=title)
pl.show()
