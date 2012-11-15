"""
===============================================
Compute all-to-all connectivity in sensor space
===============================================

Computes the Phase Locking Index (PLI) between all gradiometers and shows the
connectivity in 3D using the helmet geometry.
"""

# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)


print __doc__

import numpy as np

import mne
from mne import fiff
from mne.connectivity import spectral_connectivity
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

# Set up pick list
exclude = raw.info['bads'] + ['MEG 2443']  # bads + 1 more

# Pick MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                        exclude=exclude)

# Create epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))

# Compute connectivity for alpha band. We exclude the baseline period
fmin, fmax = 8., 13.
sfreq = raw.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period
con, freqs, n_epochs, n_tapers = spectral_connectivity(epochs,
    method='pli', sfreq=sfreq, fmin=fmin, fmax=fmax,
    faverage=True, tmin=tmin, adaptive=False, n_jobs=2)

# the epochs contain an EOG channel, which we remove now
ch_names = epochs.ch_names
idx = [ch_names.index(name) for name in ch_names if name.startswith('MEG')]
con = con[idx][:, idx]

# con is a 3D array where the last dimension is size one since we averaged
# over frequencies in a single band. Here we make it 2D
con = con[:, :, 0]

# Now, visualize the connectivity in 3D
from enthought.mayavi import mlab
mlab.figure(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))

# Plot the sensor locations
sens_loc = [raw.info['chs'][picks[i]]['loc'][:3] for i in idx]
sens_loc = np.array(sens_loc)

pts = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2],
                    color=(0, 0, 1), opacity=0.5, scale_factor=0.01)

# Get the strongest connections
n_con = 50  # show up to 50 connections
min_dist = 0.1  # exlude sensors that are less than 10cm apart
threshold = np.sort(con, axis=None)[-n_con]
ii, jj = np.where(con >= threshold)

# Remove close connections
con_nodes = list()
con_val = list()
for i, j in zip(ii, jj):
    if np.linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
        con_nodes.append((i, j))
        con_val.append(con[i, j])

con_val = np.array(con_val)

# Show the connections as tubes between sensors
vmax = np.max(con_val)
vmin = np.min(con_val)
for val, nodes in zip(con_val, con_nodes):
    x1, y1, z1 = sens_loc[nodes[0]]
    x2, y2, z2 = sens_loc[nodes[1]]
    mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                vmin=vmin, vmax=vmax, tube_radius=0.002)

mlab.scalarbar(title='Phase Locking Index (PLI)', nb_labels=4)

view = (144.4, 92.7, 0.63, np.array([-5.0e-05, 1.6e-02, 2.3e-02]))
roll = 86.4
mlab.view(*view)
mlab.roll(roll)
