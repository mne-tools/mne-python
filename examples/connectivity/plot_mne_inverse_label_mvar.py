"""
=========================================================================
Compute source space connectivity and visualize it using a circular graph
=========================================================================

This example computes the all-to-all connectivity between 68 regions in
source space based on dSPM inverse solutions and a FreeSurfer cortical
parcellation. The connectivity is visualized using a circular graph which
is ordered based on the locations of the regions.
"""

# Authors: Martin Billinger <martin.billinger@tugraz.at>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
import mne
from mne.datasets import sample
from mne.io import Raw
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import mvar_connectivity
from mne.viz import (circular_layout, plot_connectivity_circle,
                     plot_connectivity_matrix)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Load data
inverse_operator = read_inverse_operator(fname_inv)
raw = Raw(fname_raw)
events = mne.read_events(fname_event)

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# Pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# Compute inverse solution and for each epoch.
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_annot('sample', parc='aparc', subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]
label_names = [label.name for label in labels]

# Average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we return a generator
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip',
                                         return_generator=False)

# Now we are ready to compute connectivity in the alpha and beta bands.
band_names = ('alpha', 'beta')
band_colors = ('hot', 'bone')
fmin = (8., 16.)
fmax = (13., 24.)
mvar_order = 30  # model order determines frequency resolution
sfreq = raw.info['sfreq']  # the sampling frequency
con_methods = ['PDC', 'COH']
con, freqs = mvar_connectivity(label_ts, con_methods, mvar_order,
                               sfreq=sfreq, fmin=fmin, fmax=fmax)

con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c

# First visualize directed (effective) connectivity matrix
plot_connectivity_matrix(con_res['PDC'][:, :, 0], label_names,
                         node_colors=label_colors, title='All-to-All '
                         'Connectivity left-Auditory Condition (PDC)')

# Now, we visualize the connectivity using a circular graph layout

# Reorder the labels based on their location in the left hemi
lh_labels = [name for name in label_names if name.endswith('lh')]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (ypos, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

import matplotlib.pyplot as plt

# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
fig = plt.figure(num=None, figsize=(8, 4), facecolor='black')
for i, (b, cm) in enumerate(zip(band_names, band_colors)):
    plot_connectivity_circle(con_res['COH'][:, :, i], label_names, n_lines=300,
                             node_angles=node_angles, node_colors=label_colors,
                             fig=fig, subplot=(1, 2, i + 1), colormap=cm,
                             title='All-to-All Connectivity left-Auditory '
                                   'Condition (COH, {} band)'.format(b))

plt.show()
