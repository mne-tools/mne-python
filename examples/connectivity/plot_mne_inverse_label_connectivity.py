"""
=========================================================================
Compute source space connectivity and visualize it using a circular graph
=========================================================================

This example computes connectivity between 68 regions in source space based on
dSPM inverse solutions and a FreeSurfer cortical parcellation. All-to-all
functional and effective connectivity measures are obtained from two different
methods: non-parametric spectral estimates and multivariate autoregressive
(MVAR) models. The connectivity is visualized using a circular graph which is
ordered based on the locations of the regions.

MVAR connectivity is computed with the Source Connectivity Toolbox (SCoT), see
http://scot-dev.github.io/scot-doc/index.html for details.
"""

# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Martin Billinger <martin.billinger@tugraz.at>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Nicolas P. Rougier (graph code borrowed from his matplotlib gallery)
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np

import mne
from mne.datasets import sample
from mne.io import Raw
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity, mvar_connectivity
from mne.viz import (circular_layout, plot_connectivity_circle,
                     plot_connectivity_inoutcircles)
from mne.externals.scot.connectivity_statistics import significance_fdr

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

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list.
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_annot('sample', parc='aparc', subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

# Average the source estimates within each label using sign-flips to reduce
# signal cancellations. We do not return a generator, because we want to use
# the estimates repeatedly.
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip',
                                         return_generator=False)

# First, compute connectivity from spectral estimates in the alpha band.
fmin = 8.
fmax = 13.
sfreq = raw.info['sfreq']  # the sampling frequency

spec_methods = ['wpli2_debiased', 'coh']
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts,
        method=spec_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=2)

# con is a 3D array, get the connectivity for the first (and only) freq. band
# for each method
con_spec = dict()
for method, c in zip(spec_methods, con):
    con_spec[method] = c[:, :, 0]

# Second, compute connectivity from multivariate autoregressive models.
mvar_methods = ['PDC', 'COH']
con, freqs, order, p_vals = mvar_connectivity(label_ts, mvar_methods,
                                              sfreq=sfreq, fmin=fmin,
                                              fmax=fmax, ridge=10,
                                              n_surrogates=100, n_jobs=-1)

# Get connectivity for the first frequency band. Set connectivity to 0 if not
# significant, while compensating for multiple testing by controlling the false
# discovery rate.
con_mvar = dict()
for method, c, p in zip(mvar_methods, con, p_vals):
    con_mvar[method] = c[:, :, 0] * significance_fdr(p[:, :, 0], 0.01)

# Now, we visualize the connectivity using a circular graph layout

# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]

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

# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
plot_connectivity_circle(con_spec['wpli2_debiased'], label_names, n_lines=300,
                         node_angles=node_angles, node_colors=label_colors,
                         title='All-to-All Connectivity left-Auditory '
                               'Condition (WPLI^2, debiased)')
import matplotlib.pyplot as plt
plt.savefig('circle.png', facecolor='black')

# Compare coherence from both estimation methods
fig = plt.figure(num=None, figsize=(8, 4), facecolor='black')
for ii, (con, method) in enumerate(zip([con_spec['coh'], con_mvar['COH']],
                                       ['Spectral', 'MVAR'])):
    plot_connectivity_circle(con, label_names, n_lines=300,
                             node_angles=node_angles, node_colors=label_colors,
                             title=method, padding=0, fontsize_colorbar=6,
                             fig=fig, subplot=(1, 2, ii + 1), show_names=False)
plt.suptitle('All-to-all coherence', color='white', fontsize=14)

# Show effective (directed) connectivity for one node
plot_connectivity_inoutcircles(con_mvar['PDC'], 'superiortemporal-lh',
                               label_names, node_angles=node_angles, padding=0,
                               node_colors=label_colors, show_names=False,
                               title='Effective connectivity (PDC)')

plt.show()
