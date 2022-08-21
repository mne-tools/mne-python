# -*- coding: utf-8 -*-
"""
.. _ex-source-leakage:

============================================================
Visualize source leakage among labels using a circular graph
============================================================

This example computes all-to-all pairwise leakage among 68 regions in
source space based on MNE inverse solutions and a FreeSurfer cortical
parcellation. Label-to-label leakage is estimated as the correlation among the
labels' point-spread functions (PSFs). It is visualized using a circular graph
which is ordered based on the locations of the regions in the axial plane.
"""
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nicolas P. Rougier (graph code borrowed from his matplotlib gallery)
#
# License: BSD-3-Clause

# %%

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import (read_inverse_operator,
                              make_inverse_resolution_matrix,
                              get_point_spread)

from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle


print(__doc__)

# %%
# Load forward solution and inverse operator
# ------------------------------------------
#
# We need a matching forward solution and inverse operator to compute
# resolution matrices for different methods.

data_path = sample.data_path()
subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
fname_fwd = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = meg_path / 'sample_audvis-meg-oct-6-meg-fixed-inv.fif'
forward = mne.read_forward_solution(fname_fwd)
# Convert forward solution to fixed source orientations
mne.convert_forward_solution(
    forward, surf_ori=True, force_fixed=True, copy=False)
inverse_operator = read_inverse_operator(fname_inv)

# Compute resolution matrices for MNE
rm_mne = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method='MNE', lambda2=1. / 3.**2)
src = inverse_operator['src']
del forward, inverse_operator  # save memory

# %%
# Read and organise labels for cortical parcellation
# --------------------------------------------------
#
# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)
n_labels = len(labels)
label_colors = [label.color for label in labels]
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
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# %%
# Compute point-spread function summaries (PCA) for all labels
# ------------------------------------------------------------
#
# We summarise the PSFs per label by their first five principal components, and
# use the first component to evaluate label-to-label leakage below.

# Compute first PCA component across PSFs within labels.
# Note the differences in explained variance, probably due to different
# spatial extents of labels.
n_comp = 5
stcs_psf_mne, pca_vars_mne = get_point_spread(
    rm_mne, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=True)
n_verts = rm_mne.shape[0]
del rm_mne

# %%
# We can show the explained variances of principal components per label. Note
# how they differ across labels, most likely due to their varying spatial
# extent.

with np.printoptions(precision=1):
    for [name, var] in zip(label_names, pca_vars_mne):
        print(f'{name}: {var.sum():.1f}% {var}')

# %%
# The output shows the summed variance explained by the first five principal
# components as well as the explained variances of the individual components.
#
# Evaluate leakage based on label-to-label PSF correlations
# ---------------------------------------------------------
#
# Note that correlations ignore the overall amplitude of PSFs, i.e. they do
# not show which region will potentially be the bigger "leaker".

# get PSFs from Source Estimate objects into matrix
psfs_mat = np.zeros([n_labels, n_verts])
# Leakage matrix for MNE, get first principal component per label
for [i, s] in enumerate(stcs_psf_mne):
    psfs_mat[i, :] = s.data[:, 0]
# Compute label-to-label leakage as Pearson correlation of PSFs
# Sign of correlation is arbitrary, so take absolute values
leakage_mne = np.abs(np.corrcoef(psfs_mat))

# Save the plot order and create a circular layout
node_order = lh_labels[::-1] + rh_labels  # mirror label order across hemis
node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])
# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 200 strongest connections.
fig, ax = plt.subplots(
    figsize=(8, 8), facecolor='black', subplot_kw=dict(projection='polar'))
plot_connectivity_circle(leakage_mne, label_names, n_lines=200,
                         node_angles=node_angles, node_colors=label_colors,
                         title='MNE Leakage', ax=ax)

# %%
# Most leakage occurs for neighbouring regions, but also for deeper regions
# across hemispheres.
#
# Save the figure (optional)
# --------------------------
#
# Matplotlib controls figure facecolor separately for interactive display
# versus for saved figures. Thus when saving you must specify ``facecolor``,
# else your labels, title, etc will not be visible::
#
#     >>> fname_fig = meg_path / 'plot_label_leakage.png'
#     >>> fig.savefig(fname_fig, facecolor='black')
#
# Plot PSFs for individual labels
# -------------------------------
#
# Let us confirm for left and right lateral occipital lobes that there is
# indeed no leakage between them, as indicated by the correlation graph.
# We can plot the summary PSFs for both labels to examine the spatial extent of
# their leakage.

# left and right lateral occipital
idx = [22, 23]
stc_lh = stcs_psf_mne[idx[0]]
stc_rh = stcs_psf_mne[idx[1]]

# Maximum for scaling across plots
max_val = np.max([stc_lh.data, stc_rh.data])

# %%
# Point-spread function for the lateral occipital label in the left hemisphere

brain_lh = stc_lh.plot(subjects_dir=subjects_dir, subject='sample',
                       hemi='both', views='caudal',
                       clim=dict(kind='value',
                                 pos_lims=(0, max_val / 2., max_val)))
brain_lh.add_text(0.1, 0.9, label_names[idx[0]], 'title', font_size=16)

# %%
# and in the right hemisphere.

brain_rh = stc_rh.plot(subjects_dir=subjects_dir, subject='sample',
                       hemi='both', views='caudal',
                       clim=dict(kind='value',
                                 pos_lims=(0, max_val / 2., max_val)))
brain_rh.add_text(0.1, 0.9, label_names[idx[1]], 'title', font_size=16)

# %%
# Both summary PSFs are confined to their respective hemispheres, indicating
# that there is indeed low leakage between these two regions.
