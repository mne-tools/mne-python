"""
===========================================================================
Compute source leakage among labels and visualize it using a circular graph
===========================================================================

This example computes all-to-all pairwise leakage between 68 regions in
source space based on MNE inverse solutions and a FreeSurfer cortical
parcellation. Label-to-label leakage is estimated as the correlation among the
labels' point-spread functions. It is visualized using a circular graph
which is ordered based on the locations of the regions in the axial plane.
"""
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nicolas P. Rougier (graph code borrowed from his matplotlib gallery)
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import (read_inverse_operator,
                              make_inverse_resolution_matrix,
                              get_point_spread)

from mne.viz import circular_layout, plot_connectivity_circle

print(__doc__)

###############################################################################
# Load our data
# -------------
#
# We need matching forward solution and inverse operator to compute the
# resolution matrix, and labels to estimate label-to-label leakage.

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-fixed-inv.fif'

# read forward solution
forward = mne.read_forward_solution(fname_fwd)
# forward operator with fixed source orientations
forward = mne.convert_forward_solution(forward, surf_ori=True,
                                       force_fixed=True)

# Load inverse operator
inverse_operator = read_inverse_operator(fname_inv)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)
n_labels = len(labels)
label_colors = [label.color for label in labels]

# Source space used for inverse operator
src = inverse_operator['src']

###############################################################################
# Read and organise labels for cortical parcellation
# --------------------------------------------------


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

###############################################################################
# Compute point-spread function summaries (PCA) for all labels
# ------------------------------------------------------------

# Compute resolution matrix for MNE and sLORETA
lambda2 = 1. / 3.**2
method = 'MNE'
rm_mne = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method=method, lambda2=lambda2)
method = 'sLORETA'
rm_lor = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method=method, lambda2=lambda2)

# Compute first SVD component across PSFs within labels
# Note the differences in explained variance, probably due to different
# spatial extents of labels
n_comp = 5
stcs_psf_mne, pca_vars_mne = get_point_spread(
    rm_mne, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=True)
stcs_psf_lor, pca_vars_lor = get_point_spread(
    rm_lor, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=True)

n_verts = rm_mne.shape[0]

###############################################################################
# We can show the explained variances of principal components per label. Note
# how they differ across labels, most likely due to their varying spatial
# extent.
# The output shows the summed variance explained by the first five principal
# components as well as the explained variances of the individual components.
np.set_printoptions(precision=1)
for [name, var] in zip(label_names, pca_vars_mne):

    print('%s: %.1f%%' % (name, var.sum()))
    print(var)

###############################################################################
# Evaluate leakage based on label-to-label PSF correlations
# ---------------------------------------------------------
# Note that correlations ignore the overall amplitude of PSFs, i.e. they do
# not show which region will potentially be the bigger "leaker".

# get PSFs from Source Estimate objects into matrix
psfs_mat = np.zeros([n_labels, n_verts])

# Leakage matrix for MNE, get first principal component per label
for [i, s] in enumerate(stcs_psf_mne):
    psfs_mat[i, :] = s.data[:, 0]

# compute label-to-label leakage as Pearson correlation of PSFs
leakage_mne = np.corrcoef(psfs_mat)
# sign of correlation is arbitrary, so take absolute values
leakage_mne = np.abs(leakage_mne)

# Leakage matrix for sLORETA
for [i, s] in enumerate(stcs_psf_lor):
    psfs_mat[i, :] = s.data[:, 0]

# compute label-to-label leakage as Pearson correlation of PSFs
leakage_lor = np.corrcoef(psfs_mat)
# sign of correlation is arbitrary, so take absolute values
leakage_lor = np.abs(leakage_lor)

# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 200 strongest connections.
plt.ion()

fig = plt.figure(num=None, figsize=(8, 4), facecolor='black')

plot_connectivity_circle(leakage_mne, label_names, n_lines=200,
                         node_angles=node_angles, node_colors=label_colors,
                         title='MNE Leakage', fig=fig, subplot=(1, 2, 1))

no_names = [''] * len(label_names)

plot_connectivity_circle(leakage_lor, no_names, n_lines=200,
                         node_angles=node_angles, node_colors=label_colors,
                         title='sLORETA Leakage', padding=0,
                         fontsize_colorbar=6, fig=fig, subplot=(1, 2, 2))
###############################################################################
# The leakage patterns for MNE and sLORETA are very similar. Most leakage
# occurs for neighbouring regions, but also for deeper regions across
# hemispheres.

###############################################################################
# Save the figure (optional)
# --------------------------
#
# By default matplotlib does not save using the facecolor, even though this was
# set when the figure was generated. If not set via savefig, the labels, title,
# and legend will be cut off from the output png file.

fname_fig = data_path + '/MEG/sample/plot_label_leakage.png'
fig.savefig(fname_fig, facecolor='black')


###############################################################################
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

# maximum for scaling across plots
max_val = np.max([stc_lh.data, stc_rh.data])

brain_lh = stc_lh.plot(subjects_dir=subjects_dir, subject='sample',
                       hemi='both', views='caudal',
                       clim=dict(kind='value',
                                 pos_lims=(0, max_val / 2., max_val)))
brain_lh.add_text(0.1, 0.9, label_names[idx[0]], 'title', font_size=16)

brain_rh = stc_rh.plot(subjects_dir=subjects_dir, subject='sample',
                       hemi='both', views='caudal',
                       clim=dict(kind='value',
                                 pos_lims=(0, max_val / 2., max_val)))
brain_rh.add_text(0.1, 0.9, label_names[idx[1]], 'title', font_size=16)

###############################################################################
# Both summary PSFs are confined to their respective hemispheres, indicating
# that there is indeed low leakage between these two regions.
