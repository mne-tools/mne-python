"""
========================================
Extracting a functional label from a STC
========================================

We first apply a dSPM inverse operator to get source activations.
Then we make a functional label from the contiguous patch of activity within
33% of the peak activation level, at the peak activation time.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
subjects_dir = data_path + '/subjects'
perc_thresh = 0.33

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)
src = inverse_operator['src']

# Compute inverse solution
stc = apply_inverse(evoked, inverse_operator, lambda2, method)

# Figure out the index of the maximum, and plot it
max_vert_idx, max_t_idx = np.unravel_index(np.argmax(stc.data), stc.data.shape)
if max_vert_idx < len(stc.vertices[0]):
    hemi = 'lh'
    max_vert = stc.vertices[0][max_vert_idx]
else:
    hemi = 'rh'
    max_vert = stc.vertices[1][max_vert_idx - len(stc.vertices[0])]
max_t = stc.times[max_t_idx]
max_val = stc.data[max_vert_idx, max_t_idx]
threshold = max_val * perc_thresh
stc.crop(max_t, max_t)  # only consider the peak time now
kwargs = dict(hemi='lh', subjects_dir=subjects_dir, smoothing_steps=5,
              clim=dict(kind='value',
                        lims=[threshold - 0.1, threshold, max_val]))
brain = stc.plot(**kwargs)
brain.add_foci(max_vert, coords_as_verts=True, hemi=hemi)

##############################################################################
# Figure out where we should draw our functional ROI boundaries

# Use our clustering function to find contiguous segments, using a
# pass-through stat fun and percentage threshold. All we care about here
# are the clusters:

connectivity = mne.spatial_src_connectivity(src, verbose='error')  # holes
_, clusters, _, _ = mne.stats.spatio_temporal_cluster_1samp_test(
    np.array([stc.data]), threshold, n_permutations=1, tail=1,
    stat_fun=lambda x: x.mean(0),  # passes through stc.data from [stc.data]
    connectivity=connectivity)
for cluster in clusters:
    cluster = cluster[0]  # just care about space indices
    if max_vert_idx in cluster:
        break  # found our cluster
else:  # in case we did not "break"
    raise RuntimeError('Clustering failed somehow!')
if hemi == 'lh':
    verts = stc.vertices[0][cluster]
else:
    verts = stc.vertices[1][cluster - len(stc.vertices[0])]
label = mne.Label(verts, hemi=hemi, subject=stc.subject)
label = label.fill(src)  # optional, for plotting purposes
brain = stc.plot(**kwargs)
brain.add_label(label, color='k', alpha=0.75, borders=True)
