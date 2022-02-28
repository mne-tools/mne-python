# -*- coding: utf-8 -*-
"""
.. _ex-label-time-series:

====================================================
Extracting the time series of activations in a label
====================================================

We first apply a dSPM inverse operator to get signed activations in a label
(with positive and negative values) and we then compare different strategies
to average the times series in a label. We compare a simple average, with an
averaging using the dipoles normal (flip mode) and then a PCA,
also using a sign flip.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
label = 'Aud-lh'
meg_path = data_path / 'MEG' / 'sample'
label_fname = meg_path / 'labels' / f'{label}.label'
fname_inv = meg_path / 'sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = meg_path / 'sample_audvis-ave.fif'

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)
src = inverse_operator['src']

# %%
# Compute inverse solution
# ------------------------
pick_ori = "normal"  # Get signed values to see the effect of sign flip
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_ori=pick_ori)

label = mne.read_label(label_fname)

stc_label = stc.in_label(label)
modes = ('mean', 'mean_flip', 'pca_flip')
tcs = dict()
for mode in modes:
    tcs[mode] = stc.extract_label_time_course(label, src, mode=mode)
print("Number of vertices : %d" % len(stc_label.data))

# %%
# View source activations
# -----------------------

fig, ax = plt.subplots(1)
t = 1e3 * stc_label.times
ax.plot(t, stc_label.data.T, 'k', linewidth=0.5, alpha=0.5)
pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5),
      path_effects.Normal()]
for mode, tc in tcs.items():
    ax.plot(t, tc[0], linewidth=3, label=str(mode), path_effects=pe)
xlim = t[[0, -1]]
ylim = [-27, 22]
ax.legend(loc='upper right')
ax.set(xlabel='Time (ms)', ylabel='Source amplitude',
       title='Activations in Label %r' % (label.name),
       xlim=xlim, ylim=ylim)
mne.viz.tight_layout()

# %%
# Using vector solutions
# ----------------------
# It's also possible to compute label time courses for a
# :class:`mne.VectorSourceEstimate`, but only with ``mode='mean'``.

pick_ori = 'vector'
stc_vec = apply_inverse(evoked, inverse_operator, lambda2, method,
                        pick_ori=pick_ori)
data = stc_vec.extract_label_time_course(label, src)
fig, ax = plt.subplots(1)
stc_vec_label = stc_vec.in_label(label)
colors = ['#EE6677', '#228833', '#4477AA']
for ii, name in enumerate('XYZ'):
    color = colors[ii]
    ax.plot(t, stc_vec_label.data[:, ii].T, color=color, lw=0.5, alpha=0.5,
            zorder=5 - ii)
    ax.plot(t, data[0, ii], lw=3, color=color, label='+' + name, zorder=8 - ii,
            path_effects=pe)
ax.legend(loc='upper right')
ax.set(xlabel='Time (ms)', ylabel='Source amplitude',
       title='Mean vector activations in Label %r' % (label.name,),
       xlim=xlim, ylim=ylim)
mne.viz.tight_layout()
