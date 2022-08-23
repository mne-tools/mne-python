# -*- coding: utf-8 -*-
"""
.. _ex-vector-mne-solution:

============================================
Plotting the full vector-valued MNE solution
============================================

The source space that is used for the inverse computation defines a set of
dipoles, distributed across the cortex. When visualizing a source estimate, it
is sometimes useful to show the dipole directions in addition to their
estimated magnitude. This can be accomplished by computing a
:class:`mne.VectorSourceEstimate` and plotting it with
:meth:`stc.plot <mne.VectorSourceEstimate.plot>`, which uses
:func:`~mne.viz.plot_vector_source_estimates` under the hood rather than
:func:`~mne.viz.plot_source_estimates`.

It can also be instructive to visualize the actual dipole/activation locations
in 3D space in a glass brain, as opposed to activations imposed on an inflated
surface (as typically done in :meth:`mne.SourceEstimate.plot`), as it allows
you to get a better sense of the underlying source geometry.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD-3-Clause

# %%

import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path / 'subjects'
smoothing_steps = 7

# Read evoked data
meg_path = data_path / 'MEG' / 'sample'
fname_evoked = meg_path / 'sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

# Read inverse solution
fname_inv = meg_path / 'sample_audvis-meg-oct-6-meg-inv.fif'
inv = read_inverse_operator(fname_inv)

# Apply inverse solution, set pick_ori='vector' to obtain a
# :class:`mne.VectorSourceEstimate` object
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc = apply_inverse(evoked, inv, lambda2, 'dSPM', pick_ori='vector')

# Use peak getter to move visualization to the time point of the peak magnitude
_, peak_time = stc.magnitude().get_peak(hemi='lh')

# %%
# Plot the source estimate:

# sphinx_gallery_thumbnail_number = 2
brain = stc.plot(
    initial_time=peak_time, hemi='lh', subjects_dir=subjects_dir,
    smoothing_steps=smoothing_steps)

# You can save a brain movie with:
# brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16, framerate=10,
#                  interpolation='linear', time_viewer=True)

# %%
# Plot the activation in the direction of maximal power for this data:

stc_max, directions = stc.project('pca', src=inv['src'])
# These directions must by design be close to the normals because this
# inverse was computed with loose=0.2
print('Absolute cosine similarity between source normals and directions: '
      f'{np.abs(np.sum(directions * inv["source_nn"][2::3], axis=-1)).mean()}')
brain_max = stc_max.plot(
    initial_time=peak_time, hemi='lh', subjects_dir=subjects_dir,
    time_label='Max power', smoothing_steps=smoothing_steps)

# %%
# The normal is very similar:

brain_normal = stc.project('normal', inv['src'])[0].plot(
    initial_time=peak_time, hemi='lh', subjects_dir=subjects_dir,
    time_label='Normal', smoothing_steps=smoothing_steps)

# %%
# You can also do this with a fixed-orientation inverse. It looks a lot like
# the result above because the ``loose=0.2`` orientation constraint keeps
# sources close to fixed orientation:

fname_inv_fixed = (
    meg_path / 'sample_audvis-meg-oct-6-meg-fixed-inv.fif')
inv_fixed = read_inverse_operator(fname_inv_fixed)
stc_fixed = apply_inverse(
    evoked, inv_fixed, lambda2, 'dSPM', pick_ori='vector')
brain_fixed = stc_fixed.plot(
    initial_time=peak_time, hemi='lh', subjects_dir=subjects_dir,
    smoothing_steps=smoothing_steps)
