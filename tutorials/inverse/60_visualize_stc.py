# -*- coding: utf-8 -*-
"""
.. _tut-viz-stcs:

====================================
Visualize source time courses (stcs)
====================================

This tutorial focuses on visualization of :term:`source estimates <STC>`.

Surface Source Estimates
------------------------
First, we get the paths for the evoked data and the source time courses (stcs).
"""

# %%

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample, fetch_hcp_mmp_parcellation
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne import read_evokeds

data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
subjects_dir = data_path / 'subjects'

fname_evoked = meg_path / 'sample_audvis-ave.fif'
fname_stc = meg_path / 'sample_audvis-meg'
fetch_hcp_mmp_parcellation(subjects_dir)

# %%
# Then, we read the stc from file.
stc = mne.read_source_estimate(fname_stc, subject='sample')

# %%
# This is a :class:`SourceEstimate <mne.SourceEstimate>` object.
print(stc)

# %%
# The SourceEstimate object is in fact a *surface* source estimate. MNE also
# supports volume-based source estimates but more on that later.
#
# We can plot the source estimate using the
# :func:`stc.plot <mne.SourceEstimate.plot>` just as in other MNE
# objects. Note that for this visualization to work, you must have ``PyVista``
# installed on your machine.
initial_time = 0.1
brain = stc.plot(subjects_dir=subjects_dir, initial_time=initial_time,
                 clim=dict(kind='value', lims=[3, 6, 9]),
                 smoothing_steps=7)

# %%
# You can also morph it to fsaverage and visualize it using a flatmap.

# sphinx_gallery_thumbnail_number = 3
stc_fs = mne.compute_source_morph(stc, 'sample', 'fsaverage', subjects_dir,
                                  smooth=5, verbose='error').apply(stc)
brain = stc_fs.plot(subjects_dir=subjects_dir, initial_time=initial_time,
                    clim=dict(kind='value', lims=[3, 6, 9]),
                    surface='flat', hemi='both', size=(1000, 500),
                    smoothing_steps=5, time_viewer=False,
                    add_data_kwargs=dict(
                        colorbar_kwargs=dict(label_font_size=10)))

# to help orient us, let's add a parcellation (red=auditory, green=motor,
# blue=visual)
brain.add_annotation('HCPMMP1_combined', borders=2)

# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16,
#                  interpolation='linear', framerate=10)

# %%
# Note that here we used ``initial_time=0.1``, but we can also browse through
# time using ``time_viewer=True``.
#
# In case ``PyVista`` is not available, we also offer a ``matplotlib``
# backend. Here we use verbose='error' to ignore a warning that not all
# vertices were used in plotting.
mpl_fig = stc.plot(subjects_dir=subjects_dir, initial_time=initial_time,
                   backend='matplotlib', verbose='error', smoothing_steps=7)

# %%
#
# Volume Source Estimates
# -----------------------
# We can also visualize volume source estimates (used for deep structures).
#
# Let us load the sensor-level evoked data. We select the MEG channels
# to keep things simple.
evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
evoked.pick_types(meg=True, eeg=False).crop(0.05, 0.15)
# this risks aliasing, but these data are very smooth
evoked.decimate(10, verbose='error')

# %%
# Then, we can load the precomputed inverse operator from a file.
fname_inv = meg_path / 'sample_audvis-meg-vol-7-meg-inv.fif'
inv = read_inverse_operator(fname_inv)
src = inv['src']
mri_head_t = inv['mri_head_t']

# %%
# The source estimate is computed using the inverse operator and the
# sensor-space data.
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stc = apply_inverse(evoked, inv, lambda2, method)
del inv

# %%
# This time, we have a different container
# (:class:`VolSourceEstimate <mne.VolSourceEstimate>`) for the source time
# course.
print(stc)

# %%
# This too comes with a convenient plot method.
stc.plot(src, subject='sample', subjects_dir=subjects_dir)

# %%
# For this visualization, ``nilearn`` must be installed.
# This visualization is interactive. Click on any of the anatomical slices
# to explore the time series. Clicking on any time point will bring up the
# corresponding anatomical map.
#
# We could visualize the source estimate on a glass brain. Unlike the previous
# visualization, a glass brain does not show us one slice but what we would
# see if the brain was transparent like glass, and
# :term:`maximum intensity projection`) is used:
stc.plot(src, subject='sample', subjects_dir=subjects_dir, mode='glass_brain')

# %%
# You can also extract label time courses using volumetric atlases. Here we'll
# use the built-in ``aparc+aseg.mgz``:

fname_aseg = subjects_dir / 'sample' / 'mri' / 'aparc+aseg.mgz'
label_names = mne.get_volume_labels_from_aseg(fname_aseg)
label_tc = stc.extract_label_time_course(fname_aseg, src=src)

lidx, tidx = np.unravel_index(np.argmax(label_tc), label_tc.shape)
fig, ax = plt.subplots(1)
ax.plot(stc.times, label_tc.T, 'k', lw=1., alpha=0.5)
xy = np.array([stc.times[tidx], label_tc[lidx, tidx]])
xytext = xy + [0.01, 1]
ax.annotate(
    label_names[lidx], xy, xytext, arrowprops=dict(arrowstyle='->'), color='r')
ax.set(xlim=stc.times[[0, -1]], xlabel='Time (s)', ylabel='Activation')
for key in ('right', 'top'):
    ax.spines[key].set_visible(False)
fig.tight_layout()

# %%
# We can plot several labels with the most activation in their time course
# for a more fine-grained view of the anatomical loci of activation.
labels = [label_names[idx] for idx in np.argsort(label_tc.max(axis=1))[:7]
          if 'unknown' not in label_names[idx].lower()]  # remove catch-all
brain = mne.viz.Brain('sample', hemi='both', surf='pial', alpha=0.5,
                      cortex='low_contrast', subjects_dir=subjects_dir)
brain.add_volume_labels(aseg='aparc+aseg', labels=labels)
brain.show_view(azimuth=250, elevation=40, distance=400)

# %%
# And we can project these label time courses back to their original
# locations and see how the plot has been smoothed:

stc_back = mne.labels_to_stc(fname_aseg, label_tc, src=src)
stc_back.plot(src, subjects_dir=subjects_dir, mode='glass_brain')

# %%
# Vector Source Estimates
# -----------------------
# If we choose to use ``pick_ori='vector'`` in
# :func:`apply_inverse <mne.minimum_norm.apply_inverse>`
fname_inv = (
    data_path / 'MEG' / 'sample' / 'sample_audvis-meg-oct-6-meg-inv.fif'
)
inv = read_inverse_operator(fname_inv)
stc = apply_inverse(evoked, inv, lambda2, 'dSPM', pick_ori='vector')
brain = stc.plot(subject='sample', subjects_dir=subjects_dir,
                 initial_time=initial_time, brain_kwargs=dict(
                     silhouette=True), smoothing_steps=7)

# %%
# Dipole fits
# -----------
# For computing a dipole fit, we need to load the noise covariance, the BEM
# solution, and the coregistration transformation files. Note that for the
# other methods, these were already used to generate the inverse operator.
fname_cov = meg_path / 'sample_audvis-cov.fif'
fname_bem = subjects_dir / 'sample' / 'bem' / 'sample-5120-bem-sol.fif'
fname_trans = meg_path / 'sample_audvis_raw-trans.fif'

##############################################################################
# Dipoles are fit independently for each time point, so let us crop our time
# series to visualize the dipole fit for the time point of interest.
evoked.crop(0.1, 0.1)
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

##############################################################################
# Finally, we can visualize the dipole.

dip.plot_locations(fname_trans, 'sample', subjects_dir)
