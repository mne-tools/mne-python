"""
.. _tut_viz_epochs:

Visualize Source time courses
=============================

"""
# sphinx_gallery_thumbnail_number = 7

import os

import mne
from mne.datasets import sample

data_path = sample.data_path()
sample_dir = os.path.join(data_path, 'MEG', 'sample')
subjects_dir = os.path.join(data_path, 'subjects')

fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

fname_stc = os.path.join(sample_dir, 'sample_audvis-meg')

###############################################################################
# Load example data

# Read stc from file
stc = mne.read_source_estimate(fname_stc, subject='sample')

###############################################################################
# This tutorial focuses on visualization of source time courses.
# All of the surface plots introduced here are based on PySurfer. PySurfer
# in turn uses the 3D visualization capabilities of Mayavi.
# As in the rest of ...
stc.plot(subjects_dir=subjects_dir, initial_time=0.1)

###############################################################################
# Note that here we used initial_time=0.1, but we can also browse through
# time using ``time_viewer=True``.

###############################################################################
# We can also visualize volume source estimates. Let us first compute
# the source estimate from the inverse operator on a volume source space.

from mne.minimum_norm import apply_inverse, read_inverse_operator  # noqa
from mne import read_evokeds  # noqa

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-vol-7-meg-inv.fif'

evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inv = read_inverse_operator(fname_inv)
src = inv['src']

# Compute inverse solution
stc = apply_inverse(evoked, inv, lambda2, method)
stc.crop(0.0, 0.2)

###############################################################################
# Then, we can plot the stc. For this visualization, nilearn must be installed.
# This visualization is interactive. Click on any of the slices to explore
# the time series. Clicking on any time point will bring up the corresponding
# anatomical map.

# XXX: subject=None should work with stc object (?)
stc.plot(src, subject='sample', subjects_dir=subjects_dir)

###############################################################################
# We can also plot the vector source estimates, i.e., plot not only the
# magnitude but also the direction.
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'

inv = read_inverse_operator(fname_inv)
stc = apply_inverse(evoked, inv, lambda2, 'dSPM', pick_ori='vector')
stc.plot(subject='sample', subjects_dir=subjects_dir)

###############################################################################
# We can also plot the vector source estimates, i.e., plot not only the
# magnitude but also the direction.
fname_cov = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_bem = os.path.join(subjects_dir, 'sample', 'bem',
                         'sample-5120-bem-sol.fif')
fname_trans = os.path.join(data_path, 'MEG', 'sample',
                           'sample_audvis_raw-trans.fif')

evoked.pick_types(meg=True, eeg=False)
evoked_full = evoked.copy()
evoked.crop(0.07, 0.08)
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

# Plot the result in 3D brain with the MRI image.
dip.plot_locations(fname_trans, 'sample', subjects_dir, mode='orthoview')
