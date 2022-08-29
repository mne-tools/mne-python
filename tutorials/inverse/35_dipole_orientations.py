# -*- coding: utf-8 -*-
"""
.. _tut-dipole-orientations:

==================================================================
The role of dipole orientations in distributed source localization
==================================================================

When performing source localization in a distributed manner
(i.e., using MNE/dSPM/sLORETA/eLORETA),
the source space is defined as a grid of dipoles that spans a large portion of
the cortex. These dipoles have both a position and an orientation. In this
tutorial, we will look at the various options available to restrict the
orientation of the dipoles and the impact on the resulting source estimate.

See :ref:`inverse_orientation_constraints` for related information.
"""

# %%
# Load data
# ---------
# Load everything we need to perform source localization on the sample dataset.

import mne
import numpy as np
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
evokeds = mne.read_evokeds(meg_path / 'sample_audvis-ave.fif')
left_auditory = evokeds[0].apply_baseline()
fwd = mne.read_forward_solution(
    meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif')
mne.convert_forward_solution(fwd, surf_ori=True, copy=False)
noise_cov = mne.read_cov(meg_path / 'sample_audvis-cov.fif')
subject = 'sample'
subjects_dir = data_path / 'subjects'
trans_fname = meg_path / 'sample_audvis_raw-trans.fif'

# %%
# The source space
# ----------------
# Let's start by examining the source space as constructed by the
# :func:`mne.setup_source_space` function. Dipoles are placed along fixed
# intervals on the cortex, determined by the ``spacing`` parameter. The source
# space does not define the orientation for these dipoles.

lh = fwd['src'][0]  # Visualize the left hemisphere
verts = lh['rr']  # The vertices of the source space
tris = lh['tris']  # Groups of three vertices that form triangles
dip_pos = lh['rr'][lh['vertno']]  # The position of the dipoles
dip_ori = lh['nn'][lh['vertno']]
dip_len = len(dip_pos)
dip_times = [0]
white = (1.0, 1.0, 1.0)  # RGB values for a white color

actual_amp = np.ones(dip_len)  # misc amp to create Dipole instance
actual_gof = np.ones(dip_len)  # misc GOF to create Dipole instance
dipoles = mne.Dipole(dip_times, dip_pos, actual_amp, dip_ori, actual_gof)
trans = mne.read_trans(trans_fname)

fig = mne.viz.create_3d_figure(size=(600, 400), bgcolor=white)
coord_frame = 'mri'

# Plot the cortex
mne.viz.plot_alignment(
    subject=subject, subjects_dir=subjects_dir, trans=trans, surfaces='white',
    coord_frame=coord_frame, fig=fig)

# Mark the position of the dipoles with small red dots
mne.viz.plot_dipole_locations(
    dipoles=dipoles, trans=trans, mode='sphere', subject=subject,
    subjects_dir=subjects_dir, coord_frame=coord_frame, scale=7e-4, fig=fig)

mne.viz.set_3d_view(figure=fig, azimuth=180, distance=0.25)

# %%
# .. _plot_dipole_orientations_fixed_orientations:
#
# Fixed dipole orientations
# -------------------------
# While the source space defines the position of the dipoles, the inverse
# operator defines the possible orientations of them. One of the options is to
# assign a fixed orientation. Since the neural currents from which MEG and EEG
# signals originate flows mostly perpendicular to the cortex
# :footcite:`HamalainenEtAl1993`, restricting the orientation of the dipoles
# accordingly places a useful restriction on the source estimate.
#
# By specifying ``fixed=True`` when calling
# :func:`mne.minimum_norm.make_inverse_operator`, the dipole orientations are
# fixed to be orthogonal to the surface of the cortex, pointing outwards. Let's
# visualize this:

fig = mne.viz.create_3d_figure(size=(600, 400))

# Plot the cortex
mne.viz.plot_alignment(
    subject=subject, subjects_dir=subjects_dir, trans=trans,
    surfaces='white', coord_frame='head', fig=fig)

# Show the dipoles as arrows pointing along the surface normal
mne.viz.plot_dipole_locations(
    dipoles=dipoles, trans=trans, mode='arrow', subject=subject,
    subjects_dir=subjects_dir, coord_frame='head', scale=7e-4, fig=fig)

mne.viz.set_3d_view(figure=fig, azimuth=180, distance=0.1)

# %%
# Restricting the dipole orientations in this manner leads to the following
# source estimate for the sample data:

# Compute the source estimate for the left auditory condition in the sample
# dataset.
inv = make_inverse_operator(left_auditory.info, fwd, noise_cov, fixed=True)
stc = apply_inverse(left_auditory, inv, pick_ori=None)

# Visualize it at the moment of peak activity.
_, time_max = stc.get_peak(hemi='lh')
brain_fixed = stc.plot(surface='white', subjects_dir=subjects_dir,
                       initial_time=time_max, time_unit='s', size=(600, 400))
mne.viz.set_3d_view(figure=brain_fixed, focalpoint=(0., 0., 50))

# %%
# The direction of the estimated current is now restricted to two directions:
# inward and outward. In the plot, blue areas indicate current flowing inwards
# and red areas indicate current flowing outwards. Given the curvature of the
# cortex, groups of dipoles tend to point in the same direction: the direction
# of the electromagnetic field picked up by the sensors.

# %%
# .. _plot_dipole_orientations_fLOC_orientations:
#
# Loose dipole orientations
# -------------------------
# Forcing the source dipoles to be strictly orthogonal to the cortex makes the
# source estimate sensitive to the spacing of the dipoles along the cortex,
# since the curvature of the cortex changes within each ~10 square mm patch.
# Furthermore, misalignment of the MEG/EEG and MRI coordinate frames is more
# critical when the source dipole orientations are strictly constrained
# :footcite:`LinEtAl2006`. To lift the restriction on the orientation of the
# dipoles, the inverse operator has the ability to place not one, but three
# dipoles at each location defined by the source space. These three dipoles are
# placed orthogonally to form a Cartesian coordinate system. Let's visualize
# this:
fig = mne.viz.create_3d_figure(size=(600, 400))

# Plot the cortex
mne.viz.plot_alignment(
    subject=subject, subjects_dir=subjects_dir, trans=trans,
    surfaces='white', coord_frame='head', fig=fig)

# Show the three dipoles defined at each location in the source space
mne.viz.plot_alignment(
    subject=subject, subjects_dir=subjects_dir, trans=trans, fwd=fwd,
    surfaces='white', coord_frame='head', fig=fig)

mne.viz.set_3d_view(figure=fig, azimuth=180, distance=0.1)

# %%
# When computing the source estimate, the activity at each of the three dipoles
# is collapsed into the XYZ components of a single vector, which leads to the
# following source estimate for the sample data:

# Make an inverse operator with loose dipole orientations
inv = make_inverse_operator(left_auditory.info, fwd, noise_cov, fixed=False,
                            loose=1.0)

# Compute the source estimate, indicate that we want a vector solution
stc = apply_inverse(left_auditory, inv, pick_ori='vector')

# Visualize it at the moment of peak activity.
_, time_max = stc.magnitude().get_peak(hemi='lh')
brain_mag = stc.plot(subjects_dir=subjects_dir, initial_time=time_max,
                     time_unit='s', size=(600, 400), overlay_alpha=0)
mne.viz.set_3d_view(figure=brain_mag, focalpoint=(0., 0., 50))

# %%
# .. _plot_dipole_orientations_vLOC_orientations:
#
# Limiting orientations, but not fixing them
# ------------------------------------------
# Often, the best results will be obtained by allowing the dipoles to have
# somewhat free orientation, but not stray too far from a orientation that is
# perpendicular to the cortex. The ``loose`` parameter of the
# :func:`mne.minimum_norm.make_inverse_operator` allows you to specify a value
# between 0 (fixed) and 1 (unrestricted or "free") to indicate the amount the
# orientation is allowed to deviate from the surface normal.

# Set loose to 0.2, the default value
inv = make_inverse_operator(left_auditory.info, fwd, noise_cov, fixed=False,
                            loose=0.2)
stc = apply_inverse(left_auditory, inv, pick_ori='vector')

# Visualize it at the moment of peak activity.
_, time_max = stc.magnitude().get_peak(hemi='lh')
brain_loose = stc.plot(subjects_dir=subjects_dir, initial_time=time_max,
                       time_unit='s', size=(600, 400), overlay_alpha=0)
mne.viz.set_3d_view(figure=brain_loose, focalpoint=(0., 0., 50))

# %%
# Discarding dipole orientation information
# -----------------------------------------
# Often, further analysis of the data does not need information about the
# orientation of the dipoles, but rather their magnitudes. The ``pick_ori``
# parameter of the :func:`mne.minimum_norm.apply_inverse` function allows you
# to specify whether to return the full vector solution (``'vector'``) or
# rather the magnitude of the vectors (``None``, the default) or only the
# activity in the direction perpendicular to the cortex (``'normal'``).

# Only retain vector magnitudes
stc = apply_inverse(left_auditory, inv, pick_ori=None)

# Visualize it at the moment of peak activity
_, time_max = stc.get_peak(hemi='lh')
brain = stc.plot(surface='white', subjects_dir=subjects_dir,
                 initial_time=time_max, time_unit='s', size=(600, 400))
mne.viz.set_3d_view(figure=brain, focalpoint=(0., 0., 50))

# %%
# References
# ----------
# .. footbibliography::
