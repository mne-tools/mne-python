# -*- coding: utf-8 -*-
"""
.. _tut_dipole_orentiations:

The role of dipole orientations in distributed source localization
==================================================================

When performing source localization in a distributed manner (MNE/dSPM/sLORETA),
the source space is defined as a grid of dipoles that spans a large portion of
the cortex. These dipoles have both a position and an orientation. In this
tutorial, we will look at the various options available to restrict the
orientation of the dipoles and the impact on the resulting source estimate.
"""

###############################################################################
# Loading data
# ------------
# Load everything we need to perform source localization on the sample dataset.

from mayavi import mlab
import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

data_path = sample.data_path()
evokeds = mne.read_evokeds(data_path + '/MEG/sample/sample_audvis-ave.fif')
left_auditory = evokeds[0].apply_baseline()
fwd = mne.read_forward_solution(
    data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif',
    surf_ori=True)
noise_cov = mne.read_cov(data_path + '/MEG/sample/sample_audvis-cov.fif')
subjects_dir = data_path + '/subjects'

###############################################################################
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
white = (1.0, 1.0, 1.0)  # RGB values for a white color
gray = (0.5, 0.5, 0.5)  # RGB values for a gray color
red = (1.0, 0.0, 0.0)  # RGB valued for a red color

mlab.figure(size=(600, 400), bgcolor=white)

# Plot the cortex
mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], tris, color=gray)

# Mark the position of the dipoles with small red dots
mlab.points3d(dip_pos[:, 0], dip_pos[:, 1], dip_pos[:, 2], color=red,
              scale_factor=1E-3)

mlab.view(azimuth=180, distance=0.25)

###############################################################################
# Fixed dipole orientations
# -------------------------
# While the source space defines the position of the dipoles, the inverse
# operator defines the possible orientations of them. One of the options is to
# assign a fixed orientation. Since the neural currents from which MEG and EEG
# signals originate flows mostly perpendicular to the cortex [1]_, restricting
# the orientation of the dipoles accordingly places a useful restriction on the
# source estimate.
#
# By specifying ``fixed=True`` when calling
# :func:`mne.minimum_norm.make_inverse_operator`, the dipole orientations are
# fixed to be orthogonal to the surface of the cortex, pointing outwards. Let's
# visualize this:

mlab.figure(size=(600, 400), bgcolor=white)

# Plot the cortex
mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], tris, color=gray)

# Show the dipoles as arrows pointing along the surface normal
normals = lh['nn'][lh['vertno']]
mlab.quiver3d(dip_pos[:, 0], dip_pos[:, 1], dip_pos[:, 2],
              normals[:, 0], normals[:, 1], normals[:, 2],
              color=red, scale_factor=1E-3)

mlab.view(azimuth=180, distance=0.1)

###############################################################################
# Restricting the dipole orientations in this manner leads to the following
# source estimate for the sample data:

# Compute the source estimate for the 'left - auditory' condition in the sample
# dataset.
inv = make_inverse_operator(left_auditory.info, fwd, noise_cov, fixed=True)
stc = apply_inverse(left_auditory, inv, pick_ori=None)

# Visualize it at the moment of peak activity.
_, time_max = stc.get_peak(hemi='lh')
brain = stc.plot(surface='white', subjects_dir=subjects_dir,
                 initial_time=time_max, time_unit='s', size=(600, 400))

###############################################################################
# The direction of the estimated current is now restricted to two directions:
# inward and outward. In the plot, blue areas indicate current flowing inwards
# and red areas indicate current flowing outwards. Given the curvature of the
# cortex, groups of dipoles tend to point in the same direction: the direction
# of the electromagnetic field picked up by the sensors.

###############################################################################
# Loose dipole orientations
# -------------------------
# Forcing the source dipoles to be strictly orthogonal to the cortex makes the
# source estimate sensitive to the spacing of the dipoles along the cortex,
# since the curvature of the cortex changes within each ~10 square mm patch.
# Furthermore, misalignment of the MEG/EEG and MRI coordinate frames is more
# critical when the source dipole orientations are strictly constrained [2]_.
# To lift the restriction on the orientation of the dipoles, the inverse
# operator has the ability to place not one, but three dipoles at each
# location defined by the source space. These three dipoles are placed
# orthogonally to form a Cartesian coordinate system. Let's visualize this:
mlab.figure(size=(600, 400), bgcolor=white)

# Define some more colors
green = (0.0, 1.0, 0.0)
blue = (0.0, 0.0, 1.0)

# Plot the cortex
mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], tris, color=gray)

# Make an inverse operator with loose dipole orientations
inv = make_inverse_operator(left_auditory.info, fwd, noise_cov, fixed=False,
                            loose=1.0)

# Show the three dipoles defined at each location in the source space
dip_dir = inv['source_nn'].reshape(-1, 3, 3)
dip_dir = dip_dir[:len(dip_pos)]  # Only select left hemisphere
for ori, color in zip((0, 1, 2), (red, green, blue)):
    mlab.quiver3d(dip_pos[:, 0], dip_pos[:, 1], dip_pos[:, 2],
                  dip_dir[:, ori, 0], dip_dir[:, ori, 1], dip_dir[:, ori, 2],
                  color=color, scale_factor=1E-3)

mlab.view(azimuth=180, distance=0.1)

###############################################################################
# When computing the source estimate, the activity at each of the three dipoles
# is collapsed into the XYZ components of a single vector, which leads to the
# following source estimate for the sample data:

# Compute the source estimate, indicate that we want a vector solution
stc = apply_inverse(left_auditory, inv, pick_ori='vector')

# Visualize it at the moment of peak activity.
_, time_max = stc.magnitude().get_peak(hemi='lh')
brain = stc.plot(subjects_dir=subjects_dir, initial_time=time_max,
                 time_unit='s', size=(600, 400), overlay_alpha=0)

###############################################################################
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
brain = stc.plot(subjects_dir=subjects_dir, initial_time=time_max,
                 time_unit='s', size=(600, 400), overlay_alpha=0)

###############################################################################
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

# Visualize it at the moment of peak activity.
_, time_max = stc.get_peak(hemi='lh')
brain = stc.plot(surface='white', subjects_dir=subjects_dir,
                 initial_time=time_max, time_unit='s', size=(600, 400))

###############################################################################
# References
# ----------
# .. [1] Hämäläinen, M. S., Hari, R., Ilmoniemi, R. J., Knuutila, J., &
#    Lounasmaa, O. V. "Magnetoencephalography - theory, instrumentation, and
#    applications to noninvasive studies of the working human brain", Reviews
#    of Modern Physics, 1993. http://dx.doi.org/10.1103/RevModPhys.65.413
#
# .. [2] Lin, F. H., Belliveau, J. W., Dale, A. M., & Hämäläinen, M. S. (2006).
#    Distributed current estimates using cortical orientation constraints.
#    Human Brain Mapping, 27(1), 1–13. http://doi.org/10.1002/hbm.20155
