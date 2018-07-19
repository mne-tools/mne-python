# -*- coding: utf-8 -*-
"""
Morphing Source Estimates using :class:`SourceMorph`
====================================================

In this tutorial we will morph different kinds of source estimation results
between individual subject spaces using :class:`mne.SourceMorph`.
For group level statistical analyses subject specific results have to be mapped
to a common space.

We will use precomputed data and morph surface and volume source estimates to a
common space. The common space of choice will be FreeSurfer's "fsaverage".

Furthermore we will convert our volume source estimate into a NIfTI image using
:meth:`morph.as_volume <mne.SourceMorph.as_volume>`.
"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)

import os

###############################################################################
# Setup
# -----
#
# We first import the required packages and define a list of filenames for
# various datasets we are going to use to run this tutorial.
import matplotlib.pylab as plt
import nibabel as nib
from mne import (read_evokeds, SourceMorph, read_source_estimate)
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from nilearn.image import index_img
from nilearn.plotting import plot_glass_brain

# We use the MEG and MRI setup from the MNE-sample dataset
sample_dir_raw = sample.data_path()
sample_dir = sample_dir_raw + '/MEG/sample'
subjects_dir = sample_dir_raw + '/subjects'

fname_evoked = sample_dir + '/sample_audvis-ave.fif'

fname_surf = os.path.join(sample_dir, 'sample_audvis-meg')
fname_vol = os.path.join(sample_dir,
                         'sample_audvis-grad-vol-7-fwd-sensmap-vol.w')

fname_inv_surf = os.path.join(sample_dir,
                              'sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif')
fname_inv_vol = os.path.join(sample_dir,
                             'sample_audvis-meg-vol-7-meg-inv.fif')

fname_t1_fsaverage = subjects_dir + '/fsaverage/mri/brain.mgz'

###############################################################################
# Data preparation
# ----------------
#
# First we load the respective example data for surface and volume source
# estimates
stc_surf = read_source_estimate(fname_surf, subject='sample')

# Afterwards we load the corresponding source spaces as well
src_surf = read_inverse_operator(fname_inv_surf)['src']
inv_src = read_inverse_operator(fname_inv_vol)
src_vol = inv_src['src']

# ensure subject is not None
src_vol[0]['subject_his_id'] = 'sample'

# For faster computation we redefine tmin and tmax
stc_surf.crop(0.09, 0.1)

evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

# Apply inverse operator
stc_vol = apply_inverse(evoked, inv_src, 1.0 / 3.0 ** 2, "dSPM")

# To save memory
stc_vol.crop(0.09, 0.1)

###############################################################################
# In a nutshell
# -------------
#
# For many applications the respective morph will probably look like this:

# Compute morph
# morph = SourceMorph(subject_from='sample',  # Default: None
#                     subject_to='fsaverage',  # Default: 'fsaverage'
#                     subjects_dir=subjects_dir,  # Default: None
#                     src=src_vol,  # Default: None
#                     niter_affine=(100, 100, 10),  # Default: (100, 100, 10)
#                     niter_sdr=(5, 5, 3),  # Default: (5, 5, 3)
#                     spacing=5)  # Default: 5

# Apply morph
# stc_fsaverage = morph(stc_vol)

# Make NIfTI volume variant 1
# img = morph(stc_vol,
#             as_volume=True,  # output as NIfTI
#             mri_resolution=True,  # in MRI resolution
#             mri_space=True)  # and MRI space

# Make NIfTI volume variant 2
# img = morph.as_volume(stc_fsaverage,
#                       mri_resolution=(3., 3., 3.),  # iso voxel size 3 mm
#                       mri_space=True)

# Save morph to disk
# morph.save('my-favorite-morph.h5')

# Read morph from disk
# morph = read_source_morph('my-favorite-morph.h5')

# Shortcuts
# stc_fsaverage = SourceMorph(src=src_vol)(stc_vol)
# img = SourceMorph(src=src_vol)(stc_vol, as_volume=True, mri_resolution=True)

###############################################################################
# Setting up :class:`mne.SourceMorph`
# -----------------------------------
#
# SourceMorph is a class that computes a morph operation from one subject to
# another depending on the underlying data. The result will be an instance of
# :class:`mne.SourceMorph`, that contains the mapping between the two spaces.
# At the very least the source space corresponding to the source estimate that
# is going to be morphed, has to be provided. Since stored data from both
# subjects of reference will be used, it is necessary to ensure that
# subject_from and subject_to, as well as subjects_dir are correctly set.

# SourceMorph initialization If src is not provided, the morph will not be
# pre-computed but instead will be prepared for morphing when calling. This
# works only with (Vector)SourceEstimate

morph_surf = SourceMorph(subject_from='sample',
                         subject_to='fsaverage',
                         subjects_dir=subjects_dir)

# Ideally subject_from can be inferred from src, subject_to is 'fsaverage' by
# default and subjects_dir is set in the environment. In that case SourceMorph
# can be initialized taking only src as parameter.

morph_vol = SourceMorph(subject_from='sample',
                        subject_to='fsaverage',
                        subjects_dir=subjects_dir,
                        spacing=(3., 3., 3.),  # grid spacing (3., 3., 3.) mm
                        src=src_vol)

###############################################################################
# Spacing parameter of :class:`mne.SourceMorph`
# ---------------------------------------------
#
# When morphing a surface source estimate, spacing can be an int or a list of
# two arrays. In the first case the data will be morphed to an icosahedral
# mesh, having a resolution of spacing (typically 5) using
# :func:`mne.grade_to_vertices`. In turn, when morphing a volumetric source
# estimate, spacing can be a tuple of float representing the voxel size in each
# spatial dimension or a single value (int or float) to represent the very same
# but assigning equal values to all spatial dimensions. Voxel size referring to
# the spacing of the reference volumes when computing the volumetric morph in
# mm. Note that voxel size is inverse related to computation time and accuracy
# of the morph operation.
# Changing the spacing for a volumetric morph estimation, does not affect the
# later resolution of the source estimate sfter applying the morph. It is
# rather the resolution of morph estimation and hence should increased when
# aiming for more precision. The default is an isometric voxel size of 5 mm.
# In general it might be advisable to use a spacing that is smaller or equal to
# the actual grid spacing of the source estimate.

# Estimate non-linear volumetric morph based on grid spacing of (7., 7., 7.) mm

# morph = SourceMorph(src=src_vol, spacing=(7., 7., 7.))  # equiv. to spacing=7

###############################################################################
# niter_ parameters of :class:`mne.SourceMorph`
# ---------------------------------------------
#
# Additionally, compuation time and accuray of the respective volumetric morph
# result, depend on the number of iterations per step of optimization. Under
# the hood, an Affine transformation is computed based on the mutual
# information. This metric relates structural changes in image intensity
# values. Because different still brains expose high structural similarities
# this method works quite well to relate corresponding features [1]_. The
# nonlinear transformations will be performed as Symmetric Diffeomorphic
# Registration (sdr) using the cross-correlation metric [2]_.
# Both optimization procedures are performed in "levels", passing the result
# from the first level of refinement to the next and so on. For each level, the
# number of iterations to optimize the alignment, can be defined. This is done
# be assigning a tuple to niter_affine and niter_sdr. Each tuple contains as
# many values, as desired levels of refinement and each value, represents the
# number of iterations for the respective level. The default is
# niter_affine=(100, 100, 10) and niter_sdr=(5, 5, 3). Both algorithms will be
# performed using 3 levels of refinement each and the corresponding number of
# iterations.

# Estimate non-linear volumetric morph based on grid spacing of (7., 7., 7.) mm
# and a reduced number of iterations. Note the difference in computation time.

# morph = SourceMorph(src=src_vol,
#                     spacing=(7., 7., 7.),
#                     niter_affine=(10, 10, 10),  # 3 levels a 10 iterations
#                     niter_sdr=(3, 3))  # 2 levels a 3 iterations

###############################################################################
# Applying an instance of :class:`mne.SourceMorph`
# ------------------------------------------------
#
# Once we computed the morph for our respective dataset, we can morph the data,
# by giving it as an argument to the SourceMorph instance. This operation
# applies pre-computed transforms to stc.

stc_surf_m = morph_surf(stc_surf)  # morphed surface source estimate
stc_vol_m = morph_vol(stc_vol)  # morphed volume source estimate

###############################################################################
# Transforming :class:`mne.VolSourceEstimate` into NIfTI
# ------------------------------------------------------
#
# In case of the volume source estimate, we can further ask the morph to output
# a volume of our data in the new space. We do this by calling the
# :meth:`morph.as_volume <mne.SourceMorph.as_volume>`. Note, that un-morphed
# source estimates still can be converted into a NIfTI by using
# :meth:`stc.as_volume <mne.VolSourceEstimate.as_volume>`. The shape of the
# output volume can be modified by providing the argument mri_resolution. This
# argument can be boolean, a tuple or an int. If mri_resolution=True, the MRI
# resolution, that was stored in src will be used. Setting mri_resolution to
# False, will export the volume having voxel size corresponding to the spacing
# of the computed morph. Setting a tuple or single value, will cause the output
# volume to expose a voxel size of that values in mm.

# img_mri_res = morph_vol.as_volume(stc_vol_m, mri_resolution=True)

# img_morph_res = morph_vol.as_volume(stc_vol_m, mri_resolution=False)

# img_any_res = morph_vol.as_volume(stc_vol_m, mri_resolution=3)

###############################################################################
# Reading and writing :class:`mne.VolSourceEstimate` from and to disk
# -------------------------------------------------------------------
#
# An instance of SourceMorph can be saved, by calling
# :meth:`morph.save <mne.SourceMorph.save>`. This methods allows for
# specification of a filename. The morph will be save in ".h5" format. If no
# file extension is provided, "-morph.h5" will be appended to the respective
# defined filename.
# In turn, reading a saved source morph can be achieved by using
# :func:`mne.read_source_morph`.

# morph_vol.save('my-file-name')

# -morph.h5 was attached because no file extension was provided when saving
# morph_vol = read_source_morph('my-file-name-morph.h5')

###############################################################################
# Plot results
# ------------

# Load fsaverage anatomical image
t1_fsaverage = nib.load(fname_t1_fsaverage)

# Initialize figure
fig, axes = plt.subplots()
fig.subplots_adjust(top=0.8, left=0.1, right=0.9, hspace=0.5)
fig.patch.set_facecolor('white')

# Setup nilearn plotting
display = plot_glass_brain(t1_fsaverage,
                           display_mode='ortho',
                           cut_coords=[0., 0., 0.],
                           draw_cross=False,
                           axes=axes,
                           figure=fig,
                           annotate=False)

# Transform into volume time series and use first one
overlay = index_img(morph_vol.as_volume(stc_vol_m, mri_resolution=True), 0)

display.add_overlay(overlay, alpha=0.75)
display.annotate(size=8)
axes.set_title('Morphed to fsaverage', color='white', fontsize=20)

plt.text(plt.xlim()[1], plt.ylim()[0], 't = 0.09s', color='white')
plt.show()

del stc_vol_m, morph_vol, morph_surf

surfer_kwargs = dict(
    hemi='lh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=0.09, time_unit='s', size=(800, 800),
    smoothing_steps=5)
brain = stc_surf_m.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=20)

del stc_surf_m

###############################################################################
# References
# ----------
# .. [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., &
#         Eubank, W. (2003). PET-CT image registration in the chest using
#         free-form deformations. IEEE transactions on medical imaging, 22(1),
#         120-128.
#
# .. [2] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
#         Symmetric Diffeomorphic Image Registration with Cross- Correlation:
#         Evaluating Automated Labeling of Elderly and Neurodegenerative Brain,
#         12(1), 26-41.
