# -*- coding: utf-8 -*-
"""
Morphing Source Estimates using SourceMorph
===========================================

In this tutorial we will morph different kinds of source estimation results
between individual subject spaces using :class:`mne.SourceMorph`.
For group level statistical analyses subject specific results have to be mapped
to a common space.

We will use precomputed data and morph surface and volume source estimates to a
common space. The common space of choice will be FreeSurfer's "fsaverage".

Furthermore we will convert our volume source estimate into a NIfTI image using
:meth:`morph.as_volume <mne.SourceMorph.as_volume>`.

For a more detailed version of this tutorial, including some background
information about morphing different source estimate representations, see
:ref:`sphx_glr_auto_tutorials_plot_background_morph.py`
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
# estimates. In order to save computation time we crop our time series to a
# short period around the peak time, that we already know. For a real case
# scenario this might apply as well if a narrow time window of interest is
# known in advance.

stc_surf = read_source_estimate(fname_surf, subject='sample')

# The surface source space
src_surf = read_inverse_operator(fname_inv_surf)['src']

# The volume inverse operator
inv_src = read_inverse_operator(fname_inv_vol)

# The volume source space
src_vol = inv_src['src']

# Ensure subject is not None
src_vol[0]['subject_his_id'] = 'sample'

# For faster computation we redefine tmin and tmax
stc_surf.crop(0.09, 0.1)  # our prepared surface source estimate

# Read pre-computed evoked data
evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

# Apply inverse operator
stc_vol = apply_inverse(evoked, inv_src, 1.0 / 3.0 ** 2, "dSPM")

# For faster computation we redefine tmin and tmax
stc_vol.crop(0.09, 0.1)  # our prepared volume source estimate

###############################################################################
# Setting up SourceMorph for SourceEstimate
# -----------------------------------------
#
# :class:`SourceMorph <mne.SourceMorph>` initialization - If src is not
# provided, the morph will not be pre-computed but instead will be prepared for
# morphing when calling. This works only with (Vector)
# :class:`SourceEstimate <mne.SourceEstimate>`

morph_surf = SourceMorph(subject_from='sample',  # Default: None
                         subject_to='fsaverage',  # Default
                         subjects_dir=subjects_dir)  # Default: None

###############################################################################
# Setting up SourceMorph for VolSourceEstimate
# --------------------------------------------
#
# Ideally subject_from can be inferred from src, subject_to is 'fsaverage' by
# default and subjects_dir is set in the environment. In that case
# :class:`SourceMorph <mne.SourceMorph>` can be initialized taking only src as
# argument (for better understanding more keyword arguments are defined).

morph_vol = SourceMorph(subject_from='sample',  # Default: None
                        subject_to='fsaverage',  # Default
                        subjects_dir=subjects_dir,  # Default: None
                        src=src_vol)  # Default: None

###############################################################################
# Applying an instance of SourceMorph
# -----------------------------------
#
# Once we computed the morph for our respective dataset, we can morph the data,
# by giving it as an argument to the :class:`SourceMorph <mne.SourceMorph>`
# instance. This operation applies pre-computed transforms to stc.

stc_surf_m = morph_surf(stc_surf)  # SourceEstimate | VectorSourceEstimate
stc_vol_m = morph_vol(stc_vol)  # VolSourceEstimate

###############################################################################
# Reading and writing :class:`mne.SourceMorph` from and to disk
# -------------------------------------------------------------
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
# Additional functionality and shortcuts
# --------------------------------------
#
# In addition to the functionality, demonstrated above, SourceMorph can be used
# slightly different as well, in order to enhance user comfort.
#
# For instance, it is possible to directly obtain a NIfTI image when calling
# the SourceMorph instance, but setting 'as_volume=True'. If so, the __call__()
# function takes the same input arguments as
# :meth:`morph.as_volume <mne.SourceMorph.as_volume>`.
#
# Moreover it can be decided whether to actually apply the morph or not by
# setting the 'apply_morph' argument to True
#
# img_fsaverage = morph(stc, as_volume=True, apply_morph=True)
#
# Since once the environment is set up correctly, SourceMorph can be used
# without assigning an instance to a variable. Instead the __init__ and
# __call__ methods of SourceMorph can be combined into a handy one-liner:
#
# stc_fsaverage = mne.SourceMorph(src=src)(stc)

###############################################################################
# Plot results
# ------------

# Plot morphed volume source estiamte

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
axes.set_title('Morphed to fsaverage', color='black', fontsize=16)

plt.text(plt.xlim()[1], plt.ylim()[0], 't = 0.09s', color='black')
plt.show()

# save some memory
del stc_vol_m, morph_vol, morph_surf, t1_fsaverage

# Plot morphed surface source estiamte

surfer_kwargs = dict(
    hemi='lh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=0.09, time_unit='s', size=(800, 800),
    smoothing_steps=5)
brain = stc_surf_m.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=16)
