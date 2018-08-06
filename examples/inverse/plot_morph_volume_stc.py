"""
================================
Morph volumetric source estimate
================================

This example demonstrates how to morph an individual subject
:class:`mne.SourceEstimate` to a common reference space. For this purpose we
will use :class:`mne.SourceMorph`. Pre-computed data will be morphed based on
an affine transformation and a nonlinear morph, estimated based on respective
transformation from the subject's anatomical T1 (brain) to fsaverage T1
(brain).

Afterwards the transformation will be applied to the volumetric source
estimate. The result will be a plot showing the fsaverage T1 overlaid with the
morphed volumetric source estimate.

.. note:: For a tutorial about morphing see:
          :ref:`sphx_glr_auto_tutorials_plot_morph_stc.py`.
"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)
import os
import warnings

import nibabel as nib
from mne import read_evokeds, SourceMorph
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from nilearn.plotting import plot_glass_brain

print(__doc__)

###############################################################################
# Setup paths
sample_dir_raw = sample.data_path()
sample_dir = os.path.join(sample_dir_raw, 'MEG', 'sample')
subjects_dir = os.path.join(sample_dir_raw, 'subjects')

fname_evoked = os.path.join(sample_dir, 'sample_audvis-ave.fif')
fname_inv = os.path.join(sample_dir, 'sample_audvis-meg-vol-7-meg-inv.fif')

fname_t1_fsaverage = os.path.join(subjects_dir, 'fsaverage', 'mri',
                                  'brain.mgz')

###############################################################################
# Compute example data. For reference see
# :ref:`sphx_glr_auto_examples_inverse_plot_compute_mne_inverse_volume.py`

# Load data
evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)

# Apply inverse operator
stc = apply_inverse(evoked, inverse_operator, 1.0 / 3.0 ** 2, "dSPM")

# To save memory
stc.crop(0.09, 0.09)

###############################################################################
# Morph VolSourceEstimate

# Initialize SourceMorph for VolSourceEstimate
morph = SourceMorph(subject_from='sample',
                    subject_to='fsaverage',
                    subjects_dir=subjects_dir,
                    src=inverse_operator['src'])

# Morph data
stc_fsaverage = morph(stc)

###############################################################################
# Plot results

# Load fsaverage anatomical image
with warnings.catch_warnings(record=False):  # nib<->numpy
    t1_fsaverage = nib.load(fname_t1_fsaverage)

# Create mri-resolution volume of results
img_fsaverage = morph.as_volume(stc_fsaverage, mri_resolution=2)

# Plot glass brain (change to plot_anat to display an overlaid anatomical T1)
display = plot_glass_brain(t1_fsaverage,
                           title='subject results to fsaverage',
                           draw_cross=False,
                           annotate=True)

# Add functional data as overlay
display.add_overlay(img_fsaverage, alpha=0.75)
