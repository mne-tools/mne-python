"""
.. _ex-morph-volume:

================================
Morph volumetric source estimate
================================

This example demonstrates how to morph an individual subject's
:class:`mne.VolSourceEstimate` to a common reference space. We achieve this
using :class:`mne.SourceMorph`. Pre-computed data will be morphed based on
an affine transformation and a nonlinear registration method
known as Symmetric Diffeomorphic Registration (SDR) by Avants et al. [1]_.

Transformation is estimated from the subject's anatomical T1 weighted MRI
(brain) to `FreeSurfer's 'fsaverage' T1 weighted MRI (brain)
<https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage>`__.

Afterwards the transformation will be applied to the volumetric source
estimate. The result will be plotted, showing the fsaverage T1 weighted
anatomical MRI, overlaid with the morphed volumetric source estimate.

References
----------
.. [1] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
       Symmetric Diffeomorphic Image Registration with Cross- Correlation:
       Evaluating Automated Labeling of Elderly and Neurodegenerative
       Brain, 12(1), 26-41.

.. note:: For a tutorial about morphing see :ref:`ch_morph`.
"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)
import os

import nibabel as nib
import mne
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
#
# Load data:
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)

# Apply inverse operator
stc = apply_inverse(evoked, inverse_operator, 1.0 / 3.0 ** 2, "dSPM")

# To save time
stc.crop(0.09, 0.09)

###############################################################################
# Get a SourceMorph object for VolSourceEstimate
# ----------------------------------------------
#
# ``subject_from`` can typically be inferred from
# :class:`src <mne.SourceSpaces>`,
# and ``subject_to`` is  set to 'fsaverage' by default. ``subjects_dir`` can be
# None when set in the environment. In that case SourceMorph can be initialized
# taking ``src`` as only argument. See :class:`mne.SourceMorph` for more
# details.
#
# The default parameter setting for *spacing* will cause the reference volumes
# to be resliced before computing the transform. A value of '5' would cause
# the function to reslice to an isotropic voxel size of 5 mm. The higher this
# value the less accurate but faster the computation will be.
#
# A standard usage for volumetric data reads:

morph = mne.compute_source_morph(inverse_operator['src'],
                                 subject_from='sample', subject_to='fsaverage',
                                 subjects_dir=subjects_dir)

###############################################################################
# Apply morph to VolSourceEstimate
# --------------------------------
#
# The morph can be applied to the source estimate data, by giving it as the
# first argument to the :meth:`morph.apply() <mne.SourceMorph.apply>` method:

stc_fsaverage = morph.apply(stc)

###############################################################################
# Convert morphed VolSourceEstimate into NIfTI
# --------------------------------------------
#
# We can convert our morphed source estimate into a NIfTI volume using
# :meth:`morph.apply(..., output='nifti1') <mne.SourceMorph.apply>`.

# Create mri-resolution volume of results
img_fsaverage = morph.apply(stc, mri_resolution=2, output='nifti1')

###############################################################################
# Plot results
# ------------

# Load fsaverage anatomical image
t1_fsaverage = nib.load(fname_t1_fsaverage)

# Plot glass brain (change to plot_anat to display an overlaid anatomical T1)
display = plot_glass_brain(t1_fsaverage,
                           title='subject results to fsaverage',
                           draw_cross=False,
                           annotate=True)

# Add functional data as overlay
display.add_overlay(img_fsaverage, alpha=0.75)


###############################################################################
# Reading and writing SourceMorph from and to disk
# ------------------------------------------------
#
# An instance of SourceMorph can be saved, by calling
# :meth:`morph.save <mne.SourceMorph.save>`.
#
# This methods allows for specification of a filename under which the ``morph``
# will be save in ".h5" format. If no file extension is provided, "-morph.h5"
# will be appended to the respective defined filename::
#
#     >>> morph.save('my-file-name')
#
# Reading a saved source morph can be achieved by using
# :func:`mne.read_source_morph`::
#
#     >>> morph = mne.read_source_morph('my-file-name-morph.h5')
#
# Once the environment is set up correctly, no information such as
# ``subject_from`` or ``subjects_dir`` must be provided, since it can be
# inferred from the data and used morph to 'fsaverage' by default. SourceMorph
# can further be used without creating an instance and assigning it to a
# variable. Instead :func:`mne.compute_source_morph` and
# :meth:`mne.SourceMorph.apply` can be
# easily chained into a handy one-liner. Taking this together the shortest
# possible way to morph data directly would be:

stc_fsaverage_new = mne.compute_source_morph(
    inverse_operator['src'], subject_from='sample',
    subjects_dir=subjects_dir).apply(stc)
