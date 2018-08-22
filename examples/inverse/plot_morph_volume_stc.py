"""
================================
Morph volumetric source estimate
================================

This example demonstrates how to morph an individual subject's
:class:`mne.VolSourceEstimate` to a common reference space. We achieve this
using :class:`mne.SourceMorph`. Pre-computed data will be morphed based on
an affine transformation and a nonlinear morph, estimated on the respective
transformation from the subject's anatomical T1 weighted MRI (brain) to
FreeSurfer's 'fsaverage' T1 weighted MRI (brain).See
https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage .

Afterwards the transformation will be applied to the volumetric source
estimate. The result will be plotted, showing the fsaverage T1 weighted
anatomical MRI, overlaid with the morphed volumetric source estimate.

.. note:: For a tutorial about morphing see:
          :ref:`sphx_glr_auto_tutorials_plot_morph_stc.py`.
"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)
import os

import nibabel as nib
from mne import read_evokeds, SourceMorph, read_source_morph
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
evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)

# Apply inverse operator
stc = apply_inverse(evoked, inverse_operator, 1.0 / 3.0 ** 2, "dSPM")

# To save memory
stc.crop(0.09, 0.09)

###############################################################################
# Get a SourceMorph object for VolSourceEstimate
# ----------------------------------------------
#
# ``subject_from`` can be inferred from :class:`src <mne.SourceSpaces>`,
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

src = inverse_operator['src']
morph = SourceMorph(subject_from='sample',
                    subject_to='fsaverage',
                    subjects_dir=subjects_dir,
                    src=src)

###############################################################################
# Apply morph to VolSourceEstimate
# --------------------------------
#
# The morph will be applied to the source estimate data, by giving it as the
# first argument to the morph we computed above. Note that
# :meth:`morph() <mne.SourceMorph.__call__>` can take the same input arguments
# as :meth:`morph.as_volume() <mne.SourceMorph.as_volume>` to return a NIfTI
# image instead of a MNE-Python representation of the source estimate.

# Morph data
stc_fsaverage = morph(stc)

###############################################################################
# Convert morphed VolSourceEstimate into NIfTI
# --------------------------------------------
#
# We can convert our morphed source estimate into a NIfTI volume using
# :meth:`morph.as_volume() <mne.SourceMorph.as_volume>`. We provided our
# morphed source estimate as first argument. All following keyword arguments
# can be used to modify the output image.
#
# Note that ``apply_morph=False``, that is the morph will not be applied
# because the data has already been morphed. Set ``apply_morph=True`` to output
# un-morphed data as a morphed volume. Further
# :meth:`morph() <mne.SourceMorph.__call__>` can be used to output a volume as
# well, taking the same input arguments. Provide ``as_volume=True`` when
# calling the :class:`mne.SourceMorph` instance. In that case however
# apply_morph will of course be True by default.

# Create mri-resolution volume of results
img_fsaverage = morph.as_volume(stc_fsaverage, mri_resolution=2)

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
# will be appended to the respective defined filename.

morph.save('my-file-name')

# Reading a saved source morph can be achieved by using
# :func:`mne.read_source_morph`:

morph = read_source_morph('my-file-name-morph.h5')

###############################################################################
# Additional Info
# ===============
#
# In addition to the functionality, demonstrated above,
# :class:`mne.SourceMorph` can be used slightly differently as well, in order
# to enhance user comfort.
#
# For instance, it is possible to directly obtain a NIfTI image when calling
# the SourceMorph instance, but setting ``as_volume=True``. If so, the
# :meth:`morph() <mne.SourceMorph.__call__>` function takes the same input
# arguments as :meth:`morph.as_volume <mne.SourceMorph.as_volume>`.
#
# Moreover it can be decided whether to actually apply the morph or not. This
# way SourceMorph can be used to output un-morphed data as a volume as well. By
# setting ``apply_morph`` and ``as_volume`` to True, the source estimate will
# be morphed and convert it into a volume in one go:

img = morph(stc, as_volume=True, apply_morph=True)

###############################################################################
# Once the environment is set up correctly, no information such as
# ``subject_from`` or ``subjects_dir`` must be provided, since it can be
# inferred from the data and use morph to 'fsaverage' by default. SourceMorph
# can further be used without creating an instance and assigning it to a
# variable. Instead the :class:`__init__ <mne.SourceMorph>` and
# :meth:`__call__ <mne.SourceMorph.__call__>` methods of SourceMorph can be
# easily chained into a handy one-liner. Taking this together the shortest
# possible way to morph data directly would be:

src[0]['subject_his_id'] = 'sample'  # not needed for new MNE versions
os.environ['SUBJECTS_DIR'] = subjects_dir
stc_fsaverage = SourceMorph(src=src)(stc)
