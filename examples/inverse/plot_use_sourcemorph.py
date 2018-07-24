"""
================================
Demonstrate usage of SourceMorph
================================

This example demonstrates how to morph an individual subject source estimate to
a common reference space. It will be demonstrated using the SourceMorp class.
Pre-computed data will be morphed based on an affine transformation and a
nonlinear morph, estimated based on respective transformation from the
subject's anatomical T1 (brain) to fsaverage T1 (brain).
Afterwards the transformation will be applied to the
source estimate. The result will be a plot showing the fsaverage T1 overlaid
with the morphed source estimate. To see how morphing source estimates works,
see :ref:`sphx_glr_auto_tutorials_plot_morph.py` or for a more detailed
information :ref:`sphx_glr_auto_tutorials_plot_background_morph.py`

"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)
import os

import matplotlib.pylab as plt
import nibabel as nib
import numpy as np
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
stc.crop(0.087, 0.087)

###############################################################################
# Morph VolSourceEstimate

# Initialize SourceMorph for VolSourceEstimate
source_morph = SourceMorph(subject_from='sample',
                           subject_to='fsaverage',
                           subjects_dir=subjects_dir,
                           src=inverse_operator['src'],
                           spacing=5)

# Obtain absolute value for plotting
# To not copy the data into a new memory location, out=stc.data is set
np.abs(stc.data, out=stc.data)

# Morph data
stc_fsaverage = source_morph(stc)

###############################################################################
# Plot results

# Load fsaverage anatomical image
t1_fsaverage = nib.load(fname_t1_fsaverage)

# Create mri-resolution volume of results
img_fsaverage = source_morph.as_volume(stc_fsaverage, mri_resolution=2)

fig, axes = plt.subplots()
fig.subplots_adjust(top=0.8, left=0.1, right=0.9, hspace=0.5)
fig.patch.set_facecolor('white')

display = plot_glass_brain(t1_fsaverage, display_mode='ortho',
                           cut_coords=[0., 0., 0.],
                           draw_cross=False,
                           axes=axes,
                           figure=fig,
                           annotate=False)

display.add_overlay(img_fsaverage, alpha=0.75)
display.annotate(size=8)
axes.set_title('subject results to fsaverage', color='black', fontsize=12)

plt.text(plt.xlim()[1], plt.ylim()[0], 't = 0.087s', color='black')
plt.show()
