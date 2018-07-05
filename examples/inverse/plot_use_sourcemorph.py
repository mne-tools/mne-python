"""
================================
Demonstrate usage of SourceMorph
================================

This example demonstrates how to morph an individual subject source estimate to
a common reference space. It will be demonstrated using the SourceMorp class.
The example uses parts of the MNE example
:ref:`sphx_glr_auto_examples_inverse_plot_lcmv_beamformer_volume.py` and
:ref:`sphx_glr_auto_examples_inverse_plot_lcmv_beamformer.py`.
The respective result will be morphed based on an affine transformation and a
nonlinear morph, estimated based on respective transformation from the
subject's anatomical T1 (brain) to fsaverage T1 (brain) in VolSourceEstimate
case and using an affine transform in the SourceEstimate or
VectorSourceEstimate case. Afterwards the transformation will be applied to the
beamformer result. The result will be a plot showing the morphed result
overlaying the fsaverage T1. Uncomment at the respective location to plot the
result of the surface morph.

"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pylab as plt
import nibabel as nib
import numpy as np
from mne import read_evokeds, SourceMorph
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from nilearn.plotting import plot_anat

print(__doc__)

###############################################################################
# Setup paths
sample_dir = sample.data_path() + '/MEG/sample'
subjects_dir = sample.data_path() + '/subjects'

fname_evoked = sample_dir + '/sample_audvis-ave.fif'
fname_inv = sample_dir + '/sample_audvis-meg-vol-7-meg-inv.fif'

fname_t1_fsaverage = subjects_dir + '/fsaverage/mri/brain.mgz'

###############################################################################
# Compute example data. For reference see
# :ref:`<sphx_glr_auto_examples_inverse_plot_compute_mne_inverse_volume.py>`

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
source_morph = SourceMorph(inverse_operator['src'],
                           subject_from='sample',
                           subject_to='fsaverage',
                           subjects_dir=subjects_dir,
                           grid_spacing=(5., 5., 5.))

# Save and load SourceMorph
# source_morph.save('vol')
# source_morph = mne.read_source_morph('vol-morph.h5')

# Morph data
np.abs(stc.data, out=stc.data)  # for plotting
stc_fsaverage = source_morph(stc)

###############################################################################
# Plot results

# Load fsaverage anatomical image
t1_fsaverage = nib.load(fname_t1_fsaverage)

# Create mri-resolution volume of results
img_fsaverage = source_morph.as_volume(stc_fsaverage, mri_resolution=True)

fig, axes = plt.subplots()
fig.subplots_adjust(top=0.8, left=0.1, right=0.9, hspace=0.5)
fig.patch.set_facecolor('black')

display = plot_anat(t1_fsaverage, display_mode='ortho',
                    cut_coords=[0., 0., 0.],
                    draw_cross=False, axes=axes, figure=fig, annotate=False)

display.add_overlay(img_fsaverage, alpha=0.75)
display.annotate(size=8)
axes.set_title('subject results to fsaverage', color='white', fontsize=12)

plt.text(plt.xlim()[1], plt.ylim()[0], 't = 0.087s', color='white')
plt.show()
