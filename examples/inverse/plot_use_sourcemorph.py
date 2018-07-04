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
import mne
import nibabel as nib
import numpy as np
from mne import SourceMorph
from mne.beamformer import make_lcmv, apply_lcmv
from mne.datasets import sample
from nilearn.plotting import plot_anat

print(__doc__)

# Setup paths
data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

t1_sample = data_path + '/subjects/sample/mri/brain.mgz'
t1_fsaverage = data_path + '/subjects/fsaverage/mri/brain.mgz'

###############################################################################
# Compute example data. For reference see :ref:`lcmv beamformer
# <sphx_glr_auto_examples_inverse_plot_lcmv_beamformer.py>` and
# :ref:`lcmv beamformer volume
# <sphx_glr_auto_examples_inverse_plot_lcmv_beamformer_volume.py>`

fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name
event_id, tmin, tmax = [1, 2], -0.2, 0.2

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                       exclude='bads')

# Pick the channels of interest
raw.pick_channels([raw.ch_names[pick] for pick in picks])

# Re-normalize our empty-room projectors, so they are fine after subselection
raw.info.normalize_proj()

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), preload=True, proj=True,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()

# Read forward model
forward = mne.read_forward_solution(fname_fwd)
forward_surf = mne.convert_forward_solution(forward, surf_ori=True)

# Compute regularized noise and data covariances
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method='shrunk')
data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                  method='shrunk')

# Compute unit-noise-gain beamformer with whitening of the leadfield and
# data (enabled by passing a noise covariance matrix)
filters = make_lcmv(evoked.info, forward_surf, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='nai')

# Apply this spatial filter to source-reconstruct the evoked data
stc_surf = apply_lcmv(evoked, filters, max_ori_out='signed')

# To save memory
stc_surf.crop(0.087, 0.087)

# Compute VolSourceEstimate
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'

# Read forward model
forward_vol = mne.read_forward_solution(fname_fwd)

filters = make_lcmv(evoked.info, forward_vol, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='nai')

# Apply this spatial filter to the evoked data.
stc_vol = apply_lcmv(evoked, filters, max_ori_out='signed')

# To save memory
stc_vol.crop(0.087, 0.087)

###############################################################################
# Morph SourceEstimate

# Initialize morpher for SourceEstimate
src_surf = forward_surf['src']
source_morph_surf = SourceMorph(src_surf, subjects_dir=subjects_dir)

# Save and load SourceMorph if desired
# source_morph_surf.save('surf')
# source_morph_surf = mne.read_source_morph('surf-morph.h5')

# Morph data
np.abs(stc_surf.data, out=stc_surf.data)  # for plotting
stc_surf_fsaverage = source_morph_surf(stc_surf)

###############################################################################
# Morph VolSourceEstimate

# Initialize morpher for VolSourceEstimate
src_vol = forward_vol['src']
src_vol[0]['subject_his_id'] = 'sample'
source_morph_vol = SourceMorph(src_vol, subjects_dir=subjects_dir,
                               grid_spacing=(5., 5., 5.))

# Save and load SourceMorph if desired
# source_morph_vol.save('vol')
# source_morph_vol = mne.read_source_morph('vol-morph.h5')

# Morph data
np.abs(stc_vol.data, out=stc_vol.data)  # for plotting
stc_vol_fsaverage = source_morph_vol(stc_vol)

###############################################################################
# Plot results

# Plot morphed surface
# brain = stc_surf_fsaverage.plot(hemi='lh', subjects_dir=subjects_dir,
#                                 initial_time=0.087, time_unit='s')
# brain.show_view('lateral')
# brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=20)

# Plot morphed volumes

# Create mri-resolution volume of results
img = source_morph_vol.as_volume(stc_vol_fsaverage, mri_resolution=True)

fig, axes = plt.subplots()
fig.subplots_adjust(top=0.8, left=0.1, right=0.9, hspace=0.5)
fig.patch.set_facecolor('black')

display = plot_anat(nib.load(t1_fsaverage),
                    display_mode='ortho',
                    cut_coords=[0., 0., 0.],
                    draw_cross=False,
                    axes=axes,
                    figure=fig,
                    annotate=False)

display.add_overlay(img, alpha=0.75)
display.annotate(size=8)
axes.set_title('morphed subject results to fsaverage', color='white',
               fontsize=12)
plt.text(plt.xlim()[1], plt.ylim()[0], 't = 0.087s', color='white')
plt.show()
