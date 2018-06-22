"""
======================================
Demonstrate morphing VolSourceEstimate
======================================

This example demonstrates how to morph an individual subject source space to a
common reference space. For this purpose `dipy
<http://nipy.org/dipy>`_ will be
used in order to perform the necessary transforms to a fsaverage serving as
reference space. The example uses parts of the MNE example
:ref:`LCMV beamformer pipeline
<sphx_glr_auto_examples_inverse_plot_lcmv_beamformer_volume.py>`.
The respective result will be morphed based on an affine transformation and a
nonlinear morph, estimated based on respective transformation from the
subject's anatomical T1 (brain) to fsaverage T1 (brain). Afterwards the
transformation will be applied to the beamformer result. Affine transformations
are computed based on the mutual information. This metric relates structural
changes in image intensity values. Because different still brains expose high
structural similarities this method works quite well to relate corresponding
features [1]_. The nonlinear transformations will be performed as
Symmetric Diffeomorphic Registration using the cross-correlation metric [2]_.

.. note:: This example applies downsampling to all volumes in order to speed up
        computation. In a real case scenario you might want to reconsider if
        and to what extend resliceing might be necessary.

References
----------

.. [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., &
        Eubank, W. (2003). PET-CT image registration in the chest using
        free-form deformations. IEEE transactions on medical imaging, 22(1),
        120-128.

.. [2] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
        Symmetric Diffeomorphic Image Registration with Cross- Correlation:
        Evaluating Automated Labeling of Elderly and Neurodegenerative Brain,
        12(1), 26-41.

"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)

from os import makedirs, path

import matplotlib.pylab as plt
import mne
import nibabel as nib
import numpy as np
from dipy.align import imaffine, imwarp, reslice
from mne.beamformer import make_lcmv, apply_lcmv
from mne.datasets import sample
from mne.externals.h5io import write_hdf5, read_hdf5
from nilearn.image import index_img
from nilearn.plotting import plot_anat

print(__doc__)


###############################################################################
# from :ref:`LCMV beamformer inverse example
# <sphx_glr_auto_examples_inverse_plot_lcmv_beamformer_volume.py>`. Note, that
# in order to save the data fname needs to be set

def compute_lcmv_example_data(data_path, fname=None):
    raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
    fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
    # Get epochs
    event_id, tmin, tmax = [1, 2], -0.2, 0.5

    # Setup for reading the raw data
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bad channels
    events = mne.read_events(event_fname)

    # Set up pick list: gradiometers and magnetometers, excluding bad channels
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                           exclude='bads')

    # Pick the channels of interest
    raw.pick_channels([raw.ch_names[pick] for pick in picks])

    # Re-normalize our empty-room projectors, so they are fine after
    # subselection
    raw.info.normalize_proj()
    # Read epochs
    proj = False  # already applied
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), preload=True, proj=proj,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
    evoked = epochs.average()

    # Read regularized noise covariance and compute regularized data covariance
    noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0,
                                       method='shrunk')
    data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                      method='shrunk')
    # Read forward model
    forward = mne.read_forward_solution(fname_fwd)

    # Compute weights of free orientation (vector) beamformer with weight
    # normalization (neural activity index, NAI). Providing a noise covariance
    # matrix enables whitening of the data and forward solution. Source
    # orientation is optimized by setting pick_ori to 'max-power'.
    # weight_norm can also be set to 'unit-noise-gain'. Source orientation can
    # also be 'normal' (but only when using a surface-based source space) or
    # None, which computes a vector beamfomer. Note, however, that not all
    # combinations of orientation selection and weight normalization are
    # implemented yet.
    filters = make_lcmv(evoked.info, forward, data_cov, reg=0.05,
                        noise_cov=noise_cov, pick_ori='max-power',
                        weight_norm='nai')

    # Apply this spatial filter to the evoked data.
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')

    # take absolute values for plotting
    stc.data[:, :] = np.abs(stc.data)

    # Save result in stc files if desired
    if fname is not None:
        stc.save('lcmv-vol')

    # select time window (tmin, tmax) in ms - consider changing for real data
    # scenario, since those values were chosen to optimize computation time
    stc.crop(0.087, 0.087)

    src = forward['src']

    # Save result in a 4D nifti file
    img = mne.save_stc_as_volume(fname, stc, src,
                                 mri_resolution=True)

    return img, stc, src


###############################################################################
# Save non linear mapping data

def write_morph(fname, morph, overwrite=True):
    morph_out = dict()

    # dissolve object structure
    for key, value in morph.iteritems():
        # save type for order independent decomposition
        if hasattr(value, '__dict__'):
            value = value.__dict__
        morph_out[key] = value
    if not fname.endswith('.h5'):
        fname += '.h5'
    write_hdf5(fname, morph_out, overwrite=overwrite)


###############################################################################
# Load non linear mapping data

def read_morph(fname):
    if not fname.endswith('.h5'):
        fname += '.h5'
    morph_in = read_hdf5(fname)

    morph = dict()
    # create new instances
    morph['mapping'] = imwarp.DiffeomorphicMap(None, [])
    morph['mapping'].__dict__ = morph_in.get('mapping')
    morph['affine'] = imaffine.AffineMap(None)
    morph['affine'].__dict__ = morph_in.get('affine')

    morph['affine_reg'] = morph_in.get('affine_reg')
    morph['domain_shape'] = morph_in.get('domain_shape')

    return morph


###############################################################################
# Execute example

# Settings
voxel_size = (5., 5., 5.)  # of the destination volume

# Setup path
data_path = sample.data_path()

t1_sample = data_path + '/subjects/sample/mri/brain.mgz'
t1_fsaverage = data_path + '/subjects/fsaverage/mri/brain.mgz'

# compute LCMV beamformer inverse example
img, stc, src = compute_lcmv_example_data(data_path)

###############################################################################
# Compute Morph Matrix and Dispersion Volume
morph = mne.source_estimate.compute_morph_sdr(
    t1_sample,
    t1_fsaverage,
    voxel_size=voxel_size)

write_morph('SDR-sample-fsaverage', morph)

###############################################################################
# Morph Pre - Computed

morph_precomputed = read_morph('SDR-sample-fsaverage')

stc_vol_to_preC = stc.morph_precomputed('sample', 'fsaverage', src,
                                        morph_precomputed,
                                        as_volume=False)

img_vol_to_preC = stc.morph_precomputed('sample', 'fsaverage', src,
                                        morph_precomputed,
                                        as_volume=True)

###############################################################################
# Morph
stc_vol_to_drct = stc.morph('sample',
                            'fsaverage',
                            t1_sample,
                            t1_fsaverage,
                            src,
                            voxel_size=voxel_size,
                            as_volume=False)

img_vol_to_drct = stc.morph('sample',
                            'fsaverage',
                            t1_sample,
                            t1_fsaverage,
                            src,
                            voxel_size=voxel_size,
                            as_volume=True)

###############################################################################
# Plot results

# load fsaverage brain (Static)
t1_s_img = nib.load(t1_fsaverage)

# reslice Static
t1_s_img_res, t1_s_img_res_affine = reslice.reslice(
    t1_s_img.get_data(),
    t1_s_img.affine,
    t1_s_img.header.get_zooms()[:3],
    voxel_size)

t1_s_img_res = nib.Nifti1Image(t1_s_img_res, t1_s_img_res_affine)

# select image overlay
imgs = [index_img(img_vol_to_preC, 0), index_img(img_vol_to_drct, 0)]

# select anatomical background images
t1_imgs = [t1_s_img_res, t1_s_img_res]

# slices to show for Static volume
slices_s = (0, 0, 0)

slices = [slices_s, slices_s]

# define titles for plots
titles = ['fsaverage brain precomputed', 'fsaverage brain direct morph']

# plot results
figure, (axes1, axes2) = plt.subplots(2, 1)
figure.subplots_adjust(top=0.8, left=0.1, right=0.9, hspace=0.5)
figure.patch.set_facecolor('black')

for axes, img, t1_img, cut_coords, title in zip([axes1, axes2],
                                                imgs, t1_imgs, slices, titles):
    display = plot_anat(t1_img,
                        display_mode='ortho',
                        cut_coords=cut_coords,
                        draw_cross=False,
                        axes=axes,
                        figure=figure,
                        annotate=False)

    display.add_overlay(img, alpha=0.75)
    display.annotate(size=8)
    axes.set_title(title, color='white', fontsize=12)

plt.text(plt.xlim()[1], plt.ylim()[0], 't = 0.087s', color='white')
plt.suptitle('morph subject results to fsaverage', color='white', fontsize=16)
plt.show()
