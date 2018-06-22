"""
==================================================================
Demonstrate morphing subject source volume to fsaverage using dipy
==================================================================

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

from dipy.align import imaffine, imwarp, metrics, reslice, transforms

import matplotlib.pylab as plt

import mne
from mne.beamformer import make_lcmv, apply_lcmv
from mne.datasets import sample
from mne.externals.h5io import read_hdf5, write_hdf5

import nibabel as nib

from nilearn.image import index_img
from nilearn.plotting import plot_anat

import numpy as np

from os import path, makedirs

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

    # Save result in a 4D nifti file
    img = mne.save_stc_as_volume(fname, stc, forward['src'],
                                 mri_resolution=True)

    return img


###############################################################################
# Compute non-linear morph mapping

def compute_morph_map(img_m, img_s=None, niter_affine=(100, 100, 10),
                      niter_sdr=(5, 5, 3)):
    # get Static to world transform
    img_s_grid2world = img_s.affine

    # output Static as ndarray
    img_s = img_s.dataobj[:, :, :]

    # normalize values
    img_s = img_s.astype('float') / img_s.max()

    # get Moving to world transform
    img_m_grid2world = img_m.affine

    # output Moving as ndarray
    img_m = img_m.dataobj[:, :, :]

    # normalize values
    img_m = img_m.astype('float') / img_m.max()

    # compute center of mass
    c_of_mass = imaffine.transform_centers_of_mass(img_s, img_s_grid2world,
                                                   img_m, img_m_grid2world)

    nbins = 32

    # set up Affine Registration
    affreg = imaffine.AffineRegistration(
        metric=imaffine.MutualInformationMetric(nbins, None),
        level_iters=list(niter_affine),
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1])

    # translation
    translation = affreg.optimize(img_s, img_m,
                                  transforms.TranslationTransform3D(), None,
                                  img_s_grid2world, img_m_grid2world,
                                  starting_affine=c_of_mass.affine)

    # rigid body transform (translation + rotation)
    rigid = affreg.optimize(img_s, img_m,
                            transforms.RigidTransform3D(), None,
                            img_s_grid2world, img_m_grid2world,
                            starting_affine=translation.affine)

    # affine transform (translation + rotation + scaling)
    affine = affreg.optimize(img_s, img_m,
                             transforms.AffineTransform3D(), None,
                             img_s_grid2world, img_m_grid2world,
                             starting_affine=rigid.affine)

    # apply affine transformation
    img_m_affine = affine.transform(img_m)

    # set up Symmetric Diffeomorphic Registration (metric, iterations)
    sdr = imwarp.SymmetricDiffeomorphicRegistration(
        metrics.CCMetric(3), list(niter_sdr))

    # compute mapping
    mapping = sdr.optimize(img_s, img_m_affine)

    return mapping, affine


###############################################################################
# Save non linear mapping data

def save_mapping(fname, data, overwrite=True):
    out = dict()

    # dissolve object structure
    for d in data:
        # save type for order independent decomposition
        out[type(d).__name__] = d.__dict__

    write_hdf5(fname + '.h5', out, overwrite=overwrite)


###############################################################################
# Load non linear mapping data

def load_mapping(fname):
    # create new instances
    mapping = imwarp.DiffeomorphicMap(None, [])
    affine = imaffine.AffineMap(None)

    data = read_hdf5(fname + '.h5')

    mapping.__dict__ = data.get(type(mapping).__name__)
    affine.__dict__ = data.get(type(affine).__name__)

    return mapping, affine


###############################################################################
# Apply non-linear morph mapping

def morph_precomputed(img, affine, mapping):
    # morph img data
    img_sdr_affine = np.zeros(img.shape)
    for vol in range(img.shape[3]):
        img_sdr_affine[:, :, :, vol] = mapping.transform(
            affine.transform(img.dataobj[:, :, :, vol]))

    return img_sdr_affine


###############################################################################
# Morph 3D coordinates. This is used to transform the respective slices shown
# in the output such that the field of view if more or less similar. Note that
# this is necessary, because shown volumes will be positioned differently in
# the volume bounding box.

def morph_slice_indices(slice_indices, transmat):
    # make sure array is 2D
    slice_indices = np.atleast_2d(slice_indices)

    # add a column of ones
    slice_indices = np.append(slice_indices,
                              np.ones((slice_indices.shape[0], 1)), axis=1)

    # matrix multiplication with transformation matrix
    slice_indices_out = np.einsum('...j,ij', slice_indices, transmat)

    # round to select valid slices and remove last column
    return np.round(slice_indices_out[:, :-1])


###############################################################################
# Prepare volumetric data for morphing

def prepare_volume_example_data(img_in, t1_m_path, t1_s_path,
                                voxel_size=(3., 3., 3.)):
    # load lcmv inverse
    if isinstance(img_in, str):
        img_vol = nib.load(img_in)
    else:
        img_vol = img_in

    # reslice lcmv inverse
    img_vol_res, img_vol_res_affine = reslice.reslice(
        img_vol.get_data(),
        img_vol.affine,
        img_vol.header.get_zooms()[:3],
        voxel_size)

    img_vol_res = nib.Nifti1Image(img_vol_res, img_vol_res_affine)

    # load subject brain (Moving)
    t1_m_img = nib.load(t1_m_path)

    # reslice Moving
    t1_m_img_res, t1_m_img_res_affine = reslice.reslice(
        t1_m_img.get_data(),
        t1_m_img.affine,
        t1_m_img.header.get_zooms()[:3],
        voxel_size)

    t1_m_img_res = nib.Nifti1Image(t1_m_img_res, t1_m_img_res_affine)

    # load fsaverage brain (Static)
    t1_s_img = nib.load(t1_s_path)

    # reslice Static
    t1_s_img_res, t1_s_img_res_affine = reslice.reslice(
        t1_s_img.get_data(),
        t1_s_img.affine,
        t1_s_img.header.get_zooms()[:3],
        voxel_size)

    t1_s_img_res = nib.Nifti1Image(t1_s_img_res, t1_s_img_res_affine)

    return img_vol_res, t1_m_img_res, t1_s_img_res


###############################################################################
# Execute example

# Setup path
data_path = sample.data_path()
results_path = data_path + '/subjects/LCMV-results'

# create results directory if it doesn't exist
if not path.exists(results_path):
    makedirs(results_path)

# compute LCMV beamformer inverse example
img = compute_lcmv_example_data(data_path)

# load and reslice volumes
img_vol_res, t1_m_img_res, t1_s_img_res = prepare_volume_example_data(
    img,
    data_path + '/subjects/sample/mri/brain.mgz',
    data_path + '/subjects/fsaverage/mri/brain.mgz',
    voxel_size=(5., 5., 5.))

# compute morph map from Moving to Static
mapping, affine = compute_morph_map(t1_m_img_res, t1_s_img_res)

save_mapping(results_path + '/volume_morph', [mapping, affine])

mapping, affine = load_mapping(results_path + '/volume_morph')

# apply morph map (test if saving and loading worked)
img_vol_morphed = morph_precomputed(img_vol_res, mapping, affine)

# make transformed ndarray a nifti
img_vol_morphed = nib.Nifti1Image(img_vol_morphed,
                                  affine=t1_s_img_res.affine)

# save morphed result
nib.save(img_vol_morphed, results_path + '/lcmv-fsaverage.nii.gz')

###############################################################################
# Plot results

# select image overlay
imgs = [index_img(img_vol_res, 0), index_img(img_vol_res, 0),
        index_img(img_vol_morphed, 0)]

# select anatomical background images
t1_imgs = [t1_m_img_res, t1_s_img_res, t1_s_img_res]

# slices to show for Static volume
slices_s = (-10, 0, 0)

# slices to show for Moving volume
# to show roughly the same view, we transform the selected Static slices using
# the inverse affine transformation. Note that due to rotation and the
# non-linear transform, both views do not overlap perfectly (pre computed for
# this example)
slices_m = morph_slice_indices(np.atleast_2d(slices_s), affine.affine_inv)[0]

slices = [slices_s, slices_s, tuple(slices_m)]

# define titles for plots
titles = ['subject brain', 'fsaverage brain', 'fsaverage brain morphed result']

# plot results
figure, (axes1, axes2, axes3) = plt.subplots(3, 1)
figure.subplots_adjust(top=0.8, left=0.1, right=0.9, hspace=0.5)
figure.patch.set_facecolor('black')

for axes, img, t1_img, cut_coords, title in zip([axes1, axes2, axes3],
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
