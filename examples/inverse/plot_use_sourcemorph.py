"""
================================
Demonstrate usage of SourceMorph
================================

This example demonstrates how to morph an individual subject source estimate to
a common reference space. It will be demonstrated using the SourceMorp class.
The example uses parts of the MNE example
:ref:`sphx_glr_auto_examples_inverse_plot_lcmv_beamformer_volume.py` and
:ref:`sphx_glr_auto_examples_inverse_plot_mne_dspm_source_localization.py`.
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
import nibabel  as nib
import numpy as np
from mne import SourceMorph
from mne.beamformer import make_lcmv, apply_lcmv
from mne.minimum_norm import make_inverse_operator, apply_inverse
from nilearn.plotting import plot_anat

print(__doc__)


###############################################################################
# from :ref:`sphx_glr_auto_examples_inverse_plot_mne_dspm_source_localization.py`.
# Note, that in order to save the data fname needs to be set
def compute_lcmv_example_data_surf(data_path, fname=None):
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

    raw = mne.io.read_raw_fif(raw_fname)  # already has an average reference
    events = mne.find_events(raw, stim_channel='STI 014')

    event_id = dict(aud_r=1)  # event trigger and conditions
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.2  # end of each epoch (500ms after the trigger)
    raw.info['bads'] = ['MEG 2443', 'EEG 053']
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                           exclude='bads')
    baseline = (None, 0)  # means from the first instant to t = 0
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,
                        baseline=baseline, reject=reject)
    noise_cov = mne.compute_covariance(
        epochs, tmax=0., method=['shrunk', 'empirical'], verbose=True)

    evoked = epochs.average().pick_types(meg=True)

    del epochs  # to save memory

    # Read the forward solution and compute the inverse operator
    fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
    fwd = mne.read_forward_solution(fname_fwd)

    # make an MEG inverse operator
    info = evoked.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                             loose=0.2, depth=0.8)

    method = "dSPM"
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori=None)
    if fname is not None:
        stc.save(fname)

    return stc


###############################################################################
# from :ref:`sphx_glr_auto_examples_inverse_plot_lcmv_beamformer_volume.py`.
# Note, that in order to save the data fname needs to be set
def compute_lcmv_example_data_vol(data_path, fname=None):
    raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
    fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
    # Get epochs
    event_id, tmin, tmax = [1, 2], -0.2, 0.2

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

    # select time window (tmin, tmax) in ms - consider changing for real data
    # scenario, since those values were chosen to optimize computation time
    stc.crop(0.087, 0.087)

    src = forward['src']

    if fname is not None:
        stc.save(fname)

    return stc, src


# Setup path
data_path = mne.datasets.sample.data_path()
t1_sample = data_path + '/subjects/sample/mri/brain.mgz'
t1_fsaverage = data_path + '/subjects/fsaverage/mri/brain.mgz'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
subjects_dir = data_path + '/subjects'

# read precomputed data
compute_lcmv_example_data_surf(data_path, fname='lcmv-surf')
compute_lcmv_example_data_vol(data_path, fname='lcmv-vol')

src = mne.read_forward_solution(fname_fwd)['src']
stc = mne.read_source_estimate('lcmv-vol-vl.stc')
stc_rh = mne.read_source_estimate('lcmv-surf-rh.stc')

###############################################################################
# Morph VolSourceEstimate

# initialize morpher for VolSourceEstimate (src=src)
morpher = SourceMorph('sample',
                      'fsaverage',
                      src=src,
                      subjects_dir=subjects_dir,
                      grid_spacing=(5., 5., 5.))

# save morph
morpher.save('vol')

# load morph by initializing with file name
morpher_loaded = SourceMorph(None, None, fname='vol-morph.h5')

# morph data
stc_morphed = morpher_loaded(stc)

###############################################################################
# Morph SourceEstimate

fs_vertices = [np.arange(10242)] * 2

# initialize morpher for SourceEstimate (src='surf')
morpher_rh_surf = SourceMorph('sample', 'fsaverage',
                              data_from=stc_rh.vertices,
                              data_to=fs_vertices,
                              subjects_dir=subjects_dir,
                              src='surf')

# save morph
morpher_rh_surf.save('surf_rh')

# load morph by initializing with file name
morpher_rh_surf_loaded = SourceMorph(None, None, None,
                                     fname='surf_rh-morph.h5')

# morph data
stc_rh_morphed = morpher_rh_surf_loaded(stc_rh)

###############################################################################
# Plot results

# plot morphed surface
# vertno_max, time_max = stc_rh_morphed.get_peak(hemi='rh')

# surfer_kwargs = dict(
#     hemi='rh', subjects_dir=subjects_dir,
#     clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
#     initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5)

# brain = stc_rh_morphed.plot(**surfer_kwargs)
# brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=20)

# plot morphed volumes

# create mri-resolution volume of results
img = morpher_loaded.as_volume(stc_morphed, mri_resolution=True)

figure, axes = plt.subplots()
figure.subplots_adjust(top=0.8, left=0.1, right=0.9, hspace=0.5)
figure.patch.set_facecolor('black')

display = plot_anat(nib.load(t1_fsaverage),
                    display_mode='ortho',
                    cut_coords=[0., 0., 0.],
                    draw_cross=False,
                    axes=axes,
                    figure=figure,
                    annotate=False)

display.add_overlay(img, alpha=0.75)
display.annotate(size=8)
axes.set_title('morphed subject results to fsaverage', color='white',
               fontsize=12)
plt.text(plt.xlim()[1], plt.ylim()[0], 't = 0.087s', color='white')
plt.show()
