# -*- coding: utf-8 -*-
"""
=================================================
Compute cross-talk functions for LCMV beamformers
=================================================

Visualise cross-talk functions at one vertex for LCMV beamformers computed
with different data covariance matrices, which affects their cross-talk
functions.
"""
# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.beamformer import make_lcmv, make_lcmv_resolution_matrix
from mne.minimum_norm import get_cross_talk

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects/'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fname_evo = data_path + '/MEG/sample/sample_audvis-ave.fif'
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Read raw data
raw = mne.io.read_raw_fif(raw_fname)

# only pick good EEG/MEG sensors
raw.info['bads'] += ['EEG 053']  # bads + 1 more
picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')

# Find events
events = mne.find_events(raw)

# event_id = {'aud/l': 1, 'aud/r': 2, 'vis/l': 3, 'vis/r': 4}
event_id = {'vis/l': 3, 'vis/r': 4}

tmin, tmax = -.2, .25  # epoch duration
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=(-.2, 0.), preload=True)
del raw

# covariance matrix for pre-stimulus interval
tmin, tmax = -.2, 0.
cov_pre = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax,
                                 method='empirical')

# covariance matrix for post-stimulus interval (around main evoked responses)
tmin, tmax = 0.05, .25
cov_post = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax,
                                  method='empirical')
info = epochs.info
del epochs

# read forward solution
forward = mne.read_forward_solution(fname_fwd)
# use forward operator with fixed source orientations
mne.convert_forward_solution(forward, surf_ori=True,
                             force_fixed=True, copy=False)

# read noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# regularize noise covariance (we used 'empirical' above)
noise_cov = mne.cov.regularize(noise_cov, info, mag=0.1, grad=0.1,
                               eeg=0.1, rank='info')

##############################################################################
# Compute LCMV filters with different data covariance matrices
# ------------------------------------------------------------

# compute LCMV beamformer filters for pre-stimulus interval
filters_pre = make_lcmv(info, forward, cov_pre, reg=0.05,
                        noise_cov=noise_cov,
                        pick_ori=None, rank=None,
                        weight_norm=None,
                        reduce_rank=False,
                        verbose=False)

# compute LCMV beamformer filters for post-stimulus interval
filters_post = make_lcmv(info, forward, cov_post, reg=0.05,
                         noise_cov=noise_cov,
                         pick_ori=None, rank=None,
                         weight_norm=None,
                         reduce_rank=False,
                         verbose=False)

##############################################################################
# Compute resolution matrices for the two LCMV beamformers
# --------------------------------------------------------

rm_pre = make_lcmv_resolution_matrix(filters_pre, forward, info)

rm_post = make_lcmv_resolution_matrix(filters_post, forward, info)

# compute cross-talk functions (CTFs) for one target vertex
sources = [3000]

stc_pre = get_cross_talk(rm_pre, forward['src'], sources, norm=True)

stc_post = get_cross_talk(rm_post, forward['src'], sources, norm=True)
verttrue = [forward['src'][0]['vertno'][sources[0]]]  # pick one vertex
del forward

##############################################################################
# Visualize
# ---------
# Pre:

brain_pre = stc_pre.plot('sample', 'inflated', 'lh', subjects_dir=subjects_dir,
                         figure=1, clim=dict(kind='value', lims=(0, .2, .4)))

brain_pre.add_text(0.1, 0.9, 'LCMV beamformer with pre-stimulus\ndata '
                   'covariance matrix', 'title', font_size=16)

# mark true source location for CTFs
brain_pre.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

###############################################################################
# Post:

brain_post = stc_post.plot('sample', 'inflated', 'lh',
                           subjects_dir=subjects_dir,
                           figure=2, clim=dict(kind='value', lims=(0, .2, .4)))

brain_post.add_text(0.1, 0.9, 'LCMV beamformer with post-stimulus\ndata '
                    'covariance matrix', 'title', font_size=16)

brain_post.add_foci(verttrue, coords_as_verts=True, scale_factor=1.,
                    hemi='lh', color='green')

###############################################################################
# The pre-stimulus beamformer's CTF has lower values in parietal regions
# suppressed alpha activity?) but larger values in occipital regions (less
# suppression of visual activity?).
