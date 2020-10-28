"""
===================================================================
Compute source power estimate by projecting the covariance with MNE
===================================================================

We can apply the MNE inverse operator to a covariance matrix to obtain
an estimate of source power. This is computationally more efficient than first
estimating the source timecourses and then computing their power. This
code is based on the code from :footcite:`Sabbagh2020` and has been useful to
correct for individual field spread using source localization in the context of
predictive modeling.

References
----------
.. footbibliography::
"""
# Author: Denis A. Engemann <denis-alexander.engemann@inria.fr>
#         Luke Bloy <luke.bloy@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse_cov

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)

###############################################################################
# Compute empty-room covariance
# -----------------------------
# First we compute an empty-room covariance, which captures noise from the
# sensors and environment.

raw_empty_room_fname = op.join(
    data_path, 'MEG', 'sample', 'ernoise_raw.fif')
raw_empty_room = mne.io.read_raw_fif(raw_empty_room_fname)
raw_empty_room.crop(0, 60)
raw_empty_room.info['bads'] = ['MEG 2443']
raw_empty_room.info['projs'] = raw.info['projs']
noise_cov = mne.compute_raw_covariance(
    raw_empty_room, method=['empirical', 'shrunk'])
del raw_empty_room

###############################################################################
# Epoch the data
# --------------

raw.info['bads'] = ['MEG 2443', 'EEG 053']
raw.load_data().filter(4, 12)
events = mne.find_events(raw, stim_channel='STI 014')
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
tmin, tmax = -0.2, 0.5
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
epochs = mne.Epochs(raw.copy().filter(4, 12), events, event_id, tmin, tmax,
                    proj=True, picks=('meg', 'eog'), baseline=None,
                    reject=reject, preload=True)
del raw

###############################################################################
# Compute and plot covariances
# ----------------------------
# In addition to the empty-room covariance above, we compute two additional
# covariances:
#
# 1. Baseline covariance, which captures signals not of interest in our
#    analysis (e.g., sensor noise, environmental noise, physiological
#    artifacts, and also resting-state-like brain activity / "noise").
# 2. Data covariance, which captures our activation of interest (in addition
#    to noise sources).

base_cov = mne.compute_covariance(
    epochs, tmin=-0.2, tmax=0, method=['shrunk', 'empirical'], rank=None,
    verbose=True)
data_cov = mne.compute_covariance(
    epochs, tmin=0., tmax=0.2, method=['shrunk', 'empirical'], rank=None,
    verbose=True)

fig_noise_cov = mne.viz.plot_cov(noise_cov, epochs.info, show_svd=False)
fig_base_cov = mne.viz.plot_cov(base_cov, epochs.info, show_svd=False)
fig_data_cov = mne.viz.plot_cov(data_cov, epochs.info, show_svd=False)

###############################################################################
# We can also look at the covariances using topomaps, here we just show the
# baseline and data covariances, followed by the data covariance whitened
# by the baseline covariance:

evoked = epochs.average().pick('meg')
evoked.drop_channels(evoked.info['bads'])
evoked.plot(time_unit='s')
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='mag')
noise_cov.plot_topomap(evoked.info, 'grad', title='Noise')
data_cov.plot_topomap(evoked.info, 'grad', title='Data')
data_cov.plot_topomap(evoked.info, 'grad', noise_cov=noise_cov,
                      title='Whitened data')

###############################################################################
# Apply inverse operator to covariance
# ------------------------------------
# Finally, we can construct an inverse using the empty-room noise covariance:

# Read the forward solution and compute the inverse operator
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

# make an MEG inverse operator
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

###############################################################################
# Project our data and baseline covariance to source space:

stc_data = apply_inverse_cov(data_cov, evoked.info, inverse_operator,
                             nave=len(epochs), method='dSPM', verbose=True)
stc_base = apply_inverse_cov(base_cov, evoked.info, inverse_operator,
                             nave=len(epochs), method='dSPM', verbose=True)

###############################################################################
# And visualize power is relative to the baseline:

# sphinx_gallery_thumbnail_number = 9

stc_data /= stc_base
brain = stc_data.plot(subject='sample', subjects_dir=subjects_dir,
                      clim=dict(kind='percent', lims=(50, 90, 98)))
