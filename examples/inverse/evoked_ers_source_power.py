# -*- coding: utf-8 -*-
"""
.. _ex-source-loc-methods:

=====================================================================
Compute evoked ERS source power using DICS, LCMV beamformer, and dSPM
=====================================================================

Here we examine 3 ways of localizing event-related synchronization (ERS) of
beta band activity in this dataset: :ref:`somato-dataset` using
:term:`DICS`, :term:`LCMV beamformer`, and :term:`dSPM` applied to active and
baseline covariance matrices.
"""
# Authors: Luke Bloy <luke.bloy@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

import numpy as np
import mne
from mne.cov import compute_covariance
from mne.datasets import somato
from mne.time_frequency import csd_tfr
from mne.beamformer import (make_dics, apply_dics_csd, make_lcmv,
                            apply_lcmv_cov)
from mne.minimum_norm import (make_inverse_operator, apply_inverse_cov)

print(__doc__)

# %%
# Reading the raw data and creating epochs:

data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = (data_path / 'sub-{}'.format(subject) / 'meg' /
             'sub-{}_task-{}_meg.fif'.format(subject, task))

# crop to 5 minutes to save memory
raw = mne.io.read_raw_fif(raw_fname).crop(0, 300)

# we are interested in the beta band (12-30 Hz)
raw.load_data().filter(12, 30)

# the DICS beamformer currently only supports a single sensor type,
# we'll use the gradiometers in this example
picks = mne.pick_types(raw.info, meg='grad', exclude='bads')

# read epochs
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2, picks=picks,
                    preload=True, decim=3)

# read forward operator and point to freesurfer subject directory
fname_fwd = (data_path / 'derivatives' / 'sub-{}'.format(subject) /
             'sub-{}_task-{}-fwd.fif'.format(subject, task))
subjects_dir = data_path / 'derivatives' / 'freesurfer' / 'subjects'

fwd = mne.read_forward_solution(fname_fwd)

# %%
# Compute covariances and cross-spectral density
# ----------------------------------------------
# ERS activity starts at 0.5 seconds after stimulus onset. Because these
# data have been processed by MaxFilter directly (rather than MNE-Python's
# version), we have to be careful to compute the rank with a more conservative
# threshold in order to get the correct data rank (64). Once this is used in
# combination with an advanced covariance estimator like "shrunk", the rank
# will be correctly preserved.

rank = mne.compute_rank(epochs, tol=1e-6, tol_kind='relative')
active_win = (0.5, 1.5)
baseline_win = (-1, 0)
baseline_cov = compute_covariance(epochs, tmin=baseline_win[0],
                                  tmax=baseline_win[1], method='shrunk',
                                  rank=rank, verbose=True)
active_cov = compute_covariance(epochs, tmin=active_win[0], tmax=active_win[1],
                                method='shrunk', rank=rank, verbose=True)

# weighted averaging is already in the addition of covariance objects
common_cov = baseline_cov + active_cov
baseline_cov.plot(epochs.info)

# compute cross-spectral density matrices
freqs = np.logspace(np.log10(12), np.log10(30), 9)

# time-frequency decomposition
epochs_tfr = mne.time_frequency.tfr_morlet(
    epochs, freqs=freqs, n_cycles=freqs / 2, return_itc=False,
    average=False, output='complex')
epochs_tfr.decimate(20)  # decimate for speed

csd = csd_tfr(epochs_tfr, tmin=-1, tmax=1.5)
baseline_csd = csd_tfr(epochs_tfr, tmin=baseline_win[0], tmax=baseline_win[1])
ers_csd = csd_tfr(epochs_tfr, tmin=active_win[0], tmax=active_win[1])

baseline_csd.plot()

# %%
# Compute some source estimates
# -----------------------------
# Here we will use DICS, LCMV beamformer, and dSPM.
#
# See :ref:`ex-inverse-source-power` for more information about DICS.


def _gen_dics(csd, ers_csd, baseline_csd, fwd):
    filters = make_dics(epochs.info, fwd, csd.mean(), pick_ori='max-power',
                        reduce_rank=True, real_filter=True, rank=rank)
    stc_base, freqs = apply_dics_csd(baseline_csd.mean(), filters)
    stc_act, freqs = apply_dics_csd(ers_csd.mean(), filters)
    stc_act /= stc_base
    return stc_act


# generate lcmv source estimate
def _gen_lcmv(active_cov, baseline_cov, common_cov, fwd):
    filters = make_lcmv(epochs.info, fwd, common_cov, reg=0.05,
                        noise_cov=None, pick_ori='max-power')
    stc_base = apply_lcmv_cov(baseline_cov, filters)
    stc_act = apply_lcmv_cov(active_cov, filters)
    stc_act /= stc_base
    return stc_act


# generate mne/dSPM source estimate
def _gen_mne(active_cov, baseline_cov, common_cov, fwd, info, method='dSPM'):
    inverse_operator = make_inverse_operator(info, fwd, common_cov)
    stc_act = apply_inverse_cov(active_cov, info, inverse_operator,
                                method=method, verbose=True)
    stc_base = apply_inverse_cov(baseline_cov, info, inverse_operator,
                                 method=method, verbose=True)
    stc_act /= stc_base
    return stc_act


# compute source estimates
stc_dics = _gen_dics(csd, ers_csd, baseline_csd, fwd)
stc_lcmv = _gen_lcmv(active_cov, baseline_cov, common_cov, fwd)
stc_dspm = _gen_mne(active_cov, baseline_cov, common_cov, fwd, epochs.info)

# %%
# Plot source estimates
# ---------------------
# DICS:

brain_dics = stc_dics.plot(
    hemi='rh', subjects_dir=subjects_dir, subject=subject,
    time_label='DICS source power in the 12-30 Hz frequency band')

# %%
# LCMV:

brain_lcmv = stc_lcmv.plot(
    hemi='rh', subjects_dir=subjects_dir, subject=subject,
    time_label='LCMV source power in the 12-30 Hz frequency band')

# %%
# dSPM:

brain_dspm = stc_dspm.plot(
    hemi='rh', subjects_dir=subjects_dir, subject=subject,
    time_label='dSPM source power in the 12-30 Hz frequency band')

# %%
# Use volume source estimate with time-frequency resolution
# ---------------------------------------------------------

# make a volume source space
surface = subjects_dir / subject / 'bem' / 'inner_skull.surf'
vol_src = mne.setup_volume_source_space(
    subject=subject, subjects_dir=subjects_dir, surface=surface,
    pos=10, add_interpolator=False)  # just for speed!

conductivity = (0.3,)  # one layer for MEG
model = mne.make_bem_model(subject=subject, ico=3,  # just for speed
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

trans = fwd['info']['mri_head_t']
vol_fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=vol_src, bem=bem, meg=True, eeg=True,
    mindist=5.0, n_jobs=1, verbose=True)

# Compute source estimate using MNE solver
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'MNE'  # use MNE method (could also be dSPM or sLORETA)

# make a different inverse operator for each frequency so as to properly
# whiten
inverse_operator = list()
for freq_idx in range(epochs_tfr.freqs.size):
    baseline_cov = baseline_csd.get_data(index=freq_idx, as_cov=True)
    baseline_cov['data'] = baseline_cov['data'].real  # only normalize by real
    inverse_operator.append(mne.minimum_norm.make_inverse_operator(
        epochs.info, vol_fwd, baseline_cov))

stcs = mne.minimum_norm.apply_inverse_tfr_epochs(
    epochs_tfr, inverse_operator, lambda2, method=method,
    pick_ori='vector')

# %%
# Plot volume source estimates
# ----------------------------

# note, here frequencies are the outer list, opposite of the beamformer
# here, we used pick_ori='vector' so we have an orientation dimension

# compute power, take the average over epochs and cast to integers to save
# memory, the GUI can also handle complex data across epochs if your
# computer has enough RAM but this really lowers the memory usage
data = np.array([(np.mean(
    [(stc.data * stc.data.conj()).real for stc in tfr_stcs],
    axis=0, keepdims=True) * 1e32).astype(np.uint64) for tfr_stcs in stcs])
data = data.transpose((1, 2, 3, 0, 4))  # move frequencies to penultimate

# gain normalize
data = data // data.mean(axis=-1, keepdims=True)

viewer = mne.gui.view_stc(data, subject=subject, subjects_dir=subjects_dir,
                          src=vol_src, inst=epochs_tfr)
viewer.go_to_max()  # show the maximum intensity source vertex
viewer.update_cmap(vmin=0.6, vmid=0.8)
viewer.set_3d_view(azimuth=250, elevation=70, distance=30)
