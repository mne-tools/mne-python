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
#          Alex Rockhill <aprockhill@mailbox.org>
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
subjects_dir = data_path / 'derivatives' / 'freesurfer' / 'subjects'
task = 'somato'
raw_fname = (data_path / 'sub-{}'.format(subject) / 'meg' /
             'sub-{}_task-{}_meg.fif'.format(subject, task))

# crop to 5 minutes to save memory
raw = mne.io.read_raw_fif(raw_fname).crop(0, 300)

# We are interested in the beta band (12-30 Hz)
raw.load_data().filter(12, 30)

# The DICS beamformer currently only supports a single sensor type.
# We'll use the gradiometers in this example.
picks = mne.pick_types(raw.info, meg='grad', exclude='bads')

# Read epochs
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2, picks=picks,
                    preload=True, decim=3)

# Read forward operator and point to freesurfer subject directory
fwd_fname = (data_path / 'derivatives' / 'sub-{}'.format(subject) /
             'sub-{}_task-{}-fwd.fif'.format(subject, task))
fwd = mne.read_forward_solution(fwd_fname)

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
win_active = (0.5, 1.5)
win_baseline = (-1, 0)
cov_baseline = compute_covariance(epochs, tmin=win_baseline[0],
                                  tmax=win_baseline[1], method='shrunk',
                                  rank=rank, verbose=True)
cov_active = compute_covariance(epochs, tmin=win_active[0], tmax=win_active[1],
                                method='shrunk', rank=rank, verbose=True)

# when the covariance objects are added together, they are scaled by the size
# of the window used to create them so that the average is properly weighted
cov_common = cov_baseline + cov_active
cov_baseline.plot(epochs.info)

freqs = np.logspace(np.log10(12), np.log10(30), 9)

# time-frequency decomposition
epochs_tfr = mne.time_frequency.tfr_morlet(
    epochs, freqs=freqs, n_cycles=freqs / 2, return_itc=False,
    average=False, output='complex')
epochs_tfr.decimate(20)  # decimate for speed

# compute cross-spectral density matrices
csd = csd_tfr(epochs_tfr, tmin=-1, tmax=1.5)
csd_baseline = csd_tfr(epochs_tfr, tmin=win_baseline[0], tmax=win_baseline[1])
csd_ers = csd_tfr(epochs_tfr, tmin=win_active[0], tmax=win_active[1])

csd_baseline.plot()

# %%
# Compute some source estimates
# -----------------------------
# Here we will use DICS, LCMV beamformer, and dSPM.
#
# See :ref:`ex-inverse-source-power` for more information about DICS.


def _gen_dics(csd, ers_csd, csd_baseline, fwd):
    filters = make_dics(epochs.info, fwd, csd.mean(), pick_ori='max-power',
                        reduce_rank=True, real_filter=True, rank=rank)
    stc_base, freqs = apply_dics_csd(csd_baseline.mean(), filters)
    stc_act, freqs = apply_dics_csd(csd_ers.mean(), filters)
    stc_act /= stc_base
    return stc_act


# generate lcmv source estimate
def _gen_lcmv(active_cov, cov_baseline, common_cov, fwd):
    filters = make_lcmv(epochs.info, fwd, common_cov, reg=0.05,
                        noise_cov=None, pick_ori='max-power')
    stc_base = apply_lcmv_cov(cov_baseline, filters)
    stc_act = apply_lcmv_cov(cov_active, filters)
    stc_act /= stc_base
    return stc_act


# generate mne/dSPM source estimate
def _gen_mne(cov_active, cov_baseline, cov_common, fwd, info, method='dSPM'):
    inverse_operator = make_inverse_operator(info, fwd, cov_common)
    stc_act = apply_inverse_cov(cov_active, info, inverse_operator,
                                method=method, verbose=True)
    stc_base = apply_inverse_cov(cov_baseline, info, inverse_operator,
                                 method=method, verbose=True)
    stc_act /= stc_base
    return stc_act


# Compute source estimates
stc_dics = _gen_dics(csd, csd_ers, csd_baseline, fwd)
stc_lcmv = _gen_lcmv(cov_active, cov_baseline, cov_common, fwd)
stc_dspm = _gen_mne(cov_active, cov_baseline, cov_common, fwd, epochs.info)

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
# whiten the sensor data
inverse_operator = list()
for freq_idx in range(epochs_tfr.freqs.size):
    # for each frequency, compute a separate covariance matrix
    cov_baseline = csd_baseline.get_data(index=freq_idx, as_cov=True)
    cov_baseline['data'] = cov_baseline['data'].real  # only normalize by real
    # then use that covariance matrix as normalization for the inverse
    # operator
    inverse_operator.append(mne.minimum_norm.make_inverse_operator(
        epochs.info, vol_fwd, cov_baseline))

# finally, compute the stcs for each epoch and frequency
stcs = mne.minimum_norm.apply_inverse_tfr_epochs(
    epochs_tfr, inverse_operator, lambda2, method=method,
    pick_ori='vector')

# %%
# Plot volume source estimates
# ----------------------------

viewer = mne.gui.view_vol_stc(stcs, subject=subject, subjects_dir=subjects_dir,
                              src=vol_src, inst=epochs_tfr)
viewer.go_to_extreme()  # show the maximum intensity source vertex
viewer.set_cmap(vmin=0.25, vmid=0.8)
viewer.set_3d_view(azimuth=40, elevation=35, distance=350)
