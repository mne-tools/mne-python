"""
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
# License: BSD (3-clause)
import os.path as op

import numpy as np
import mne
from mne.cov import compute_covariance
from mne.datasets import somato
from mne.time_frequency import csd_morlet
from mne.beamformer import (make_dics, apply_dics_csd, make_lcmv,
                            apply_lcmv_cov)
from mne.minimum_norm import (make_inverse_operator, apply_inverse_cov)

print(__doc__)

###############################################################################
# Reading the raw data and creating epochs:
data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = op.join(data_path, 'sub-{}'.format(subject), 'meg',
                    'sub-{}_task-{}_meg.fif'.format(subject, task))

raw = mne.io.read_raw_fif(raw_fname)

# We are interested in the beta band (12-30 Hz)
raw.load_data().filter(12, 30)

# The DICS beamformer currently only supports a single sensor type.
# We'll use the gradiometers in this example.
picks = mne.pick_types(raw.info, meg='grad', exclude='bads')

# Read epochs
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2, picks=picks,
                    preload=True)

# Read forward operator and point to freesurfer subject directory
fname_fwd = op.join(data_path, 'derivatives', 'sub-{}'.format(subject),
                    'sub-{}_task-{}-fwd.fif'.format(subject, task))
subjects_dir = op.join(data_path, 'derivatives', 'freesurfer', 'subjects')

fwd = mne.read_forward_solution(fname_fwd)

###############################################################################
# Compute covariances
# -------------------
# ERS activity starts at 0.5 seconds after stimulus onset.

active_win = (0.5, 1.5)
baseline_win = (-1, 0)

baseline_cov = compute_covariance(epochs, tmin=baseline_win[0],
                                  tmax=baseline_win[1], method='shrunk',
                                  rank=None)
active_cov = compute_covariance(epochs, tmin=active_win[0], tmax=active_win[1],
                                method='shrunk', rank=None)

# Weighted averaging is already in the addition of covariance objects.
common_cov = baseline_cov + active_cov


###############################################################################
# Compute some source estimates
# -----------------------------
# Here we will use DICS, LCMV beamformer, and dSPM.
#
# See :ref:`ex-inverse-source-power` for more information about DICS.

def _gen_dics(active_win, baseline_win, epochs):
    freqs = np.logspace(np.log10(12), np.log10(30), 9)
    csd = csd_morlet(epochs, freqs, tmin=-1, tmax=1.5, decim=20)
    csd_baseline = csd_morlet(epochs, freqs, tmin=baseline_win[0],
                              tmax=baseline_win[1], decim=20)
    csd_ers = csd_morlet(epochs, freqs, tmin=active_win[0], tmax=active_win[1],
                         decim=20)
    filters = make_dics(epochs.info, fwd, csd.mean(), pick_ori='max-power')
    stc_base, freqs = apply_dics_csd(csd_baseline.mean(), filters)
    stc_act, freqs = apply_dics_csd(csd_ers.mean(), filters)
    stc_act /= stc_base
    return stc_act


# generate lcmv source estimate
def _gen_lcmv(active_cov, baseline_cov, common_cov):
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


# Compute source estimates
stc_dics = _gen_dics(active_win, baseline_win, epochs)
stc_lcmv = _gen_lcmv(active_cov, baseline_cov, common_cov)
stc_dspm = _gen_mne(active_cov, baseline_cov, common_cov, fwd, epochs.info)

###############################################################################
# Plot source estimates
# ---------------------

for method, stc in zip(['DICS', 'LCMV', 'dSPM'],
                       [stc_dics, stc_lcmv, stc_dspm]):
    title = '%s source power in the 12-30 Hz frequency band' % method
    brain = stc.plot(hemi='rh', subjects_dir=subjects_dir,
                     subject=subject, time_label=title)
