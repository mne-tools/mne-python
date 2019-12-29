"""

=========================================
Compute evoked ERS source power using LCMV beamfomer and MNE inverse
=========================================

Compute a Dynamic Imaging of Coherent Sources (DICS) [1]_ filter from
single-trial activity to estimate source power across a frequency band. This
example demonstrates how to source localize the event-related synchronization
(ERS) of beta band activity in this dataset: :ref:`somato-dataset`

References
----------
.. [1] Gross et al. Dynamic imaging of coherent sources: Studying neural
       interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#         Roman Goj <roman.goj@gmail.com>
#         Denis Engemann <denis.engemann@gmail.com>
#         Stefan Appelhoff <stefan.appelhoff@mailbox.org>
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

# We are interested in the beta band. So we can filter
raw.load_data().filter(12, 30)

# Set picks, use a single sensor type
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

# fix the subject in fwd['src']
for src in fwd['src']:
    src['subject_his_id'] = subject

###############################################################################
# now we can compute some source estimates using different methods.
#
# define the active and baseline windows in seconds
# ERS activity starts at 0.5 seconds after stimulus onset
active_win = (0.5, 1.5)
baseline_win = (-1, 0)


# generate a dics source estimate - see example/plot_dics_source_power.py for
# more information
def _gen_dics(active_win, baseline_win, epochs):
    freqs = np.logspace(np.log10(12), np.log10(30), 9)
    csd = csd_morlet(epochs, freqs, tmin=-1, tmax=1.5, decim=20)
    csd_baseline = csd_morlet(epochs, freqs, tmin=baseline_win[0],
                              tmax=baseline_win[1], decim=20)
    csd_ers = csd_morlet(epochs, freqs, tmin=active_win[0], tmax=active_win[1],
                         decim=20)
    filters = make_dics(epochs.info, fwd, csd.mean(), pick_ori='max-power')
    baseline_source_power, freqs = apply_dics_csd(csd_baseline.mean(), filters)
    beta_source_power, freqs = apply_dics_csd(csd_ers.mean(), filters)
    stc = beta_source_power / baseline_source_power
    return stc


def _gen_lcmv(active_cov, baseline_cov, common_cov):
    filters = make_lcmv(epochs.info, fwd, common_cov, reg=0.05,
                        noise_cov=None, pick_ori='max-power',
                        weight_norm='nai', rank=None)
    stc_base = apply_lcmv_cov(baseline_cov, filters)
    stc_act = apply_lcmv_cov(active_cov, filters)
    stc_act /= stc_base
    return stc_act


def _gen_mne(active_cov, baseline_cov, common_cov, fwd, info, method):
    inverse_operator = make_inverse_operator(info, fwd, common_cov,
                                             loose=0.2, depth=0.8)

    stc_act = apply_inverse_cov(
        active_cov, info, 1, inverse_operator,
        method=method, pick_ori=None,
        lambda2=1. / 9.,
        verbose=True, dB=False)

    stc_base = apply_inverse_cov(
        baseline_cov, info, 1, inverse_operator,
        method=method, pick_ori=None,
        lambda2=1. / 9.,
        verbose=True, dB=False)
    stc_act /= stc_base
    return stc_act


# compute covariances for the lcmv and MNE methods.
baseline_cov = compute_covariance(epochs, tmin=baseline_win[0],
                                  tmax=baseline_win[1], method='shrunk',
                                  rank=None)
active_cov = compute_covariance(epochs, tmin=active_win[0], tmax=active_win[1],
                                method='shrunk', rank=None)

# compute a common covariance (weighted average of time windows)
# to use in making the filter.
ws = np.array([baseline_win[1] - baseline_win[0],
               active_win[1] - active_win[0]])
ws /= ws.sum()
# this isn't supported
# common_cov = ws[0] * baseline_cov + ws[1] * active_cov
common_cov = baseline_cov + active_cov

# Compute source estimates
stc_dics = _gen_dics(active_win, baseline_win, epochs)
stc_lcmv = _gen_lcmv(active_cov, baseline_cov, common_cov)
stc_mne = _gen_mne(active_cov, baseline_cov, common_cov, fwd, epochs.info,
                   'MNE')
stc_dspm = _gen_mne(active_cov, baseline_cov, common_cov, fwd, epochs.info,
                    'dSPM')

for method, stc in zip(['DICS', 'LCMV', 'MNE', 'dSPM'],
                       [stc_dics, stc_lcmv, stc_mne, stc_dspm]):
    title = '%s source power in the 12-30 Hz frequency band' % method
    brain = stc.plot(hemi='rh', subjects_dir=subjects_dir,
                     subject=subject, time_label=title)
