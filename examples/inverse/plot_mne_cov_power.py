"""
===================================================================
Compute source power estimate by projecting the covariance with MNE
===================================================================

We can use the MNE operator like a beamformer to get a power
estimate by source localizing the bandpass filtered
covariance matrix.

"""
# Author: Denis A. Engemann <denis-alexander.engemann@inria.fr>
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
raw = mne.io.read_raw_fif(raw_fname)  # already has an average reference
raw.load_data()

raw_empty_room_fname = op.join(
    data_path, 'MEG', 'sample', 'ernoise_raw.fif')
raw_empty_room = mne.io.read_raw_fif(raw_empty_room_fname)
raw_empty_room.crop(0, 60)
raw_empty_room.info['bads'] = ['MEG 2443']
raw_empty_room.info['projs'] = raw.info['projs']

events = mne.find_events(raw, stim_channel='STI 014')

# event trigger and conditions
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
raw.info['bads'] = ['MEG 2443', 'EEG 053']
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

epochs_noise = mne.Epochs(raw, events, event_id, tmin, tmax=0., proj=True,
                          picks=('meg', 'eog'), baseline=None, reject=reject)

noise_cov = mne.compute_raw_covariance(
    raw_empty_room, method=['empirical', 'shrunk'])

epochs = mne.Epochs(raw.copy().filter(4, 12), events, event_id, tmin, tmax,
                    proj=True, picks=('meg', 'eog'), baseline=None,
                    reject=reject)

data_cov = mne.compute_covariance(
    epochs, tmin=0., tmax=0.2, method=['shrunk', 'empirical'], rank=None,
    verbose=True)

base_cov = mne.compute_covariance(
    epochs, tmin=-0.2, tmax=0, method=['shrunk', 'empirical'], rank=None,
    verbose=True)

fig_noise_cov = mne.viz.plot_cov(noise_cov, raw.info, show_svd=False)
fig_cov = mne.viz.plot_cov(data_cov, raw.info, show_svd=False)

evoked = epochs.average().pick('meg')
evoked.drop_channels(evoked.info['bads'])
evoked.plot(time_unit='s')
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='mag',
                    time_unit='s')

evoked_noise_cov = mne.EvokedArray(data=np.diag(noise_cov['data'])[:, None],
                                   info=evoked.info)

evoked_data_cov = mne.EvokedArray(data=np.diag(data_cov['data'])[:, None],
                                  info=evoked.info)

evoked_data_cov_white = mne.whiten_evoked(evoked_data_cov, noise_cov)


def plot_cov_diag_topomap(evoked, ch_type='grad'):
    evoked.plot_topomap(
        ch_type=ch_type, times=[0],
        vmin=np.min, vmax=np.max, cmap='viridis',
        units=dict(mag='None', grad='None'),
        scalings=dict(mag=1, grad=1),
        cbar_fmt=None)


plot_cov_diag_topomap(evoked_noise_cov, 'grad')
plot_cov_diag_topomap(evoked_data_cov, 'grad')
plot_cov_diag_topomap(evoked_data_cov_white, 'grad')

# Read the forward solution and compute the inverse operator
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

# make an MEG inverse operator
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

stc_er = apply_inverse_cov(
    data_cov, evoked.info, 1 / 9, inverse_operator,
    method='dSPM', pick_ori=None,
    lambda2=1.,
    verbose=True, dB=False)

stc_base = apply_inverse_cov(
    base_cov, evoked.info, 1 / 9, inverse_operator,
    method='dSPM', pick_ori=None,
    lambda2=1.,
    verbose=True, dB=False)

# Power is relative to the baseline
stc = stc_er / stc_base

stc.plot(subject='sample', subjects_dir=subjects_dir, hemi='both',
         clim=dict(kind='percent', lims=(50, 90, 98)))
