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
from mne.minimum_norm import make_inverse_operator

data_path = sample.data_path()

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


def _apply_inverse_cov(cov, info, nave, inverse_operator, lambda2=1 / 9,
                       method="dSPM", pick_ori=None, prepared=False,
                       label=None, method_params=None, return_residual=False,
                       verbose=None, log=True):
    """Apply inverse operator to evoked data HACKED
    """
    from mne.minimum_norm.inverse import _check_reference
    from mne.minimum_norm.inverse import _check_ori
    from mne.minimum_norm.inverse import _check_ch_names
    from mne.minimum_norm.inverse import _check_or_prepare
    from mne.minimum_norm.inverse import _pick_channels_inverse_operator
    from mne.minimum_norm.inverse import _assemble_kernel
    from mne.minimum_norm.inverse import _subject_from_inverse
    from mne.minimum_norm.inverse import _get_src_type
    from mne.minimum_norm.inverse import _make_stc
    from mne.utils import _check_option
    from mne.utils import logger
    from mne.io.constants import FIFF

    INVERSE_METHODS = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']

    class fake_evoked:
        info = info

    _check_reference(fake_evoked, inverse_operator['info']['ch_names'])
    _check_option('method', method, INVERSE_METHODS)
    if method == 'eLORETA' and return_residual:
        raise ValueError('eLORETA does not currently support return_residual')
    _check_ori(pick_ori, inverse_operator['source_ori'])
    #
    #   Set up the inverse according to the parameters
    #

    _check_ch_names(inverse_operator, info)

    inv = _check_or_prepare(inverse_operator, nave, lambda2, method,
                            method_params, prepared)

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(cov['names'], inv)
    logger.info('Applying inverse operator to cov...')
    logger.info('    Picked %d channels from the data' % len(sel))
    logger.info('    Computing inverse...')

    K, noise_norm, vertno, source_nn = _assemble_kernel(inv, label, method,
                                                        pick_ori)

    # apply imaging kernel
    sol = np.einsum('ij,ij->i', K, (cov.data[sel] @ K.T).T)[:, None]

    is_free_ori = (inverse_operator['source_ori'] ==
                   FIFF.FIFFV_MNE_FREE_ORI and pick_ori != 'normal')

    if is_free_ori and pick_ori != 'vector':
        logger.info('    Combining the current components...')
        sol = sol[0::3] + sol[1::3] + sol[2::3]

    if noise_norm is not None:
        logger.info('    %s...' % (method,))
        if is_free_ori and pick_ori == 'vector':
            noise_norm = noise_norm.repeat(3, axis=0)
        sol *= noise_norm

    tstep = 1.0 / info['sfreq']
    tmin = 0.0
    subject = _subject_from_inverse(inverse_operator)

    src_type = _get_src_type(inverse_operator['src'], vertno)
    if log:
        sol = np.log10(sol, out=sol)

    stc = _make_stc(sol, vertno, tmin=tmin, tstep=tstep, subject=subject,
                    vector=(pick_ori == 'vector'), source_nn=source_nn,
                    src_type=src_type)
    logger.info('[done]')

    return stc

subjects_dir = data_path + '/subjects'
# make an MEG inverse operator
inverse_operator_er = make_inverse_operator(info, fwd, noise_cov,
                                            loose=0.2, depth=0.8)

stc_er = _apply_inverse_cov(
    data_cov, evoked.info, 1 / 9, inverse_operator_er,
    method='dSPM', pick_ori=None,
    lambda2=1.,
    verbose=True, log=False)

stc_base = _apply_inverse_cov(
    base_cov, evoked.info, 1 / 9, inverse_operator_er,
    method='dSPM', pick_ori=None,
    lambda2=1.,
    verbose=True, log=False)

# Power is relative to the baseline
stc = stc_er / stc_base

stc.plot(subject='sample', subjects_dir=subjects_dir, hemi='both',
         clim=dict(kind='percent', lims=(50, 90, 98)))
