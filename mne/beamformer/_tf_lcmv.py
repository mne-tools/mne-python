import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('mne')

import mne

from ..fiff.pick import pick_channels_cov
from ..forward import _subject_from_forward
from ..cov import compute_whitener
from ..source_estimate import SourceEstimate
from ._lcmv import _prepare_beamformer_input


def iter_filter_epochs(raw, freq_bins, events, event_id, tmin, tmax, baseline,
                       n_jobs, picks):
    """Filters raw data, creates and yields epochs
    """
    # Getting picks based on rejections prior to filtering
    tmp_epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            proj=True, baseline=baseline, preload=True)
    picks = tmp_epochs.picks

    for freq_bin in freq_bins:
        raw_band = raw.copy()
        raw_band.filter(l_freq=freq_bin[0], h_freq=freq_bin[1], picks=picks,
                        n_jobs=n_jobs)
        # TODO: Which of these parameters should really be exposed? All?
        # defaults taken from mne.Epochs?
        epochs_band = mne.Epochs(raw_band, events, event_id, tmin, tmax,
                                 proj=True, picks=picks, baseline=baseline,
                                 preload=True,
                                 reject=dict(grad=4000e-13, mag=4e-12))

        yield epochs_band, freq_bin


def tf_lcmv(epochs_band, forward, label, tmin, tmax, tstep, win_length,
            control, reg):
    # TODO: Check win_length and freq_bin match in length
    # TODO: Check that no time window is longer than tstep
    single_sols = []
    overlap_sol = []
    # TODO: Note that 0.3 / 0.05 produces 5.999! So n_overlap will be 5
    # instead of the 6 that it should be, absurd! How to deal with this
    # better than by multiplying by 1e3?!?
    n_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))
    n_overlap = int((win_length * 1e3) // (tstep * 1e3))

    # Calculating noise covariance
    noise_cov = mne.compute_covariance(epochs_band, tmin=control[0],
                                       tmax=control[1])
    noise_cov = mne.cov.regularize(noise_cov, epochs_band.info, mag=0.05,
                                   grad=0.05, eeg=0.1, proj=True)

    for i_time in range(n_steps):
        win_tmin = tmin + i_time * tstep
        win_tmax = win_tmin + win_length

        if win_tmax < tmax + (epochs_band.times[-1] - epochs_band.times[-2]):
            # Calculating data covariance in current time window
            data_cov = mne.compute_covariance(epochs_band, tmin=win_tmin,
                                              tmax=win_tmax)

            stc = _lcmv_source_power(epochs_band.info, forward, noise_cov,
                                     data_cov, reg=0.001, label=label)
            single_sols.append(stc.data[:, 0])

        # Average over all time windows that contain the current time
        # point, which is the current time window along with n_overlap
        # others
        if i_time - n_overlap < 0:
            curr_sol = np.mean(single_sols[0:i_time + 1], axis=0)
        else:
            curr_sol = np.mean(single_sols[i_time - n_overlap + 1:
                                           i_time + 1], axis=0)

        # The final values for the current time point averaged over all
        # time windows that contain it
        overlap_sol.append(curr_sol)

    sol = np.array(overlap_sol)
    return SourceEstimate(sol.T, vertices=stc.vertno, tmin=tmin, tstep=tstep,
                          subject=stc.subject)


def _lcmv_source_power(info, forward, noise_cov, data_cov, reg=0.01,
                       label=None, picks=None, pick_ori=None, verbose=None):
    # TODO: Results of tf_dics are really weird, this function's results should
    # be examined.
    is_free_ori, picks, ch_names, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    # Handle whitening
    whitener, _ = compute_whitener(noise_cov, info, picks)

    # whiten the leadfield
    G = np.dot(whitener, G)

    # Apply SSPs + whitener to data covariance
    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))
    Cm = np.dot(whitener, np.dot(Cm, whitener.T))

    # Calculating regularized inverse, equivalent to an inverse operation after
    # the following regularization:
    # Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
    Cm_inv = linalg.pinv(Cm, reg)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    source_power = np.zeros((n_sources, 1))
    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

        # Noise normalization
        noise_norm = np.dot(Wk, Wk.T)
        noise_norm = noise_norm.trace()

        # Calculating source power
        sp_temp = np.dot(np.dot(Wk, Cm), Wk.T)
        sp_temp /= noise_norm
        if pick_ori == 'normal':
            source_power[k, 0] = sp_temp[2, 2]
        else:
            source_power[k, 0] = sp_temp.trace()

    logger.info('[done]')

    subject = _subject_from_forward(forward)
    return SourceEstimate(source_power, vertices=vertno, tmin=1,
                          tstep=1, subject=subject)
