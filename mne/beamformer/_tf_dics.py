"""5D time-frequency beamforming based on DICS
"""

# Authors: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import logging
logger = logging.getLogger('mne')

from ..time_frequency import compute_epochs_csd
from ..source_estimate import SourceEstimate
from .. import verbose
from . import dics_source_power


@verbose
def tf_dics(epochs, forward, tmin, tmax, tstep, win_lengths, freq_bins,
            mode='fourier', mt_bandwidths=None, mt_adaptive=False,
            mt_low_bias=True, reg=0.01, label=None, pick_ori=None,
            verbose=None):
    """5D time-frequency beamforming based on DICS.

    Calculate source power in time-frequency windows using a spatial filter
    based on the Dynamic Imaging of Coherent Sources (DICS) beamforming
    approach. For each time window and frequency bin combination cross-spectral
    density (CSD) is computed and used to create a beamformer spatial filter
    with noise CSD calculated from the baseline period being used for
    normalization.

    NOTE: Currently noise CSD can only be calculated in the baseline period
    with a time window of length equal to that used for calculating the data
    CSD. It should also be possible to use a longer time window (e.g. 600 ms)
    for the noise CSD estimate, but this is not implemented and was not tested.

    The original reference is:
    Dalal et al. Five-dimensional neuroimaging: Localization of the
    time-frequency dynamics of cortical activity.
    NeuroImage (2008) vol. 40 (4) pp. 1686-1700

    NOTE : Dalal et al. used a synthetic aperture magnetometry beamformer (SAM)
    in each time-frequency window instead of DICS.
    """
    # TODO: Comment that multitaper mode is not yet well tested for use in
    # tf_dics

    if len(win_lengths) != len(freq_bins):
        raise ValueError('One time window length expected per frequency bin')
    if mt_bandwidths is not None and len(mt_bandwidths) != len(freq_bins):
        raise ValueError('When using multitaper mode and specifying '
                         'multitaper transform bandwidth, one value must be '
                         'provided per frequency bin')
    if mt_bandwidths is None:
        mt_bandwidths = [None] * len(freq_bins)

    # TODO: Note that 0.3 / 0.05 produces 5.99! So result of // or np.floor
    # will be 5 instead of 6. How to deal with this better than by multiplying
    # by 1e3?
    # Note: Multiplying by 1e3 to avoid numerical issues, e.g. 0.3 // 0.05 == 5
    n_time_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))

    sol_final = []
    for i_freq, freq_bin in enumerate(freq_bins):
        win_length = win_lengths[i_freq]
        n_overlap = int((win_length * 1e3) // (tstep * 1e3))

        # Calculating noise CSD
        noise_csd = compute_epochs_csd(epochs, mode=mode, fmin=freq_bin[0],
                                       fmax=freq_bin[1], fsum=True, tmin=tmin,
                                       tmax=tmin + win_length,
                                       mt_bandwidth=mt_bandwidths[i_freq],
                                       mt_low_bias=mt_low_bias)

        sol_single = []
        sol_overlap = []
        for i_time in range(n_time_steps):
            win_tmin = tmin + i_time * tstep
            win_tmax = win_tmin + win_length

            logger.info('Computing time-frequency DICS beamformer for time '
                        'window %d to %d ms, in frequency range %d to %d Hz'
                        % (win_tmin * 1e3, win_tmax * 1e3, freq_bin[0],
                           freq_bin[1]))

            if win_tmax < tmax + (epochs.times[-1] - epochs.times[-2]):
                # Calculating data CSD in current time window
                data_csd = compute_epochs_csd(epochs, mode=mode,
                                              fmin=freq_bin[0],
                                              fmax=freq_bin[1], fsum=True,
                                              tmin=win_tmin, tmax=win_tmax,
                                              mt_bandwidth=
                                              mt_bandwidths[i_freq],
                                              mt_low_bias=mt_low_bias)

                stc = dics_source_power(epochs.info, forward, noise_csd,
                                        data_csd, reg=reg, label=label,
                                        pick_ori=pick_ori)
                sol_single.append(stc.data[:, 0])

            # Average over all time windows that contain the current time
            # point, which is the current time window along with
            # n_overlap - 1 previous ones
            if i_time - n_overlap < 0:
                curr_sol = np.mean(sol_single[0:i_time + 1], axis=0)
            else:
                curr_sol =\
                    np.mean(sol_single[i_time - n_overlap + 1:i_time + 1],
                            axis=0)

            # The final result for the current time point in the current
            # frequency bin
            sol_overlap.append(curr_sol)

        # Gathering solutions for all time points for current frequency bin
        sol_final.append(sol_overlap)

    sol_final = np.array(sol_final)

    # Creating stc objects containing all time points for each frequency bin
    stcs = []
    for i_freq, _ in enumerate(freq_bins):
        stc = SourceEstimate(sol_final[i_freq, :, :].T, vertices=stc.vertno,
                             tmin=tmin, tstep=tstep, subject=stc.subject)
        stcs.append(stc)

    return stcs
