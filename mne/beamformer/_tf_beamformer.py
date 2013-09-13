import numpy as np

import logging
logger = logging.getLogger('mne')

from ..time_frequency import compute_epochs_csd
from . import dics_source_power
from ..source_estimate import SourceEstimate


def tf_dics(epochs, forward, label, tmin, tmax, tstep, win_lengths, freq_bins):
    # TODO: Check win_lengths and freq_bins match in length
    # TODO: Check that no time window is longer than tstep
    sol = []
    n_steps = int(np.floor((tmax - tmin) / tstep))  # TODO: Use // and 1e3
    for i_freq, freq_bin in enumerate(freq_bins):
        single_sols = []
        overlap_sol = []
        win_length = win_lengths[i_freq]
        # TODO: Note that 0.3 / 0.05 produces 5.999! So n_overlap will be 5
        # instead of the 6 that it should be, absurd! How to deal with this
        # better than by multiplying by 1e3?!?
        n_overlap = int((win_length * 1e3) // (tstep * 1e3))
        #n_filters = int(((timax - win_length - tmin) * 1e3) // (tstep * 1e3))
        for i_time in range(n_steps):
            win_tmin = tmin + i_time * tstep
            win_tmax = win_tmin + win_length

            # TODO: Improve logging, which should be fairly informative,
            # because this is going to be taking a lot of time
            logger.info(str((i_time, i_freq)) + ' ' +
                        str(((win_tmin, win_tmax), freq_bin)))

            # Calculating data and noise CSD matrices for current time window
            if win_tmax < tmax + (epochs.times[-1] - epochs.times[-2]):
                # TODO: Allow selection of multitaper bandwidth for each window
                # length
                data_csd = compute_epochs_csd(epochs, mode='multitaper',
                                              tmin=win_tmin, tmax=win_tmax,
                                              fmin=freq_bin[0],
                                              fmax=freq_bin[1], fsum=True,
                                              mt_bandwidth=15)
                # TODO: Do not manually set control/baseline window!
                noise_csd = compute_epochs_csd(epochs, mode='multitaper',
                                               tmin=-0.15, tmax=-0.001,
                                               fmin=freq_bin[0],
                                               fmax=freq_bin[1], fsum=True,
                                               mt_bandwidth=15)
                stc = dics_source_power(epochs.info, forward, noise_csd,
                                        data_csd, reg=0.001, label=label)
                single_sols.append(stc.data[:, 0])

            # Average over all time windows that contain the current time
            # point, which is the current time window along with n_
            if i_time - n_overlap < 0:
                curr_sol = np.mean(single_sols[0:i_time + 1], axis=0)
            # TODO: Unnecessary, the case below solves this already!
            #elif i_time > n_filters:
            #    curr_sol = np.mean(single_sols[i_time - n_overlap + 1:])
            else:
                curr_sol =\
                    np.mean(single_sols[i_time - n_overlap + 1:i_time + 1],
                            axis=0)

            # The final values for the current time point averaged over all
            # time windows that contain it
            overlap_sol.append(curr_sol)

        # Gathering solutions for all time points for current frequency bin
        sol.append(overlap_sol)

        #for i_freq, _ in enumerate(freq_bins):
        #    sol[i_time].append(stc.data[:, i_freq])

    sol = np.array(sol)
    stcs = []
    for i_freq, _ in enumerate(freq_bins):
        stc = SourceEstimate(sol[i_freq, :, :].T, vertices=stc.vertno,
                             tmin=tmin, tstep=tstep, subject=stc.subject)
        stcs.append(stc)

    return stcs
