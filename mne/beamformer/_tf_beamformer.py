import numpy as np

import logging
logger = logging.getLogger('mne')

from ..time_frequency import compute_epochs_csd
from . import dics_source_power
from ..source_estimate import SourceEstimate


def tf_dics(epochs, forward, label, tmin, tmax, tstep, freq_bins):
    sol = []
    n_steps = int(np.floor((tmax - tmin) / tstep))
    for i_time in range(n_steps):
        sol.append([])
        win_tmin = tmin + i_time * tstep
        win_tmax = tmin + (i_time + 1) * tstep

        # Calculating data and noise CSD matrices for current time window
        data_csds = []
        noise_csds = []
        for i_freq, freq_bin in enumerate(freq_bins):
            # TODO: Improve logging, which should be fairly informative,
            # because this is going to be taking a lot of time
            logger.info((i_time, freq_bin))

            data_csd = compute_epochs_csd(epochs, mode='multitaper',
                                          tmin=win_tmin, tmax=win_tmax,
                                          fmin=freq_bin[0], fmax=freq_bin[1],
                                          fsum=True, mt_bandwidth=10)
            data_csds.append(data_csd)

            noise_csd = compute_epochs_csd(epochs, mode='multitaper',
                                           tmin=-0.15, tmax=-0.001,
                                           fmin=freq_bin[0], fmax=freq_bin[1],
                                           fsum=True, mt_bandwidth=10)
            noise_csds.append(noise_csd)

        stc = dics_source_power(epochs.info, forward, noise_csds, data_csds,
                                reg=0.001, label=label)

        for i_freq, _ in enumerate(freq_bins):
            sol[i_time].append(stc.data[:, i_freq])

    sol = np.array(sol)
    stcs = []
    for i_freq, _ in enumerate(freq_bins):
        stc = SourceEstimate(sol[:, i_freq, :].T, vertices=stc.vertno,
                             tmin=tmin, tstep=tstep, subject=stc.subject)
        stcs.append(stc)

    return stcs
