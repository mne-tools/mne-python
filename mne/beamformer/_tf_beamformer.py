import numpy as np

import logging
logger = logging.getLogger('mne')

from ..time_frequency import compute_epochs_csd
from . import dics_source_power


def tf_dics(epochs, forward, label, tmin, tmax, tstep, freq_bins):
    n_steps = int(np.floor((tmax - tmin) / tstep))
    stcs = []
    for i_time in range(n_steps):
        win_tmin = tmin + i_time * tstep
        win_tmax = tmin + (i_time + 1) * tstep

        # Calculating data and noise CSD matrices for current time window
        data_csds = []
        noise_csds = []
        for freq_bin in freq_bins:
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
        #stc = dics_source_power(epochs.info, forward, noise_csds, data_csds,
        #                        reg=0.001)
        stcs.append(stc)

    return stcs
