"""Util function to baseline correct data
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

from .utils import logger, verbose


def _log_rescale(baseline, mode='mean'):
    """Helper to log the rescaling method"""
    if baseline is not None:
        valid_modes = ('logratio', 'ratio', 'zscore', 'mean', 'percent',
                       'zlogratio')
        if mode not in valid_modes:
            raise Exception('mode should be any of : %s' % (valid_modes, ))
        msg = 'Applying baseline correction (mode: %s)' % mode
    else:
        msg = 'No baseline correction applied'
    logger.info(msg)


@verbose
def rescale(data, times, baseline, mode='mean', copy=True, verbose=None):
    """Rescale aka baseline correct data

    Parameters
    ----------
    data : array
        It can be of any shape. The only constraint is that the last
        dimension should be time.
    times : 1D array
        Time instants is seconds.
    baseline : tuple or list of length 2, or None
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used. If None, no correction is applied.
    mode : 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent' | 'zlogratio'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline)).
        logratio is the same an mean but in log-scale, zlogratio is the
        same as zscore but data is rendered in log-scale first.
    copy : bool
        Whether to return a new instance or modify in place.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    data_scaled: array
        Array of same shape as data after rescaling.
    """
    data = data.copy() if copy else data
    _log_rescale(baseline, mode)
    if baseline is None:
        return data

    bmin, bmax = baseline
    if bmin is None:
        imin = 0
    else:
        imin = int(np.where(times >= bmin)[0][0])
    if bmax is None:
        imax = len(times)
    else:
        imax = int(np.where(times <= bmax)[0][-1]) + 1

    # avoid potential "empty slice" warning
    if data.shape[-1] > 0:
        mean = np.mean(data[..., imin:imax], axis=-1)[..., None]
    else:
        mean = 0  # otherwise we get an ugly nan
    if mode == 'mean':
        data -= mean
    if mode == 'logratio':
        data /= mean
        data = np.log10(data)  # a value of 1 means 10 times bigger
    if mode == 'ratio':
        data /= mean
    elif mode == 'zscore':
        std = np.std(data[..., imin:imax], axis=-1)[..., None]
        data -= mean
        data /= std
    elif mode == 'percent':
        data -= mean
        data /= mean
    elif mode == 'zlogratio':
        data /= mean
        data = np.log10(data)
        std = np.std(data[..., imin:imax], axis=-1)[..., None]
        data /= std

    return data
