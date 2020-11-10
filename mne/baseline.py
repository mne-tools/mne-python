"""Utility functions to baseline-correct data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np

from .utils import logger, verbose, _check_option


def _log_rescale(baseline, mode='mean'):
    """Log the rescaling method."""
    if baseline is not None:
        _check_option('mode', mode, ['logratio', 'ratio', 'zscore', 'mean',
                                     'percent', 'zlogratio'])
        msg = 'Applying baseline correction (mode: %s)' % mode
    else:
        msg = 'No baseline correction applied'
    return msg


@verbose
def rescale(data, times, baseline, mode='mean', copy=True, picks=None,
            verbose=None):
    """Rescale (baseline correct) data.

    Parameters
    ----------
    data : array
        It can be of any shape. The only constraint is that the last
        dimension should be time.
    times : 1D array
        Time instants is seconds.
    %(rescale_baseline)s
    mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by

        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')

    copy : bool
        Whether to return a new instance or modify in place.
    picks : list of int | None
        Data to process along the axis=-2 (None, default, processes all).
    %(verbose)s

    Returns
    -------
    data_scaled: array
        Array of same shape as data after rescaling.
    """
    data = data.copy() if copy else data
    msg = _log_rescale(baseline, mode)
    logger.info(msg)
    if baseline is None or data.shape[-1] == 0:
        return data

    bmin, bmax = baseline
    if bmin is None:
        imin = 0
    else:
        imin = np.where(times >= bmin)[0]
        if len(imin) == 0:
            raise ValueError('bmin is too large (%s), it exceeds the largest '
                             'time value' % (bmin,))
        imin = int(imin[0])
    if bmax is None:
        imax = len(times)
    else:
        imax = np.where(times <= bmax)[0]
        if len(imax) == 0:
            raise ValueError('bmax is too small (%s), it is smaller than the '
                             'smallest time value' % (bmax,))
        imax = int(imax[-1]) + 1
    if imin >= imax:
        raise ValueError('Bad rescaling slice (%s:%s) from time values %s, %s'
                         % (imin, imax, bmin, bmax))

    # technically this is inefficient when `picks` is given, but assuming
    # that we generally pick most channels for rescaling, it's not so bad
    mean = np.mean(data[..., imin:imax], axis=-1, keepdims=True)

    if mode == 'mean':
        def fun(d, m):
            d -= m
    elif mode == 'ratio':
        def fun(d, m):
            d /= m
    elif mode == 'logratio':
        def fun(d, m):
            d /= m
            np.log10(d, out=d)
    elif mode == 'percent':
        def fun(d, m):
            d -= m
            d /= m
    elif mode == 'zscore':
        def fun(d, m):
            d -= m
            d /= np.std(d[..., imin:imax], axis=-1, keepdims=True)
    elif mode == 'zlogratio':
        def fun(d, m):
            d /= m
            np.log10(d, out=d)
            d /= np.std(d[..., imin:imax], axis=-1, keepdims=True)

    if picks is None:
        fun(data, mean)
    else:
        for pi in picks:
            fun(data[..., pi, :], mean[..., pi, :])
    return data
