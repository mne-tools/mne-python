"""Util function to baseline correct data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

from .utils import logger, verbose


def _log_rescale(baseline, mode='mean'):
    """Log the rescaling method."""
    if baseline is not None:
        valid_modes = ('logratio', 'ratio', 'zscore', 'mean', 'percent',
                       'zlogratio')
        if mode not in valid_modes:
            raise Exception('mode should be any of : %s' % (valid_modes, ))
        msg = 'Applying baseline correction (mode: %s)' % mode
    else:
        msg = 'No baseline correction applied'
    return msg


@verbose
def rescale(data, times, baseline, mode='mean', copy=True, verbose=None):
    """Rescale (baseline correct) data.

    Parameters
    ----------
    data : array
        It can be of any shape. The only constraint is that the last
        dimension should be time.
    times : 1D array
        Time instants is seconds.
    baseline : tuple or list of length 2, or None
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is ``(bmin, bmax)``
        the interval is between ``bmin`` (s) and ``bmax`` (s).
        If ``bmin is None`` the beginning of the data is used
        and if ``bmax is None`` then ``bmax`` is set to the end of the
        interval. If baseline is ``(None, None)`` the entire time
        interval is used. If baseline is None, no correction is applied.
    mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio' | None
        Perform baseline correction by

          - subtracting the mean baseline power ('mean')
          - dividing by the mean baseline power ('ratio')
          - dividing by the mean baseline power and taking the log ('logratio')
          - subtracting the mean baseline power followed by dividing by the
            mean baseline power ('percent')
          - subtracting the mean baseline power and dividing by the standard
            deviation of the baseline power ('zscore')
          - dividing by the mean baseline power, taking the log, and dividing
            by the standard deviation of the baseline power ('zlogratio')

        If None no baseline correction is applied.
    copy : bool
        Whether to return a new instance or modify in place.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    data_scaled: array
        Array of same shape as data after rescaling.
    """  # noqa: E501
    data = data.copy() if copy else data
    msg = _log_rescale(baseline, mode)
    logger.info(msg)
    if baseline is None:
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

    # avoid potential "empty slice" warning
    if data.shape[-1] > 0:
        mean = np.mean(data[..., imin:imax], axis=-1)[..., None]
    else:
        mean = 0  # otherwise we get an ugly nan
    if mode == 'mean':
        data -= mean
    elif mode == 'ratio':
        data /= mean
    elif mode == 'logratio':
        data /= mean
        data = np.log10(data)
    elif mode == 'percent':
        data -= mean
        data /= mean
    elif mode == 'zscore':
        std = np.std(data[..., imin:imax], axis=-1)[..., None]
        data -= mean
        data /= std
    elif mode == 'zlogratio':
        data /= mean
        data = np.log10(data)
        std = np.std(data[..., imin:imax], axis=-1)[..., None]
        data /= std

    return data
