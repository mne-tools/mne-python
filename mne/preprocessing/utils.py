# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import numpy as np

from ..utils import logger


# adapted from SciPy

def check_cola(win, nperseg, noverlap, tol=1e-10):
    """Check whether the Constant OverLap Add (COLA) constraint is met."""
    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    noverlap = int(noverlap)
    step = nperseg - noverlap
    binsums = np.sum([win[ii * step:(ii + 1) * step]
                      for ii in range(nperseg // step)], axis=0)
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]
    deviation = binsums - np.median(binsums)
    return np.max(np.abs(deviation)) < tol


def _get_lims_cola(n_samp, n_times, sfreq):
    from scipy.signal import get_window
    if n_samp > n_times:
        raise ValueError('Effective duration (%s) must be at most the '
                         'duration of the raw instance (%s)'
                         % (n_samp / sfreq, n_times / sfreq))
    # Eventually this could be configurable
    window = 'hann'
    step = n_samp // 2
    win = get_window(window, n_samp)
    n_overlap = n_samp - step
    if not check_cola(win, n_samp, n_overlap, tol=1e-2):
        raise RuntimeError('COLA not met')
    starts = np.arange(0, n_times - n_samp + 1, step)
    stops = starts + n_samp
    delta = n_times - stops[-1]
    stops[-1] = n_times
    pl = 's' if len(starts) != 1 else ''
    logger.info('    Processing %s data chunk%s of (at least) %0.1f sec with '
                '%s windowing'
                % (len(starts), pl, n_samp / sfreq, window))
    if delta > 0:
        logger.info('    The final %0.3f sec will be lumped into the final '
                    'window' % (delta / sfreq,))
    windows = list(np.tile(win[np.newaxis], (len(starts), 1)))
    # First and last windows are special, fix them
    windows[0][:n_overlap] = 1.
    windows[-1] = np.concatenate([windows[-1], np.ones(delta)])
    windows[-1][n_overlap:] = 1.
    return starts, stops, windows
