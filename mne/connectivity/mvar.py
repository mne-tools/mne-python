# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import numpy as np
from ..parallel import parallel_func
from ..utils import logger
from scot.var import VARBase
from scot.builtin.var import VAR
from scot.connectivity import connectivity


def acm(x, l):
    """Calculates autocorrelation matrix of x at lag l.
    """
    if l == 0:
        a, b = x, x
    else:
        a = x[:, l:]
        b = x[:, 0:-l]

    return np.dot(a[:, :], b[:, :].T) / a.shape[1]


def _epoch_autocorrelations(epoch, max_lag):
    return [acm(epoch, l) for l in range(max_lag + 1)]


def _get_n_epochs(epochs, n):
    """Generator that returns lists with at most n epochs"""
    epochs_out = []
    for e in epochs:
        epochs_out.append(e)
        if len(epochs_out) >= n:
            yield epochs_out
            epochs_out = []
    yield epochs_out


def _fit_mvar_lsq(data, order, delta):
    var = VAR(order, delta)
    #todo: only convert if data is a generator
    data = np.asarray(list(data)).transpose([2, 1, 0])
    var.fit(data)
    return var


def _fit_mvar_yw(data, order, n_jobs=1, verbose=None):

    parallel, my_epoch_autocorrelations, _ = \
                parallel_func(_epoch_autocorrelations, n_jobs,
                              verbose=verbose)
    n_epochs = 0
    logger.info('Accumulating autocovariance matrices...')
    for epoch_block in _get_n_epochs(data, n_jobs):
        out = parallel(my_epoch_autocorrelations(epoch, order)
                       for epoch in epoch_block)
        if n_epochs == 0:
            acm_estimates = np.sum(out, 0)
        else:
            acm_estimates += np.sum(out, 0)
        n_epochs += len(epoch_block)
    acm_estimates /= n_epochs

    var = VARBase(order)
    var.from_yw(acm_estimates)

    return var


def mvar_connectivity(data, method, order, fitting_mode='yw', sfreq=2, fmin=0,
                      fmax=np.inf, nfft=512):
    """Estimate connectivity from multivariate autoregressive (MVAR) models.

    Parameters
    ----------
    data : array, shape=(n_epochs, n_signals, n_times)
           or list/generator of array, shape =(n_signals, n_times)
        The data from which to compute connectivity.
    method : string | list of string
        Connectivity measure(s) to compute.
    order : int
        order (length) of the underlying MVAR model
    fitting_mode : str
        Determines how to fit the MVAR model.
        'yw' : Solve Yule-Walker equations
        'lsq' : Least-Squares fitting
    sfreq : float
        The sampling frequency.
    fmin : float | tuple of floats
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 16.) for two bands with 8Hz and 16Hz lower freq.
    fmax : float | tuple of floats
        The upper frequency of interest. Multiple bands are defined using
        a tuple, e.g. (12., 24.) for two band with 12Hz and 24Hz upper freq.
    """

    if not isinstance(method, (list, tuple)):
        method = [method]

    fmin = np.asarray((fmin,)).ravel()
    fmax = np.asarray((fmax,)).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

    logger.info('MVAR fitting...')
    if fitting_mode == 'yw':
        var = _fit_mvar_yw(data, order)
    elif fitting_mode == 'lsq':
        var = _fit_mvar_lsq(data, order, 0)

    freqs, fmask = [], []
    freq_range = np.linspace(0, sfreq/2, nfft)
    for fl, fh in zip(fmin, fmax):
        fmask.append(np.logical_and(fl <= freq_range, freq_range <= fh))
        freqs.append(freq_range[fmask[-1]])

    logger.info('Connectivity computation...')
    results = []
    c = connectivity(method, var.coef, var.rescov, nfft)
    for mth in method:
        bands = [np.mean(c[mth][:, :, fm], axis=2) for fm in fmask]
        bands = np.abs(bands).transpose((1, 2, 0))
        results.append(bands)

    return results, freqs