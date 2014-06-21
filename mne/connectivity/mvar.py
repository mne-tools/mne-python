# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import numpy as np

from ..parallel import parallel_func
from ..utils import logger
from ..externals.scot.varbase import VARBase
from ..externals.scot.var import VAR
from ..externals.scot.connectivity import connectivity
from ..externals.scot.connectivity_statistics import surrogate_connectivity
from ..externals.scot.xvschema import make_nfold


def _acm(x, l):
    """Calculates autocorrelation matrix of x at lag l.
    """
    if l == 0:
        a, b = x, x
    else:
        a = x[:, l:]
        b = x[:, 0:-l]

    return np.dot(a[:, :], b[:, :].T) / a.shape[1]


def _epoch_autocorrelations(epoch, max_lag):
    return [_acm(epoch, l) for l in range(max_lag + 1)]


def _get_n_epochs(epochs, n):
    """Generator that returns lists with at most n epochs"""
    epochs_out = []
    for e in epochs:
        epochs_out.append(e)
        if len(epochs_out) >= n:
            yield epochs_out
            epochs_out = []
    yield epochs_out


def _fit_mvar_lsq(data, pmin, pmax, delta):
    var = VAR(pmin, delta, xvschema=make_nfold(10))
    if pmin != pmax:
        logger.info('MVAR order selection...')
        #todo: joblib!
        var.optimize_order(data, pmin, pmax)
    #todo: only convert if data is a generator
    data = np.asarray(list(data)).transpose([2, 1, 0])
    var.fit(data)
    return var


def _fit_mvar_yw(data, pmin, pmax, n_jobs=1, verbose=None):
    if pmin != pmax:
        raise NotImplementedError('Yule-Walker fitting does not support '
                                  'automatic model order selection.')
    order = pmin

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


def mvar_connectivity(data, method, order=(1, None), fitting_mode='lsq',
                      ridge=0, sfreq=2 * np.pi, fmin=0, fmax=np.inf, nfft=512,
                      n_surrogates=None):
    """Estimate connectivity from multivariate autoregressive (MVAR) models.

    This function uses routines from SCoT [1] to fit MVAR models and compute
    connectivity measures.

    Parameters
    ----------
    data : array, shape=(n_epochs, n_signals, n_times)
           or list/generator of array, shape =(n_signals, n_times)
        The data from which to compute connectivity.
    method : string | list of string
        Connectivity measure(s) to compute. Supported measures:
        'COH' : coherence
        'pCOH' : partial coherence
        'PDC' : partial directed coherence
        'PDCF' : partial directed coherence factor
        'GPDC' : generalized partial directed coherence
        'DTF' : directed transfer function
        'ffDTF' : full-frequency directed transfer function
        'dDTF' : "direct" directed transfer function
        'GDTF' : generalized directed transfer function
    order : int | (int, int)
        order (length) of the underlying MVAR model. If order is a tuple
        (p0, p1) of two ints, the function selects the best model order between
        p0 and p1. p1 can be None, which causes the order selection to stop at
        the lowest candidate.
    fitting_mode : str
        Determines how to fit the MVAR model.
        'lsq' : Least-Squares fitting
        'yw' : Solve Yule-Walker equations
        Yule-Walker equations can utilize data generators, which makes them
        more memory efficient than least-squares. However, yw-estimation may
        fail if `order` or `n_signals` is too high for the amount of data
        available.
    ridge : float
        Ridge-regression coefficient (l2 penalty) for least-squares fitting.
        This parameter is ignored for Yule-Walker fitting.
    sfreq : float
        The sampling frequency.
    fmin : float | tuple of floats
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 16.) for two bands with 8Hz and 16Hz lower freq.
    fmax : float | tuple of floats
        The upper frequency of interest. Multiple bands are defined using
        a tuple, e.g. (12., 24.) for two band with 12Hz and 24Hz upper freq.
    n_surrogates : int | None
        If set to None, no statistics are calculated. Otherwise, `surrogates`
        is the number of surrogate datasets on which the chance level is
        calculated. In this case the *p*-values are returned, which are related
        to the probability that the observed connectivity is not caused by
        chance. See scot.connectivity_statistics.surrogate_connectivity for
        details on the procedure.
        **Warning**: Correction for multiple testing is required if the
        *p*-values are used as basis for significance testing.

    Returns
    -------
    con : array | list of arrays
        Computed connectivity measure(s). The shape of each array is
        (n_signals, n_signals, n_frequencies)
    freqs : array
        Frequency points at which the connectivity was computed.
    var_order : int
        MVAR model order that was used for fitting the model.
    p_values : array | list of arrays | None
        *p*-values of connectivity measure(s). The shape of each array is
        (n_signals, n_signals, n_frequencies). `p_values` is returned as  None
        if no statistics are calculated (i.e. `n_surrogates` evaluates to
        False).

    References
    ----------
    [1] M. Billinger, C.Brunner, G. R. Mueller-Putz. "SCoT: a Python toolbox
        for EEG source connectivity", Frontiers in Neuroinformatics, 8:22, 2014
    """

    if not isinstance(method, (list, tuple)):
        method = [method]

    fmin = np.asarray((fmin,)).ravel()
    fmax = np.asarray((fmax,)).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

    try:
        pmin, pmax = order[0], order[1]
    except TypeError:
        pmin, pmax = order, order

    logger.info('MVAR fitting...')
    if fitting_mode == 'yw':
        var = _fit_mvar_yw(data, pmin, pmax)
    elif fitting_mode == 'lsq':
        var = _fit_mvar_lsq(data, pmin, pmax, ridge)
    else:
        raise ValueError('Unknown fitting mode: %s' % fitting_mode)

    freqs, fmask = [], []
    freq_range = np.linspace(0, sfreq/2, nfft)
    for fl, fh in zip(fmin, fmax):
        fmask.append(np.logical_and(fl <= freq_range, freq_range <= fh))
        freqs.append(freq_range[fmask[-1]])

    logger.info('Connectivity computation...')
    results = []
    con = connectivity(method, var.coef, var.rescov, nfft)
    for mth in method:
        bands = [np.mean(np.abs(con[mth][:, :, fm]), axis=2) for fm in fmask]
        results.append(np.transpose(bands, (1, 2, 0)))

    if n_surrogates:
        logger.info('Computing connectivity statistics...')
        data = np.asarray(list(data)).transpose([2, 1, 0])

        blocksize = 10
        n_blocks = n_surrogates // blocksize

        p_vals = []
        for i in range(n_blocks):
            #todo: joblib!
            scon = surrogate_connectivity(method, data, var, nfft, blocksize)

            for m, mth in enumerate(method):
                c, sc = np.abs(con[mth]), np.abs(scon[mth])
                bands = [np.mean(c[:, :, fm], axis=-1) for fm in fmask]
                sbands = [np.mean(sc[:, :, :, fm], axis=-1) for fm in fmask]

                p = [np.sum(bs >= b, axis=0) for b, bs in zip(bands, sbands)]
                p = np.array(p).transpose(1, 2, 0) / (n_blocks * blocksize)
                if i == 0:
                    p_vals.append(p)
                else:
                    p_vals[m] += p
    else:
        p_vals = None

    return results, freqs, var.p, p_vals

