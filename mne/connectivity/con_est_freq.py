# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from warnings import warn

import numpy as np
from scipy.fftpack import fftfreq

import logging
logger = logging.getLogger('mne')

from .utils import check_idx
from .. import SourceEstimate
from ..time_frequency.multitaper import dpss_windows, _mt_spectra,\
                                        _psd_from_mt, _csd_from_mt,\
                                        _psd_from_mt_adaptive
from .. import verbose


def _coh_acc(csd_xy):
    """Accumulator function for coherency"""
    return csd_xy


def _coh_norm(acc_mean, psd_xx, psd_yy, n_epochs):
    """Normalization function for coherency"""
    return acc_mean / np.sqrt(psd_xx * psd_yy)


def _pli_acc(csd_xy):
    """Accumulator function for PLI"""
    return np.sign(np.imag(csd_xy))


def _pli_norm(acc_mean, psd_xx, psd_yy, n_epochs):
    """Normalization function for PLI"""
    return np.abs(acc_mean)


@verbose
def freq_connectivity(data, method='coh', idx=None, sfreq=2*np.pi, fmin=0,
                      fmax=np.inf, bandwidth=None, adaptive=False,
                      low_bias=True, verbose=None):
    """Compute various frequency-domain connectivity measures

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD_ :math:`S_{XY}(f)` and :math:`S_{XX}(f), S_{YY}(f)`,
    respectively, which are computed using a multi-taper method.

    By default, the connectivity between all signals is computed (only
    connections corresponding to the lower-triangular part of the
    connectivity matrix). If one is only interested in the connectivity
    between some signals, the "idx" parameter can be used. For example,
    to compute the connectivity between the signal with index 0 and signals
    "2, 3, 4" (a total of 3 connections) one can use the following:

    idx = (np.array([0, 0, 0],    # row indices
           np.array([2, 3, 4])))  # col indices

    con_flat = freq_connectivity(data, 'coh', idx=idx, ...)

    This is equivalent, but more efficient, to

    coh = freq_connectivity(data, 'coh', idx=None, ...)
    coh_flat = coh[idx]  # idx defined above

    Supported Connectivity Measures
    -------------------------------
    The connectivity method(s) is specified using the "method" parameter. The
    following methods are supported (note: E[] denotes average over epochs).
    Multiple measures can be computed at once by using a list/tuple, e.g.
    "['coh', 'pli']" to compute coherency and PLI.

    'coh' : Coherency
        The coherency is given by

        .. math:: C_{XY}(f) = \frac{E[S_{XY}](f)}
                                   {\sqrt{E[S_{XX}]E[(f) S_{YY}(f)}]}

    'pli' : Phase Locking Index (PLI)
        The PLI is given by

        .. math:: PLI_{XY}(f) = |E[sign(Im(S_{XY}(f)))]|

    Defining Custom Connectivity Measures
    -------------------------------------
    It is possible to define custom connectivity measures by passing tuples
    with function handles to the "method" parameter. Assume we want to
    re-implement the PLI method (see above) ourselves.

    Frist, we define an accumulator and normalization function

    def pli_acc(csd_xy):
        # The function receives the CSD for one signal pair
        return np.sign(np.imag(csd_xy))

    def pli_norm(acc, psd_xx, psd_yy, n_epochs):
        # acc is the output of pli_acc defined above averaged over epochs
        # for this measure we ignore the PSD and n_epochs parameter
        return np.abs(acc)

    Now, we define our custom PLI method which we can pass to the "method"
    parameter:

    my_pli = (pli_acc, pli_norm)

    Note: The function pli_acc receives the CSD which is an array of length
    n_freq. The returned array can have an arbitrary shape and data
    type, which makes it possible to e.g. compute phase histograms etc.
    (in this case the accumulator function could return a (n_freq, n_bin)
    array).

    Parameters
    ----------
    data : array, shape=(n_epochs, n_signals, n_times) | list of SourceEstimate
        The data from which to compute coherency.
    method : (string | tuple with two function handles) or a list thereof
        Connectivity measure(s) to compute.
    idx : tuple of arrays | None
        Two arrays with indices of connections for which to compute
        connectivity. If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    con : array | list of arrays
        Computed connectivity measure(s). If "idx" is None, the first
        two dimensions have shape (n_signals, n_signals) otherwise the
        first dimension is len(idx[0]). The remaining dimensions are
        method depenend.
    freqs : array
        Frequency points at which the coherency was computed.
    n_epochs : int
        Number of epochs used for computation.
    n_tapers : int
        The number of DPSS tapers used.
    """

    # assign functions to various methods
    if not isinstance(method, (list, tuple)):
        method = [method]

    n_methods = len(method)

    accumulator_fun = []
    normalization_fun = []
    for m in method:
        if m == 'coh':
            accumulator_fun.append(_coh_acc)
            normalization_fun.append(_coh_norm)
        elif m == 'pli':
            accumulator_fun.append(_pli_acc)
            normalization_fun.append(_pli_norm)
        elif isinstance(m, (tuple, list)):
            if len(m) != 2 or not all([callable(fun) for fun in m]):
                raise ValueError('custom method must be defined using a '
                                 'list/tuple with two function handles')
            accumulator_fun.append(m[0])
            normalization_fun.append(m[1])
        else:
            raise ValueError('invalid value for method')

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays or SourceEstimates

    logger.info('Connectivity computation...')
    for epoch_idx, data_i in enumerate(data):

        if isinstance(data_i, SourceEstimate):
            # allow data to be a list of source estimates
            data_i = data_i.data

        if epoch_idx == 0:
            # initialize things
            n_signals, n_times = data_i.shape

            # compute standardized half-bandwidth
            if bandwidth is not None:
                half_nbw = float(bandwidth) * n_times / (2 * sfreq)
            else:
                half_nbw = 4

            # compute dpss windows
            n_tapers_max = int(2 * half_nbw)
            dpss, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                         low_bias=low_bias)
            n_tapers = len(eigvals)
            logger.info('    using %d DPSS windows' % n_tapers)

            if adaptive and len(eigvals) < 3:
                warn('Not adaptively combining the spectral estimators '
                     'due to a low number of tapers.')
                adaptive = False

            if idx is None:
                # only compute r for lower-triangular region
                idx_use = np.tril_indices(n_signals, -1)
            else:
                idx_use = check_idx(idx)

            # number of connectivities to compute
            n_con = len(idx_use[0])

            logger.info('    computing connectivity for %d connections'
                        % n_con)

            # decide which frequencies to keep
            freqs = fftfreq(n_times, 1. / sfreq)
            freqs = freqs[freqs >= 0]
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            freqs = freqs[freq_mask]
            n_freqs = np.sum(freq_mask)
            logger.info('    frequencies: %0.1fHz..%0.1fHz (%d points)'
                        % (freqs[0], freqs[-1], n_freqs))

            # unique signals for which we actually need to compute PSD etc.
            sig_idx = np.unique(np.r_[idx_use[0], idx_use[1]])

            # map idx to unique indices
            idx_map = [np.searchsorted(sig_idx, ind) for ind in idx_use]

            # allocate space to accumulate PSD and con. score over epochs
            psd = np.zeros((len(sig_idx), n_freqs))
            con_accumulators = []
            tmp = np.zeros(n_freqs, dtype=np.complex128)
            for fun in accumulator_fun:
                out = fun(tmp)
                con_accumulators.append(np.zeros((n_con,) + out.shape,
                                        dtype=out.dtype))
            del tmp, out

        # epoch processing starts here
        if data_i.shape != (n_signals, n_times):
            raise ValueError('all epochs must have the same shape')

        logger.info('    computing connectivity for epoch %d'
                    % (epoch_idx + 1))

        # compute tapered spectra
        x_mt, _ = _mt_spectra(data_i[sig_idx], dpss, sfreq)

        if not adaptive:
            x_mt = x_mt[:, :, freq_mask]
            weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
            this_psd = _psd_from_mt(x_mt, weights)

            # accumulate PSD
            psd += this_psd

            # accumulate connectivity scores
            for i in xrange(n_con):
                csd = _csd_from_mt(x_mt[idx_map[0][i]], x_mt[idx_map[1][i]],
                                   weights, weights)[0]

                for fun, acc in zip(accumulator_fun, con_accumulators):
                    acc[i] += fun(csd)

        else:
            # compute PSD and adaptive weights
            this_psd, weights = _psd_from_mt_adaptive(x_mt, eigvals, freq_mask,
                                                      return_weights=True)

            # accumulate PSD
            psd += this_psd

            # only keep freqs of interest
            x_mt = x_mt[:, :, freq_mask]

            # compute CSD
            for i in xrange(n_con):
                csd = _csd_from_mt(x_mt[idx_map[0][i]], x_mt[idx_map[1][i]],
                                   weights[idx_map[0][i]],
                                   weights[idx_map[1][i]])

                for fun, acc in zip(accumulator_fun, con_accumulators):
                    acc[i] += fun(csd)

    # normalize
    n_epochs = epoch_idx + 1
    psd /= n_epochs
    for acc in con_accumulators:
        acc /= float(n_epochs)

    # compute final connectivity scores
    con = []
    for fun, acc in zip(normalization_fun, con_accumulators):
        # detect the shape and dtype of the output
        tmp = fun(acc[0], psd[0], psd[0], n_epochs)
        this_con = np.empty((n_con,) + tmp.shape, dtype=tmp.dtype)
        del tmp
        for i in xrange(n_con):
            this_con[i] = fun(acc[i], psd[idx_map[0][i]], psd[idx_map[1][i]],
                              n_epochs)

        con.append(this_con)

    if idx is None:
        # return all-to-all connectivity matrices
        logger.info('    assembling 3D connectivity matrix')
        con_flat = con
        con = []
        for this_con_flat in con_flat:
            this_con = np.zeros((n_signals, n_signals)
                                + this_con_flat.shape[1:],
                                dtype=this_con_flat.dtype)
            this_con[idx_use] = this_con_flat
            con.append(this_con)

    logger.info('[Connectivity computation done]')

    if n_methods == 1:
        # for a single method return connectivity directly
        con = con[0]

    return con, freqs, n_epochs, n_tapers
