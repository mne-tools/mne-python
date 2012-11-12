# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from warnings import warn

import numpy as np
from scipy.fftpack import fftfreq

import logging
logger = logging.getLogger('mne')

from .utils import check_indices
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


def _spli_acc(csd_xy):
    """Accumulator function for sPLI"""
    return np.sign(np.imag(csd_xy))


# TODO:
# - parallel processing of epochs
@verbose
def freq_connectivity(data, method='coh', indices=None, sfreq=2*np.pi, fmin=0,
                      fmax=np.inf, fskip=0, faverage=False, bandwidth=None,
                      adaptive=False, low_bias=True, block_size=1000,
                      verbose=None):
    """Compute various frequency-domain connectivity measures

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD_ :math:`S_{XY}(f)` and :math:`S_{XX}(f), S_{YY}(f)`,
    respectively, which are computed using a multi-taper method.

    By default, the connectivity between all signals is computed (only
    connections corresponding to the lower-triangular part of the
    connectivity matrix). If one is only interested in the connectivity
    between some signals, the "indices" parameter can be used. For example,
    to compute the connectivity between the signal with index 0 and signals
    "2, 3, 4" (a total of 3 connections) one can use the following:

    indices = (np.array([0, 0, 0],    # row indices
           np.array([2, 3, 4])))  # col indices

    con_flat = freq_connectivity(data, 'coh', indices=indices, ...)

    This is equivalent, but more efficient, to

    coh = freq_connectivity(data, 'coh', indices=None, ...)
    coh_flat = coh[indices]  # indices defined above

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

    'spli' : signed Phase Locking Index (sPLI)
        The sPLI is given by

        .. math:: PLI_{XY}(f) = E[sign(Im(S_{XY}(f)))]

    Defining Custom Connectivity Measures
    -------------------------------------
    It is possible to define custom connectivity measures by passing tuples
    with function handles to the "method" parameter. Assume we want to
    implement the PLI method (unsiged version of PLI above) ourselves.

    Frist, we define an accumulator and normalization function

    def pli_acc(csd_xy):
        # The function receives the CSD csd_xy.shape = (n_pairs, n_freq)
        return np.sign(np.imag(csd_xy))

    def pli_norm(acc, psd_xx, psd_yy, n_epochs):
        # acc is the output of pli_acc defined above averaged over epochs
        # for this measure we ignore the PSD and n_epochs parameter
        return np.abs(acc)

    Now, we define our custom PLI method which we can pass to the "method"
    parameter:

    my_pli = (pli_acc, pli_norm)

    Note: The function pli_acc receives the CSD which is an array with shape
    (n_pairs, n_freq). The first dimension of the returned array must be
    n_pairs. All other dimensions and the data type can be arbitrary. This
    makes it possible to e.g. compute phase histograms etc. For example,
    the accumulator function could return an (n_pairs, n_freq, n_bin) array.
    If no normalization is needed, the normalization function can be None.

    Parameters
    ----------
    data : array, shape=(n_epochs, n_signals, n_times) | list of SourceEstimate
        The data from which to compute coherency.
    method : (string | tuple with two function handles) or a list thereof
        Connectivity measure(s) to compute.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which to compute
        connectivity. If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    fmin : float | tuple of floats
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 20.) for two bands with 8Hz and 20Hz lower freq.
    fmax : float | tuple of floats
        The upper frequency of interest. Multiple bands are dedined using
        a tuple, e.g. (13., 30.) for two band with 13Hz and 30Hz upper freq.
    fskip : int
        Omit every "(fskip + 1)-th" frequency bin to decimate in frequency
        domain.
    faverage : boolean
        Average connectivity scores for each frequency band. If True,
        the output freqs will be a list with arrays of the frequencies
        that were averaged.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    block_size : int
        How many connections to compute at once (higher numbers are faster
        but require more memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    con : array | list of arrays
        Computed connectivity measure(s). If "indices" is None, the first
        two dimensions have shape (n_signals, n_signals) otherwise the
        first dimension is len(indices[0]). The remaining dimensions are
        method depenend.
    freqs : array
        Frequency points at which the coherency was computed.
    n_epochs : int
        Number of epochs used for computation.
    n_tapers : int
        The number of DPSS tapers used.
    """

    # format fmin and fmax and check inputs
    fmin = np.asarray((fmin,)).ravel()
    fmax = np.asarray((fmax,)).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

    n_bands = len(fmin)

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
        elif m == 'spli':
            accumulator_fun.append(_spli_acc)
            normalization_fun.append(None)
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

            if indices is None:
                # only compute r for lower-triangular region
                indices_use = np.tril_indices(n_signals, -1)
            else:
                indices_use = check_indices(indices)

            # number of connectivities to compute
            n_con = len(indices_use[0])

            logger.info('    computing connectivity for %d connections'
                        % n_con)

            # decide which frequencies to keep
            freqs = fftfreq(n_times, 1. / sfreq)
            freqs = freqs[freqs >= 0]

            freq_mask_bands = []
            for f_lower, f_upper in zip(fmin, fmax):
                this_mask = (freqs >= f_lower) & (freqs <= f_upper)
                if fskip > 0:
                    # only keep every (fskip + 1)-th frequency point
                    for pos in xrange(fskip):
                        this_mask[pos + 1::fskip + 1] = False
                freq_mask_bands.append(this_mask)

            # combined mask for all bands
            freq_mask = freq_mask_bands[0].copy()
            for mask in freq_mask_bands[1:]:
                freq_mask |= mask

            # frequencies for each band
            freqs_bands = [freqs[mask] for mask in freq_mask_bands]

            n_freqs = np.sum(freq_mask)
            if n_bands == 1:
                logger.info('    frequencies: %0.1fHz..%0.1fHz (%d points)'
                            % (freqs_bands[0][0], freqs_bands[0][-1], n_freqs))
            else:
                logger.info('    computing connectivity for the bands:')
                for i, bfreqs in enumerate(freqs_bands):
                    logger.info('    band %d: %0.1fHz..%0.1fHz (%d points)'
                                % (i + 1, bfreqs[0], bfreqs[-1], len(bfreqs)))

            # freqs output argument
            freqs_out = freqs[freq_mask]
            if faverage:
                logger.info('    connectivity scores will be averaged for '
                            'each band')
                # we will need the indices to average over for each band
                freq_idx_bands = [np.searchsorted(freqs_out, f)
                                  for f in freqs_bands]
                # for each band we return the frequencies that were averaged
                freqs_out = freqs_bands

            # unique signals for which we actually need to compute PSD etc.
            sig_idx = np.unique(np.r_[indices_use[0], indices_use[1]])

            # map indices to unique indices
            idx_map = [np.searchsorted(sig_idx, ind) for ind in indices_use]

            # allocate space to accumulate PSD and con. score over epochs
            psd = np.zeros((len(sig_idx), n_freqs))
            con_accumulators = []
            tmp_csd = np.zeros((3, n_freqs), dtype=np.complex128)
            tmp_psd = np.ones((3, n_freqs))
            for acc, norm  in zip(accumulator_fun, normalization_fun):
                out = acc(tmp_csd)
                if out.shape[0] != tmp_csd.shape[0]:
                    raise ValueError('Accumulator function must return output '
                                     'with shape[0] == csd.shape[0]')
                faverage_acc = False
                if faverage:
                    # test if averaging over frequencies works
                    if norm is not None:
                        out = norm(out, tmp_psd, tmp_psd, 3)
                    else:
                        faverage_acc = True  # we can directly average
                    if out.shape[1] != n_freqs:
                        raise ValueError('Averaging over freq. not possible '
                                         'normalization function must return '
                                         'output with shape[1] == n_freqs')
                if faverage_acc:
                    # for this method we can average over freq. for each epoch
                    this_acc = np.zeros((n_con, n_freqs) + out.shape[2:],
                                         dtype=out.dtype)
                else:
                    # no averaging or averaging at the end
                    this_acc = np.zeros((n_con,) + out.shape[1:],
                                        dtype=out.dtype)

                con_accumulators.append(this_acc)
            del tmp_csd, tmp_psd, out

        # epoch processing starts here
        if data_i.shape != (n_signals, n_times):
            raise ValueError('all epochs must have the same shape')

        logger.info('    computing connectivity for epoch %d'
                    % (epoch_idx + 1))

        # compute tapered spectra
        x_mt, _ = _mt_spectra(data_i[sig_idx], dpss, sfreq)

        if adaptive:
            # compute PSD and adaptive weights
            this_psd, weights = _psd_from_mt_adaptive(x_mt, eigvals, freq_mask,
                                                      return_weights=True)

            # only keep freqs of interest
            x_mt = x_mt[:, :, freq_mask]
        else:
            # do not use adaptive weights
            x_mt = x_mt[:, :, freq_mask]
            weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
            this_psd = _psd_from_mt(x_mt, weights)

        # accumulate PSD
        psd += this_psd

        # accumulate connectivity scores
        for i in xrange(0, n_con, block_size):
            if adaptive:
                csd = _csd_from_mt(x_mt[idx_map[0][i:i + block_size]],
                                   x_mt[idx_map[1][i:i + block_size]],
                                   weights[idx_map[0][i:i + block_size]],
                                   weights[idx_map[1][i:i + block_size]])
            else:
                csd = _csd_from_mt(x_mt[idx_map[0][i:i + block_size]],
                                   x_mt[idx_map[1][i:i + block_size]],
                                   weights, weights)

            for fun, norm, acc in zip(accumulator_fun, normalization_fun,
                                      con_accumulators):
                this_acc = fun(csd)
                if faverage and norm is None:
                    # average over each frequency band
                    for j in xrange(n_bands):
                        acc[i:i + block_size, j] +=\
                            np.mean(this_acc[:, freq_idx_bands[j]], axis=1)
                else:
                    acc[i:i + block_size] += this_acc

    # normalize
    n_epochs = epoch_idx + 1
    psd /= n_epochs
    for acc in con_accumulators:
        acc /= float(n_epochs)

    # compute final connectivity scores
    con = []
    for fun, acc in zip(normalization_fun, con_accumulators):
        if fun is None:
            # we don't need normalization
            con.append(acc)
            continue

        # detect the shape and dtype of the output
        tmp = fun(acc[:2], psd[:2], psd[:2], n_epochs)
        if faverage:
            this_con = np.empty((n_con, n_bands) + tmp.shape[2:],
                                 dtype=tmp.dtype)
        else:
            this_con = np.empty((n_con,) + tmp.shape[1:], dtype=tmp.dtype)
        del tmp

        for i in xrange(0, n_con, block_size):
            this_block = fun(acc[i:i + block_size],
                             psd[idx_map[0][i:i + block_size]],
                             psd[idx_map[1][i:i + block_size]], n_epochs)
            if faverage:
                for j in xrange(n_bands):
                    this_con[i:i + block_size, j] =\
                        np.mean(this_block[:, freq_idx_bands[j]], axis=1)
            else:
                this_con[i:i + block_size] = this_block

        con.append(this_con)

    if indices is None:
        # return all-to-all connectivity matrices
        logger.info('    assembling 3D connectivity matrix')
        con_flat = con
        con = []
        for this_con_flat in con_flat:
            this_con = np.zeros((n_signals, n_signals)
                                + this_con_flat.shape[1:],
                                dtype=this_con_flat.dtype)
            this_con[indices_use] = this_con_flat
            con.append(this_con)

    logger.info('[Connectivity computation done]')

    if n_methods == 1:
        # for a single method return connectivity directly
        con = con[0]

    return con, freqs_out, n_epochs, n_tapers
