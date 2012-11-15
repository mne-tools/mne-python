# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from warnings import warn

import numpy as np
from scipy.fftpack import fftfreq

import logging
logger = logging.getLogger('mne')


from .utils import check_indices
from ..parallel import parallel_func
from .. import Epochs, SourceEstimate
from ..time_frequency.multitaper import dpss_windows, _mt_spectra,\
                                        _psd_from_mt, _csd_from_mt,\
                                        _psd_from_mt_adaptive
from .. import verbose


########################################################################
# Accumulator and normalization functions for various methods

def _coh_acc(csd_xy):
    """Accumulator function for coherence, coherency etc"""
    return csd_xy


def _coh_norm(acc_mean, psd_xx, psd_yy, n_epochs):
    """Normalization function for coherence"""
    return np.abs(acc_mean) / np.sqrt(psd_xx * psd_yy)


def _cohy_norm(acc_mean, psd_xx, psd_yy, n_epochs):
    """Normalization function for coherency"""
    return acc_mean / np.sqrt(psd_xx * psd_yy)


def _imcoh_norm(acc_mean, psd_xx, psd_yy, n_epochs):
    """Normalization function for imaginary coherence"""
    return np.imag(acc_mean) / np.sqrt(psd_xx * psd_yy)


def _pli_acc(csd_xy):
    """Accumulator function for PLI"""
    return np.sign(np.imag(csd_xy))


def _pli_norm(acc_mean, psd_xx, psd_yy, n_epochs):
    """Normalization function for PLI"""
    return np.abs(acc_mean)

########################################################################


def _epoch_spectral_connectivity(data, sfreq, dpss, eigvals, freq_mask,
                                 adaptive, faverage, freq_idx_bands,
                                 idx_map, block_size, accumulator_fun,
                                 normalization_fun, con_accumulators,
                                 psd, con_acc_info=None,
                                 accumulate_inplace=True):
    """Connectivity estimation for one epoch see spectral_connectivity"""
    if not accumulate_inplace:
        # the con_acc_info are tuples of shape and dtype, allocate space
        acc_shapes = [acc[0] for acc in con_acc_info]
        acc_dtypes = [acc[1] for acc in con_acc_info]
        con_accumulators = [np.zeros(shape, dtype=dtype)
                            for shape, dtype in zip(acc_shapes, acc_dtypes)]

    n_con = con_accumulators[0].shape[0]
    x_mt, _ = _mt_spectra(data, dpss, sfreq)

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

    # accumulate or retrun psd
    if accumulate_inplace:
        psd += this_psd
    else:
        psd = this_psd

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
                for j, freq_idx in enumerate(freq_idx_bands):
                    acc[i:i + block_size, j] +=\
                        np.mean(this_acc[:, freq_idx], axis=1)
            else:
                acc[i:i + block_size] += this_acc

    return con_accumulators, psd


def _get_n_epochs(epochs, n):
    """Generator that returns lists with at most n epochs"""
    epochs_out = []
    for e in epochs:
        epochs_out.append(e)
        if len(epochs_out) >= n:
            yield epochs_out
            epochs_out = []
    yield epochs_out


def _check_method(method):
    """Check user defined method"""
    if not isinstance(method, (list, tuple)):
        return False
    if not len(method) == 2:
        return False
    if not callable(method[0]):
        return False
    if method[1] is not None and not callable(method[1]):
        return False
    return True


@verbose
def spectral_connectivity(data, method='coh', indices=None, sfreq=2 * np.pi,
                          fmin=0, fmax=np.inf, fskip=0, faverage=False,
                          tmin=None, tmax=None, bandwidth=None, adaptive=False,
                          low_bias=True, block_size=1000, n_jobs=1,
                          verbose=None):
    """Compute various frequency-domain connectivity measures

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD) Sxy(f) and Sxx(f), Syy(f), respectively,
    which are computed using a multi-taper method.

    By default, the connectivity between all signals is computed (only
    connections corresponding to the lower-triangular part of the
    connectivity matrix). If one is only interested in the connectivity
    between some signals, the "indices" parameter can be used. For example,
    to compute the connectivity between the signal with index 0 and signals
    "2, 3, 4" (a total of 3 connections) one can use the following:

    indices = (np.array([0, 0, 0],    # row indices
               np.array([2, 3, 4])))  # col indices

    con_flat = spectral_connectivity(data, method='coh', indices=indices, ...)

    In this case con_flat.shape = (3, n_freqs). The connectivity scores are
    in the same order as defined indices.

    Supported Connectivity Measures
    -------------------------------
    The connectivity method(s) is specified using the "method" parameter. The
    following methods are supported (note: E[] denotes average over epochs).
    Multiple measures can be computed at once by using a list/tuple, e.g.
    "['coh', 'pli']" to compute coherence and PLI.

    'coh' : Coherence given by

                     | E[Sxy(f)] |
        C(f) = ---------------------------
               sqrt(E[Sxx(f)] * E[Syy(f)])

    'cohy' : Coherency given by

                       E[Sxy(f)]
        C(f) = ---------------------------
               sqrt(E[Sxx(f)] * E[Syy(f)])

    'imcoh' : Imaginary coherence given by

                      Im(E[Sxy(f)])
        C(f) = --------------------------
               sqrt(E[Sxx(f)] * E[Syy(f)])


    'pli' : Phase Locking Index (PLI) given by

        PLI(f) = |E[sign(Im(Sxy(f)))]|

    Defining Custom Connectivity Measures
    -------------------------------------
    It is possible to define custom connectivity measures by passing tuples
    with two functions to the "method" parameter. Specifically, any measure
    of the form

    Con = f_norm(E[f_acc(Sxy(f))], E[Sxx(f)], E[Syy(f)], n_epochs)

    can be implemented by defining the accumulator and normalization functions
    f_acc and f_norm, respectively. For example, if we want to re-implement
    the PLI method ourselves we we define an accumulator and normalization
    functions as follows

    def pli_acc(csd_xy):
        # The function receives the CSD csd_xy.shape = (n_pairs, n_freq)
        return np.sign(np.imag(csd_xy))

    def pli_norm(acc_mean, psd_xx, psd_yy, n_epochs):
        # acc_norm is the output of pli_acc defined above averaged over epochs
        # for this measure we ignore the PSD and n_epochs parameters
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
    data : array, shape=(n_epochs, n_signals, n_times)
           or list/generator of SourceEstimate
           or Epochs
        The data from which to compute connectivity.
    method : (string | tuple with two functions) or a list thereof
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
    tmin : float | None
        Time to start connectivity estimation. Only supported if data is
        Epochs or a list of SourceEstimate
    tmax : float | None
        Time to end connectivity estimation. Only supported if data is
        Epochs or a list of SourceEstimate.
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
    n_jobs : int
        How many epochs to process in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    con : array | list of arrays
        Computed connectivity measure(s). If "indices" is None, the first
        two dimensions have shape (n_signals, n_signals) otherwise the
        first dimension is len(indices[0]). The remaining dimensions are
        method dependent.
    freqs : array
        Frequency points at which the coherency was computed.
    n_epochs : int
        Number of epochs used for computation.
    n_tapers : int
        The number of DPSS tapers used.
    """

    if n_jobs > 1:
        parallel, my_epoch_spectral_connectivity, _ = \
                parallel_func(_epoch_spectral_connectivity, n_jobs,
                              verbose=verbose)

    # format fmin and fmax and check inputs
    fmin = np.asarray((fmin,)).ravel()
    fmax = np.asarray((fmax,)).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

    n_bands = len(fmin)

    # assign functions to various methods
    if isinstance(method, (list, tuple)):
        # check if the user has defined a single custom method
        if _check_method(method):
            method = [method]
    else:
        # make it a tuple
        method = [method]

    n_methods = len(method)

    accumulator_fun = []
    normalization_fun = []
    for m in method:
        if m == 'coh':
            accumulator_fun.append(_coh_acc)
            normalization_fun.append(_coh_norm)
        elif m == 'cohy':
            accumulator_fun.append(_coh_acc)
            normalization_fun.append(_cohy_norm)
        elif m == 'imcoh':
            accumulator_fun.append(_coh_acc)
            normalization_fun.append(_imcoh_norm)
        elif m == 'pli':
            accumulator_fun.append(_pli_acc)
            normalization_fun.append(_pli_norm)
        elif isinstance(m, (tuple, list)):
            if not _check_method(m):
                raise ValueError('custom method must be defined using a '
                                 'list/tuple with two functions')
            accumulator_fun.append(m[0])
            normalization_fun.append(m[1])
        else:
            raise ValueError('invalid value for method')

    # by default we assume time starts at zero
    tmintmax_support = False
    tmin_idx = None
    tmax_idx = None
    tmin_true = None
    tmax_true = None

    if isinstance(data, Epochs):
        tmin_true = data.times[0]
        tmax_true = data.times[-1]
        if tmin is not None:
            tmin_idx = np.argmin(np.abs(data.times - tmin))
            tmin_true = data.times[tmin_idx]
        if tmax is not None:
            tmax_idx = np.argmin(np.abs(data.times - tmax))
            tmax_true = data.times[tmax_idx]
        tmintmax_support = True

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays or SourceEstimates
    epoch_idx = 0
    logger.info('Connectivity computation...')
    for epoch_block in _get_n_epochs(data, n_jobs):

        if epoch_idx == 0:
            first_epoch = epoch_block[0]

            if isinstance(first_epoch, SourceEstimate):
                tmin_true = first_epoch.times[0]
                tmax_true = first_epoch.times[-1]
                if tmin is not None:
                    tmin_idx = np.argmin(np.abs(first_epoch.times - tmin))
                    tmin_true = first_epoch.times[tmin_idx]
                if tmax is not None:
                    tmax_idx = np.argmin(np.abs(first_epoch.times - tmax))
                    tmax_true = first_epoch.times[tmax_idx]
                tmintmax_support = True
                first_epoch = first_epoch.data

            if not tmintmax_support and (tmin is not None or tmax is not None):
                raise ValueError('tmin and tmax are only supported if data is '
                                 'Epochs or a list of SourceEstimate')

            # we want to include the sample at tmax_idx
            tmax_idx = tmax_idx + 1 if tmax_idx is not None else None
            n_signals, n_times = first_epoch[:, tmin_idx:tmax_idx].shape

            # if we are not using Epochs or SourceEstimate, we assume time
            # starts at zero
            if tmin_true is None:
                tmin_true = 0.
            if tmax_true is None:
                tmax_true = n_times / float(sfreq)

            logger.info('    using t=%0.3fs..%0.3fs for estimation (%d points)'
                        % (tmin_true, tmax_true, n_times))

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
            freqs_all = fftfreq(n_times, 1. / sfreq)
            freqs_all = freqs_all[freqs_all >= 0]

            # create a frequency mask for all bands
            freq_mask = np.zeros(len(freqs_all), dtype=np.bool)
            for f_lower, f_upper in zip(fmin, fmax):
                freq_mask |= (freqs_all >= f_lower) & (freqs_all <= f_upper)

            # possibly skip frequency points
            for pos in xrange(fskip):
                freq_mask[pos + 1::fskip + 1] = False

            # the frequency points where we compute connectivity
            freqs = freqs_all[freq_mask]

            # get the freq. indices and points for each band
            freq_idx_bands = [np.where((freqs >= fl) & (freqs <= fu))[0]
                              for fl, fu in zip(fmin, fmax)]
            freqs_bands = [freqs[freq_idx] for freq_idx in freq_idx_bands]

            n_freqs = np.sum(freq_mask)
            if n_bands == 1:
                logger.info('    frequencies: %0.1fHz..%0.1fHz (%d points)'
                            % (freqs_bands[0][0], freqs_bands[0][-1], n_freqs))
            else:
                logger.info('    computing connectivity for the bands:')
                for i, bfreqs in enumerate(freqs_bands):
                    logger.info('     band %d: %0.1fHz..%0.1fHz (%d points)'
                                % (i + 1, bfreqs[0], bfreqs[-1], len(bfreqs)))

            if faverage:
                logger.info('    connectivity scores will be averaged for '
                            'each band')

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
                        out2 = norm(out, tmp_psd, tmp_psd, 3)
                    else:
                        out2 = out
                        faverage_acc = True  # we can directly average
                    if out2.ndim < 2 or out2.shape[1] != n_freqs:
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

            if n_jobs > 1:
                # we will need the info about the accumulators
                con_acc_info = [(acc.shape, acc.dtype)
                                for acc in con_accumulators]

        for i, this_epoch in enumerate(epoch_block):
            if isinstance(this_epoch, SourceEstimate):
                # allow data to be a list of source estimates
                epoch_block[i] = this_epoch.data[:, tmin_idx:tmax_idx]
            else:
                epoch_block[i] = this_epoch[:, tmin_idx:tmax_idx]

        # check dimensions
        for this_epoch in epoch_block:
            if this_epoch.shape != (n_signals, n_times):
                raise ValueError('all epochs must have the same shape')

        if n_jobs == 1:
            # no parallel processing
            for this_epoch in epoch_block:
                if this_epoch.shape != (n_signals, n_times):
                    raise ValueError('all epochs must have the same shape')

                logger.info('    computing connectivity for epoch %d'
                            % (epoch_idx + 1))

                # con_accumulators and psd are updated inplace
                _epoch_spectral_connectivity(this_epoch[sig_idx], sfreq, dpss,
                    eigvals, freq_mask, adaptive, faverage, freq_idx_bands,
                    idx_map, block_size, accumulator_fun, normalization_fun,
                    con_accumulators, psd)
                epoch_idx += 1
        else:
            # process epochs in parallel
            logger.info('    computing connectivity for epochs %d..%d'
                        % (epoch_idx + 1, epoch_idx + len(epoch_block)))

            out = parallel(my_epoch_spectral_connectivity(epoch[sig_idx],
                    sfreq, dpss, eigvals, freq_mask, adaptive, faverage,
                    freq_idx_bands, idx_map, block_size, accumulator_fun,
                    normalization_fun, None, None, con_acc_info=con_acc_info,
                    accumulate_inplace=False) for epoch in epoch_block)

            # do the accumulation
            for this_out in out:
                for acc, this_acc in zip(con_accumulators, this_out[0]):
                    acc += this_acc
                psd += this_out[1]

            epoch_idx += len(epoch_block)

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

    if faverage:
        # for each band we return the frequencies that were averaged
        freqs = freqs_bands

    return con, freqs, n_epochs, n_tapers
