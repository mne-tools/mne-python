# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from functools import partial
from inspect import getmembers

import numpy as np
from scipy.fftpack import fftfreq

from .utils import check_indices
from ..fixes import _get_args
from ..parallel import parallel_func
from ..source_estimate import _BaseSourceEstimate
from ..epochs import BaseEpochs
from ..time_frequency.multitaper import (dpss_windows, _mt_spectra,
                                         _psd_from_mt, _csd_from_mt,
                                         _psd_from_mt_adaptive)
from ..time_frequency.tfr import morlet, cwt
from ..utils import logger, verbose, _time_mask, warn
from ..externals.six import string_types

########################################################################
# Various connectivity estimators


class _AbstractConEstBase(object):
    """ABC for connectivity estimators."""

    def start_epoch(self):
        raise RuntimeError('start_epoch method not implemented')

    def accumulate(self, con_idx, csd_xy):
        raise RuntimeError('accumulate method not implemented')

    def combine(self, other):
        raise RuntimeError('combine method not implemented')

    def compute_con(self, con_idx, n_epochs):
        raise RuntimeError('compute_con method not implemented')


class _EpochMeanConEstBase(_AbstractConEstBase):
    """Base class for methods that estimate connectivity as mean epoch-wise."""

    def __init__(self, n_cons, n_freqs, n_times):
        self.n_cons = n_cons
        self.n_freqs = n_freqs
        self.n_times = n_times

        if n_times == 0:
            self.csd_shape = (n_cons, n_freqs)
        else:
            self.csd_shape = (n_cons, n_freqs, n_times)

        self.con_scores = None

    def start_epoch(self):  # noqa: D401
        """This method is called at the start of each epoch."""
        pass  # for this type of con. method we don't do anything

    def combine(self, other):
        """Include con. accumated for some epochs in this estimate."""
        self._acc += other._acc


class _CohEstBase(_EpochMeanConEstBase):
    """Base Estimator for Coherence, Coherency, Imag. Coherence."""

    def __init__(self, n_cons, n_freqs, n_times):
        super(_CohEstBase, self).__init__(n_cons, n_freqs, n_times)

        # allocate space for accumulation of CSD
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate CSD for some connections."""
        self._acc[con_idx] += csd_xy


class _CohEst(_CohEstBase):
    """Coherence Estimator."""

    name = 'Coherence'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.abs(csd_mean) / np.sqrt(psd_xx * psd_yy)


class _CohyEst(_CohEstBase):
    """Coherency Estimator."""

    name = 'Coherency'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape,
                                       dtype=np.complex128)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = csd_mean / np.sqrt(psd_xx * psd_yy)


class _ImCohEst(_CohEstBase):
    """Imaginary Coherence Estimator."""

    name = 'Imaginary Coherence'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.imag(csd_mean) / np.sqrt(psd_xx * psd_yy)


class _PLVEst(_EpochMeanConEstBase):
    """PLV Estimator."""

    name = 'PLV'

    def __init__(self, n_cons, n_freqs, n_times):
        super(_PLVEst, self).__init__(n_cons, n_freqs, n_times)

        # allocate accumulator
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        self._acc[con_idx] += csd_xy / np.abs(csd_xy)

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        plv = np.abs(self._acc / n_epochs)
        self.con_scores[con_idx] = plv


class _PLIEst(_EpochMeanConEstBase):
    """PLI Estimator."""

    name = 'PLI'

    def __init__(self, n_cons, n_freqs, n_times):
        super(_PLIEst, self).__init__(n_cons, n_freqs, n_times)

        # allocate accumulator
        self._acc = np.zeros(self.csd_shape)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        self._acc[con_idx] += np.sign(np.imag(csd_xy))

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        pli_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.abs(pli_mean)


class _PLIUnbiasedEst(_PLIEst):
    """Unbiased PLI Square Estimator."""

    name = 'Unbiased PLI Square'

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        pli_mean = self._acc[con_idx] / n_epochs

        # See Vinck paper Eq. (30)
        con = (n_epochs * pli_mean ** 2 - 1) / (n_epochs - 1)

        self.con_scores[con_idx] = con


class _WPLIEst(_EpochMeanConEstBase):
    """WPLI Estimator."""

    name = 'WPLI'

    def __init__(self, n_cons, n_freqs, n_times):
        super(_WPLIEst, self).__init__(n_cons, n_freqs, n_times)

        # store  both imag(csd) and abs(imag(csd))
        acc_shape = (2,) + self.csd_shape
        self._acc = np.zeros(acc_shape)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        im_csd = np.imag(csd_xy)
        self._acc[0, con_idx] += im_csd
        self._acc[1, con_idx] += np.abs(im_csd)

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)

        num = np.abs(self._acc[0, con_idx])
        denom = self._acc[1, con_idx]

        # handle zeros in denominator
        z_denom = np.where(denom == 0.)
        denom[z_denom] = 1.

        con = num / denom

        # where we had zeros in denominator, we set con to zero
        con[z_denom] = 0.

        self.con_scores[con_idx] = con


class _WPLIDebiasedEst(_EpochMeanConEstBase):
    """Debiased WPLI Square Estimator."""

    name = 'Debiased WPLI Square'

    def __init__(self, n_cons, n_freqs, n_times):
        super(_WPLIDebiasedEst, self).__init__(n_cons, n_freqs, n_times)
        # store imag(csd), abs(imag(csd)), imag(csd)^2
        acc_shape = (3,) + self.csd_shape
        self._acc = np.zeros(acc_shape)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        im_csd = np.imag(csd_xy)
        self._acc[0, con_idx] += im_csd
        self._acc[1, con_idx] += np.abs(im_csd)
        self._acc[2, con_idx] += im_csd ** 2

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)

        # note: we use the trick from fieldtrip to compute the
        # the estimate over all pairwise epoch combinations
        sum_im_csd = self._acc[0, con_idx]
        sum_abs_im_csd = self._acc[1, con_idx]
        sum_sq_im_csd = self._acc[2, con_idx]

        denom = sum_abs_im_csd ** 2 - sum_sq_im_csd

        # handle zeros in denominator
        z_denom = np.where(denom == 0.)
        denom[z_denom] = 1.

        con = (sum_im_csd ** 2 - sum_sq_im_csd) / denom

        # where we had zeros in denominator, we set con to zero
        con[z_denom] = 0.

        self.con_scores[con_idx] = con


class _PPCEst(_EpochMeanConEstBase):
    """Pairwise Phase Consistency (PPC) Estimator."""

    name = 'PPC'

    def __init__(self, n_cons, n_freqs, n_times):
        super(_PPCEst, self).__init__(n_cons, n_freqs, n_times)

        # store csd / abs(csd)
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        denom = np.abs(csd_xy)
        z_denom = np.where(denom == 0.)
        denom[z_denom] = 1.
        this_acc = csd_xy / denom
        this_acc[z_denom] = 0.  # handle division by zero

        self._acc[con_idx] += this_acc

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)

        # note: we use the trick from fieldtrip to compute the
        # the estimate over all pairwise epoch combinations
        con = ((self._acc[con_idx] * np.conj(self._acc[con_idx]) - n_epochs) /
               (n_epochs * (n_epochs - 1.)))

        self.con_scores[con_idx] = np.real(con)


###############################################################################
def _epoch_spectral_connectivity(data, sig_idx, tmin_idx, tmax_idx, sfreq,
                                 mode, window_fun, eigvals, wavelets,
                                 freq_mask, mt_adaptive, idx_map, block_size,
                                 psd, accumulate_psd, con_method_types,
                                 con_methods, n_signals, n_times,
                                 accumulate_inplace=True):
    """Connectivity estimation for one epoch see spectral_connectivity."""
    n_cons = len(idx_map[0])

    if wavelets is not None:
        n_times_spectrum = n_times
        n_freqs = len(wavelets)
    else:
        n_times_spectrum = 0
        n_freqs = np.sum(freq_mask)

    if not accumulate_inplace:
        # instantiate methods only for this epoch (used in parallel mode)
        con_methods = [mtype(n_cons, n_freqs, n_times_spectrum)
                       for mtype in con_method_types]

    if len(sig_idx) == n_signals:
        # we use all signals: use a slice for faster indexing
        sig_idx = slice(None, None)

    # compute tapered spectra
    if mode in ['multitaper', 'fourier']:
        x_mt = list()
        this_psd = list()
        sig_pos_start = 0
        for this_data in data:
            this_n_sig = this_data.shape[0]
            sig_pos_end = sig_pos_start + this_n_sig
            if not isinstance(sig_idx, slice):
                this_sig_idx = sig_idx[(sig_idx >= sig_pos_start) &
                                       (sig_idx < sig_pos_end)] - sig_pos_start
            else:
                this_sig_idx = sig_idx
            if isinstance(this_data, _BaseSourceEstimate):
                _mt_spectra_partial = partial(_mt_spectra, dpss=window_fun,
                                              sfreq=sfreq)
                this_x_mt = this_data.transform_data(
                    _mt_spectra_partial, idx=this_sig_idx, tmin_idx=tmin_idx,
                    tmax_idx=tmax_idx)
            else:
                this_x_mt, _ = _mt_spectra(this_data[this_sig_idx,
                                                     tmin_idx:tmax_idx],
                                           window_fun, sfreq)

            if mt_adaptive:
                # compute PSD and adaptive weights
                _this_psd, weights = _psd_from_mt_adaptive(
                    this_x_mt, eigvals, freq_mask, return_weights=True)

                # only keep freqs of interest
                this_x_mt = this_x_mt[:, :, freq_mask]
            else:
                # do not use adaptive weights
                this_x_mt = this_x_mt[:, :, freq_mask]
                if mode == 'multitaper':
                    weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
                else:
                    # hack to so we can sum over axis=-2
                    weights = np.array([1.])[:, None, None]

                if accumulate_psd:
                    _this_psd = _psd_from_mt(this_x_mt, weights)

            x_mt.append(this_x_mt)
            if accumulate_psd:
                this_psd.append(_this_psd)

        x_mt = np.concatenate(x_mt, axis=0)
        if accumulate_psd:
            this_psd = np.concatenate(this_psd, axis=0)

        # advance position
        sig_pos_start = sig_pos_end

    elif mode == 'cwt_morlet':
        # estimate spectra using CWT
        x_cwt = list()
        this_psd = list()
        sig_pos_start = 0
        for this_data in data:
            this_n_sig = this_data.shape[0]
            sig_pos_end = sig_pos_start + this_n_sig
            if not isinstance(sig_idx, slice):
                this_sig_idx = sig_idx[(sig_idx >= sig_pos_start) &
                                       (sig_idx < sig_pos_end)] - sig_pos_start
            else:
                this_sig_idx = sig_idx
            if isinstance(this_data, _BaseSourceEstimate):
                cwt_partial = partial(cwt, Ws=wavelets, use_fft=True,
                                      mode='same')
                this_x_cwt = this_data.transform_data(
                    cwt_partial, idx=this_sig_idx, tmin_idx=tmin_idx,
                    tmax_idx=tmax_idx)
            else:
                this_x_cwt = cwt(this_data[this_sig_idx, tmin_idx:tmax_idx],
                                 wavelets, use_fft=True, mode='same')

            if accumulate_psd:
                this_psd.append((this_x_cwt * this_x_cwt.conj()).real)

            x_cwt.append(this_x_cwt)

            # advance position
            sig_pos_start = sig_pos_end

        x_cwt = np.concatenate(x_cwt, axis=0)
        if accumulate_psd:
            this_psd = np.concatenate(this_psd, axis=0)
    else:
        raise RuntimeError('invalid mode')

    # accumulate or return psd
    if accumulate_psd:
        if accumulate_inplace:
            psd += this_psd
        else:
            psd = this_psd
    else:
        psd = None

    # tell the methods that a new epoch starts
    for method in con_methods:
        method.start_epoch()

    # accumulate connectivity scores
    if mode in ['multitaper', 'fourier']:
        for i in range(0, n_cons, block_size):
            con_idx = slice(i, i + block_size)
            if mt_adaptive:
                csd = _csd_from_mt(x_mt[idx_map[0][con_idx]],
                                   x_mt[idx_map[1][con_idx]],
                                   weights[idx_map[0][con_idx]],
                                   weights[idx_map[1][con_idx]])
            else:
                csd = _csd_from_mt(x_mt[idx_map[0][con_idx]],
                                   x_mt[idx_map[1][con_idx]],
                                   weights, weights)

            for method in con_methods:
                method.accumulate(con_idx, csd)
    else:
        # cwt_morlet mode
        for i in range(0, n_cons, block_size):
            con_idx = slice(i, i + block_size)

            csd = x_cwt[idx_map[0][con_idx]] * \
                np.conjugate(x_cwt[idx_map[1][con_idx]])
            for method in con_methods:
                method.accumulate(con_idx, csd)

    return con_methods, psd


def _get_n_epochs(epochs, n):
    """Generator that returns lists with at most n epochs."""
    epochs_out = []
    for e in epochs:
        if not isinstance(e, (list, tuple)):
            e = (e,)
        epochs_out.append(e)
        if len(epochs_out) >= n:
            yield epochs_out
            epochs_out = []
    yield epochs_out


def _check_method(method):
    """Test if a method implements the required interface."""
    interface_members = [m[0] for m in getmembers(_AbstractConEstBase)
                         if not m[0].startswith('_')]
    method_members = [m[0] for m in getmembers(method)
                      if not m[0].startswith('_')]

    for member in interface_members:
        if member not in method_members:
            return False, member
    return True, None


def _get_and_verify_data_sizes(data, n_signals=None, n_times=None, times=None):
    """Helper function to get and/or verify the data sizes and time scales."""
    if not isinstance(data, (list, tuple)):
        raise ValueError('data has to be a list or tuple')
    n_signals_tot = 0
    for this_data in data:
        this_n_signals, this_n_times = this_data.shape
        if n_times is not None:
            if this_n_times != n_times:
                raise ValueError('all input time series must have the same '
                                 'number of time points')
        else:
            n_times = this_n_times
        n_signals_tot += this_n_signals

        if hasattr(this_data, 'times'):
            this_times = this_data.times
            if times is not None:
                if np.any(times != this_times):
                    warn('time scales of input time series do not match')
            else:
                times = this_times

    if n_signals is not None:
        if n_signals != n_signals_tot:
            raise ValueError('the number of time series has to be the same in '
                             'each epoch')
    n_signals = n_signals_tot

    return n_signals, n_times, times


# map names to estimator types
_CON_METHOD_MAP = {'coh': _CohEst, 'cohy': _CohyEst, 'imcoh': _ImCohEst,
                   'plv': _PLVEst, 'ppc': _PPCEst, 'pli': _PLIEst,
                   'pli2_unbiased': _PLIUnbiasedEst, 'wpli': _WPLIEst,
                   'wpli2_debiased': _WPLIDebiasedEst}


@verbose
def spectral_connectivity(data, method='coh', indices=None, sfreq=2 * np.pi,
                          mode='multitaper', fmin=None, fmax=np.inf,
                          fskip=0, faverage=False, tmin=None, tmax=None,
                          mt_bandwidth=None, mt_adaptive=False,
                          mt_low_bias=True, cwt_frequencies=None,
                          cwt_n_cycles=7, block_size=1000, n_jobs=1,
                          verbose=None):
    """Compute frequency- and time-frequency-domain connectivity measures.

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD) Sxy and Sxx, Syy.

    The spectral densities can be estimated using a multitaper method with
    digital prolate spheroidal sequence (DPSS) windows, a discrete Fourier
    transform with Hanning windows, or a continuous wavelet transform using
    Morlet wavelets. The spectral estimation mode is specified using the
    "mode" parameter.

    By default, the connectivity between all signals is computed (only
    connections corresponding to the lower-triangular part of the
    connectivity matrix). If one is only interested in the connectivity
    between some signals, the "indices" parameter can be used. For example,
    to compute the connectivity between the signal with index 0 and signals
    "2, 3, 4" (a total of 3 connections) one can use the following::

        indices = (np.array([0, 0, 0]),    # row indices
                   np.array([2, 3, 4]))    # col indices

        con_flat = spectral_connectivity(data, method='coh',
                                         indices=indices, ...)

    In this case con_flat.shape = (3, n_freqs). The connectivity scores are
    in the same order as defined indices.

    **Supported Connectivity Measures**

    The connectivity method(s) is specified using the "method" parameter. The
    following methods are supported (note: ``E[]`` denotes average over
    epochs). Multiple measures can be computed at once by using a list/tuple,
    e.g., ``['coh', 'pli']`` to compute coherence and PLI.

        'coh' : Coherence given by::

                     | E[Sxy] |
            C = ---------------------
                sqrt(E[Sxx] * E[Syy])

        'cohy' : Coherency given by::

                       E[Sxy]
            C = ---------------------
                sqrt(E[Sxx] * E[Syy])

        'imcoh' : Imaginary coherence [1]_ given by::

                      Im(E[Sxy])
            C = ----------------------
                sqrt(E[Sxx] * E[Syy])

        'plv' : Phase-Locking Value (PLV) [2]_ given by::

            PLV = |E[Sxy/|Sxy|]|

        'ppc' : Pairwise Phase Consistency (PPC), an unbiased estimator
        of squared PLV [3]_.

        'pli' : Phase Lag Index (PLI) [4]_ given by::

            PLI = |E[sign(Im(Sxy))]|

        'pli2_unbiased' : Unbiased estimator of squared PLI [5]_.

        'wpli' : Weighted Phase Lag Index (WPLI) [5]_ given by::

                      |E[Im(Sxy)]|
            WPLI = ------------------
                      E[|Im(Sxy)|]

        'wpli2_debiased' : Debiased estimator of squared WPLI [5].


    References
    ----------

    .. [1] Nolte et al. "Identifying true brain interaction from EEG data using
           the imaginary part of coherency" Clinical neurophysiology, vol. 115,
           no. 10, pp. 2292-2307, Oct. 2004.

    .. [2] Lachaux et al. "Measuring phase synchrony in brain signals" Human
           brain mapping, vol. 8, no. 4, pp. 194-208, Jan. 1999.

    .. [3] Vinck et al. "The pairwise phase consistency: a bias-free measure of
           rhythmic neuronal synchronization" NeuroImage, vol. 51, no. 1,
           pp. 112-122, May 2010.

    .. [4] Stam et al. "Phase lag index: assessment of functional connectivity
           from multi channel EEG and MEG with diminished bias from common
           sources" Human brain mapping, vol. 28, no. 11, pp. 1178-1193,
           Nov. 2007.

    .. [5] Vinck et al. "An improved index of phase-synchronization for
           electro-physiological data in the presence of volume-conduction,
           noise and sample-size bias" NeuroImage, vol. 55, no. 4,
           pp. 1548-1565, Apr. 2011.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | Epochs
        The data from which to compute connectivity. Note that it is also
        possible to combine multiple signals by providing a list of tuples,
        e.g., data = [(arr_0, stc_0), (arr_1, stc_1), (arr_2, stc_2)],
        corresponds to 3 epochs, and arr_* could be an array with the same
        number of time points as stc_*. The array-like object can also
        be a list/generator of array, shape =(n_signals, n_times),
        or a list/generator of SourceEstimate or VolSourceEstimate objects.
    method : string | list of string
        Connectivity measure(s) to compute.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which to compute
        connectivity. If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    mode : str
        Spectrum estimation mode can be either: 'multitaper', 'fourier', or
        'cwt_morlet'.
    fmin : float | tuple of floats
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 20.) for two bands with 8Hz and 20Hz lower freq.
        If None the frequency corresponding to an epoch length of 5 cycles
        is used.
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
        Time to start connectivity estimation. Note: when "data" is an array,
        the first sample is assumed to be at time 0. For other types
        (Epochs, etc.), the time information contained in the object is used
        to compute the time indices.
    tmax : float | None
        Time to end connectivity estimation. Note: when "data" is an array,
        the first sample is assumed to be at time 0. For other types
        (Epochs, etc.), the time information contained in the object is used
        to compute the time indices.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    cwt_frequencies : array
        Array of frequencies of interest. Only used in 'cwt_morlet' mode.
    cwt_n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency. Only used in
        'cwt_morlet' mode.
    block_size : int
        How many connections to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many epochs to process in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    con : array | list of arrays
        Computed connectivity measure(s). The shape of each array is either
        (n_signals, n_signals, n_frequencies) mode: 'multitaper' or 'fourier'
        (n_signals, n_signals, n_frequencies, n_times) mode: 'cwt_morlet'
        when "indices" is None, or
        (n_con, n_frequencies) mode: 'multitaper' or 'fourier'
        (n_con, n_frequencies, n_times) mode: 'cwt_morlet'
        when "indices" is specified and "n_con = len(indices[0])".
    freqs : array
        Frequency points at which the connectivity was computed.
    times : array
        Time points for which the connectivity was computed.
    n_epochs : int
        Number of epochs used for computation.
    n_tapers : int
        The number of DPSS tapers used. Only defined in 'multitaper' mode.
        Otherwise None is returned.
    """
    if n_jobs != 1:
        parallel, my_epoch_spectral_connectivity, _ = \
            parallel_func(_epoch_spectral_connectivity, n_jobs,
                          verbose=verbose)

    # format fmin and fmax and check inputs
    if fmin is None:
        fmin = -np.inf  # set it to -inf, so we can adjust it later

    fmin = np.asarray((fmin,)).ravel()
    fmax = np.asarray((fmax,)).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

    n_bands = len(fmin)

    # assign names to connectivity methods
    if not isinstance(method, (list, tuple)):
        method = [method]  # make it a list so we can iterate over it

    n_methods = len(method)
    con_method_types = []
    for m in method:
        if m in _CON_METHOD_MAP:
            method = _CON_METHOD_MAP[m]
            con_method_types.append(method)
        elif isinstance(m, string_types):
            raise ValueError('%s is not a valid connectivity method' % m)
        else:
            # add custom method
            method_valid, msg = _check_method(m)
            if not method_valid:
                raise ValueError('The supplied connectivity method does '
                                 'not have the method %s' % msg)
            con_method_types.append(m)

    # determine how many arguments the compute_con_function needs
    n_comp_args = [len(_get_args(mtype.compute_con))
                   for mtype in con_method_types]

    # we only support 3 or 5 arguments
    if any(n not in (3, 5) for n in n_comp_args):
        raise ValueError('The compute_con function needs to have either '
                         '3 or 5 arguments')

    # if none of the comp_con functions needs the PSD, we don't estimate it
    accumulate_psd = any(n == 5 for n in n_comp_args)

    if isinstance(data, BaseEpochs):
        times_in = data.times  # input times for Epochs input type
        sfreq = data.info['sfreq']

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays or SourceEstimates
    epoch_idx = 0
    logger.info('Connectivity computation...')
    for epoch_block in _get_n_epochs(data, n_jobs):

        if epoch_idx == 0:
            # initialize everything
            first_epoch = epoch_block[0]

            # get the data size and time scale
            n_signals, n_times_in, times_in = \
                _get_and_verify_data_sizes(first_epoch)

            if times_in is None:
                # we are not using Epochs or SourceEstimate(s) as input
                times_in = np.linspace(0.0, n_times_in / sfreq, n_times_in,
                                       endpoint=False)

            n_times_in = len(times_in)
            mask = _time_mask(times_in, tmin, tmax, sfreq=sfreq)
            tmin_idx, tmax_idx = np.where(mask)[0][[0, -1]]
            tmax_idx += 1
            tmin_true = times_in[tmin_idx]
            tmax_true = times_in[tmax_idx - 1]  # time of last point used

            times = times_in[tmin_idx:tmax_idx]
            n_times = len(times)

            if indices is None:
                # only compute r for lower-triangular region
                indices_use = np.tril_indices(n_signals, -1)
            else:
                indices_use = check_indices(indices)

            # number of connectivities to compute
            n_cons = len(indices_use[0])

            logger.info('    computing connectivity for %d connections'
                        % n_cons)

            logger.info('    using t=%0.3fs..%0.3fs for estimation (%d points)'
                        % (tmin_true, tmax_true, n_times))

            # get frequencies of interest for the different modes
            if mode in ['multitaper', 'fourier']:
                # fmin fmax etc is only supported for these modes
                # decide which frequencies to keep
                freqs_all = fftfreq(n_times, 1. / sfreq)
                freqs_all = freqs_all[freqs_all >= 0]
            elif mode == 'cwt_morlet':
                # cwt_morlet mode
                if cwt_frequencies is None:
                    raise ValueError('define frequencies of interest using '
                                     'cwt_frequencies')
                else:
                    cwt_frequencies = cwt_frequencies.astype(np.float)
                if any(cwt_frequencies > (sfreq / 2.)):
                    raise ValueError('entries in cwt_frequencies cannot be '
                                     'larger than Nyquist (sfreq / 2)')
                freqs_all = cwt_frequencies
            else:
                raise ValueError('mode has an invalid value')

            # check that fmin corresponds to at least 5 cycles
            five_cycle_freq = 5. * sfreq / float(n_times)

            if len(fmin) == 1 and fmin[0] == -np.inf:
                # we use the 5 cycle freq. as default
                fmin = [five_cycle_freq]
            else:
                if any(fmin < five_cycle_freq):
                    warn('fmin corresponds to less than 5 cycles, '
                         'spectrum estimate will be unreliable')

            # create a frequency mask for all bands
            freq_mask = np.zeros(len(freqs_all), dtype=np.bool)
            for f_lower, f_upper in zip(fmin, fmax):
                freq_mask |= ((freqs_all >= f_lower) & (freqs_all <= f_upper))

            # possibly skip frequency points
            for pos in range(fskip):
                freq_mask[pos + 1::fskip + 1] = False

            # the frequency points where we compute connectivity
            freqs = freqs_all[freq_mask]
            n_freqs = len(freqs)

            # get the freq. indices and points for each band
            freq_idx_bands = [np.where((freqs >= fl) & (freqs <= fu))[0]
                              for fl, fu in zip(fmin, fmax)]
            freqs_bands = [freqs[freq_idx] for freq_idx in freq_idx_bands]

            # make sure we don't have empty bands
            for i, n_f_band in enumerate([len(f) for f in freqs_bands]):
                if n_f_band == 0:
                    raise ValueError('There are no frequency points between '
                                     '%0.1fHz and %0.1fHz. Change the band '
                                     'specification (fmin, fmax) or the '
                                     'frequency resolution.'
                                     % (fmin[i], fmax[i]))

            if n_bands == 1:
                logger.info('    frequencies: %0.1fHz..%0.1fHz (%d points)'
                            % (freqs_bands[0][0], freqs_bands[0][-1],
                               n_freqs))
            else:
                logger.info('    computing connectivity for the bands:')
                for i, bfreqs in enumerate(freqs_bands):
                    logger.info('     band %d: %0.1fHz..%0.1fHz '
                                '(%d points)' % (i + 1, bfreqs[0],
                                                 bfreqs[-1], len(bfreqs)))

            if faverage:
                logger.info('    connectivity scores will be averaged for '
                            'each band')

            # get the window function, wavelets, etc for different modes
            if mode == 'multitaper':
                # compute standardized half-bandwidth
                if mt_bandwidth is not None:
                    half_nbw = float(mt_bandwidth) * n_times / (2 * sfreq)
                else:
                    half_nbw = 4

                # compute dpss windows
                n_tapers_max = int(2 * half_nbw)
                window_fun, eigvals = dpss_windows(n_times, half_nbw,
                                                   n_tapers_max,
                                                   low_bias=mt_low_bias)
                n_tapers = len(eigvals)
                logger.info('    using multitaper spectrum estimation with '
                            '%d DPSS windows' % n_tapers)

                if mt_adaptive and len(eigvals) < 3:
                    warn('Not adaptively combining the spectral estimators '
                         'due to a low number of tapers.')
                    mt_adaptive = False

                n_times_spectrum = 0  # this method only uses the freq. domain
                wavelets = None
            elif mode == 'fourier':
                logger.info('    using FFT with a Hanning window to estimate '
                            'spectra')

                window_fun = np.hanning(n_times)
                mt_adaptive = False
                eigvals = 1.
                n_tapers = None
                n_times_spectrum = 0  # this method only uses the freq. domain
                wavelets = None
            elif mode == 'cwt_morlet':
                logger.info('    using CWT with Morlet wavelets to estimate '
                            'spectra')

                # reformat cwt_n_cycles if we have removed some frequencies
                # using fmin, fmax, fskip
                cwt_n_cycles = np.asarray((cwt_n_cycles,)).ravel()
                if len(cwt_n_cycles) > 1:
                    if len(cwt_n_cycles) != len(cwt_frequencies):
                        raise ValueError('cwt_n_cycles must be float or an '
                                         'array with the same size as '
                                         'cwt_frequencies')
                    cwt_n_cycles = cwt_n_cycles[freq_mask]

                # get the Morlet wavelets
                wavelets = morlet(sfreq, freqs, n_cycles=cwt_n_cycles,
                                  zero_mean=True)
                eigvals = None
                n_tapers = None
                window_fun = None
                n_times_spectrum = n_times
            else:
                raise ValueError('mode has an invalid value')

            # unique signals for which we actually need to compute PSD etc.
            sig_idx = np.unique(np.r_[indices_use[0], indices_use[1]])

            # map indices to unique indices
            idx_map = [np.searchsorted(sig_idx, ind) for ind in indices_use]

            # allocate space to accumulate PSD
            if accumulate_psd:
                if n_times_spectrum == 0:
                    psd_shape = (len(sig_idx), n_freqs)
                else:
                    psd_shape = (len(sig_idx), n_freqs, n_times_spectrum)
                psd = np.zeros(psd_shape)
            else:
                psd = None

            # create instances of the connectivity estimators
            con_methods = [mtype(n_cons, n_freqs, n_times_spectrum)
                           for mtype in con_method_types]

            sep = ', '
            metrics_str = sep.join([meth.name for meth in con_methods])
            logger.info('    the following metrics will be computed: %s'
                        % metrics_str)

        # check dimensions and time scale
        for this_epoch in epoch_block:
            _get_and_verify_data_sizes(this_epoch, n_signals, n_times_in,
                                       times_in)

        if n_jobs == 1:
            # no parallel processing
            for this_epoch in epoch_block:
                logger.info('    computing connectivity for epoch %d'
                            % (epoch_idx + 1))

                # con methods and psd are updated inplace
                _epoch_spectral_connectivity(
                    this_epoch, sig_idx, tmin_idx,
                    tmax_idx, sfreq, mode, window_fun, eigvals, wavelets,
                    freq_mask, mt_adaptive, idx_map, block_size, psd,
                    accumulate_psd, con_method_types, con_methods,
                    n_signals, n_times, accumulate_inplace=True)
                epoch_idx += 1
        else:
            # process epochs in parallel
            logger.info('    computing connectivity for epochs %d..%d'
                        % (epoch_idx + 1, epoch_idx + len(epoch_block)))

            out = parallel(my_epoch_spectral_connectivity(
                this_epoch, sig_idx,
                tmin_idx, tmax_idx, sfreq, mode, window_fun, eigvals,
                wavelets, freq_mask, mt_adaptive, idx_map, block_size, psd,
                accumulate_psd, con_method_types, None, n_signals, n_times,
                accumulate_inplace=False) for this_epoch in epoch_block)

            # do the accumulation
            for this_out in out:
                for method, parallel_method in zip(con_methods, this_out[0]):
                    method.combine(parallel_method)
                if accumulate_psd:
                    psd += this_out[1]

            epoch_idx += len(epoch_block)

    # normalize
    n_epochs = epoch_idx
    if accumulate_psd:
        psd /= n_epochs

    # compute final connectivity scores
    con = []
    for method, n_args in zip(con_methods, n_comp_args):
        if n_args == 3:
            # compute all scores at once
            method.compute_con(slice(0, n_cons), n_epochs)
        else:
            # compute scores block-wise to save memory
            for i in range(0, n_cons, block_size):
                con_idx = slice(i, i + block_size)
                psd_xx = psd[idx_map[0][con_idx]]
                psd_yy = psd[idx_map[1][con_idx]]
                method.compute_con(con_idx, n_epochs, psd_xx, psd_yy)

        # get the connectivity scores
        this_con = method.con_scores

        if this_con.shape[0] != n_cons:
            raise ValueError('First dimension of connectivity scores must be '
                             'the same as the number of connections')
        if faverage:
            if this_con.shape[1] != n_freqs:
                raise ValueError('2nd dimension of connectivity scores must '
                                 'be the same as the number of frequencies')
            con_shape = (n_cons, n_bands) + this_con.shape[2:]
            this_con_bands = np.empty(con_shape, dtype=this_con.dtype)
            for band_idx in range(n_bands):
                this_con_bands[:, band_idx] =\
                    np.mean(this_con[:, freq_idx_bands[band_idx]], axis=1)
            this_con = this_con_bands

        con.append(this_con)

    if indices is None:
        # return all-to-all connectivity matrices
        logger.info('    assembling connectivity matrix')
        con_flat = con
        con = []
        for this_con_flat in con_flat:
            this_con = np.zeros((n_signals, n_signals) +
                                this_con_flat.shape[1:],
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

    return con, freqs, times, n_epochs, n_tapers
