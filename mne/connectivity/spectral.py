# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from functools import partial
from inspect import getmembers

import numpy as np

from .utils import check_indices
from ..utils import _check_option
from ..fixes import _get_args, rfftfreq
from ..parallel import parallel_func
from ..source_estimate import _BaseSourceEstimate
from ..epochs import BaseEpochs
from ..time_frequency.multitaper import (_mt_spectra, _compute_mt_params,
                                         _psd_from_mt, _csd_from_mt,
                                         _psd_from_mt_adaptive)
from ..time_frequency.tfr import morlet, cwt
from ..utils import logger, verbose, _time_mask, warn

########################################################################
# Various connectivity estimators


class _AbstractConEstBase(object):
    """ABC for connectivity estimators."""

    def start_epoch(self):
        raise NotImplementedError('start_epoch method not implemented')

    def accumulate(self, con_idx, csd_xy):
        raise NotImplementedError('accumulate method not implemented')

    def combine(self, other):
        raise NotImplementedError('combine method not implemented')

    def compute_con(self, con_idx, n_epochs):
        raise NotImplementedError('compute_con method not implemented')


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
        """Called at the start of each epoch."""
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

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):  # lgtm
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.abs(csd_mean) / np.sqrt(psd_xx * psd_yy)


class _CohyEst(_CohEstBase):
    """Coherency Estimator."""

    name = 'Coherency'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):  # lgtm
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape,
                                       dtype=np.complex128)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = csd_mean / np.sqrt(psd_xx * psd_yy)


class _ImCohEst(_CohEstBase):
    """Imaginary Coherence Estimator."""

    name = 'Imaginary Coherence'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):  # lgtm
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


class _ciPLVEst(_EpochMeanConEstBase):
    """corrected imaginary PLV Estimator."""

    name = 'ciPLV'

    def __init__(self, n_cons, n_freqs, n_times):
        super(_ciPLVEst, self).__init__(n_cons, n_freqs, n_times)

        # allocate accumulator
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        self._acc[con_idx] += csd_xy / np.abs(csd_xy)

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        imag_plv = np.abs(np.imag(self._acc)) / n_epochs
        real_plv = np.real(self._acc) / n_epochs
        real_plv = np.clip(real_plv, -1, 1)  # bounded from -1 to 1
        mask = (np.abs(real_plv) == 1)  # avoid division by 0
        real_plv[mask] = 0
        corrected_imag_plv = imag_plv / np.sqrt(1 - real_plv ** 2)
        self.con_scores[con_idx] = corrected_imag_plv


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
    """Estimate connectivity for one epoch (see spectral_connectivity)."""
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

    _check_option('mode', mode, ('cwt_morlet', 'multitaper', 'fourier'))
    if len(sig_idx) == n_signals:
        # we use all signals: use a slice for faster indexing
        sig_idx = slice(None, None)

    # compute tapered spectra
    x_t = list()
    this_psd = list()
    for this_data in data:
        if mode in ('multitaper', 'fourier'):
            if isinstance(this_data, _BaseSourceEstimate):
                _mt_spectra_partial = partial(_mt_spectra, dpss=window_fun,
                                              sfreq=sfreq)
                this_x_t = this_data.transform_data(
                    _mt_spectra_partial, idx=sig_idx, tmin_idx=tmin_idx,
                    tmax_idx=tmax_idx)
            else:
                this_x_t, _ = _mt_spectra(
                    this_data[sig_idx, tmin_idx:tmax_idx],
                    window_fun, sfreq)

            if mt_adaptive:
                # compute PSD and adaptive weights
                _this_psd, weights = _psd_from_mt_adaptive(
                    this_x_t, eigvals, freq_mask, return_weights=True)

                # only keep freqs of interest
                this_x_t = this_x_t[:, :, freq_mask]
            else:
                # do not use adaptive weights
                this_x_t = this_x_t[:, :, freq_mask]
                if mode == 'multitaper':
                    weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
                else:
                    # hack to so we can sum over axis=-2
                    weights = np.array([1.])[:, None, None]

                if accumulate_psd:
                    _this_psd = _psd_from_mt(this_x_t, weights)
        else:  # mode == 'cwt_morlet'
            if isinstance(this_data, _BaseSourceEstimate):
                cwt_partial = partial(cwt, Ws=wavelets, use_fft=True,
                                      mode='same')
                this_x_t = this_data.transform_data(
                    cwt_partial, idx=sig_idx, tmin_idx=tmin_idx,
                    tmax_idx=tmax_idx)
            else:
                this_x_t = cwt(this_data[sig_idx, tmin_idx:tmax_idx],
                               wavelets, use_fft=True, mode='same')
            _this_psd = (this_x_t * this_x_t.conj()).real

        x_t.append(this_x_t)
        if accumulate_psd:
            this_psd.append(_this_psd)

    x_t = np.concatenate(x_t, axis=0)
    if accumulate_psd:
        this_psd = np.concatenate(this_psd, axis=0)

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
                csd = _csd_from_mt(x_t[idx_map[0][con_idx]],
                                   x_t[idx_map[1][con_idx]],
                                   weights[idx_map[0][con_idx]],
                                   weights[idx_map[1][con_idx]])
            else:
                csd = _csd_from_mt(x_t[idx_map[0][con_idx]],
                                   x_t[idx_map[1][con_idx]],
                                   weights, weights)

            for method in con_methods:
                method.accumulate(con_idx, csd)
    else:  # mode == 'cwt_morlet'  # reminder to add alternative TFR methods
        for i_block, i in enumerate(range(0, n_cons, block_size)):
            con_idx = slice(i, i + block_size)
            # this codes can be very slow
            csd = (x_t[idx_map[0][con_idx]] *
                   x_t[idx_map[1][con_idx]].conjugate())

            for method in con_methods:
                method.accumulate(con_idx, csd)
                # future estimator types need to be explicitly handled here

    return con_methods, psd


def _get_n_epochs(epochs, n):
    """Generate lists with at most n epochs."""
    epochs_out = list()
    for epoch in epochs:
        if not isinstance(epoch, (list, tuple)):
            epoch = (epoch,)
        epochs_out.append(epoch)
        if len(epochs_out) >= n:
            yield epochs_out
            epochs_out = list()
    if 0 < len(epochs_out) < n:
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
    """Get and/or verify the data sizes and time scales."""
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
                   'plv': _PLVEst, 'ciplv': _ciPLVEst, 'ppc': _PPCEst,
                   'pli': _PLIEst, 'pli2_unbiased': _PLIUnbiasedEst,
                   'wpli': _WPLIEst, 'wpli2_debiased': _WPLIDebiasedEst}


def _check_estimators(method, mode):
    """Check construction of connectivity estimators."""
    n_methods = len(method)
    con_method_types = list()
    for this_method in method:
        if this_method in _CON_METHOD_MAP:
            con_method_types.append(_CON_METHOD_MAP[this_method])
        elif isinstance(this_method, str):
            raise ValueError('%s is not a valid connectivity method' %
                             this_method)
        else:
            # support for custom class
            method_valid, msg = _check_method(this_method)
            if not method_valid:
                raise ValueError('The supplied connectivity method does '
                                 'not have the method %s' % msg)
            con_method_types.append(this_method)

    # determine how many arguments the compute_con_function needs
    n_comp_args = [len(_get_args(mtype.compute_con))
                   for mtype in con_method_types]

    # we currently only support 3 arguments
    if any(n not in (3, 5) for n in n_comp_args):
        raise ValueError('The .compute_con method needs to have either '
                         '3 or 5 arguments')
    # if none of the comp_con functions needs the PSD, we don't estimate it
    accumulate_psd = any(n == 5 for n in n_comp_args)
    return con_method_types, n_methods, accumulate_psd, n_comp_args


@verbose
def spectral_connectivity(data, method='coh', indices=None, sfreq=2 * np.pi,
                          mode='multitaper', fmin=None, fmax=np.inf,
                          fskip=0, faverage=False, tmin=None, tmax=None,
                          mt_bandwidth=None, mt_adaptive=False,
                          mt_low_bias=True, cwt_freqs=None,
                          cwt_n_cycles=7, block_size=1000, n_jobs=1,
                          verbose=None):
    """Compute frequency- and time-frequency-domain connectivity measures.

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD) Sxy and Sxx, Syy.

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
    method : str | list of str
        Connectivity measure(s) to compute.
    indices : tuple of array | None
        Two arrays with indices of connections for which to compute
        connectivity. If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    mode : str
        Spectrum estimation mode can be either: 'multitaper', 'fourier', or
        'cwt_morlet'.
    fmin : float | tuple of float
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 20.) for two bands with 8Hz and 20Hz lower freq.
        If None the frequency corresponding to an epoch length of 5 cycles
        is used.
    fmax : float | tuple of float
        The upper frequency of interest. Multiple bands are dedined using
        a tuple, e.g. (13., 30.) for two band with 13Hz and 30Hz upper freq.
    fskip : int
        Omit every "(fskip + 1)-th" frequency bin to decimate in frequency
        domain.
    faverage : bool
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
        Only use tapers with more than 90%% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    cwt_freqs : array
        Array of frequencies of interest. Only used in 'cwt_morlet' mode.
    cwt_n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency. Only used in
        'cwt_morlet' mode.
    block_size : int
        How many connections to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many epochs to process in parallel.
    %(verbose)s

    Returns
    -------
    con : array | list of array
        Computed connectivity measure(s). The shape of each array is either
        (n_signals, n_signals, n_freqs) mode: 'multitaper' or 'fourier'
        (n_signals, n_signals, n_freqs, n_times) mode: 'cwt_morlet'
        when "indices" is None, or
        (n_con, n_freqs) mode: 'multitaper' or 'fourier'
        (n_con, n_freqs, n_times) mode: 'cwt_morlet'
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

    Notes
    -----
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

        'ciplv' : corrected imaginary PLV (icPLV) [3]_ given by::

                             |E[Im(Sxy/|Sxy|)]|
            ciPLV = ------------------------------------
                     sqrt(1 - |E[real(Sxy/|Sxy|)]| ** 2)

        'ppc' : Pairwise Phase Consistency (PPC), an unbiased estimator
        of squared PLV [4]_.

        'pli' : Phase Lag Index (PLI) [5]_ given by::

            PLI = |E[sign(Im(Sxy))]|

        'pli2_unbiased' : Unbiased estimator of squared PLI [6]_.

        'wpli' : Weighted Phase Lag Index (WPLI) [6]_ given by::

                      |E[Im(Sxy)]|
            WPLI = ------------------
                      E[|Im(Sxy)|]

        'wpli2_debiased' : Debiased estimator of squared WPLI [6]_.

    References
    ----------
    .. [1] Nolte et al. "Identifying true brain interaction from EEG data using
           the imaginary part of coherency" Clinical neurophysiology, vol. 115,
           no. 10, pp. 2292-2307, Oct. 2004.
    .. [2] Lachaux et al. "Measuring phase synchrony in brain signals" Human
           brain mapping, vol. 8, no. 4, pp. 194-208, Jan. 1999.
    .. [3] Bruña et al. "Phase locking value revisited: teaching new tricks to
           an old dog" Journal of Neural Engineering, vol. 15, no. 5, pp.
           056011 , Jul. 2018.
    .. [4] Vinck et al. "The pairwise phase consistency: a bias-free measure of
           rhythmic neuronal synchronization" NeuroImage, vol. 51, no. 1,
           pp. 112-122, May 2010.
    .. [5] Stam et al. "Phase lag index: assessment of functional connectivity
           from multi channel EEG and MEG with diminished bias from common
           sources" Human brain mapping, vol. 28, no. 11, pp. 1178-1193,
           Nov. 2007.
    .. [6] Vinck et al. "An improved index of phase-synchronization for
           electro-physiological data in the presence of volume-conduction,
           noise and sample-size bias" NeuroImage, vol. 55, no. 4,
           pp. 1548-1565, Apr. 2011.
    """
    if n_jobs != 1:
        parallel, my_epoch_spectral_connectivity, _ = \
            parallel_func(_epoch_spectral_connectivity, n_jobs,
                          verbose=verbose)

    # format fmin and fmax and check inputs
    if fmin is None:
        fmin = -np.inf  # set it to -inf, so we can adjust it later

    fmin = np.array((fmin,), dtype=float).ravel()
    fmax = np.array((fmax,), dtype=float).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

    n_bands = len(fmin)

    # assign names to connectivity methods
    if not isinstance(method, (list, tuple)):
        method = [method]  # make it a list so we can iterate over it

    # handle connectivity estimators
    (con_method_types, n_methods, accumulate_psd,
     n_comp_args) = _check_estimators(method=method, mode=mode)

    if isinstance(data, BaseEpochs):
        times_in = data.times  # input times for Epochs input type
        sfreq = data.info['sfreq']

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays or SourceEstimates
    epoch_idx = 0
    logger.info('Connectivity computation...')
    for epoch_block in _get_n_epochs(data, n_jobs):
        if epoch_idx == 0:
            # initialize everything times and frequencies
            (n_cons, times, n_times, times_in, n_times_in, tmin_idx,
             tmax_idx, n_freqs, freq_mask, freqs, freqs_bands, freq_idx_bands,
             n_signals, indices_use) = _prepare_connectivity(
                epoch_block=epoch_block, tmin=tmin, tmax=tmax, fmin=fmin,
                fmax=fmax, sfreq=sfreq, indices=indices, mode=mode,
                fskip=fskip, n_bands=n_bands,
                cwt_freqs=cwt_freqs, faverage=faverage)

            # get the window function, wavelets, etc for different modes
            (spectral_params, mt_adaptive, n_times_spectrum,
             n_tapers) = _assemble_spectral_params(
                mode=mode, n_times=n_times, mt_adaptive=mt_adaptive,
                mt_bandwidth=mt_bandwidth, sfreq=sfreq,
                mt_low_bias=mt_low_bias, cwt_n_cycles=cwt_n_cycles,
                cwt_freqs=cwt_freqs, freqs=freqs, freq_mask=freq_mask)

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

        call_params = dict(
            sig_idx=sig_idx, tmin_idx=tmin_idx,
            tmax_idx=tmax_idx, sfreq=sfreq, mode=mode,
            freq_mask=freq_mask, idx_map=idx_map, block_size=block_size,
            psd=psd, accumulate_psd=accumulate_psd,
            mt_adaptive=mt_adaptive,
            con_method_types=con_method_types,
            con_methods=con_methods if n_jobs == 1 else None,
            n_signals=n_signals, n_times=n_times,
            accumulate_inplace=True if n_jobs == 1 else False)
        call_params.update(**spectral_params)

        if n_jobs == 1:
            # no parallel processing
            for this_epoch in epoch_block:
                logger.info('    computing connectivity for epoch %d'
                            % (epoch_idx + 1))
                # con methods and psd are updated inplace
                _epoch_spectral_connectivity(data=this_epoch, **call_params)
                epoch_idx += 1
        else:
            # process epochs in parallel
            logger.info('    computing connectivity for epochs %d..%d'
                        % (epoch_idx + 1, epoch_idx + len(epoch_block)))

            out = parallel(my_epoch_spectral_connectivity(
                           data=this_epoch, **call_params)
                           for this_epoch in epoch_block)
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
    con = list()
    for method, n_args in zip(con_methods, n_comp_args):
        # future estimators will need to be handled here
        if n_args == 3:
            # compute all scores at once
            method.compute_con(slice(0, n_cons), n_epochs)
        elif n_args == 5:
            # compute scores block-wise to save memory
            for i in range(0, n_cons, block_size):
                con_idx = slice(i, i + block_size)
                psd_xx = psd[idx_map[0][con_idx]]
                psd_yy = psd[idx_map[1][con_idx]]
                method.compute_con(con_idx, n_epochs, psd_xx, psd_yy)
        else:
            raise RuntimeError('This should never happen.')

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
        logger.info('    assembling connectivity matrix '
                    '(filling the upper triangular region of the matrix)')
        con_flat = con
        con = list()
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


def _prepare_connectivity(epoch_block, tmin, tmax, fmin, fmax, sfreq, indices,
                          mode, fskip, n_bands,
                          cwt_freqs, faverage):
    """Check and precompute dimensions of results data."""
    first_epoch = epoch_block[0]

    # get the data size and time scale
    n_signals, n_times_in, times_in = _get_and_verify_data_sizes(first_epoch)

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
        logger.info('only using indices for lower-triangular matrix')
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
    if mode in ('multitaper', 'fourier'):
        # fmin fmax etc is only supported for these modes
        # decide which frequencies to keep
        freqs_all = rfftfreq(n_times, 1. / sfreq)
    elif mode == 'cwt_morlet':
        # cwt_morlet mode
        if cwt_freqs is None:
            raise ValueError('define frequencies of interest using '
                             'cwt_freqs')
        else:
            cwt_freqs = cwt_freqs.astype(np.float64)
        if any(cwt_freqs > (sfreq / 2.)):
            raise ValueError('entries in cwt_freqs cannot be '
                             'larger than Nyquist (sfreq / 2)')
        freqs_all = cwt_freqs
    else:
        raise ValueError('mode has an invalid value')

    # check that fmin corresponds to at least 5 cycles
    dur = float(n_times) / sfreq
    five_cycle_freq = 5. / dur
    if len(fmin) == 1 and fmin[0] == -np.inf:
        # we use the 5 cycle freq. as default
        fmin = np.array([five_cycle_freq])
    else:
        if np.any(fmin < five_cycle_freq):
            warn('fmin=%0.3f Hz corresponds to %0.3f < 5 cycles '
                 'based on the epoch length %0.3f sec, need at least %0.3f '
                 'sec epochs or fmin=%0.3f. Spectrum estimate will be '
                 'unreliable.' % (np.min(fmin), dur * np.min(fmin), dur,
                                  5. / np.min(fmin), five_cycle_freq))

    # create a frequency mask for all bands
    freq_mask = np.zeros(len(freqs_all), dtype=bool)
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

    return (n_cons, times, n_times, times_in, n_times_in, tmin_idx,
            tmax_idx, n_freqs, freq_mask, freqs, freqs_bands, freq_idx_bands,
            n_signals, indices_use)


def _assemble_spectral_params(mode, n_times, mt_adaptive, mt_bandwidth, sfreq,
                              mt_low_bias, cwt_n_cycles, cwt_freqs,
                              freqs, freq_mask):
    """Prepare time-frequency decomposition."""
    spectral_params = dict(
        eigvals=None, window_fun=None, wavelets=None)
    n_tapers = None
    n_times_spectrum = 0
    if mode == 'multitaper':
        window_fun, eigvals, mt_adaptive = _compute_mt_params(
            n_times, sfreq, mt_bandwidth, mt_low_bias, mt_adaptive)
        spectral_params.update(window_fun=window_fun, eigvals=eigvals)
    elif mode == 'fourier':
        logger.info('    using FFT with a Hanning window to estimate '
                    'spectra')
        spectral_params.update(window_fun=np.hanning(n_times), eigvals=1.)
    elif mode == 'cwt_morlet':
        logger.info('    using CWT with Morlet wavelets to estimate '
                    'spectra')

        # reformat cwt_n_cycles if we have removed some frequencies
        # using fmin, fmax, fskip
        cwt_n_cycles = np.array((cwt_n_cycles,), dtype=float).ravel()
        if len(cwt_n_cycles) > 1:
            if len(cwt_n_cycles) != len(cwt_freqs):
                raise ValueError('cwt_n_cycles must be float or an '
                                 'array with the same size as cwt_freqs')
            cwt_n_cycles = cwt_n_cycles[freq_mask]

        # get the Morlet wavelets
        spectral_params.update(
            wavelets=morlet(sfreq, freqs,
                            n_cycles=cwt_n_cycles, zero_mean=True))
        n_times_spectrum = n_times
    else:
        raise ValueError('mode has an invalid value')
    return spectral_params, mt_adaptive, n_times_spectrum, n_tapers
