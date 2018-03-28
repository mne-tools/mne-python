# -*- coding: utf-8 -*-
# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Susanna Aro <susanna.aro@aalto.fi>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import copy as cp
import numbers

import numpy as np
from .tfr import cwt, morlet
from ..io.pick import pick_types, pick_channels
from ..utils import logger, verbose, warn, copy_function_doc_to_method_doc
from ..viz.misc import plot_csd
from ..time_frequency.multitaper import (dpss_windows, _mt_spectra,
                                         _csd_from_mt, _psd_from_mt_adaptive)
from ..parallel import parallel_func
from ..externals.h5io import read_hdf5, write_hdf5


class CrossSpectralDensity(object):
    """Cross-spectral density.

    Given a list of time series, the CSD matrix denotes for each pair of time
    series, the cross-spectral density. This matrix is symmetric and internally
    stored as a vector.

    This object can store multiple CSD matrices: one for each frequency.
    Use ``.get_data(freq)`` to obtain an CSD matrix as an ndarray.

    Parameters
    ----------
    data : ndarray, shape ((n_channels**2 + n_channels) / 2, n_frequencies)
        For each frequency, the cross-spectral density matrix in vector format.
    names : list of string
        List of string descriptions for each time series (e.g. channel names)
    tmin : float
        Start of the time window for which CSD was calculated in seconds.
    tmax : float
        End of the time window for which CSD was calculated in seconds.
    frequencies : float | list of float | list of list of float
        Frequency or frequencies for which the CSD matrix was calculated. When
        averaging across frequencies (see the :func:`CrossSpectralDensity.mean`
        function), this will be a list of lists that contains for each
        frequency bin, the frequencies that were averaged. Frequencies should
        always be sorted.
    n_fft : int
        The number of FFT points or samples that have been used in the
        computation of this CSD.
    projs : list of Projection | None
        List of projectors to apply to timeseries data when using this CSD
        object to compute a DICS beamformer. Defaults to None, which means no
        projectors will be applied.

    See Also
    --------
    csd_epochs
    csd_array
    """

    def __init__(self, data, names, tmin, tmax, frequencies, n_fft,
                 projs=None):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        elif data.ndim > 2:
            raise ValueError('`data` should be either a 1D or 2D array.')
        self._data = data

        if len(names) != _n_dims_from_triu(len(data)):
            raise ValueError('Number of names does not match the number of '
                             'time series in the CSD matrix.')
        self.names = names
        self.tmin = tmin
        self.tmax = tmax

        if isinstance(frequencies, numbers.Number):
            frequencies = [frequencies]
        if len(frequencies) != data.shape[1]:
            raise ValueError('Number of frequencies does not match the number '
                             'of CSD matrices in the data array (%d != %d).' %
                             (len(frequencies), data.shape[1]))
        self.frequencies = frequencies

        self.n_fft = n_fft
        self.projs = cp.deepcopy(projs)

    @property
    def n_channels(self):
        """Number of time series defined in this CSD object."""
        return len(self.names)

    @property
    def is_sum(self):
        """Whether the CSD matrix represents a sum (or average) of freqs."""
        # If the CSD is an average, the frequencies will be stored as a list
        # of lists (or like-like objects) instead of plain numbers.
        return not isinstance(self.frequencies[0], numbers.Number)

    def __repr__(self):  # noqa: D105
        # Make a pretty string representation of the frequencies
        freq_strs = []
        for f in self.frequencies:
            if isinstance(f, numbers.Number):
                freq_strs.append(str(f))
            elif len(f) == 1:
                freq_strs.append(str(f[0]))
            else:
                freq_strs.append('{}-{}'.format(np.min(f), np.max(f)))
        freq_str = ', '.join(freq_strs) + ' Hz.'

        return (
            '<CrossSpectralDensity  |  '
            'n_channels={}, time={} to {} s, frequencies={}>'
        ).format(self.n_channels, self.tmin, self.tmax, freq_str)

    def sum(self, fmin=None, fmax=None):
        """Calculate the sum CSD in the given frequency range(s).

        If the exact given frequencies are not available, the nearest
        frequencies will be chosen.

        Parameters
        ----------
        fmin : float | list of float | None
            Lower bound of the frequency range in Hertz. Defaults to the lowest
            frequency available. When a list of frequencies is given, these are
            used as the lower bounds (inclusive) of frequency bins and the sum
            is taken for each bin.
        fmax : float | list of float | None
            Upper bound of the frequency range in Hertz. Defaults to the
            highest frequency available. When a list of frequencies is given,
            these are used as the upper bounds (inclusive) of frequency bins
            and the sum is taken for each bin.

        Returns
        -------
        csd : Instance of CrossSpectralDensity
            The CSD matrix, summed across the given frequency range(s).
        """
        if self.is_sum:
            raise RuntimeError('This CSD matrix already represents a mean or '
                               'sum across frequencies.')

        # Deal with the various ways in which fmin and fmax can be specified
        if fmin is None and fmax is None:
            fmin = [self.frequencies[0]]
            fmax = [self.frequencies[-1]]
        else:
            if isinstance(fmin, numbers.Number):
                fmin = [fmin]
            if isinstance(fmax, numbers.Number):
                fmax = [fmax]
            if fmin is None:
                fmin = [self.frequencies[0]] * len(fmax)
            if fmax is None:
                fmax = [self.frequencies[-1]] * len(fmin)

        if any(fmin_ > fmax_ for fmin_, fmax_ in zip(fmin, fmax)):
            raise ValueError('Some lower bounds are higher than the '
                             'corresponding upper bounds.')

        # Find the index of the lower bound of each frequency bin
        fmin_inds = [self._get_frequency_index(f) for f in fmin]
        fmax_inds = [self._get_frequency_index(f) + 1 for f in fmax]

        if len(fmin_inds) != len(fmax_inds):
            raise ValueError('The length of fmin does not match the '
                             'length of fmax.')

        # Sum across each frequency bin
        n_bins = len(fmin_inds)
        new_data = np.zeros((self._data.shape[0], n_bins),
                            dtype=self._data.dtype)
        new_frequencies = []
        for i, (min_ind, max_ind) in enumerate(zip(fmin_inds, fmax_inds)):
            new_data[:, i] = self._data[:, min_ind:max_ind].sum(axis=1)
            new_frequencies.append(self.frequencies[min_ind:max_ind])

        csd_out = CrossSpectralDensity(data=new_data, names=self.names,
                                       tmin=self.tmin, tmax=self.tmax,
                                       frequencies=new_frequencies,
                                       n_fft=self.n_fft)
        return csd_out

    def mean(self, fmin=None, fmax=None):
        """Calculate the mean CSD in the given frequency range(s).

        Parameters
        ----------
        fmin : float | list of float | None
            Lower bound of the frequency range in Hertz. Defaults to the lowest
            frequency available. When a list of frequencies is given, these are
            used as the lower bounds (inclusive) of frequency bins and the mean
            is taken for each bin.
        fmax : float | list of float | None
            Upper bound of the frequency range in Hertz. Defaults to the
            highest frequency available. When a list of frequencies is given,
            these are used as the upper bounds (inclusive) of frequency bins
            and the mean is taken for each bin.

        Returns
        -------
        csd : Instance of CrossSpectralDensity
            The CSD matrix, averaged across the given frequency range(s).
        """
        csd = self.sum(fmin, fmax)
        for i, f in enumerate(csd.frequencies):
            csd._data[:, i] /= len(f)
        return csd

    def _get_frequency_index(self, freq):
        """Find the index of the given frequency in ``self.frequencies``.

        If the exact given frequency is not available, the nearest frequencies
        will be chosen, up to a difference of 1 Hertz.

        Parameters
        ----------
        freq : float
            The frequency to find the index for

        Returns
        -------
        index : int
            The index of the frequency nearest to the requested frequency.
        """
        if self.is_sum:
            raise ValueError('This CSD object represents a mean across '
                             'frequencies. Cannot select a specific '
                             'frequency.')

        distance = np.abs(np.asarray(self.frequencies) - freq)
        index = np.argmin(distance)
        min_dist = distance[index]
        if min_dist > 1:
            raise IndexError('Frequency %f is not available.' % freq)
        return index

    def pick_frequency(self, freq=None, index=None):
        """Get a CrossSpectralDensity object with only the given frequency.

        Parameters
        ----------
        freq : float | None
            Return the CSD matrix for a specific frequency. Only available
            when no averaging across frequencies has been done.
        index : int | None
            Return the CSD matrix for the frequency or frequency-bin with the
            given index.

        Returns
        -------
        csd : instance of CrossSpectralDensity
            A CSD object containing a single CSD matrix that corresponds to the
            requested frequency or frequency-bin.

        See Also
        --------
        get_data
        """
        if freq is None and index is None:
            raise ValueError('Use either the "freq" or "index" parameter to '
                             'select the desired frequency.')

        elif freq is not None:
            if index is not None:
                raise ValueError('Cannot specify both a frequency and index.')

            index = self._get_frequency_index(freq)

        return self[index]

    def get_data(self, frequency=None, index=None):
        """Get the CSD matrix for a given frequency as NumPy array.

        If there is only one matrix defined in the CSD object, calling this
        method without any parameters will return it. If multiple matrices are
        defined, use either the ``frequency`` or ``index`` parameter to select
        one.

        Parameters
        ----------
        frequency : float | None
            Return the CSD matrix for a specific frequency. Only available when
            no averaging across frequencies has been done.
        index : int | None
            Return the CSD matrix for the frequency or frequency-bin with the
            given index.

        Returns
        -------
        csd : ndarray, shape (n_channels, n_channels)
            The CSD matrix corresponding to the requested frequency.

        See Also
        --------
        pick_frequency
        """
        if frequency is None and index is None:
            if self._data.shape[1] > 1:
                raise ValueError('Specify either the frequency or index of '
                                 'the frequency bin for which to obtain the '
                                 'CSD matrix.')
            index = 0
        elif frequency is not None:
            if index is not None:
                raise ValueError('Cannot specify both a frequency and index.')
            index = self._get_frequency_index(frequency)

        return _vector_to_sym_mat(self._data[:, index])

    @copy_function_doc_to_method_doc(plot_csd)
    def plot(self, info=None, mode='csd', colorbar=True, cmap='viridis',
             n_cols=None, show=True):
        return plot_csd(self, info=info, mode=mode, colorbar=colorbar,
                        cmap=cmap, n_cols=n_cols, show=show)

    def __setstate__(self, state):  # noqa: D105
        self._data = state['data']
        self.tmin = state['tmin']
        self.tmax = state['tmax']
        self.names = state['names']
        self.frequencies = state['frequencies']
        self.n_fft = state['n_fft']

    def __getstate__(self):  # noqa: D105
        return dict(
            data=self._data,
            tmin=self.tmin,
            tmax=self.tmax,
            names=self.names,
            frequencies=self.frequencies,
            n_fft=self.n_fft,
        )

    def __getitem__(self, sel):  # noqa: D105
        return CrossSpectralDensity(
            data=self._data[:, sel], names=self.names, tmin=self.tmin,
            tmax=self.tmax,
            frequencies=np.atleast_1d(self.frequencies)[sel].tolist(),
            n_fft=self.n_fft,
        )

    def save(self, fname):
        """Save the CSD to an HDF5 file.

        Parameters
        ----------
        fname : str
            The name of the file to save the CSD to. The extension '.h5' will
            be appended if the given filename doesn't have it already.

        See Also
        --------
        read_csd : For reading CSD objects from a file.
        """
        if not fname.endswith('.h5'):
            fname += '.h5'

        write_hdf5(fname, self.__getstate__(), overwrite=True, title='conpy')

    def copy(self):
        """Return copy of the CrossSpectralDensity object."""
        return cp.deepcopy(self)


def _n_dims_from_triu(n):
    """Compute matrix dims from number of elements in the upper triangle.

    Parameters
    ----------
    n : int
        Number of elements in the upper triangle of the symmetric matrix.

    Returns
    -------
    dim : int
        The dimensions of the symmetric matrix.
    """
    return int(np.ceil(np.sqrt(n * 2))) - 1


def _vector_to_sym_mat(vec):
    """Convert vector to a symmetric matrix.

    The upper triangle of the matrix (including the diagonal) will be filled
    with the values of the vector.

    Parameters
    ----------
    vec : list or 1d-array
        The vector to convert to a symmetric matrix.

    Returns
    -------
    mat : 2d-array
        The symmetric matrix.

    See Also
    --------
    _sym_mat_to_vector
    """
    dim = _n_dims_from_triu(len(vec))
    mat = np.zeros((dim, dim) + vec.shape[1:], dtype=vec.dtype)

    # Fill the upper triangle of the matrix
    mat[np.triu_indices(dim)] = vec

    # Fill out the lower triangle (make conjugate to ensure matix is hermitian)
    mat = mat + np.rollaxis(mat, 1).conj()

    # We counted the diagonal twice
    if np.issubdtype(mat.dtype, np.integer):
        mat[np.diag_indices(dim)] //= 2
    else:
        mat[np.diag_indices(dim)] /= 2

    return mat


def _sym_mat_to_vector(mat):
    """Convert a symmetric matrix to a vector.

    The upper triangle of the matrix (including the diagonal) will be used
    as the values of the vector.

    Parameters
    ----------
    mat : 2d-array
        The symmetric matrix to convert to a vector

    Returns
    -------
    vec : 1d-array
        A vector consisting of the values of the upper triangle of the matrix.

    See Also
    --------
    _vector_to_sym_mat
    """
    return mat[np.triu_indices_from(mat)]


def read_csd(fname):
    """Read a CrossSpectralDensity object from an HDF5 file.

    Parameters
    ----------
    fname : str
        The name of the file to read the CSD from. The extension '.h5' will be
        appended if the given filename doesn't have it already.

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The CSD that was stored in the file.

    See Also
    --------
    CrossSpectralDensity.save : For saving CSD objects
    """
    if not fname.endswith('.h5'):
        fname += '.h5'

    csd_dict = read_hdf5(fname, title='conpy')
    return CrossSpectralDensity(**csd_dict)


def pick_channels_csd(csd, include=[], exclude=[]):
    """Pick channels from covariance matrix.

    Parameters
    ----------
    csd : instance of CrossSpectralDensity
        The CSD object to select the channels from.
    include : list of string
        List of channels to include (if empty, include all available).
    exclude : list of string
        Channels to exclude (if empty, do not exclude any).

    Returns
    -------
    res : instance of CrossSpectralDensity
        Cross-spectral density restricted to selected channels.
    """
    sel = pick_channels(csd.names, include=include, exclude=exclude)
    data = []
    for vec in csd._data.T:
        mat = _vector_to_sym_mat(vec)
        mat = mat[sel, :][:, sel]
        data.append(_sym_mat_to_vector(mat))
    names = [csd.names[i] for i in sel]

    return CrossSpectralDensity(
        data=np.array(data).T,
        names=names,
        tmin=csd.tmin,
        tmax=csd.tmax,
        frequencies=csd.frequencies,
        n_fft=csd.n_fft,
    )


@verbose
def csd_epochs(epochs, mode='multitaper', fmin=0, fmax=np.inf,
               frequencies=None, fsum=True, tmin=None, tmax=None, n_fft=None,
               mt_bandwidth=None, mt_adaptive=False, mt_low_bias=True,
               cwt_n_cycles=7, decim=1, picks=None, projs=None, n_jobs=1,
               verbose=None):
    """Estimate cross-spectral density from epochs.

    The cross-spectral density (CSD) is the covariance between two signals, for
    example the activity recorded at two sensors, in the frequency domain. This
    function computes CSD matrices that contain the CSD between all channels
    defined in the epochs object, for multiple frequencies.

    Note: Baseline correction should be used when creating the Epochs.
          Otherwise the computed cross-spectral density will be inaccurate.

    Note: Results are scaled by sampling frequency for compatibility with
          Matlab.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    mode : 'multitaper' | 'fourier' | 'cwt_morlet'
        Spectrum estimation mode. Defaults to 'multitaper'.
    fmin : float | None
        Minimum frequency of interest, in Hertz.
        Only used in 'multitaper' or 'fourier' mode. For 'cwt_morlet' mode, use
        the ``frequencies`` parameter instead.
    fmax : float | np.inf | None
        Maximum frequency of interest, in Hertz.
        Only used in 'multitaper' or 'fourier' mode. For 'cwt_morlet' mode, use
        the ``frequencies`` parameter instead.
    frequencies : list of float | None
        The frequencies of interest, in Hertz. For each frequency, a Morlet
        wavelet will be created. Only used in 'cwt_morlet' mode. For other
        modes, use the ``fmin`` and ``fmax`` parameters instead.
    fsum : bool
        Sum CSD values for the frequencies of interest. Summing is performed
        instead of averaging so that accumulated power is comparable to power
        in the time domain. Defaults to True.

        .. note:: Summing or averaging across frequencies can also be performed
                  later using the :meth:`CrossSpectralDensity.sum` and
                  :meth:`CrossSpectralDensity.mean` methods.
    tmin : float | None
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    n_fft : int | None
        Length of the FFT. If ``None``, the exact number of samples between
        ``tmin`` and ``tmax`` will be used.
        Only used in 'multitaper' or 'fourier' mode.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
        Only used in 'multitaper' mode.
    cwt_n_cycles: float | list of float | None
        Number of cycles to use when constructing Morlet wavelets. Fixed number
        or one per frequency. Defaults to 7.
        Only used in 'cwt_morlet' mode.
    decim : int | slice
        To reduce memory usage, decimation factor during time-frequency
        decomposition. Defaults to 1 (no decimation).
        Only used in 'cwt_morlet' mode.

        If `int`, uses tfr[..., ::decim].
        If `slice`, uses tfr[..., decim].

    picks : list of int | None
        The indices of the channels to include in the CSD computation. By
        default, only MEG and EEG channels are used (excluding bad channels).
    projs : list of Projection | None
        List of projectors to store in the CSD object, or None to indicate that
        the projectors from the epochs should be inherited. Defaults to None.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.
    verbose : bool | str | int | None
        If not ``None``, override default verbose level
        (see :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
        for more).

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.

    See Also
    --------
    csd_array
    """
    # Portions of this code adapted from mne/connectivity/spectral.py

    # Check correctness of input data and parameters
    if epochs.baseline is None and epochs.info['highpass'] < 0.1:
        warn('Epochs are not baseline corrected or enough highpass filtered. '
             'Cross-spectral density may be inaccurate.')

    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, eog=False,
                           ref_meg=False, exclude='bads')
    ch_names = [epochs.ch_names[k] for k in picks]

    # Apply time window and channel selection
    epochs = epochs.copy().pick_channels(ch_names)

    sfreq = epochs.info['sfreq']

    if projs is None:
        projs = epochs.info['projs']

    return csd_array(X=epochs.get_data(), sfreq=sfreq, t0=epochs.tmin,
                     mode=mode, fmin=fmin, fmax=fmax, frequencies=frequencies,
                     fsum=fsum, tmin=tmin, tmax=tmax, names=ch_names,
                     n_fft=n_fft, mt_bandwidth=mt_bandwidth,
                     mt_adaptive=mt_adaptive, mt_low_bias=mt_low_bias,
                     projs=projs, decim=decim, cwt_n_cycles=cwt_n_cycles,
                     n_jobs=n_jobs, verbose=verbose)


@verbose
def csd_array(X, sfreq, t0=0, mode='multitaper', fmin=0, fmax=np.inf,
              frequencies=None, fsum=True, tmin=None, tmax=None, names=None,
              n_fft=None, mt_bandwidth=None, mt_adaptive=False,
              mt_low_bias=True, cwt_n_cycles=7, decim=1, projs=None, n_jobs=1,
              verbose=None):
    """Estimate cross-spectral density from an array.

    The cross-spectral density (CSD) is the covariance between two signals, for
    example the activity recorded at two sensors, in the frequency domain. This
    function computes CSD matrices that contain the CSD between all signals
    defined in the given array, for multiple frequencies.

    .. note:: Results are scaled by sampling frequency for compatibility with
              Matlab.

    Parameters
    ----------
    X : array-like, shape (n_epochs, n_channels, n_times)
        The time series data consisting of n_epochs separate observations
        of signals with n_channels time-series of length n_times.
    sfreq : float
        Sampling frequency of observations.
    t0 : float
        Time of the first sample relative to the onset of the epoch, in
        seconds. Defaults to 0.
    mode : 'multitaper' | 'fourier' | 'cwt_morlet'
        Spectrum estimation mode. Defaults to 'multitaper'.
    fmin : float | None
        Minimum frequency of interest, in Hertz.
        Only used in 'multitaper' or 'fourier' mode. For 'cwt_morlet' mode, use
        the ``frequencies`` parameter instead.
    fmax : float | np.inf | None
        Maximum frequency of interest, in Hertz.
        Only used in 'multitaper' or 'fourier' mode. For 'cwt_morlet' mode, use
        the ``frequencies`` parameter instead.
    frequencies : list of float | None
        The frequencies of interest, in Hertz.
        Only used in 'cwt_morlet' mode. For other modes, use the ``fmin`` and
        ``fmax`` parameters instead.
    fsum : bool
        Sum CSD values for the frequencies of interest. Summing is performed
        instead of averaging so that accumulated power is comparable to power
        in the time domain. Defaults to True.

        .. note:: Summing or averaging across frequencies can also be performed
                  later using the :meth:`CrossSpectralDensity.sum` and
                  :meth:`CrossSpectralDensity.mean` methods.
    tmin : float | None
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    names : list of str | None
        A name for each time series. If ``None`` (the default), the series will
        be named 'SERIES###'.
    n_fft : int | None
        Length of the FFT. If ``None``, the exact number of samples between
        ``tmin`` and ``tmax`` will be used.
        Only used in 'multitaper' or 'fourier' mode.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
        Only used in 'multitaper' mode.
    cwt_n_cycles: float | list of float | None
        Number of cycles to use when constructing Morlet wavelets. Fixed number
        or one per frequency. Defaults to 7.
        Only used in 'cwt_morlet' mode.
    decim : int | slice
        To reduce memory usage, decimation factor during time-frequency
        decomposition. Defaults to 1 (no decimation).
        Only used in 'cwt_morlet' mode.

        If `int`, uses tfr[..., ::decim].
        If `slice`, uses tfr[..., decim].

    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to None, which
        means no projectors are stored.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.
    verbose : bool | str | int | None
        If not ``None``, override default verbose level
        (see :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
        for more).

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.

    See Also
    --------
    csd_epochs
    """
    from scipy.fftpack import fftfreq  # Local import to keep "import mne" fast

    # Check correctness of input data and parameters
    if mode == 'cwt_morlet' and frequencies is None:
        raise ValueError('When using "cwt_morlet" mode, you need to specify '
                         'the "frequencies" parameter.')
    else:
        if fmax < fmin:
            raise ValueError('fmax must be larger than fmin')
    if tmax is not None and tmin is not None:
        if tmax < tmin:
            raise ValueError('tmax must be larger than tmin')

    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X must be n_replicates x n_channels x n_times.")

    n_replicates, n_channels, n_times = X.shape
    tstep = 1. / sfreq
    times = np.arange(n_times) * tstep + t0
    if names is None:
        names = ['SERIES%03d' % (i + 1) for i in range(n_channels)]

    if tmin is not None and tmin < times[0] - tstep:
        raise ValueError('tmin should be larger than the smallest data time '
                         'point')
    if tmax is not None and tmax > times[-1] + tstep:
        raise ValueError('tmax should be smaller than the largest data time '
                         'point')

    # Preparing for computing CSD
    logger.info('Computing cross-spectral density from epochs...')

    if mode == 'multitaper' or mode == 'fourier':
        # Slice X to the requested time window
        tstart = None if tmin is None else np.searchsorted(times, tmin - 1e-10)
        tstop = None if tmax is None else np.searchsorted(times, tmax + 1e-10)
        X = X[:, :, tstart:tstop]
        times = times[tstart:tstop]
        n_times = len(times)
        n_fft = n_times if n_fft is None else n_fft

        window_fun, eigvals, n_tapers, mt_adaptive = _compute_mt_params(
            n_times, sfreq, mode, mt_bandwidth, mt_low_bias, mt_adaptive)

        # Preparing frequencies of interest
        orig_frequencies = fftfreq(n_fft, 1. / sfreq)
        freq_mask = (orig_frequencies > fmin) & (orig_frequencies < fmax)

        mt_frequencies = np.fft.rfftfreq(n_fft, 1. / sfreq)
        freq_mask_mt = (mt_frequencies > fmin) & (mt_frequencies < fmax)
        frequencies = mt_frequencies[freq_mask_mt]

        if len(frequencies) == 0:
            raise ValueError('No discrete fourier transform results within '
                             'the given frequency window. Please widen either '
                             'the frequency window or the time window')

        parallel, my_csd, _ = parallel_func(_csd_multitaper, n_jobs,
                                            verbose=verbose)

    elif mode == 'cwt_morlet':
        # Construct the appropriate Morlet wavelets
        wavelets = morlet(sfreq, frequencies, cwt_n_cycles)
        n_fft = 1  # _csd_morlet averages across time instead of summing

        # Slice X to the requested time window + half the length of the longest
        # wavelet.
        wave_length = len(wavelets[np.argmin(frequencies)]) // 2
        tstart = tstop = None
        if tmin is not None:
            tstart = np.searchsorted(times, tmin)
            tstart = max(0, tstart - wave_length)
        if tmax is not None:
            tstop = np.searchsorted(times, tmax)
            tstop = min(n_times, tstop + wave_length)
        X = X[:, :, tstart:tstop]
        times = times[tstart:tstop]
        n_times = len(times)

        # After CSD computation, we slice again to the requested time window.
        csd_tstart = None if tmin is None else np.searchsorted(times,
                                                               tmin - 1e-10)
        csd_tstop = None if tmax is None else np.searchsorted(times,
                                                              tmax + 1e-10)
        csd_tslice = slice(csd_tstart, csd_tstop)

        parallel, my_csd, _ = parallel_func(_csd_morlet, n_jobs,
                                            verbose=verbose)
    else:
        raise ValueError("The mode parameter must be either 'cwt_morlet', "
                         "'multitaper' or 'fourier'.")

    n_freqs = len(frequencies)
    n_freqs_in_csd = 1 if fsum else n_freqs
    csds_mean = np.zeros((n_channels * (n_channels + 1) // 2, n_freqs_in_csd),
                         dtype=np.complex)

    # Compute CSD for each trial
    n_blocks = int(np.ceil(n_replicates / float(n_jobs)))
    for i in range(n_blocks):
        epoch_block = X[i * n_jobs:(i + 1) * n_jobs]
        if n_jobs > 1:
            logger.info('    Computing CSD matrices for epochs %d..%d'
                        % (i * n_jobs + 1, (i + 1) * n_jobs))
        else:
            logger.info('    Computing CSD matrix for epoch %d' % (i + 1))

        if mode == 'multitaper' or mode == 'fourier':
            # Calculating Fourier transform using multitaper module
            csds = parallel(my_csd(this_epoch, sfreq, n_times, window_fun,
                                   eigvals, freq_mask, freq_mask_mt, n_fft,
                                   mode, mt_adaptive)
                            for this_epoch in epoch_block)

        elif mode == 'cwt_morlet':
            csds = parallel(my_csd(this_epoch, sfreq, wavelets, csd_tslice,
                                   decim)
                            for this_epoch in epoch_block)

        if fsum:  # Sum across frequencies
            csds = np.sum(csds, axis=2, keepdims=True)

        # Add CSD matrices in-place
        csds_mean += np.sum(csds, axis=0)

    csds_mean /= n_replicates
    logger.info('[done]')

    if fsum:  # CSD was computed over a frequency band
        frequencies = [frequencies]

    return CrossSpectralDensity(csds_mean, names=names, tmin=times[0],
                                tmax=times[-1], frequencies=frequencies,
                                n_fft=n_fft, projs=projs)


def _compute_mt_params(n_times, sfreq, mode, mt_bandwidth, mt_low_bias,
                       mt_adaptive):
    """Compute windowing and multitaper parameters.

    Parameters
    ----------
    n_times : int
        Number of time points.
    s_freq : int
        Sampling frequency of signal.
    mode : str
        Spectrum estimation mode can be either: 'multitaper' or 'fourier'.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.

    Returns
    -------
    window_fun : ndarray
        Window function(s) of length n_times. When 'multitaper' mode is used
        will correspond to first output of `dpss_windows` and when 'fourier'
        mode is used will be a Hanning window of length `n_times`.
    eigvals : ndarray | float
        Eigenvalues associated with window functions. Only needed when mode is
        'multitaper'. When the mode 'fourier' is used this is set to 1.
    n_tapers : int | None
        Number of tapers to use. Only used when mode is 'multitaper'.
    ret_mt_adaptive : bool
        Updated value of `mt_adaptive` argument as certain parameter values
        will not allow adaptive spectral estimators.
    """
    ret_mt_adaptive = mt_adaptive
    if mode == 'multitaper':
        # Compute standardized half-bandwidth
        if mt_bandwidth is not None:
            half_nbw = float(mt_bandwidth) * n_times / (2. * sfreq)
        else:
            half_nbw = 2.

        # Compute DPSS windows
        n_tapers_max = int(2 * half_nbw)
        window_fun, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                           low_bias=mt_low_bias)
        n_tapers = len(eigvals)
        logger.info('    using multitaper spectrum estimation with %d DPSS '
                    'windows' % n_tapers)

        if mt_adaptive and len(eigvals) < 3:
            warn('Not adaptively combining the spectral estimators due to a '
                 'low number of tapers.')
            ret_mt_adaptive = False
    elif mode == 'fourier':
        logger.info('    using FFT with a Hanning window to estimate spectra')
        window_fun = np.hanning(n_times)
        ret_mt_adaptive = False
        eigvals = 1.
        n_tapers = None
    else:
        raise ValueError('Mode has an invalid value.')

    return window_fun, eigvals, n_tapers, ret_mt_adaptive


def _csd_multitaper(X, sfreq, n_times, window_fun, eigvals, freq_mask,
                    freq_mask_mt, n_fft, mode, mt_adaptive):
    """Compute cross spectral density (CSD) using multitaper module.

    Computes the CSD for a single epoch of data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        The time series data consisting of n_channels time-series of length
        n_times.
    sfreq : float
        The sampling frequency of the data in Hertz.
    n_times : int
        Number of time samples
    window_fun : ndarray
        Window function(s) of length n_times. When 'multitaper' mode is used
        will correspond to first output of `dpss_windows` and when 'fourier'
        mode is used will be a Hanning window of length `n_times`.
    eigvals : ndarray | float
        Eigenvalues associated with window functions. Only needed when mode is
        'multitaper'. When the mode 'fourier' is used this is set to 1.
    freq_mask : ndarray
        Which frequencies to use.
    freq_mask_mt : ndarray
        Which frequencies to use after multitaper pass.
    n_fft : int
        Length of the FFT.
    mode : str
        Spectrum estimation mode can be either: 'multitaper' or 'fourier'.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    """
    x_mt, _ = _mt_spectra(X, window_fun, sfreq, n_fft)

    if mt_adaptive:
        # Compute adaptive weights
        _, weights = _psd_from_mt_adaptive(x_mt, eigvals, freq_mask_mt,
                                           return_weights=True)
        # Tiling weights so that we can easily use _csd_from_mt()
        weights = weights[:, np.newaxis, :, :]
        weights = np.tile(weights, [1, x_mt.shape[0], 1, 1])
    else:
        # Do not use adaptive weights
        if mode == 'multitaper':
            weights = np.sqrt(eigvals)[np.newaxis, np.newaxis, :, np.newaxis]
        else:
            # Hack so we can sum over axis=-2
            weights = np.array([1.])[:, np.newaxis, np.newaxis, np.newaxis]

    x_mt = x_mt[:, :, freq_mask_mt]

    # Calculating CSD
    # Tiling x_mt so that we can easily use _csd_from_mt()
    x_mt = x_mt[:, np.newaxis, :, :]
    x_mt = np.tile(x_mt, [1, x_mt.shape[0], 1, 1])
    y_mt = np.transpose(x_mt, axes=[1, 0, 2, 3])
    weights_y = np.transpose(weights, axes=[1, 0, 2, 3])
    csds = _csd_from_mt(x_mt, y_mt, weights, weights_y)

    # FIXME: don't compute full matrix in the first place
    csds = np.array([_sym_mat_to_vector(csds[:, :, i])
                     for i in range(csds.shape[-1])]).T

    # Scaling by number of samples and compensating for loss of power
    # due to windowing (see section 11.5.2 in Bendat & Piersol).
    if mode == 'fourier':
        csds /= n_times
        csds *= 8 / 3.

    # Scaling by sampling frequency for compatibility with Matlab
    csds /= sfreq

    return csds


def _csd_morlet(data, sfreq, wavelets, tslice=None, decim=1):
    """Compute cross spectral density (CSD) using the given Morlet wavelets.

    Computes the CSD for a single epoch of data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        The time series data consisting of n_channels time-series of length
        n_times.
    sfreq : float
        The sampling frequency of the data in Hertz.
    wavelets : list of ndarray
        The Morlet wavelets for which to compute the CSD's. These have been
        created by the `mne.time_frequency.tfr.morlet` function.
    tslice : slice | None
        The desired time samples to compute the CSD over. If None, defaults to
        including all time samples.
    decim : int | slice
        To reduce memory usage, decimation factor during time-frequency
        decomposition. Defaults to 1 (no decimation).
        Only used in 'cwt_morlet' mode.

        If `int`, uses tfr[..., ::decim].
        If `slice`, uses tfr[..., decim].

    Returns
    -------
    csd : ndarray, shape ((n_channels**2 + n_channels) / 2 , n_wavelets)
        For each wavelet, the upper triangle of the cross spectral density
        matrix.

    See Also
    --------
    _vector_to_sym_mat : For converting the CSD to a full matrix
    """
    # Remove best straight-line fit
    from scipy.signal import detrend  # Local import to keep "import mne" fast
    data = detrend(data)

    # Compute PSD
    psds = cwt(data, wavelets, use_fft=True, decim=decim)

    if tslice is not None:
        tstart = None if tslice.start is None else tslice.start // decim
        tstop = None if tslice.stop is None else tslice.stop // decim
        tstep = None if tslice.step is None else tslice.step // decim
        tslice = slice(tstart, tstop, tstep)
        psds = psds[:, :, tslice]

    psds_conj = np.conj(psds)

    # Compute the spectral density between all pairs of series
    n_channels = data.shape[0]
    csds = np.vstack([psds[[i]] * psds_conj[i:] for i in range(n_channels)])

    # Average along time dimension
    csds = csds.mean(axis=2)

    # Scaling by sampling frequency for compatibility with Matlab
    csds /= sfreq

    return csds
