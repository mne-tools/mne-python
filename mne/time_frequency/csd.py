# -*- coding: utf-8 -*-
# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Susanna Aro <susanna.aro@aalto.fi>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD-3-Clause

import copy as cp
import numbers

import numpy as np

from .tfr import _cwt_array, morlet, _get_nfft, EpochsTFR
from ..io.pick import pick_channels, _picks_to_idx
from ..utils import (logger, verbose, warn, copy_function_doc_to_method_doc,
                     ProgressBar, _check_fname, _import_h5io_funcs,
                     _validate_type)
from ..viz.misc import plot_csd
from ..time_frequency.multitaper import (_compute_mt_params, _mt_spectra,
                                         _csd_from_mt, _psd_from_mt_adaptive)
from ..parallel import parallel_func


def pick_channels_csd(csd, include=[], exclude=[], ordered=False, copy=True):
    """Pick channels from cross-spectral density matrix.

    Parameters
    ----------
    csd : instance of CrossSpectralDensity
        The CSD object to select the channels from.
    include : list of str
        List of channels to include (if empty, include all available).
    exclude : list of str
        Channels to exclude (if empty, do not exclude any).
    ordered : bool
        If True (default False), ensure that the order of the channels in the
        modified instance matches the order of ``include``.

        .. versionadded:: 0.20.0
    copy : bool
        If True (the default), return a copy of the CSD matrix with the
        modified channels. If False, channels are modified in-place.

        .. versionadded:: 0.20.0

    Returns
    -------
    res : instance of CrossSpectralDensity
        Cross-spectral density restricted to selected channels.
    """
    if copy:
        csd = csd.copy()

    sel = pick_channels(csd.ch_names, include=include, exclude=exclude,
                        ordered=ordered)
    data = []
    for vec in csd._data.T:
        mat = _vector_to_sym_mat(vec)
        mat = mat[sel, :][:, sel]
        data.append(_sym_mat_to_vector(mat))
    ch_names = [csd.ch_names[i] for i in sel]

    csd._data = np.array(data).T
    csd.ch_names = ch_names
    return csd


class CrossSpectralDensity(object):
    """Cross-spectral density.

    Given a list of time series, the CSD matrix denotes for each pair of time
    series, the cross-spectral density. This matrix is symmetric and internally
    stored as a vector.

    This object can store multiple CSD matrices: one for each frequency.
    Use ``.get_data(freq)`` to obtain an CSD matrix as an ndarray.

    Parameters
    ----------
    data : ndarray, shape ((n_channels**2 + n_channels) // 2, n_frequencies)
        For each frequency, the cross-spectral density matrix in vector format.
    ch_names : list of str
        List of string names for each channel.
    frequencies : float | list of float | list of list of float
        Frequency or frequencies for which the CSD matrix was calculated. When
        averaging across frequencies (see the :func:`CrossSpectralDensity.mean`
        function), this will be a list of lists that contains for each
        frequency bin, the frequencies that were averaged. Frequencies should
        always be sorted.
    n_fft : int
        The number of FFT points or samples that have been used in the
        computation of this CSD.
    tmin : float | None
        Start of the time window for which CSD was calculated in seconds. Can
        be ``None`` (the default) to indicate no timing information is
        available.
    tmax : float | None
        End of the time window for which CSD was calculated in seconds. Can be
        ``None`` (the default) to indicate no timing information is available.
    projs : list of Projection | None
        List of projectors to apply to timeseries data when using this CSD
        object to compute a DICS beamformer. Defaults to ``None``, which means
        no projectors will be applied.

    See Also
    --------
    csd_fourier
    csd_multitaper
    csd_morlet
    csd_array_fourier
    csd_array_multitaper
    csd_array_morlet
    """

    def __init__(self, data, ch_names, frequencies, n_fft, tmin=None,
                 tmax=None, projs=None):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        elif data.ndim > 2:
            raise ValueError('`data` should be either a 1D or 2D array.')
        self._data = data

        if len(ch_names) != _n_dims_from_triu(len(data)):
            raise ValueError('Number of ch_names does not match the number of '
                             'time series in the CSD matrix.')
        self.ch_names = list(ch_names)
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
        if projs is None:
            self.projs = []
        else:
            self.projs = cp.deepcopy(projs)

    @property
    def n_channels(self):
        """Number of time series defined in this CSD object."""
        return len(self.ch_names)

    @property
    def _is_sum(self):
        """Whether the CSD matrix represents a sum (or average) of freqs."""
        # If the CSD is an average, the frequencies will be stored as a list
        # of lists (or like-like objects) instead of plain numbers.
        return not isinstance(self.frequencies[0], numbers.Number)

    def __len__(self):  # noqa: D105
        """Return number of frequencies.

        Returns
        -------
        n_freqs : int
            The number of frequencies.
        """
        return len(self.frequencies)

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

        if self.tmin is not None and self.tmax is not None:
            time_str = '{} to {} s'.format(self.tmin, self.tmax)
        else:
            time_str = 'unknown'

        return (
            '<CrossSpectralDensity | '
            'n_channels={}, time={}, frequencies={}>'
        ).format(self.n_channels, time_str, freq_str)

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
        csd : instance of CrossSpectralDensity
            The CSD matrix, summed across the given frequency range(s).
        """
        if self._is_sum:
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

        csd_out = CrossSpectralDensity(data=new_data, ch_names=self.ch_names,
                                       tmin=self.tmin, tmax=self.tmax,
                                       frequencies=new_frequencies,
                                       n_fft=self.n_fft, projs=self.projs)
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
        csd : instance of CrossSpectralDensity
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
        if self._is_sum:
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

    def get_data(self, frequency=None, index=None, as_cov=False):
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
        as_cov : bool
            Whether to return the data as a numpy array (`False`, the default),
            or pack it in a :class:`mne.Covariance` object (`True`).

            .. versionadded:: 0.20

        Returns
        -------
        csd : ndarray, shape (n_channels, n_channels) | instance of Covariance
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

        data = _vector_to_sym_mat(self._data[:, index])
        if as_cov:
            # Pack the data into a Covariance object
            from ..cov import Covariance  # to avoid circular import
            return Covariance(data, self.ch_names, bads=[], projs=self.projs,
                              nfree=self.n_fft)
        else:
            return data

    @copy_function_doc_to_method_doc(plot_csd)
    def plot(self, info=None, mode='csd', colorbar=True, cmap='viridis',
             n_cols=None, show=True):
        return plot_csd(self, info=info, mode=mode, colorbar=colorbar,
                        cmap=cmap, n_cols=n_cols, show=show)

    def __setstate__(self, state):  # noqa: D105
        # Avoid circular import
        from ..proj import Projection
        self._data = state['data']
        self.tmin = state['tmin']
        self.tmax = state['tmax']
        self.ch_names = state['ch_names']
        self.frequencies = state['frequencies']
        self.n_fft = state['n_fft']
        self.projs = [Projection(**proj) for proj in state['projs']]

    def __getstate__(self):  # noqa: D105
        return dict(
            data=self._data,
            tmin=self.tmin,
            tmax=self.tmax,
            ch_names=self.ch_names,
            frequencies=self.frequencies,
            n_fft=self.n_fft,
            projs=self.projs,
        )

    def __getitem__(self, sel):  # noqa: D105
        """Subselect frequencies.

        Parameters
        ----------
        sel : ndarray
            Array of frequency indices to subselect.

        Returns
        -------
        csd : instance of CrossSpectralDensity
            A new CSD instance with the subset of frequencies.
        """
        return CrossSpectralDensity(
            data=self._data[:, sel], ch_names=self.ch_names, tmin=self.tmin,
            tmax=self.tmax,
            frequencies=np.atleast_1d(self.frequencies)[sel].tolist(),
            n_fft=self.n_fft,
            projs=self.projs,
        )

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save the CSD to an HDF5 file.

        Parameters
        ----------
        fname : str
            The name of the file to save the CSD to. The extension '.h5' will
            be appended if the given filename doesn't have it already.
        %(overwrite)s

            .. versionadded:: 1.0
        %(verbose)s

            .. versionadded:: 1.0

        See Also
        --------
        read_csd : For reading CSD objects from a file.
        """
        _, write_hdf5 = _import_h5io_funcs()
        if not fname.endswith('.h5'):
            fname += '.h5'

        fname = _check_fname(fname, overwrite=overwrite)
        write_hdf5(fname, self.__getstate__(), overwrite=True,
                   title='conpy')

    def copy(self):
        """Return copy of the CrossSpectralDensity object.

        Returns
        -------
        copy : instance of CrossSpectralDensity
            A copy of the object.
        """
        return cp.deepcopy(self)

    def pick_channels(self, ch_names, ordered=False):
        """Pick channels from this cross-spectral density matrix.

        Parameters
        ----------
        ch_names : list of str
            List of channels to keep. All other channels are dropped.
        ordered : bool
            If True (default False), ensure that the order of the channels
            matches the order of ``ch_names``.

        Returns
        -------
        csd : instance of CrossSpectralDensity.
            The modified cross-spectral density object.

        Notes
        -----
        Operates in-place.

        .. versionadded:: 0.20.0
        """
        return pick_channels_csd(self, include=ch_names, exclude=[],
                                 ordered=ordered, copy=False)


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

    # Fill out the lower tri (make conjugate to ensure matrix is hermitian)
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
    CrossSpectralDensity.save : For saving CSD objects.
    """
    read_hdf5, _ = _import_h5io_funcs()
    if not fname.endswith('.h5'):
        fname += '.h5'

    csd_dict = read_hdf5(fname, title='conpy')

    if csd_dict["projs"] is not None:
        # Avoid circular import
        from ..proj import Projection
        csd_dict["projs"] = [Projection(**proj) for proj in csd_dict["projs"]]

    return CrossSpectralDensity(**csd_dict)


@verbose
def csd_fourier(epochs, fmin=0, fmax=np.inf, tmin=None, tmax=None, picks=None,
                n_fft=None, projs=None, n_jobs=None, *, verbose=None):
    """Estimate cross-spectral density from an array using short-time fourier.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs to compute the CSD for.
    fmin : float
        Minimum frequency of interest, in Hertz.
    fmax : float | np.inf
        Maximum frequency of interest, in Hertz.
    tmin : float | None
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    %(picks_good_data_noref)s
    n_fft : int | None
        Length of the FFT. If ``None``, the exact number of samples between
        ``tmin`` and ``tmax`` will be used.
    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to ``None``,
        which means the projectors defined in the Epochs object will be copied.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.

    See Also
    --------
    csd_array_fourier
    csd_array_morlet
    csd_array_multitaper
    csd_morlet
    csd_multitaper
    """
    epochs, projs = _prepare_csd(epochs, tmin, tmax, picks, projs)
    return csd_array_fourier(epochs.get_data(), sfreq=epochs.info['sfreq'],
                             t0=epochs.tmin, fmin=fmin, fmax=fmax, tmin=tmin,
                             tmax=tmax, ch_names=epochs.ch_names, n_fft=n_fft,
                             projs=projs, n_jobs=n_jobs, verbose=verbose)


@verbose
def csd_array_fourier(X, sfreq, t0=0, fmin=0, fmax=np.inf, tmin=None,
                      tmax=None, ch_names=None, n_fft=None, projs=None,
                      n_jobs=None, *, verbose=None):
    """Estimate cross-spectral density from an array using short-time fourier.

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
    fmin : float
        Minimum frequency of interest, in Hertz.
    fmax : float | np.inf
        Maximum frequency of interest, in Hertz.
    tmin : float | None
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    ch_names : list of str | None
        A name for each time series. If ``None`` (the default), the series will
        be named 'SERIES###'.
    n_fft : int | None
        Length of the FFT. If ``None``, the exact number of samples between
        ``tmin`` and ``tmax`` will be used.
    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to ``None``,
        which means no projectors are stored.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.

    See Also
    --------
    csd_array_morlet
    csd_array_multitaper
    csd_fourier
    csd_morlet
    csd_multitaper
    """
    from scipy.fft import rfftfreq
    X, times, tmin, tmax, fmin, fmax = _prepare_csd_array(
        X, sfreq, t0, tmin, tmax, fmin, fmax)

    # Slice X to the requested time window
    tstart = None if tmin is None else np.searchsorted(times, tmin - 1e-10)
    tstop = None if tmax is None else np.searchsorted(times, tmax + 1e-10)
    X = X[:, :, tstart:tstop]
    times = times[tstart:tstop]
    n_times = len(times)
    n_fft = n_times if n_fft is None else n_fft

    # Preparing frequencies of interest
    # orig_frequencies = fftfreq(n_fft, 1. / sfreq)
    orig_frequencies = rfftfreq(n_fft, 1. / sfreq)
    freq_mask = (orig_frequencies > fmin) & (orig_frequencies < fmax)
    frequencies = orig_frequencies[freq_mask]

    if len(frequencies) == 0:
        raise ValueError('No discrete fourier transform results within '
                         'the given frequency window. Please widen either '
                         'the frequency window or the time window')

    # Compute the CSD
    return _execute_csd_function(X, times, frequencies, _csd_fourier,
                                 params=[sfreq, n_times, freq_mask, n_fft],
                                 n_fft=n_fft, ch_names=ch_names, projs=projs,
                                 n_jobs=n_jobs, verbose=verbose)


@verbose
def csd_multitaper(epochs, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                   picks=None, n_fft=None, bandwidth=None, adaptive=False,
                   low_bias=True, projs=None, n_jobs=None, *, verbose=None):
    """Estimate cross-spectral density from epochs using a multitaper method.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs to compute the CSD for.
    fmin : float | None
        Minimum frequency of interest, in Hertz.
    fmax : float | np.inf
        Maximum frequency of interest, in Hertz.
    tmin : float
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    %(picks_good_data_noref)s
    n_fft : int | None
        Length of the FFT. If ``None``, the exact number of samples between
        ``tmin`` and ``tmax`` will be used.
    bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to ``None``,
        which means the projectors defined in the Epochs object will by copied.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.

    See Also
    --------
    csd_array_fourier
    csd_array_morlet
    csd_array_multitaper
    csd_fourier
    csd_morlet
    """
    epochs, projs = _prepare_csd(epochs, tmin, tmax, picks, projs)
    return csd_array_multitaper(epochs.get_data(), sfreq=epochs.info['sfreq'],
                                t0=epochs.tmin, fmin=fmin, fmax=fmax,
                                tmin=tmin, tmax=tmax, ch_names=epochs.ch_names,
                                n_fft=n_fft, bandwidth=bandwidth,
                                adaptive=adaptive, low_bias=low_bias,
                                projs=projs, n_jobs=n_jobs, verbose=verbose)


@verbose
def csd_array_multitaper(X, sfreq, t0=0, fmin=0, fmax=np.inf, tmin=None,
                         tmax=None, ch_names=None, n_fft=None, bandwidth=None,
                         adaptive=False, low_bias=True, projs=None,
                         n_jobs=None, *, verbose=None):
    """Estimate cross-spectral density from an array using a multitaper method.

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
    fmin : float
        Minimum frequency of interest, in Hertz.
    fmax : float | np.inf
        Maximum frequency of interest, in Hertz.
    tmin : float | None
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    ch_names : list of str | None
        A name for each time series. If ``None`` (the default), the series will
        be named 'SERIES###'.
    n_fft : int | None
        Length of the FFT. If ``None``, the exact number of samples between
        ``tmin`` and ``tmax`` will be used.
    bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to ``None``,
        which means no projectors are stored.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.

    See Also
    --------
    csd_array_fourier
    csd_array_morlet
    csd_fourier
    csd_morlet
    csd_multitaper
    """
    from scipy.fft import rfftfreq
    X, times, tmin, tmax, fmin, fmax = _prepare_csd_array(
        X, sfreq, t0, tmin, tmax, fmin, fmax)

    # Slice X to the requested time window
    tstart = None if tmin is None else np.searchsorted(times, tmin - 1e-10)
    tstop = None if tmax is None else np.searchsorted(times, tmax + 1e-10)
    X = X[:, :, tstart:tstop]
    times = times[tstart:tstop]
    n_times = len(times)
    n_fft = n_times if n_fft is None else n_fft

    window_fun, eigvals, mt_adaptive = \
        _compute_mt_params(n_times, sfreq, bandwidth, low_bias, adaptive)

    # Preparing frequencies of interest
    orig_frequencies = rfftfreq(n_fft, 1. / sfreq)
    freq_mask = (orig_frequencies > fmin) & (orig_frequencies < fmax)
    frequencies = orig_frequencies[freq_mask]

    if len(frequencies) == 0:
        raise ValueError('No discrete fourier transform results within '
                         'the given frequency window. Please widen either '
                         'the frequency window or the time window')

    # Compute the CSD
    return _execute_csd_function(X, times, frequencies, _csd_multitaper,
                                 params=[sfreq, n_times, window_fun, eigvals,
                                         freq_mask, n_fft, adaptive],
                                 n_fft=n_fft, ch_names=ch_names, projs=projs,
                                 n_jobs=n_jobs, verbose=verbose)


@verbose
def csd_morlet(epochs, frequencies, tmin=None, tmax=None, picks=None,
               n_cycles=7, use_fft=True, decim=1, projs=None, n_jobs=None, *,
               verbose=None):
    """Estimate cross-spectral density from epochs using Morlet wavelets.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs to compute the CSD for.
    frequencies : list of float
        The frequencies of interest, in Hertz.
    tmin : float | None
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    %(picks_good_data_noref)s
    n_cycles : float | list of float | None
        Number of cycles to use when constructing Morlet wavelets. Fixed number
        or one per frequency. Defaults to 7.
    use_fft : bool
        Whether to use FFT-based convolution to compute the wavelet transform.
        Defaults to True.
    decim : int | slice
        To reduce memory usage, decimation factor during time-frequency
        decomposition. Defaults to 1 (no decimation).

        If `int`, uses tfr[..., ::decim].
        If `slice`, uses tfr[..., decim].

    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to ``None``,
        which means the projectors defined in the Epochs object will be copied.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.

    See Also
    --------
    csd_array_fourier
    csd_array_morlet
    csd_array_multitaper
    csd_fourier
    csd_multitaper
    """
    epochs, projs = _prepare_csd(epochs, tmin, tmax, picks, projs)
    return csd_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'],
                            frequencies=frequencies, t0=epochs.tmin, tmin=tmin,
                            tmax=tmax, ch_names=epochs.ch_names,
                            n_cycles=n_cycles, use_fft=use_fft, decim=decim,
                            projs=projs, n_jobs=n_jobs, verbose=verbose)


@verbose
def csd_array_morlet(X, sfreq, frequencies, t0=0, tmin=None, tmax=None,
                     ch_names=None, n_cycles=7, use_fft=True, decim=1,
                     projs=None, n_jobs=None, *, verbose=None):
    """Estimate cross-spectral density from an array using Morlet wavelets.

    Parameters
    ----------
    X : array-like, shape (n_epochs, n_channels, n_times)
        The time series data consisting of n_epochs separate observations
        of signals with n_channels time-series of length n_times.
    sfreq : float
        Sampling frequency of observations.
    frequencies : list of float
        The frequencies of interest, in Hertz.
    t0 : float
        Time of the first sample relative to the onset of the epoch, in
        seconds. Defaults to 0.
    tmin : float | None
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    ch_names : list of str | None
        A name for each time series. If ``None`` (the default), the series will
        be named 'SERIES###'.
    n_cycles : float | list of float | None
        Number of cycles to use when constructing Morlet wavelets. Fixed number
        or one per frequency. Defaults to 7.
    use_fft : bool
        Whether to use FFT-based convolution to compute the wavelet transform.
        Defaults to True.
    decim : int | slice
        To reduce memory usage, decimation factor during time-frequency
        decomposition. Defaults to 1 (no decimation).

        If `int`, uses tfr[..., ::decim].
        If `slice`, uses tfr[..., decim].

    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to ``None``,
        which means the projectors defined in the Epochs object will be copied.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.

    See Also
    --------
    csd_array_fourier
    csd_array_multitaper
    csd_fourier
    csd_morlet
    csd_multitaper
    """
    X, times, tmin, tmax, _, _ = _prepare_csd_array(X, sfreq, t0, tmin, tmax)
    n_times = len(times)

    # Construct the appropriate Morlet wavelets
    wavelets = morlet(sfreq, frequencies, n_cycles)

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

    # After CSD computation, we slice again to the requested time window.
    csd_tstart = None if tmin is None else np.searchsorted(times, tmin - 1e-10)
    csd_tstop = None if tmax is None else np.searchsorted(times, tmax + 1e-10)
    csd_tslice = slice(csd_tstart, csd_tstop)
    times = times[csd_tslice]

    # Compute the CSD
    nfft = _get_nfft(wavelets, X, use_fft)
    return _execute_csd_function(X, times, frequencies, _csd_morlet,
                                 params=[sfreq, wavelets, nfft, csd_tslice,
                                         use_fft, decim],
                                 n_fft=1, ch_names=ch_names, projs=projs,
                                 n_jobs=n_jobs, verbose=verbose)


def _prepare_csd(epochs, tmin=None, tmax=None, picks=None, projs=None):
    """Do some checking and preprocessing of common csd_* parameters.

    See the csd_* functions for documentation of the parameters.
    """
    tstep = epochs.times[1] - epochs.times[0]
    if tmin is not None and tmin < epochs.times[0] - tstep:
        raise ValueError('tmin should be larger than the smallest data time '
                         'point')
    if tmax is not None and tmax > epochs.times[-1] + tstep:
        raise ValueError('tmax should be smaller than the largest data time '
                         'point')
    if tmax is not None and tmin is not None:
        if tmax < tmin:
            raise ValueError('tmax must be larger than tmin')
    if epochs.baseline is None and epochs.info['highpass'] < 0.1:
        warn('Epochs are not baseline corrected or enough highpass filtered. '
             'Cross-spectral density may be inaccurate.')

    picks = _picks_to_idx(epochs.info, picks, 'data', with_ref_meg=False)
    epochs = epochs.copy().pick(picks)

    if projs is None:
        projs = epochs.info['projs']

    return epochs, projs


def _prepare_csd_array(X, sfreq, t0, tmin, tmax, fmin=None, fmax=None):
    """Do some checking and preprocessing of common csd_r=array_* parameters.

    See the csd_array_* functions for documentation of the parameters.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X must be n_epochs x n_channels x n_times.")

    n_times = X.shape[2]
    tstep = 1. / sfreq
    times = np.arange(n_times) * tstep + t0

    # Check tmin and tmax
    if tmax is None:
        tmax = times.max()
    if tmin is None:
        tmin = times.min()
    if tmax <= tmin:
        raise ValueError('tmax must be larger than tmin')
    if tmin < times[0] - tstep:
        raise ValueError('tmin should be larger than the smallest data time '
                         'point')
    if tmax > times[-1] + tstep:
        raise ValueError('tmax should be smaller than the largest data time '
                         'point')

    # Check fmin and fmax
    if fmax is not None and fmin is not None and fmax <= fmin:
        raise ValueError('fmax must be larger than fmin')

    return X, times, tmin, tmax, fmin, fmax


@verbose
def _execute_csd_function(X, times, frequencies, csd_function, params, n_fft,
                          ch_names=None, projs=None, n_jobs=None, *,
                          verbose=None):
    """Estimate cross-spectral density with a given function.

    This function will apply the given CSD function in parallel across epochs.

    Parameters
    ----------
    X : array-like, shape (n_epochs, n_channels, n_times)
        The time series data consisting of n_epochs separate observations
        of signals with n_channels time-series of length n_times.
    times : float
        Timestamps for each sample.
    frequencies : list of float
        The frequencies of interest for which the CSD is going to be computed.
    csd_function : function
        Function that performs the actual CSD computation
    params : list
        List of parameters to pass the CSD function.
    n_fft : int
        Number of FFT points. This is stored in the CSD object.
    ch_names : list of str | None
        A name for each time series. If ``None`` (the default), the series will
        be named 'SERIES###'.
    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to ``None``,
        which means the projectors defined in the Epochs object will be copied.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.
    """
    n_epochs, n_channels, _ = X.shape

    logger.info('Computing cross-spectral density from epochs...')

    n_freqs = len(frequencies)
    csds_mean = np.zeros((n_channels * (n_channels + 1) // 2, n_freqs),
                         dtype=np.complex128)

    # Prepare the function that does the actual CSD computation for parallel
    # execution.
    parallel, my_csd, n_jobs = parallel_func(
        csd_function, n_jobs, verbose=verbose)

    # Compute CSD for each trial
    n_blocks = int(np.ceil(n_epochs / float(n_jobs)))
    for i in ProgressBar(range(n_blocks), mesg='CSD epoch blocks'):
        epoch_block = X[i * n_jobs:(i + 1) * n_jobs]
        csds = parallel(my_csd(this_epoch, *params)
                        for this_epoch in epoch_block)

        # Add CSD matrices in-place
        csds_mean += np.sum(csds, axis=0)

    csds_mean /= n_epochs
    logger.info('[done]')

    if ch_names is None:
        ch_names = ['SERIES%03d' % (i + 1) for i in range(n_channels)]

    return CrossSpectralDensity(csds_mean, ch_names=ch_names, tmin=times[0],
                                tmax=times[-1], frequencies=frequencies,
                                n_fft=n_fft, projs=projs)


def _csd_fourier(X, sfreq, n_times, freq_mask, n_fft):
    """Compute cross spectral density (CSD) using short-time fourier transform.

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
    freq_mask : ndarray
        Which frequencies to use.
    n_fft : int
        Length of the FFT.
    """
    x_mt, _ = _mt_spectra(X, np.hanning(n_times), sfreq, n_fft)

    # Hack so we can sum over axis=-2
    weights = np.array([1.])[:, np.newaxis, np.newaxis, np.newaxis]

    x_mt = x_mt[:, :, freq_mask]

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
    csds /= n_times
    csds *= 8 / 3.

    # Scaling by sampling frequency for compatibility with Matlab
    csds /= sfreq

    return csds


def _csd_multitaper(X, sfreq, n_times, window_fun, eigvals, freq_mask, n_fft,
                    adaptive):
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
        Window function(s) of length n_times. This corresponds to first output
        of `dpss_windows`.
    eigvals : ndarray | float
        Eigenvalues associated with window functions.
    freq_mask : ndarray
        Which frequencies to use.
    n_fft : int
        Length of the FFT.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
    """
    x_mt, _ = _mt_spectra(X, window_fun, sfreq, n_fft)

    if adaptive:
        # Compute adaptive weights
        _, weights = _psd_from_mt_adaptive(x_mt, eigvals, freq_mask,
                                           return_weights=True)
        # Tiling weights so that we can easily use _csd_from_mt()
        weights = weights[:, np.newaxis, :, :]
        weights = np.tile(weights, [1, x_mt.shape[0], 1, 1])
    else:
        # Do not use adaptive weights
        weights = np.sqrt(eigvals)[np.newaxis, np.newaxis, :, np.newaxis]

    x_mt = x_mt[:, :, freq_mask]

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

    # Scaling by sampling frequency for compatibility with Matlab
    csds /= sfreq

    return csds


def _csd_morlet(data, sfreq, wavelets, nfft, tslice=None, use_fft=True,
                decim=1):
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
    nfft : int
        The number of FFT points.
    tslice : slice | None
        The desired time samples to compute the CSD over. If None, defaults to
        including all time samples.
    use_fft : bool
        Whether to use FFT-based convolution to compute the wavelet transform.
        Defaults to True.
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
    _vector_to_sym_mat : For converting the CSD to a full matrix.
    """
    # Compute PSD
    psds = _cwt_array(data, wavelets, nfft, mode='same', use_fft=use_fft,
                      decim=decim)

    if tslice is not None:
        tstart = None if tslice.start is None else tslice.start // decim
        tstop = None if tslice.stop is None else tslice.stop // decim
        tstep = None if tslice.step is None else tslice.step // decim
        tslice = slice(tstart, tstop, tstep)
        psds = psds[:, :, tslice]

    psds_conj = np.conj(psds)

    # Compute the spectral density between all pairs of series
    n_channels = data.shape[0]
    csds = np.vstack([np.mean(psds[[i]] * psds_conj[i:], axis=2)
                      for i in range(n_channels)])

    # Scaling by sampling frequency for compatibility with Matlab
    csds /= sfreq
    return csds


@verbose
def csd_tfr(epochs_tfr, tmin=None, tmax=None, picks=None, projs=None,
            verbose=None):
    """Compute covariance matrices across frequencies for TFR epochs.

    Parameters
    ----------
    epochs_tfr : EpochsTFR
        The time-frequency resolved epochs over which to compute the
        covariance.
    tmin : float | None
        Minimum time instant to consider, in seconds. If ``None`` start at
        first sample.
    tmax : float | None
        Maximum time instant to consider, in seconds. If ``None`` end at last
        sample.
    %(picks_good_data_noref)s
    projs : list of Projection | None
        List of projectors to store in the CSD object. Defaults to ``None``,
        which means the projectors defined in the EpochsTFR object will be
        copied.
    %(verbose)s

    Returns
    -------
    res : instance of CrossSpectralDensity
        Cross-spectral density restricted to selected channels.
    """
    _validate_type(epochs_tfr, EpochsTFR)
    epochs_tfr, projs = _prepare_csd(epochs_tfr, tmin, tmax, picks, projs)
    X = epochs_tfr.data
    times = epochs_tfr.times
    n_channels, n_freqs = len(epochs_tfr.ch_names), epochs_tfr.freqs.size
    data = np.zeros((n_channels * (n_channels + 1) // 2, n_freqs),
                    dtype=np.complex128)

    # Slice X to the requested time window
    tstart = None if tmin is None else np.searchsorted(times, tmin - 1e-10)
    tstop = None if tmax is None else np.searchsorted(times, tmax + 1e-10)
    X = X[:, :, :, tstart:tstop]

    for idx, epochs_data in enumerate(X):
        # This is equivalent to:
        # csds = np.vstack([np.mean(epochs_data[[i]] * epochs_data_conj[i:],
        #                           axis=2) for i in range(n_channels)])
        # There is a redundancy in the calculation here because we don't really
        # need the lower triangle of the matrix, but it should still be faster
        # than a loop (hopefully!).
        csds = np.einsum('xft,yft->xyf', epochs_data, np.conj(epochs_data))
        csds = csds[np.triu_indices(n_channels) + (slice(None),)]
        csds /= epochs_data.shape[-1]

        # Scaling by sampling frequency for compatibility with Matlab
        csds /= epochs_tfr.info['sfreq']
        data += csds

    # scale to compute mean
    data /= len(epochs_tfr)

    # TO DO: EpochTFR should store n_fft to be consistent
    return CrossSpectralDensity(data=data, ch_names=epochs_tfr.ch_names,
                                tmin=tmin, tmax=tmax,
                                frequencies=epochs_tfr.freqs, n_fft=None,
                                projs=projs)
