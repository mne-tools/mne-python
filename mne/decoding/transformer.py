# -*- coding: utf-8 -*-
# Authors: Mainak Jas <mainak@neuro.hut.fi>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from .mixin import TransformerMixin
from .base import BaseEstimator

from .. import pick_types
from ..filter import filter_data, _triage_filter_params
from ..time_frequency.psd import psd_array_multitaper
from ..externals import six
from ..utils import _check_type_picks


class Scaler(TransformerMixin):
    u"""Standardize data across channels.

    By default, this makes each time point (within each epoch) have
    μ=0, σ=1.

    Parameters
    ----------
    info : instance of Info
        The measurement info
    with_mean : boolean, True by default
        If True, center the data before scaling.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    info : instance of Info
        The measurement info
    ``ch_mean_`` : dict
        The mean value for each channel type
    ``std_`` : dict
        The standard deviation for each channel type
    """

    def __init__(self, info, with_mean=True, with_std=True):  # noqa: D102
        self.info = info
        self.with_mean = with_mean
        self.with_std = with_std
        self.ch_mean_ = dict()  # TODO rename attribute
        self.std_ = dict()  # TODO rename attribute

    def fit(self, epochs_data, y):
        """Standardize data across channels.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data to concatenate channels.
        y : array, shape (n_epochs,)
            The label for each epoch.

        Returns
        -------
        self : instance of Scaler
            Returns the modified instance.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        X = np.atleast_3d(epochs_data)

        picks_list = dict()
        picks_list['mag'] = pick_types(self.info, meg='mag', ref_meg=False,
                                       exclude='bads')
        picks_list['grad'] = pick_types(self.info, meg='grad', ref_meg=False,
                                        exclude='bads')
        picks_list['eeg'] = pick_types(self.info, eeg=True, ref_meg=False,
                                       meg=False, exclude='bads')

        self.picks_list_ = picks_list

        for key, this_pick in picks_list.items():
            if self.with_mean:
                if len(this_pick) == 0:
                    ch_mean = np.nan * np.ones((X.shape[0], 1, X.shape[2]))
                else:
                    ch_mean = X[:, this_pick, :].mean(axis=1, keepdims=True)
                self.ch_mean_[key] = ch_mean  # TODO rename attribute
            if self.with_std:
                if len(this_pick) == 0:
                    ch_std = np.nan * np.ones((X.shape[0], 1, X.shape[2]))
                else:
                    ch_std = np.std(X[:, this_pick, :], axis=1, keepdims=True)
                self.std_[key] = ch_std  # TODO rename attribute

        return self

    def transform(self, epochs_data, y=None):
        """Standardize data across channels.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : None | array, shape (n_epochs,)
            The label for each epoch.
            If None not used. Defaults to None.

        Returns
        -------
        X : array, shape (n_epochs, n_channels, n_times)
            The data concatenated over channels.

        Notes
        -----
        This function makes a copy of the data before the operations and the
        memory usage may be large with big data.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        X = np.atleast_3d(epochs_data).copy()

        for key, this_pick in six.iteritems(self.picks_list_):
            if self.with_mean:
                X[:, this_pick, :] -= self.ch_mean_[key]
            if self.with_std:
                X[:, this_pick, :] /= self.std_[key]

        return X

    def inverse_transform(self, epochs_data, y=None):
        """Invert standardization of data across channels.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : None | array, shape (n_epochs,)
            The label for each epoch.
            If None not used. Defaults to None.

        Returns
        -------
        X : array, shape (n_epochs, n_channels, n_times)
            The data concatenated over channels.

        Notes
        -----
        This function makes a copy of the data before the operations and the
        memory usage may be large with big data.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        X = np.atleast_3d(epochs_data).copy()

        for key, this_pick in six.iteritems(self.picks_list_):
            if self.with_std:
                X[:, this_pick, :] *= self.std_[key]
            if self.with_mean:
                X[:, this_pick, :] += self.ch_mean_[key]

        return X


class Vectorizer(TransformerMixin):
    """Transform n-dimensional array into 2D array of n_samples by n_features.

    This class reshapes an n-dimensional array into an n_samples * n_features
    array, usable by the estimators and transformers of scikit-learn.

    Examples
    --------
    clf = make_pipeline(SpatialFilter(), _XdawnTransformer(), Vectorizer(),
                        LogisticRegression())

    Attributes
    ----------
    ``features_shape_`` : tuple
         Stores the original shape of data.
    """

    def fit(self, X, y=None):
        """Store the shape of the features of X.

        Parameters
        ----------
        X : array-like
            The data to fit. Can be, for example a list, or an array of at
            least 2d. The first dimension must be of length n_samples, where
            samples are the independent samples used by the estimator
            (e.g. n_epochs for epoched data).
        y : None | array, shape (n_samples,)
            Used for scikit-learn compatibility.

        Returns
        -------
        self : Instance of Vectorizer
            Return the modified instance.
        """
        X = np.asarray(X)
        self.features_shape_ = X.shape[1:]
        return self

    def transform(self, X):
        """Convert given array into two dimensions.

        Parameters
        ----------
        X : array-like
            The data to fit. Can be, for example a list, or an array of at
            least 2d. The first dimension must be of length n_samples, where
            samples are the independent samples used by the estimator
            (e.g. n_epochs for epoched data).

        Returns
        -------
        X : array, shape (n_samples, n_features)
            The transformed data.
        """
        X = np.asarray(X)
        if X.shape[1:] != self.features_shape_:
            raise ValueError("Shape of X used in fit and transform must be "
                             "same")
        return X.reshape(len(X), -1)

    def fit_transform(self, X, y=None):
        """Fit the data, then transform in one step.

        Parameters
        ----------
        X : array-like
            The data to fit. Can be, for example a list, or an array of at
            least 2d. The first dimension must be of length n_samples, where
            samples are the independent samples used by the estimator
            (e.g. n_epochs for epoched data).
        y : None | array, shape (n_samples,)
            Used for scikit-learn compatibility.

        Returns
        -------
        X : array, shape (n_samples, -1)
            The transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Transform 2D data back to its original feature shape.

        Parameters
        ----------
        X : array-like, shape (n_samples,  n_features)
            Data to be transformed back to original shape.

        Returns
        -------
        X : array
            The data transformed into shape as used in fit. The first
            dimension is of length n_samples.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X should be of 2 dimensions but given has %s "
                             "dimension(s)" % X.ndim)
        return X.reshape((len(X),) + self.features_shape_)


class PSDEstimator(TransformerMixin):
    """Compute power spectrum density (PSD) using a multi-taper method.

    Parameters
    ----------
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    n_jobs : int
        Number of parallel jobs to use (only used if adaptive=True).
    normalization : str
        Either "full" or "length" (default). If "full", the PSD will
        be normalized by the sampling rate as well as the length of
        the signal (as in nitime).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    """

    def __init__(self, sfreq=2 * np.pi, fmin=0, fmax=np.inf, bandwidth=None,
                 adaptive=False, low_bias=True, n_jobs=1,
                 normalization='length', verbose=None):  # noqa: D102
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.low_bias = low_bias
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.normalization = normalization

    def fit(self, epochs_data, y):
        """Compute power spectrum density (PSD) using a multi-taper method.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : array, shape (n_epochs,)
            The label for each epoch

        Returns
        -------
        self : instance of PSDEstimator
            returns the modified instance
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        return self

    def transform(self, epochs_data, y=None):
        """Compute power spectrum density (PSD) using a multi-taper method.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data
        y : None | array, shape (n_epochs,)
            The label for each epoch.
            If None not used. Defaults to None.

        Returns
        -------
        psd : array, shape (n_signals, len(freqs)) or (len(freqs),)
            The computed PSD.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        psd, _ = psd_array_multitaper(
            epochs_data, sfreq=self.sfreq, fmin=self.fmin, fmax=self.fmax,
            bandwidth=self.bandwidth, adaptive=self.adaptive,
            low_bias=self.low_bias, normalization=self.normalization,
            n_jobs=self.n_jobs)
        return psd


class FilterEstimator(TransformerMixin):
    """Estimator to filter RtEpochs.

    Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
    filter to the channels selected by "picks".

    l_freq and h_freq are the frequencies below which and above which,
    respectively, to filter out of the data. Thus the uses are:

        - l_freq < h_freq: band-pass filter
        - l_freq > h_freq: band-stop filter
        - l_freq is not None, h_freq is None: low-pass filter
        - l_freq is None, h_freq is not None: high-pass filter

    If n_jobs > 1, more memory is required as "len(picks) * n_times"
    additional time points need to be temporarily stored in memory.

    Parameters
    ----------
    info : instance of Info
        Measurement info.
    l_freq : float | None
        Low cut-off frequency in Hz. If None the data are only low-passed.
    h_freq : float | None
        High cut-off frequency in Hz. If None the data are only
        high-passed.
    picks : array-like of int | None
        Indices of channels to filter. If None only the data (MEG/EEG)
        channels will be filtered.
    filter_length : str (Default: '10s') | int | None
        Length of the filter to use. If None or "len(x) < filter_length",
        the filter length used is len(x). Otherwise, if int, overlap-add
        filtering with a filter of the specified length in samples) is
        used (faster for long signals). If str, a human-readable time in
        units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
        to the shortest power-of-two length at least that duration.
    l_trans_bandwidth : float
        Width of the transition band at the low cut-off frequency in Hz.
    h_trans_bandwidth : float
        Width of the transition band at the high cut-off frequency in Hz.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fft'.
    method : str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        self.verbose.

    See Also
    --------
    TemporalFilter
    """

    def __init__(self, info, l_freq, h_freq, picks=None, filter_length='auto',
                 l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1,
                 method='fft', iir_params=None, verbose=None):  # noqa: D102
        self.info = info
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.picks = _check_type_picks(picks)
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.n_jobs = n_jobs
        self.method = method
        self.iir_params = iir_params

    def fit(self, epochs_data, y):
        """Filter data.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : array, shape (n_epochs,)
            The label for each epoch.

        Returns
        -------
        self : instance of FilterEstimator
            Returns the modified instance
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        if self.picks is None:
            self.picks = pick_types(self.info, meg=True, eeg=True,
                                    ref_meg=False, exclude=[])

        if self.l_freq == 0:
            self.l_freq = None
        if self.h_freq is not None and self.h_freq > (self.info['sfreq'] / 2.):
            self.h_freq = None
        if self.l_freq is not None and not isinstance(self.l_freq, float):
            self.l_freq = float(self.l_freq)
        if self.h_freq is not None and not isinstance(self.h_freq, float):
            self.h_freq = float(self.h_freq)

        if self.info['lowpass'] is None or (self.h_freq is not None and
                                            (self.l_freq is None or
                                             self.l_freq < self.h_freq) and
                                            self.h_freq <
                                            self.info['lowpass']):
            self.info['lowpass'] = self.h_freq

        if self.info['highpass'] is None or (self.l_freq is not None and
                                             (self.h_freq is None or
                                              self.l_freq < self.h_freq) and
                                             self.l_freq >
                                             self.info['highpass']):
            self.info['highpass'] = self.l_freq

        return self

    def transform(self, epochs_data, y=None):
        """Filter data.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : None | array, shape (n_epochs,)
            The label for each epoch.
            If None not used. Defaults to None.

        Returns
        -------
        X : array, shape (n_epochs, n_channels, n_times)
            The data after filtering
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        epochs_data = np.atleast_3d(epochs_data)
        return filter_data(
            epochs_data, self.info['sfreq'], self.l_freq, self.h_freq,
            self.picks, self.filter_length, self.l_trans_bandwidth,
            self.h_trans_bandwidth, method=self.method,
            iir_params=self.iir_params, n_jobs=self.n_jobs, copy=False,
            verbose=False)


class UnsupervisedSpatialFilter(TransformerMixin, BaseEstimator):
    """Use unsupervised spatial filtering across time and samples.

    Parameters
    ----------
    estimator : scikit-learn estimator
        Estimator using some decomposition algorithm.
    average : bool, defaults to False
        If True, the estimator is fitted on the average across samples
        (e.g. epochs).
    """

    def __init__(self, estimator, average=False):  # noqa: D102
        # XXX: Use _check_estimator #3381
        for attr in ('fit', 'transform', 'fit_transform'):
            if not hasattr(estimator, attr):
                raise ValueError('estimator must be a scikit-learn '
                                 'transformer, missing %s method' % attr)

        if not isinstance(average, bool):
            raise ValueError("average parameter must be of bool type, got "
                             "%s instead" % type(bool))

        self.estimator = estimator
        self.average = average

    def fit(self, X, y=None):
        """Fit the spatial filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data to be filtered.
        y : None | array, shape (n_samples,)
            Used for scikit-learn compatibility.

        Returns
        -------
        self : Instance of UnsupervisedSpatialFilter
            Return the modified instance.
        """
        if self.average:
            X = np.mean(X, axis=0).T
        else:
            n_epochs, n_channels, n_times = X.shape
            # trial as time samples
            X = np.transpose(X, (1, 0, 2)).reshape((n_channels, n_epochs *
                                                    n_times)).T
        self.estimator.fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Transform the data to its filtered components after fitting.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data to be filtered.
        y : None | array, shape (n_samples,)
            Used for scikit-learn compatibility.

        Returns
        -------
        X : array, shape (n_trials, n_channels, n_times)
            The transformed data.
        """
        return self.fit(X).transform(X)

    def transform(self, X):
        """Transform the data to its spatial filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data to be filtered.

        Returns
        -------
        X : array, shape (n_trials, n_channels, n_times)
            The transformed data.
        """
        n_epochs, n_channels, n_times = X.shape
        # trial as time samples
        X = np.transpose(X, [1, 0, 2]).reshape([n_channels, n_epochs *
                                                n_times]).T
        X = self.estimator.transform(X)
        X = np.reshape(X.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
        return X


class TemporalFilter(TransformerMixin):
    """Estimator to filter data array along the last dimension.

    Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
    filter to the channels.

    l_freq and h_freq are the frequencies below which and above which,
    respectively, to filter out of the data. Thus the uses are:

        - l_freq < h_freq: band-pass filter
        - l_freq > h_freq: band-stop filter
        - l_freq is not None, h_freq is None: low-pass filter
        - l_freq is None, h_freq is not None: high-pass filter

    See :func:`mne.filter.filter_data`.

    Parameters
    ----------
    l_freq : float | None
        Low cut-off frequency in Hz. If None the data are only low-passed.
    h_freq : float | None
        High cut-off frequency in Hz. If None the data are only
        high-passed.
    sfreq : float, defaults to 1.0
        Sampling frequency in Hz.
    filter_length : str | int, defaults to 'auto'
        Length of the FIR filter to use (if applicable):

            * int: specified length in samples.
            * 'auto' (default in 0.14): the filter length is chosen based
              on the size of the transition regions (7 times the reciprocal
              of the shortest transition band).
            * str: (default in 0.13 is "10s") a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.

    l_trans_bandwidth : float | str
        Width of the transition band at the low cut-off frequency in Hz
        (high pass or cutoff 1 in bandpass). Can be "auto"
        (default in 0.14) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    h_trans_bandwidth : float | str
        Width of the transition band at the high cut-off frequency in Hz
        (low pass or cutoff 2 in bandpass). Can be "auto"
        (default in 0.14) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    n_jobs : int | str, defaults to 1
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fft'.
    method : str, defaults to 'fir'
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None, defaults to None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    fir_window : str, defaults to 'hamming'
        The window to use in FIR design, can be "hamming", "hann",
        or "blackman".
    verbose : bool, str, int, or None, defaults to None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        self.verbose.

    See Also
    --------
    FilterEstimator
    Vectorizer
    mne.filter.filter_data
    """

    def __init__(self, l_freq=None, h_freq=None, sfreq=1.0,
                 filter_length='auto', l_trans_bandwidth='auto',
                 h_trans_bandwidth='auto', n_jobs=1, method='fir',
                 iir_params=None, fir_window='hamming',
                 verbose=None):  # noqa: D102
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.n_jobs = n_jobs
        self.method = method
        self.iir_params = iir_params
        self.fir_window = fir_window
        self.verbose = verbose

        if not isinstance(self.n_jobs, int) and self.n_jobs == 'cuda':
            raise ValueError('n_jobs must be int or "cuda", got %s instead.'
                             % type(self.n_jobs))

    def fit(self, X, y=None):
        """Do nothing (for scikit-learn compatibility purposes).

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times) or or shape (n_channels, n_times) # noqa
            The data to be filtered over the last dimension. The channels
            dimension can be zero when passing a 2D array.
        y : None
            Not used, for scikit-learn compatibility issues.

        Returns
        -------
        self : instance of Filterer
            Returns the modified instance.
        """
        return self

    def transform(self, X):
        """Filter data along the last dimension.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times) or shape (n_channels, n_times) # noqa
            The data to be filtered over the last dimension. The channels
            dimension can be zero when passing a 2D array.

        Returns
        -------
        X : array, shape is same as used in input.
            The data after filtering.
        """
        X = np.atleast_2d(X)

        if X.ndim > 3:
            raise ValueError("Array must be of at max 3 dimensions instead "
                             "got %s dimensional matrix" % (X.ndim))

        shape = X.shape
        X = X.reshape(-1, shape[-1])
        (X, self.sfreq, self.l_freq, self.h_freq, self.l_trans_bandwidth,
         self.h_trans_bandwidth, self.filter_length, _, self.fir_window) = \
            _triage_filter_params(X, self.sfreq, self.l_freq, self.h_freq,
                                  self.l_trans_bandwidth,
                                  self.h_trans_bandwidth, self.filter_length,
                                  self.method, phase='zero',
                                  fir_window=self.fir_window)
        X = filter_data(X, self.sfreq, self.l_freq, self.h_freq,
                        filter_length=self.filter_length,
                        l_trans_bandwidth=self.l_trans_bandwidth,
                        h_trans_bandwidth=self.h_trans_bandwidth,
                        n_jobs=self.n_jobs, method=self.method,
                        iir_params=self.iir_params, copy=False,
                        fir_window=self.fir_window,
                        verbose=self.verbose)
        return X.reshape(shape)
