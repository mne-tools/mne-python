# Authors: Mainak Jas <mainak@neuro.hut.fi>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from .mixin import TransformerMixin

from .. import pick_types
from ..filter import (low_pass_filter, high_pass_filter, band_pass_filter,
                      band_stop_filter)
from ..time_frequency import multitaper_psd
from ..externals import six
from ..utils import _check_type_picks, deprecated


class Scaler(TransformerMixin):
    """Standardizes data across channels

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
    ch_mean_ : dict
        The mean value for each channel type
    std_ : dict
        The standard deviation for each channel type
     """
    def __init__(self, info, with_mean=True, with_std=True):
        self.info = info
        self.with_mean = with_mean
        self.with_std = with_std
        self.ch_mean_ = dict()  # TODO rename attribute
        self.std_ = dict()  # TODO rename attribute

    def fit(self, epochs_data, y):
        """Standardizes data across channels

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
        picks_list['eeg'] = pick_types(self.info, eeg='grad', ref_meg=False,
                                       exclude='bads')

        self.picks_list_ = picks_list

        for key, this_pick in picks_list.items():
            if self.with_mean:
                ch_mean = X[:, this_pick, :].mean(axis=1)[:, None, :]
                self.ch_mean_[key] = ch_mean  # TODO rename attribute
            if self.with_std:
                ch_std = X[:, this_pick, :].mean(axis=1)[:, None, :]
                self.std_[key] = ch_std  # TODO rename attribute

        return self

    def transform(self, epochs_data, y=None):
        """Standardizes data across channels

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
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        X = np.atleast_3d(epochs_data)

        for key, this_pick in six.iteritems(self.picks_list_):
            if self.with_mean:
                X[:, this_pick, :] -= self.ch_mean_[key]
            if self.with_std:
                X[:, this_pick, :] /= self.std_[key]

        return X

    def inverse_transform(self, epochs_data, y=None):
        """ Inverse standardization of data across channels

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
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        X = np.atleast_3d(epochs_data)

        for key, this_pick in six.iteritems(self.picks_list_):
            if self.with_mean:
                X[:, this_pick, :] += self.ch_mean_[key]
            if self.with_std:
                X[:, this_pick, :] *= self.std_[key]

        return X


class EpochsVectorizer(TransformerMixin):
    """EpochsVectorizer transforms epoch data to fit into a scikit-learn pipeline.

    Parameters
    ----------
    info : instance of Info
        The measurement info.

    Attributes
    ----------
    n_epochs : int
        The number of epochs.
    n_channels : int
        The number of channels.
    n_times : int
        The number of time points.

    """
    def __init__(self, info=None):
        self.info = info
        self.n_epochs = None
        self.n_channels = None
        self.n_times = None

    def fit(self, epochs_data, y):
        """For each epoch, concatenate data from different channels into a single
        feature vector.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data to concatenate channels.
        y : array, shape (n_epochs,)
            The label for each epoch.

        Returns
        -------
        self : instance of ConcatenateChannels
            returns the modified instance
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        return self

    def transform(self, epochs_data, y=None):
        """For each epoch, concatenate data from different channels into a single
        feature vector.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : None | array, shape (n_epochs,)
            The label for each epoch.
            If None not used. Defaults to None.

        Returns
        -------
        X : array, shape (n_epochs, n_channels * n_times)
            The data concatenated over channels
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))

        epochs_data = np.atleast_3d(epochs_data)

        n_epochs, n_channels, n_times = epochs_data.shape
        X = epochs_data.reshape(n_epochs, n_channels * n_times)
        # save attributes for inverse_transform
        self.n_epochs = n_epochs
        self.n_channels = n_channels
        self.n_times = n_times

        return X

    def inverse_transform(self, X, y=None):
        """For each epoch, reshape a feature vector into the original data shape

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels * n_times)
            The feature vector concatenated over channels
        y : None | array, shape (n_epochs,)
            The label for each epoch.
            If None not used. Defaults to None.

        Returns
        -------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The original data
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(X))

        return X.reshape(self.n_epochs, self.n_channels, self.n_times)


@deprecated("Class 'ConcatenateChannels' has been renamed to "
            "'EpochsVectorizer' and will be removed in release 0.11.")
class ConcatenateChannels(EpochsVectorizer):
    pass


class PSDEstimator(TransformerMixin):
    """Compute power spectrum density (PSD) using a multi-taper method

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
        If not None, override default verbose level (see mne.verbose).
    """
    def __init__(self, sfreq=2 * np.pi, fmin=0, fmax=np.inf, bandwidth=None,
                 adaptive=False, low_bias=True, n_jobs=1,
                 normalization='length', verbose=None):
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
        """Compute power spectrum density (PSD) using a multi-taper method

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
        """Compute power spectrum density (PSD) using a multi-taper method

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

        epochs_data = np.atleast_3d(epochs_data)

        n_epochs, n_channels, n_times = epochs_data.shape
        X = epochs_data.reshape(n_epochs * n_channels, n_times)

        psd, _ = multitaper_psd(x=X, sfreq=self.sfreq, fmin=self.fmin,
                                fmax=self.fmax, bandwidth=self.bandwidth,
                                adaptive=self.adaptive, low_bias=self.low_bias,
                                n_jobs=self.n_jobs,
                                normalization=self.normalization,
                                verbose=self.verbose)

        _, n_freqs = psd.shape
        psd = psd.reshape(n_epochs, n_channels, n_freqs)

        return psd


class FilterEstimator(TransformerMixin):
    """Estimator to filter RtEpochs

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
        If not None, override default verbose level (see mne.verbose).
        Defaults to self.verbose.
    """
    def __init__(self, info, l_freq, h_freq, picks=None, filter_length='10s',
                 l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, n_jobs=1,
                 method='fft', iir_params=None, verbose=None):
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
        """Filters data

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
        """Filters data

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

        if self.l_freq is None and self.h_freq is not None:
            epochs_data = \
                low_pass_filter(epochs_data, self.info['sfreq'], self.h_freq,
                                filter_length=self.filter_length,
                                trans_bandwidth=self.l_trans_bandwidth,
                                method=self.method, iir_params=self.iir_params,
                                picks=self.picks, n_jobs=self.n_jobs,
                                copy=False, verbose=False)

        if self.l_freq is not None and self.h_freq is None:
            epochs_data = \
                high_pass_filter(epochs_data, self.info['sfreq'], self.l_freq,
                                 filter_length=self.filter_length,
                                 trans_bandwidth=self.h_trans_bandwidth,
                                 method=self.method,
                                 iir_params=self.iir_params,
                                 picks=self.picks, n_jobs=self.n_jobs,
                                 copy=False, verbose=False)

        if self.l_freq is not None and self.h_freq is not None:
            if self.l_freq < self.h_freq:
                epochs_data = \
                    band_pass_filter(epochs_data, self.info['sfreq'],
                                     self.l_freq, self.h_freq,
                                     filter_length=self.filter_length,
                                     l_trans_bandwidth=self.l_trans_bandwidth,
                                     h_trans_bandwidth=self.h_trans_bandwidth,
                                     method=self.method,
                                     iir_params=self.iir_params,
                                     picks=self.picks, n_jobs=self.n_jobs,
                                     copy=False, verbose=False)
            else:
                epochs_data = \
                    band_stop_filter(epochs_data, self.info['sfreq'],
                                     self.h_freq, self.l_freq,
                                     filter_length=self.filter_length,
                                     l_trans_bandwidth=self.h_trans_bandwidth,
                                     h_trans_bandwidth=self.l_trans_bandwidth,
                                     method=self.method,
                                     iir_params=self.iir_params,
                                     picks=self.picks, n_jobs=self.n_jobs,
                                     copy=False, verbose=False)
        return epochs_data
