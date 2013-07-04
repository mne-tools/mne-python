# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import numpy as np

from sklearn.base import TransformerMixin

from .. import verbose
from ..filter import low_pass_filter, high_pass_filter, band_pass_filter, \
    band_stop_filter
from ..time_frequency import multitaper_psd
from ..fiff import pick_types

import logging
logger = logging.getLogger('mne')


class RtClassifier:

    """
    TODO: complete docstrings ...

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, estimator):

        self.estimator = estimator

    def fit(self, X, y):

        self.estimator.fit(X, y)
        return self

    def predict(self, X):

        result = self.estimator.predict(X)

        return result


class Scaler(TransformerMixin):
    """
    Standardizes data across channels

    Parameters
    ----------
    info : dict
        measurement info
    with_mean : boolean, True by default
        If True, center the data before scaling.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
            unit standard deviation).

    Attributes
    ----------
    `ch_mean_` : array
        The mean value for each channel type
    `ch_std_` : array
        The standard deviation for each channel type
     """

    def __init__(self, info, with_mean=True, with_std=True):
        self.info = info
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, epochs_data, y):
        """
        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data to concatenate channels
        y : array
            The label for each epoch

        Returns
        -------
        self : instance of Scaler
            returns the modified instance
        """

        X = epochs_data

        picks_list = [pick_types(self.info, meg='mag', exclude='bads'),
                      pick_types(self.info, eeg='True', exclude='bads'),
                      pick_types(self.info, meg='grad', exclude='bads')]

        for pick_one in picks_list:
            if self.with_mean:
                self.ch_mean_ = X[:, pick_one, :].mean(axis=1)[:, None, :]
            if self.with_std:
                self.ch_std_ = X[:, pick_one, :].mean(axis=1)[:, None, :]

        return self

    def transform(self, epochs_data):
        """
        Standardizes data across channels

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray of shape (n_epochs, n_channels*n_times)
            The data concatenated over channels
        """

        X = epochs_data

        picks_list = [pick_types(self.info, meg='mag', exclude='bads'),
                      pick_types(self.info, eeg='True', exclude='bads'),
                      pick_types(self.info, meg='grad', exclude='bads')]

        for pick_one in picks_list:
            if self.with_mean:
                X[:, pick_one, :] -= self.ch_mean_
            if self.with_std:
                X[:, pick_one, :] /= self.ch_std_

        return X

    def fit_transform(self, epochs_data, y):
        """
        Standardizes data across channels

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.

        y : array
            The label for each epoch

        Returns
        -------
        ndarray : shape (n_epochs, n_channels*n_times)
            The data concatenated over channels
        """
        return self.fit(epochs_data, y).transform(epochs_data)


class ConcatenateChannels(TransformerMixin):

    def __init__(self, info=None):
        self.info = info

    def fit(self, epochs_data, y):
        """
        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data to concatenate channels
        y : array
            The label for each epoch

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
        """
        Concatenates data from different channels into a single feature vector

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray of shape (n_epochs, n_channels*n_times)
            The data concatenated over channels
        """
        n_epochs, n_channels, n_time = epochs_data.shape
        X = epochs_data.reshape(n_epochs, n_channels*n_time)

        return X

    def fit_transform(self, epochs_data, y):
        """
        Concatenates data from different channels into single feature vector

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.

        y : array
            The label for each epoch

        Returns
        -------
        ndarray : shape (n_epochs, n_channels*n_times)
            The data concatenated over channels
        """
        return self.fit(epochs_data, y).transform(epochs_data)


class PSDEstimator(TransformerMixin):
    """
    Compute power spectrum density (PSD) using a multi-taper method

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    def __init__(self, sfreq=2 * np.pi, fmin=0, fmax=np.inf, bandwidth=None,
                 adaptive=False, low_bias=True, n_jobs=1, verbose=None):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.low_bias = low_bias
        self.j_jobs = n_jobs
        self.verbose = verbose

    def fit(self, epochs_data, y):
        """
        Compute power spectrum density (PSD) using a multi-taper method

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.
        y : array
            The label for each epoch

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
        """
        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data

        Returns
        -------
        psd : array, shape=(n_signals, len(freqs)) or (len(freqs),)
            The computed PSD.
        """
        return multitaper_psd(epochs_data, self.sfreq, self.fmin, self.fmax,
                              self.bandwidth, self.adaptive, self.low_bias,
                              self.n_jobs, self.verbose)

    def fit_transform(self, epochs_data, y):
        """
        Compute power spectrum density (PSD) using a multi-taper method

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.

        y : array
            The label for each epoch

        Returns
        -------
        psd : array, shape=(n_signals, len(freqs)) or (len(freqs),)
            The computed PSD.
        """
        return self.fit(epochs_data, y).transform(epochs_data)


class FilterEstimator(TransformerMixin):
    """
    Estimator to filter Rt_Epochs
    """

    def __init__(self, info, l_freq, h_freq, picks=None, filter_length='10s',
                 l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, n_jobs=1,
                 method='fft', iir_params=dict(order=4, ftype='butter'),
                 verbose=None):
        """
        Filter a subset of channels.

        Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
        filter to the channels selected by "picks".

        l_freq and h_freq are the frequencies below which and above which,
        respectively, to filter out of the data. Thus the uses are:
            l_freq < h_freq: band-pass filter
            l_freq > h_freq: band-stop filter
            l_freq is not None, h_freq is None: low-pass filter
            l_freq is None, h_freq is not None: high-pass filter

        Note: If n_jobs > 1, more memory is required as "len(picks) * n_times"
              additional time points need to be temporarily stored in memory.

        Parameters
        ----------
        info: dict
            Measurement info.
        l_freq : float | None
            Low cut-off frequency in Hz. If None the data are only low-passed.
        h_freq : float | None
            High cut-off frequency in Hz. If None the data are only
            high-passed.
        picks : list of int | None
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
        iir_params : dict
            Dictionary of parameters to use for IIR filtering.
            See mne.filter.construct_iir_filter for details.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """

        self.info = info
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.picks = picks
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.n_jobs = n_jobs
        self.method = method
        self.iir_params = iir_params
        self.verbose = verbose

    def fit(self, epochs_data, y):
        """
        Filters data

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        self : instance of FilterEstimator
            returns the modified instance
        """

        if self.picks is None:
            self.picks = pick_types(self.info, meg=True, eeg=True, exclude=[])

        if self.l_freq == 0:
            self.l_freq = None
        if self.h_freq > (self.info['sfreq'] / 2.):
            self.h_freq = None

        if self.h_freq is not None and \
                (self.l_freq is None or self.l_freq < self.h_freq) and \
                self.h_freq < self.info['lowpass']:
            self.info['lowpass'] = self.h_freq

        if self.l_freq is not None and \
                (self.h_freq is None or self.l_freq < self.h_freq) and \
                self.l_freq > self.info['highpass']:
            self.info['highpass'] = self.l_freq

        return self

    @verbose
    def transform(self, epochs_data, y=None):
        """
        Filters data

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : array, shape=(n_epochs, n_channels, n_times)
            The data after filtering
        """
        if self.l_freq is None and self.h_freq is not None:
            logger.info('Low-pass filtering at %0.2g Hz' % self.h_freq)
            epochs_data = \
                low_pass_filter(epochs_data, self.fs, self.h_freq,
                                filter_length=self.filter_length,
                                trans_bandwidth=self.l_trans_bandwidth,
                                method=self.method, iir_params=self.iir_params,
                                picks=self.picks, n_jobs=self.n_jobs,
                                copy=False)

        if self.l_freq is not None and self.h_freq is None:
            logger.info('High-pass filtering at %0.2g Hz' % self.l_freq)

            epochs_data = \
                high_pass_filter(epochs_data, self.info['sfreq'], self.l_freq,
                                 filter_length=self.filter_length,
                                 trans_bandwidth=self.h_trans_bandwidth,
                                 method=self.method,
                                 iir_params=self.iir_params,
                                 picks=self.picks, n_jobs=self.n_jobs,
                                 copy=False)

        if self.l_freq is not None and self.h_freq is not None:
            if self.l_freq < self.h_freq:
                logger.info('Band-pass filtering from %0.2g - %0.2g Hz'
                            % (self.l_freq, self.h_freq))
                epochs_data = \
                    band_pass_filter(epochs_data, self.info['sfreq'],
                                     self.l_freq, self.h_freq,
                                     filter_length=self.filter_length,
                                     l_trans_bandwidth=self.l_trans_bandwidth,
                                     h_trans_bandwidth=self.h_trans_bandwidth,
                                     method=self.method,
                                     iir_params=self.iir_params,
                                     picks=self.picks, n_jobs=self.n_jobs,
                                     copy=False)
            else:
                logger.info('Band-stop filtering from %0.2g - %0.2g Hz'
                            % (self.h_freq, self.l_freq))
                epochs_data = \
                    band_stop_filter(epochs_data, self.info['sfreq'],
                                     self.h_freq, self.l_freq,
                                     filter_length=self.filter_length,
                                     l_trans_bandwidth=self.h_trans_bandwidth,
                                     h_trans_bandwidth=self.l_trans_bandwidth,
                                     method=self.method,
                                     iir_params=self.iir_params,
                                     picks=self.picks, n_jobs=self.n_jobs,
                                     copy=False)
        return epochs_data

    def fit_transform(self, epochs_data, y):
        """
        Filters data

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.

        y : array
            The label for each epoch

        Returns
        -------
        array, shape=(n_epochs, n_channels, n_times)
        The data after filtering
        """
        return self.fit(epochs_data, y).transform(epochs_data)
