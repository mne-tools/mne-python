# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, check_array, clone
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from .._fiff.pick import (
    _pick_data_channels,
    _picks_by_type,
    _picks_to_idx,
    pick_info,
)
from ..cov import _check_scalings_user
from ..epochs import BaseEpochs
from ..filter import filter_data
from ..time_frequency import psd_array_multitaper
from ..utils import _check_option, _validate_type, check_version, fill_doc
from ._fixes import validate_data  # TODO VERSION remove with sklearn 1.4+


class MNETransformerMixin(TransformerMixin):
    """TransformerMixin plus some helpers."""

    def _check_data(
        self,
        epochs_data,
        *,
        y=None,
        atleast_3d=True,
        fit=False,
        return_y=False,
        multi_output=False,
        check_n_features=True,
    ):
        # Sklearn calls asarray under the hood which works, but elsewhere they check for
        # __len__ then look at the size of obj[0]... which is an epoch of shape (1, ...)
        # rather than what they expect (shape (...)). So we explicitly get the NumPy
        # array to make everyone happy.
        if isinstance(epochs_data, BaseEpochs):
            epochs_data = epochs_data.get_data(copy=False)
        kwargs = dict(dtype=np.float64, allow_nd=True, order="C")
        if check_version("sklearn", "1.4"):  # TODO VERSION sklearn 1.4+
            kwargs["force_writeable"] = True
        if hasattr(self, "n_features_in_") and check_n_features:
            if y is None:
                epochs_data = validate_data(
                    self,
                    epochs_data,
                    **kwargs,
                    reset=fit,
                )
            else:
                epochs_data, y = validate_data(
                    self,
                    epochs_data,
                    y,
                    **kwargs,
                    reset=fit,
                )
        elif y is None:
            epochs_data = check_array(epochs_data, **kwargs)
        else:
            epochs_data, y = check_X_y(
                X=epochs_data, y=y, multi_output=multi_output, **kwargs
            )
        if fit:
            self.n_features_in_ = epochs_data.shape[1]
        if atleast_3d:
            epochs_data = np.atleast_3d(epochs_data)
        return (epochs_data, y) if return_y else epochs_data


class _ConstantScaler:
    """Scale channel types using constant values."""

    def __init__(self, info, scalings, do_scaling=True):
        self._scalings = scalings
        self._info = info
        self._do_scaling = do_scaling

    def fit(self, X, y=None):
        scalings = _check_scalings_user(self._scalings)
        picks_by_type = _picks_by_type(
            pick_info(self._info, _pick_data_channels(self._info, exclude=()))
        )
        std = np.ones(sum(len(p[1]) for p in picks_by_type))
        if X.shape[1] != len(std):
            raise ValueError(
                f"info had {len(std)} data channels but X has {len(X)} channels"
            )
        if self._do_scaling:  # this is silly, but necessary for completeness
            for kind, picks in picks_by_type:
                std[picks] = 1.0 / scalings[kind]
        self.std_ = std
        self.mean_ = np.zeros_like(std)
        return self

    def transform(self, X):
        return X / self.std_

    def inverse_transform(self, X, y=None):
        return X * self.std_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def _sklearn_reshape_apply(func, return_result, X, *args, **kwargs):
    """Reshape epochs and apply function."""
    _validate_type(X, np.ndarray, "X")
    if X.size == 0:
        return X.copy() if return_result else None
    orig_shape = X.shape
    X = np.reshape(X.transpose(0, 2, 1), (-1, orig_shape[1]))
    X = func(X, *args, **kwargs)
    if return_result:
        X.shape = (orig_shape[0], orig_shape[2], orig_shape[1])
        X = X.transpose(0, 2, 1)
        return X


@fill_doc
class Scaler(MNETransformerMixin, BaseEstimator):
    """Standardize channel data.

    This class scales data for each channel. It differs from scikit-learn
    classes (e.g., :class:`sklearn.preprocessing.StandardScaler`) in that
    it scales each *channel* by estimating μ and σ using data from all
    time points and epochs, as opposed to standardizing each *feature*
    (i.e., each time point for each channel) by estimating using μ and σ
    using data from all epochs.

    Parameters
    ----------
    %(info)s Only necessary if ``scalings`` is a dict or None.
    scalings : dict, str, default None
        Scaling method to be applied to data channel wise.

        * if scalings is None (default), scales mag by 1e15, grad by 1e13,
          and eeg by 1e6.
        * if scalings is :class:`dict`, keys are channel types and values
          are scale factors.
        * if ``scalings=='median'``,
          :class:`sklearn.preprocessing.RobustScaler`
          is used (requires sklearn version 0.17+).
        * if ``scalings=='mean'``,
          :class:`sklearn.preprocessing.StandardScaler`
          is used.

    with_mean : bool, default True
        If True, center the data using mean (or median) before scaling.
        Ignored for channel-type scaling.
    with_std : bool, default True
        If True, scale the data to unit variance (``scalings='mean'``),
        quantile range (``scalings='median``), or using channel type
        if ``scalings`` is a dict or None).
    """

    def __init__(self, info=None, scalings=None, with_mean=True, with_std=True):
        self.info = info
        self.with_mean = with_mean
        self.with_std = with_std
        self.scalings = scalings

    def fit(self, epochs_data, y=None):
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
            The modified instance.
        """
        epochs_data = self._check_data(epochs_data, y=y, fit=True, multi_output=True)
        assert epochs_data.ndim == 3, epochs_data.shape

        _validate_type(self.scalings, (dict, str, type(None)), "scalings")
        if isinstance(self.scalings, str):
            _check_option(
                "scalings", self.scalings, ["mean", "median"], extra="when str"
            )
        if self.scalings is None or isinstance(self.scalings, dict):
            if self.info is None:
                raise ValueError(
                    f'Need to specify "info" if scalings is {type(self.scalings)}'
                )
            self.scaler_ = _ConstantScaler(self.info, self.scalings, self.with_std)
        elif self.scalings == "mean":
            self.scaler_ = StandardScaler(
                with_mean=self.with_mean, with_std=self.with_std
            )
        else:  # scalings == 'median':
            self.scaler_ = RobustScaler(
                with_centering=self.with_mean, with_scaling=self.with_std
            )

        _sklearn_reshape_apply(self.scaler_.fit, False, epochs_data, y=y)
        return self

    def transform(self, epochs_data):
        """Standardize data across channels.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels[, n_times])
            The data.

        Returns
        -------
        X : array, shape (n_epochs, n_channels, n_times)
            The data concatenated over channels.

        Notes
        -----
        This function makes a copy of the data before the operations and the
        memory usage may be large with big data.
        """
        check_is_fitted(self, "scaler_")
        epochs_data = self._check_data(epochs_data, atleast_3d=False)
        if epochs_data.ndim == 2:  # can happen with SlidingEstimator
            if self.info is not None:
                assert len(self.info["ch_names"]) == epochs_data.shape[1]
            epochs_data = epochs_data[..., np.newaxis]
        assert epochs_data.ndim == 3, epochs_data.shape
        return _sklearn_reshape_apply(self.scaler_.transform, True, epochs_data)

    def fit_transform(self, epochs_data, y=None):
        """Fit to data, then transform it.

        Fits transformer to epochs_data and y and returns a transformed version
        of epochs_data.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : None | array, shape (n_epochs,)
            The label for each epoch.
            Defaults to None.

        Returns
        -------
        X : array, shape (n_epochs, n_channels, n_times)
            The data concatenated over channels.

        Notes
        -----
        This function makes a copy of the data before the operations and the
        memory usage may be large with big data.
        """
        return self.fit(epochs_data, y).transform(epochs_data)

    def inverse_transform(self, epochs_data):
        """Invert standardization of data across channels.

        Parameters
        ----------
        epochs_data : array, shape ([n_epochs, ]n_channels, n_times)
            The data.

        Returns
        -------
        X : array, shape (n_epochs, n_channels, n_times)
            The data concatenated over channels.

        Notes
        -----
        This function makes a copy of the data before the operations and the
        memory usage may be large with big data.
        """
        epochs_data = self._check_data(epochs_data, atleast_3d=False)
        squeeze = False
        # Can happen with CSP
        if epochs_data.ndim == 2:
            squeeze = True
            epochs_data = epochs_data[..., np.newaxis]
        assert epochs_data.ndim == 3, epochs_data.shape
        out = _sklearn_reshape_apply(self.scaler_.inverse_transform, True, epochs_data)
        if squeeze:
            out = out[..., 0]
        return out


class Vectorizer(MNETransformerMixin, BaseEstimator):
    """Transform n-dimensional array into 2D array of n_samples by n_features.

    This class reshapes an n-dimensional array into an n_samples * n_features
    array, usable by the estimators and transformers of scikit-learn.

    Attributes
    ----------
    features_shape_ : tuple
         Stores the original shape of data.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> clf = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression())
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
        self : instance of Vectorizer
            Return the modified instance.
        """
        X = self._check_data(X, y=y, atleast_3d=False, fit=True, check_n_features=False)
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
        X = self._check_data(X, atleast_3d=False)
        if X.shape[1:] != self.features_shape_:
            raise ValueError("Shape of X used in fit and transform must be same")
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
        X = self._check_data(X, atleast_3d=False, check_n_features=False)
        if X.ndim not in (2, 3):
            raise ValueError(
                f"X should be of 2 or 3 dimensions but has shape {X.shape}"
            )
        return X.reshape(X.shape[:-1] + self.features_shape_)


@fill_doc
class PSDEstimator(MNETransformerMixin, BaseEstimator):
    """Compute power spectral density (PSD) using a multi-taper method.

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
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    n_jobs : int
        Number of parallel jobs to use (only used if adaptive=True).
    %(normalization)s

    See Also
    --------
    mne.time_frequency.psd_array_multitaper
    mne.io.Raw.compute_psd
    mne.Epochs.compute_psd
    mne.Evoked.compute_psd
    """

    def __init__(
        self,
        sfreq=2 * np.pi,
        fmin=0,
        fmax=np.inf,
        bandwidth=None,
        adaptive=False,
        low_bias=True,
        n_jobs=None,
        normalization="length",
    ):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.low_bias = low_bias
        self.n_jobs = n_jobs
        self.normalization = normalization

    def fit(self, epochs_data, y=None):
        """Compute power spectral density (PSD) using a multi-taper method.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : array, shape (n_epochs,)
            The label for each epoch.

        Returns
        -------
        self : instance of PSDEstimator
            The modified instance.
        """
        self._check_data(epochs_data, y=y, fit=True)
        self.fitted_ = True  # sklearn compliance
        return self

    def transform(self, epochs_data):
        """Compute power spectral density (PSD) using a multi-taper method.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        psd : array, shape (n_signals, n_freqs) or (n_freqs,)
            The computed PSD.
        """
        epochs_data = self._check_data(epochs_data)
        psd, _ = psd_array_multitaper(
            epochs_data,
            sfreq=self.sfreq,
            fmin=self.fmin,
            fmax=self.fmax,
            bandwidth=self.bandwidth,
            adaptive=self.adaptive,
            low_bias=self.low_bias,
            normalization=self.normalization,
            n_jobs=self.n_jobs,
        )
        return psd


@fill_doc
class FilterEstimator(MNETransformerMixin, BaseEstimator):
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
    %(info_not_none)s
    %(l_freq)s
    %(h_freq)s
    %(picks_good_data)s
    %(filter_length)s
    %(l_trans_bandwidth)s
    %(h_trans_bandwidth)s
    n_jobs : int | str
        Number of jobs to run in parallel.
        Can be 'cuda' if ``cupy`` is installed properly and method='fir'.
    method : str
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR filtering.
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    %(fir_design)s

    See Also
    --------
    TemporalFilter

    Notes
    -----
    This is primarily meant for use in realtime applications.
    In general it is not recommended in a normal processing pipeline as it may result
    in edge artifacts. Use with caution.
    """

    def __init__(
        self,
        info,
        l_freq,
        h_freq,
        picks=None,
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        n_jobs=None,
        method="fir",
        iir_params=None,
        fir_design="firwin",
    ):
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
        self.fir_design = fir_design

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
            The modified instance.
        """
        self.picks_ = _picks_to_idx(self.info, self.picks)
        self._check_data(epochs_data, y=y, fit=True)

        if self.l_freq == 0:
            self.l_freq = None

        if self.info["lowpass"] is None or (
            self.h_freq is not None
            and (self.l_freq is None or self.l_freq < self.h_freq)
            and self.h_freq < self.info["lowpass"]
        ):
            with self.info._unlock():
                self.info["lowpass"] = self.h_freq

        if self.info["highpass"] is None or (
            self.l_freq is not None
            and (self.h_freq is None or self.l_freq < self.h_freq)
            and self.l_freq > self.info["highpass"]
        ):
            with self.info._unlock():
                self.info["highpass"] = self.l_freq

        return self

    def transform(self, epochs_data):
        """Filter data.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : array, shape (n_epochs, n_channels, n_times)
            The data after filtering.
        """
        return filter_data(
            self._check_data(epochs_data),
            self.info["sfreq"],
            self.l_freq,
            self.h_freq,
            self.picks_,
            self.filter_length,
            self.l_trans_bandwidth,
            self.h_trans_bandwidth,
            method=self.method,
            iir_params=self.iir_params,
            n_jobs=self.n_jobs,
            copy=False,
            fir_design=self.fir_design,
            verbose=False,
        )


class UnsupervisedSpatialFilter(MNETransformerMixin, BaseEstimator):
    """Use unsupervised spatial filtering across time and samples.

    Parameters
    ----------
    estimator : instance of sklearn.base.BaseEstimator
        Estimator using some decomposition algorithm.
    average : bool, default False
        If True, the estimator is fitted on the average across samples
        (e.g. epochs).
    """

    def __init__(self, estimator, average=False):
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
        self : instance of UnsupervisedSpatialFilter
            Return the modified instance.
        """
        # sklearn.utils.estimator_checks.check_estimator(self.estimator) is probably
        # too strict for us, given that we don't fully adhere yet, so just check attrs
        for attr in ("fit", "transform", "fit_transform"):
            if not hasattr(self.estimator, attr):
                raise ValueError(
                    "estimator must be a scikit-learn "
                    f"transformer, missing {attr} method"
                )
        _validate_type(self.average, bool, "average")
        X = self._check_data(X, y=y, fit=True)
        if self.average:
            X = np.mean(X, axis=0).T
        else:
            n_epochs, n_channels, n_times = X.shape
            # trial as time samples
            X = np.transpose(X, (1, 0, 2)).reshape((n_channels, n_epochs * n_times)).T

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X)
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
        X : array, shape (n_epochs, n_channels, n_times)
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
        X : array, shape (n_epochs, n_channels, n_times)
            The transformed data.
        """
        check_is_fitted(self.estimator_)
        X = self._check_data(X)
        return self._apply_method(X, "transform")

    def inverse_transform(self, X):
        """Inverse transform the data to its original space.

        Parameters
        ----------
        X : array, shape (n_epochs, n_components, n_times)
            The data to be inverted.

        Returns
        -------
        X : array, shape (n_epochs, n_channels, n_times)
            The transformed data.
        """
        return self._apply_method(X, "inverse_transform")

    def _apply_method(self, X, method):
        """Vectorize time samples as trials, apply method and reshape back.

        Parameters
        ----------
        X : array, shape (n_epochs, n_dims, n_times)
            The data to be inverted.

        Returns
        -------
        X : array, shape (n_epochs, n_dims, n_times)
            The transformed data.
        """
        n_epochs, n_channels, n_times = X.shape
        # trial as time samples
        X = np.transpose(X, [1, 0, 2])
        X = np.reshape(X, [n_channels, n_epochs * n_times]).T
        # apply method
        method = getattr(self.estimator_, method)
        X = method(X)
        # put it back to n_epochs, n_dimensions
        X = np.reshape(X.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
        return X


@fill_doc
class TemporalFilter(MNETransformerMixin, BaseEstimator):
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
    sfreq : float, default 1.0
        Sampling frequency in Hz.
    filter_length : str | int, default 'auto'
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
    n_jobs : int | str, default 1
        Number of jobs to run in parallel.
        Can be 'cuda' if ``cupy`` is installed properly and method='fir'.
    method : str, default 'fir'
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None, default None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    fir_window : str, default 'hamming'
        The window to use in FIR design, can be "hamming", "hann",
        or "blackman".
    fir_design : str
        Can be "firwin" (default) to use :func:`scipy.signal.firwin`,
        or "firwin2" to use :func:`scipy.signal.firwin2`. "firwin" uses
        a time-domain design technique that generally gives improved
        attenuation using fewer samples than "firwin2".

        .. versionadded:: 0.15

    See Also
    --------
    FilterEstimator
    Vectorizer
    mne.filter.filter_data
    """

    def __init__(
        self,
        l_freq=None,
        h_freq=None,
        sfreq=1.0,
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        n_jobs=None,
        method="fir",
        iir_params=None,
        fir_window="hamming",
        fir_design="firwin",
    ):
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
        self.fir_design = fir_design

    def fit(self, X, y=None):
        """Do nothing (for scikit-learn compatibility purposes).

        Parameters
        ----------
        X : array, shape ([n_epochs, ]n_channels, n_times)
            The data to be filtered over the last dimension. The channels
            dimension can be zero when passing a 2D array.
        y : None
            Not used, for scikit-learn compatibility issues.

        Returns
        -------
        self : instance of TemporalFilter
            The modified instance.
        """
        self.fitted_ = True  # sklearn compliance
        self._check_data(X, y=y, atleast_3d=False, fit=True)
        return self

    def transform(self, X):
        """Filter data along the last dimension.

        Parameters
        ----------
        X : array, shape ([n_epochs, ]n_channels, n_times)
            The data to be filtered over the last dimension. The channels
            dimension can be zero when passing a 2D array.

        Returns
        -------
        X : array
            The data after filtering.
        """  # noqa: E501
        X = self._check_data(X, atleast_3d=False)
        X = np.atleast_2d(X)

        if X.ndim > 3:
            raise ValueError(
                "Array must be of at max 3 dimensions instead "
                f"got {X.ndim} dimensional matrix"
            )

        shape = X.shape
        X = X.reshape(-1, shape[-1])
        X = filter_data(
            X,
            self.sfreq,
            self.l_freq,
            self.h_freq,
            filter_length=self.filter_length,
            l_trans_bandwidth=self.l_trans_bandwidth,
            h_trans_bandwidth=self.h_trans_bandwidth,
            n_jobs=self.n_jobs,
            method=self.method,
            iir_params=self.iir_params,
            copy=False,
            fir_window=self.fir_window,
            fir_design=self.fir_design,
        )
        return X.reshape(shape)
