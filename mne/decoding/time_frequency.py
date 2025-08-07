# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..time_frequency.tfr import _compute_tfr
from ..utils import _check_option, fill_doc
from .transformer import MNETransformerMixin


@fill_doc
class TimeFrequency(MNETransformerMixin, BaseEstimator):
    """Time frequency transformer.

    Time-frequency transform of times series along the last axis.

    Parameters
    ----------
    freqs : array-like of float, shape (n_freqs,)
        The frequencies.
    sfreq : float | int, default 1.0
        Sampling frequency of the data.
    method : 'multitaper' | 'morlet', default 'morlet'
        The time-frequency method. 'morlet' convolves a Morlet wavelet.
        'multitaper' uses Morlet wavelets windowed with multiple DPSS
        multitapers.
    n_cycles : float | array of float, default 7.0
        Number of cycles  in the Morlet wavelet. Fixed number
        or one per frequency.
    time_bandwidth : float, default None
        If None and method=multitaper, will be set to 4.0 (3 tapers).
        Time x (Full) Bandwidth product. Only applies if
        method == 'multitaper'. The number of good tapers (low-bias) is
        chosen automatically based on this to equal floor(time_bandwidth - 1).
    use_fft : bool, default True
        Use the FFT for convolutions or not.
    decim : int | slice, default 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts, yet decimation
                  is done after the convolutions.

    output : str, default 'complex'
        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
    %(n_jobs)s
        The number of epochs to process at the same time. The parallelization
        is implemented across channels.
    %(verbose)s

    See Also
    --------
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_multitaper
    """

    def __init__(
        self,
        freqs,
        sfreq=1.0,
        method="morlet",
        n_cycles=7.0,
        time_bandwidth=None,
        use_fft=True,
        decim=1,
        output="complex",
        n_jobs=1,
        verbose=None,
    ):
        """Init TimeFrequency transformer."""
        self.freqs = freqs
        self.sfreq = sfreq
        self.method = method
        self.n_cycles = n_cycles
        self.time_bandwidth = time_bandwidth
        self.use_fft = use_fft
        self.decim = decim
        # Check that output is not an average metric (e.g. ITC)
        self.output = output
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __sklearn_tags__(self):
        """Return sklearn tags."""
        out = super().__sklearn_tags__()
        from sklearn.utils import TransformerTags

        if out.transformer_tags is None:
            out.transformer_tags = TransformerTags()
        out.transformer_tags.preserves_dtype = []  # real->complex
        return out

    def fit_transform(self, X, y=None):
        """Time-frequency transform of times series along the last axis.

        Parameters
        ----------
        X : array, shape (n_samples, n_channels, n_times)
            The training data samples. The channel dimension can be zero- or
            1-dimensional.
        y : None
            For scikit-learn compatibility purposes.

        Returns
        -------
        Xt : array, shape (n_samples, n_channels, n_freqs, n_times)
            The time-frequency transform of the data, where n_channels can be
            zero- or 1-dimensional.
        """
        return self.fit(X, y).transform(X)

    def fit(self, X, y=None):  # noqa: D401
        """Do nothing (for scikit-learn compatibility purposes).

        Parameters
        ----------
        X : array, shape (n_samples, n_channels, n_times)
            The training data.
        y : array | None
            The target values.

        Returns
        -------
        self : object
            Return self.
        """
        # Check non-average output
        _check_option("output", self.output, ["complex", "power", "phase"])
        self._check_data(X, y=y, fit=True)
        self.fitted_ = True
        return self

    def transform(self, X):
        """Time-frequency transform of times series along the last axis.

        Parameters
        ----------
        X : array, shape (n_samples, [n_channels, ]n_times)
            The training data samples. The channel dimension can be zero- or
            1-dimensional.

        Returns
        -------
        Xt : array, shape (n_samples, [n_channels, ]n_freqs, n_times)
            The time-frequency transform of the data, where n_channels can be
            zero- or 1-dimensional.
        """
        X = self._check_data(X, atleast_3d=False)
        check_is_fitted(self, "fitted_")
        # Ensure 3-dimensional X
        shape = X.shape[1:-1]
        if not shape:
            X = X[:, np.newaxis, :]

        # Compute time-frequency
        Xt = _compute_tfr(
            X,
            freqs=self.freqs,
            sfreq=self.sfreq,
            method=self.method,
            n_cycles=self.n_cycles,
            zero_mean=True,
            time_bandwidth=self.time_bandwidth,
            use_fft=self.use_fft,
            decim=self.decim,
            output=self.output,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        # Back to original shape
        if not shape:
            Xt = Xt[:, 0, :]

        return Xt
