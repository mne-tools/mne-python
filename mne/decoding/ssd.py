# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from .._fiff.pick import _picks_to_idx
from ..cov import Covariance, _regularized_covariance
from ..defaults import _handle_default
from ..filter import filter_data
from ..rank import compute_rank
from ..time_frequency import psd_array_welch
from ..utils import (
    _check_option,
    _time_mask,
    _validate_type,
    _verbose_safe_false,
    fill_doc,
    logger,
)


@fill_doc
class SSD(TransformerMixin, BaseEstimator):
    """
    Signal decomposition using the Spatio-Spectral Decomposition (SSD).

    SSD seeks to maximize the power at a frequency band of interest while
    simultaneously minimizing it at the flanking (surrounding) frequency bins
    (considered noise). It extremizes the covariance matrices associated with
    signal and noise :footcite:`NikulinEtAl2011`.

    SSD can either be used as a dimensionality reduction method or a
    ‘denoised’ low rank factorization method :footcite:`HaufeEtAl2014b`.

    Parameters
    ----------
    %(info_not_none)s Must match the input data.
    filt_params_signal : dict
        Filtering for the frequencies of interest.
    filt_params_noise : dict
        Filtering for the frequencies of non-interest.
    reg : float | str | None (default)
        Which covariance estimator to use.
        If not None (same as 'empirical'), allow regularization for covariance
        estimation. If float, shrinkage is used (0 <= shrinkage <= 1). For str
        options, reg will be passed to method :func:`mne.compute_covariance`.
    n_components : int | None (default None)
        The number of components to extract from the signal.
        If None, the number of components equal to the rank of the data are
        returned (see ``rank``).
    picks : array of int | None (default None)
        The indices of good channels.
    sort_by_spectral_ratio : bool (default True)
        If set to True, the components are sorted according to the spectral
        ratio.
        See Eq. (24) in :footcite:`NikulinEtAl2011`.
    return_filtered : bool (default False)
        If return_filtered is True, data is bandpassed and projected onto the
        SSD components.
    n_fft : int (default None)
       If sort_by_spectral_ratio is set to True, then the SSD sources will be
       sorted according to their spectral ratio which is calculated based on
       :func:`mne.time_frequency.psd_array_welch`. The n_fft parameter sets the
       length of FFT used.
       See :func:`mne.time_frequency.psd_array_welch` for more information.
    cov_method_params : dict | None (default None)
        As in :class:`mne.decoding.SPoC`
        The default is None.
    rank : None | dict | ‘info’ | ‘full’
        As in :class:`mne.decoding.SPoC`
        This controls the rank computation that can be read from the
        measurement info or estimated from the data, which determines the
        maximum possible number of components.
        See Notes of :func:`mne.compute_rank` for details.
        We recommend to use 'full' when working with epoched data.

    Attributes
    ----------
    filters_ : array, shape (n_channels, n_components)
        The spatial filters to be multiplied with the signal.
    patterns_ : array, shape (n_components, n_channels)
        The patterns for reconstructing the signal from the filtered data.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        info,
        filt_params_signal,
        filt_params_noise,
        reg=None,
        n_components=None,
        picks=None,
        sort_by_spectral_ratio=True,
        return_filtered=False,
        n_fft=None,
        cov_method_params=None,
        rank=None,
    ):
        """Initialize instance."""
        dicts = {"signal": filt_params_signal, "noise": filt_params_noise}
        for param, dd in [("l", 0), ("h", 0), ("l", 1), ("h", 1)]:
            key = ("signal", "noise")[dd]
            if param + "_freq" not in dicts[key]:
                raise ValueError(
                    f"{param + '_freq'} must be defined in filter parameters for {key}"
                )
            val = dicts[key][param + "_freq"]
            if not isinstance(val, int | float):
                _validate_type(val, ("numeric",), f"{key} {param}_freq")
        # check freq bands
        if (
            filt_params_noise["l_freq"] > filt_params_signal["l_freq"]
            or filt_params_signal["h_freq"] > filt_params_noise["h_freq"]
        ):
            raise ValueError(
                "Wrongly specified frequency bands!\n"
                "The signal band-pass must be within the noise "
                "band-pass!"
            )
        self.picks = picks
        del picks
        self.info = info
        self.freqs_signal = (filt_params_signal["l_freq"], filt_params_signal["h_freq"])
        self.freqs_noise = (filt_params_noise["l_freq"], filt_params_noise["h_freq"])
        self.filt_params_signal = filt_params_signal
        self.filt_params_noise = filt_params_noise
        # check if boolean
        if not isinstance(sort_by_spectral_ratio, (bool)):
            raise ValueError("sort_by_spectral_ratio must be boolean")
        self.sort_by_spectral_ratio = sort_by_spectral_ratio
        if n_fft is None:
            self.n_fft = int(self.info["sfreq"])
        else:
            self.n_fft = int(n_fft)
        # check if boolean
        if not isinstance(return_filtered, (bool)):
            raise ValueError("return_filtered must be boolean")
        self.return_filtered = return_filtered
        self.reg = reg
        self.n_components = n_components
        self.rank = rank
        self.cov_method_params = cov_method_params

    def _check_X(self, X):
        """Check input data."""
        _validate_type(X, np.ndarray, "X")
        _check_option("X.ndim", X.ndim, (2, 3))
        n_chan = X.shape[-2]
        if n_chan != self.info["nchan"]:
            raise ValueError(
                "Info must match the input data."
                f"Found {n_chan} channels but expected {self.info['nchan']}."
            )

    def fit(self, X, y=None):
        """Estimate the SSD decomposition on raw or epoched data.

        Parameters
        ----------
        X : array, shape ([n_epochs, ]n_channels, n_times)
            The input data from which to estimate the SSD. Either 2D array
            obtained from continuous data or 3D array obtained from epoched
            data.
        y : None
            Ignored; exists for compatibility with scikit-learn pipelines.

        Returns
        -------
        self : instance of SSD
            Returns the modified instance.
        """
        ch_types = self.info.get_channel_types(picks=self.picks, unique=True)
        if len(ch_types) > 1:
            raise ValueError(
                "At this point SSD only supports fitting "
                f"single channel types. Your info has {len(ch_types)} types."
            )
        self.picks_ = _picks_to_idx(self.info, self.picks, none="data", exclude="bads")
        self._check_X(X)
        X_aux = X[..., self.picks_, :]

        X_signal = filter_data(X_aux, self.info["sfreq"], **self.filt_params_signal)
        X_noise = filter_data(X_aux, self.info["sfreq"], **self.filt_params_noise)
        X_noise -= X_signal
        if X.ndim == 3:
            X_signal = np.hstack(X_signal)
            X_noise = np.hstack(X_noise)

        # prevent rank change when computing cov with rank='full'
        cov_signal = _regularized_covariance(
            X_signal,
            reg=self.reg,
            method_params=self.cov_method_params,
            rank="full",
            info=self.info,
        )
        cov_noise = _regularized_covariance(
            X_noise,
            reg=self.reg,
            method_params=self.cov_method_params,
            rank="full",
            info=self.info,
        )

        # project cov to rank subspace
        cov_signal, cov_noise, rank_proj = _dimensionality_reduction(
            cov_signal, cov_noise, self.info, self.rank
        )

        eigvals_, eigvects_ = eigh(cov_signal, cov_noise)
        # sort in descending order
        ix = np.argsort(eigvals_)[::-1]
        self.eigvals_ = eigvals_[ix]
        # project back to sensor space
        self.filters_ = np.matmul(rank_proj, eigvects_[:, ix])
        self.patterns_ = np.linalg.pinv(self.filters_)

        # We assume that ordering by spectral ratio is more important
        # than the initial ordering. This ordering should be also learned when
        # fitting.
        X_ssd = self.filters_.T @ X[..., self.picks_, :]
        sorter_spec = Ellipsis
        if self.sort_by_spectral_ratio:
            _, sorter_spec = self.get_spectral_ratio(ssd_sources=X_ssd)
        self.sorter_spec = sorter_spec
        logger.info("Done.")
        return self

    def transform(self, X):
        """Estimate epochs sources given the SSD filters.

        Parameters
        ----------
        X : array, shape ([n_epochs, ]n_channels, n_times)
            The input data from which to estimate the SSD. Either 2D array
            obtained from continuous data or 3D array obtained from epoched
            data.

        Returns
        -------
        X_ssd : array, shape ([n_epochs, ]n_components, n_times)
            The processed data.
        """
        self._check_X(X)
        if self.filters_ is None:
            raise RuntimeError("No filters available. Please first call fit")
        if self.return_filtered:
            X_aux = X[..., self.picks_, :]
            X = filter_data(X_aux, self.info["sfreq"], **self.filt_params_signal)
        X_ssd = self.filters_.T @ X[..., self.picks_, :]
        if X.ndim == 2:
            X_ssd = X_ssd[self.sorter_spec][: self.n_components]
        else:
            X_ssd = X_ssd[:, self.sorter_spec, :][:, : self.n_components, :]
        return X_ssd

    def fit_transform(self, X, y=None, **fit_params):
        """Fit SSD to data, then transform it.

        Fits transformer to ``X`` and ``y`` with optional parameters ``fit_params``, and
        returns a transformed version of ``X``.

        Parameters
        ----------
        X : array, shape ([n_epochs, ]n_channels, n_times)
            The input data from which to estimate the SSD. Either 2D array obtained from
            continuous data or 3D array obtained from epoched data.
        y : None
            Ignored; exists for compatibility with scikit-learn pipelines.
        **fit_params : dict
            Additional fitting parameters passed to the :meth:`mne.decoding.SSD.fit`
            method. Not used for this class.

        Returns
        -------
        X_ssd : array, shape ([n_epochs, ]n_components, n_times)
            The processed data.
        """
        # use parent TransformerMixin method but with custom docstring
        return super().fit_transform(X, y=y, **fit_params)

    def get_spectral_ratio(self, ssd_sources):
        """Get the spectal signal-to-noise ratio for each spatial filter.

        Spectral ratio measure for best n_components selection
        See :footcite:`NikulinEtAl2011`, Eq. (24).

        Parameters
        ----------
        ssd_sources : array
            Data projected to SSD space.

        Returns
        -------
        spec_ratio : array, shape (n_channels)
            Array with the sprectal ratio value for each component.
        sorter_spec : array, shape (n_channels)
            Array of indices for sorting spec_ratio.

        References
        ----------
        .. footbibliography::
        """
        psd, freqs = psd_array_welch(
            ssd_sources, sfreq=self.info["sfreq"], n_fft=self.n_fft
        )
        sig_idx = _time_mask(freqs, *self.freqs_signal)
        noise_idx = _time_mask(freqs, *self.freqs_noise)
        if psd.ndim == 3:
            mean_sig = psd[:, :, sig_idx].mean(axis=2).mean(axis=0)
            mean_noise = psd[:, :, noise_idx].mean(axis=2).mean(axis=0)
            spec_ratio = mean_sig / mean_noise
        else:
            mean_sig = psd[:, sig_idx].mean(axis=1)
            mean_noise = psd[:, noise_idx].mean(axis=1)
            spec_ratio = mean_sig / mean_noise
        sorter_spec = spec_ratio.argsort()[::-1]
        return spec_ratio, sorter_spec

    def inverse_transform(self):
        """Not implemented yet."""
        raise NotImplementedError("inverse_transform is not yet available.")

    def apply(self, X):
        """Remove selected components from the signal.

        This procedure will reconstruct M/EEG signals from which the dynamics
        described by the excluded components is subtracted
        (denoised by low-rank factorization).
        See :footcite:`HaufeEtAl2014b` for more information.

        .. note:: Unlike in other classes with an apply method,
           only NumPy arrays are supported (not instances of MNE objects).

        Parameters
        ----------
        X : array, shape ([n_epochs, ]n_channels, n_times)
            The input data from which to estimate the SSD. Either 2D array
            obtained from continuous data or 3D array obtained from epoched
            data.

        Returns
        -------
        X : array, shape ([n_epochs, ]n_channels, n_times)
            The processed data.
        """
        X_ssd = self.transform(X)
        pick_patterns = self.patterns_[self.sorter_spec][: self.n_components].T
        X = pick_patterns @ X_ssd
        return X


def _dimensionality_reduction(cov_signal, cov_noise, info, rank):
    """Perform dimensionality reduction on the covariance matrices."""
    n_channels = cov_signal.shape[0]

    # find ranks of covariance matrices
    rank_signal = list(
        compute_rank(
            Covariance(
                cov_signal,
                info.ch_names,
                list(),
                list(),
                0,
                verbose=_verbose_safe_false(),
            ),
            rank,
            _handle_default("scalings_cov_rank", None),
            info,
        ).values()
    )[0]
    rank_noise = list(
        compute_rank(
            Covariance(
                cov_noise,
                info.ch_names,
                list(),
                list(),
                0,
                verbose=_verbose_safe_false(),
            ),
            rank,
            _handle_default("scalings_cov_rank", None),
            info,
        ).values()
    )[0]
    rank = np.min([rank_signal, rank_noise])  # should be identical

    if rank < n_channels:
        eigvals, eigvects = eigh(cov_signal)
        # sort in descending order
        ix = np.argsort(eigvals)[::-1]
        eigvals = eigvals[ix]
        eigvects = eigvects[:, ix]
        # compute rank subspace projection matrix
        rank_proj = np.matmul(
            eigvects[:, :rank], np.eye(rank) * (eigvals[:rank] ** -0.5)
        )
        logger.info(
            "Projecting covariance of %i channels to %i rank subspace",
            n_channels,
            rank,
        )
    else:
        rank_proj = np.eye(n_channels)
        logger.info("Preserving covariance rank (%i)", rank)

    # project covariance matrices to rank subspace
    cov_signal = np.matmul(rank_proj.T, np.matmul(cov_signal, rank_proj))
    cov_noise = np.matmul(rank_proj.T, np.matmul(cov_noise, rank_proj))
    return cov_signal, cov_noise, rank_proj
