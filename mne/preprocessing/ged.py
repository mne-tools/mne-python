"""Generalized eigendecomposition (GED) preprocessing."""

import numpy as np

from ..io import BaseRaw
from ..time_frequency import psd_array_welch
from ..utils import _ensure_int, _validate_type
from ..viz.topomap import plot_topomap


class GED:
    """Generalized eigendecomposition for contrast-based preprocessing.

    GED estimates spatial filters that maximize variance in foreground data
    relative to background data by solving a generalized eigenvalue problem.
    This can be used for contrast-based artifact attenuation, for example by
    fitting GED on artifact-dominant data and reference/background data, then
    reconstructing the signal after excluding selected GED components.

    Parameters
    ----------
    n_components : int | None
        Number of GED components to retain. If None, all components are
        retained.
    reg : float
        Regularization amount for the background covariance matrix. The
        regularized covariance is computed as::
            (1 - reg) * C_bg + reg * mean(eig(C_bg)) * I
    normalize : bool
        If True, channel-wise z-score the data before covariance estimation.
        If False, channel-wise mean-center the data only.

    Attributes
    ----------
    filters_ : ndarray, shape (n_components, n_channels)
        GED spatial filters. Each row is one component.
    patterns_ : ndarray, shape (n_components, n_channels)
        GED spatial patterns.
    eigenvalues_ : ndarray, shape (n_components,)
        Generalized eigenvalues, sorted in descending order.
    foreground_cov_ : ndarray, shape (n_channels, n_channels)
        Foreground covariance matrix.
    background_cov_ : ndarray, shape (n_channels, n_channels)
        Background covariance matrix before regularization.
    background_cov_reg_ : ndarray, shape (n_channels, n_channels)
        Regularized background covariance matrix.
    n_components_ : int
        Number of retained GED components.
    explained_ratio_ : ndarray, shape (n_components,)
        Normalized GED eigenvalues. These values summarize the relative
        contribution of each retained component to the foreground/background
        contrast.

    Notes
    -----
    This implementation is adapted from contrast-based GED cleaning for
    ambulatory EEG artifact attenuation.
    """

    def __init__(self, n_components=None, reg=0.01, normalize=False):

        self.n_components = n_components

        self.reg = reg

        self.normalize = normalize

    def _get_fit_data(self, foreground, background, picks):
        """Get fitting data from Raw objects or arrays."""
        if isinstance(foreground, BaseRaw) or isinstance(background, BaseRaw):
            if not isinstance(foreground, BaseRaw) or not isinstance(
                background, BaseRaw
            ):
                raise ValueError(
                    "foreground and background must both be Raw objects, or both "
                    "be ndarray objects"
                )

            if foreground.ch_names != background.ch_names:
                raise ValueError(
                    "foreground and background Raw objects must have the same "
                    "channel names in the same order"
                )

            foreground_types = foreground.get_channel_types()
            background_types = background.get_channel_types()
            if foreground_types != background_types:
                raise ValueError(
                    "foreground and background Raw objects must have the same "
                    "channel types in the same order"
                )

            foreground_data = foreground.get_data(picks=picks)
            background_data = background.get_data(picks=picks)

            self.info_ = foreground.copy().pick(picks).info
            self.ch_names_ = self.info_["ch_names"]
            self.sfreq_ = self.info_["sfreq"]

            return foreground_data, background_data

        return foreground, background

    def _prepare_data(self, data, name):
        """Validate, orient, and center/normalize data."""
        _validate_type(data, np.ndarray, name)

        data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got {data.ndim}D")

        # MNE convention is channels x times. This keeps compatibility with

        # the original implementation, which accepted transposed input.

        if data.shape[0] > data.shape[1]:
            data = data.T

        data = data - np.mean(data, axis=1, keepdims=True)

        if self.normalize:
            scale = np.std(data, axis=1, keepdims=True)

            if np.any(scale == 0):
                raise ValueError(f"{name} contains a channel with zero variance")

            data = data / scale

        return data

    def fit(self, foreground, background, picks=None):
        """Fit GED from foreground and background data.

        Parameters
        ----------
        foreground : instance of Raw | ndarray, shape (n_channels, n_times)
            Data containing the activity to emphasize. For artifact
            attenuation, this can be artifact-dominant data.
        background : instance of Raw | ndarray, shape (n_channels, n_times)
            Reference/background data.
        picks : array-like | None
            Channels to include when fitting GED. Only used when
            ``foreground`` and ``background`` are Raw objects. If None, all
            data channels are used.

        Returns
        -------
        self : instance of GED
            The fitted GED instance.
        """
        from scipy import linalg

        foreground, background = self._get_fit_data(foreground, background, picks)

        foreground = self._prepare_data(foreground, "foreground")
        background = self._prepare_data(background, "background")

        if foreground.shape[0] != background.shape[0]:
            raise ValueError(
                "foreground and background must have the same number of "
                f"channels, got {foreground.shape[0]} and {background.shape[0]}"
            )

        n_channels = foreground.shape[0]
        if self.n_components is None:
            n_components = n_channels
        else:
            n_components = _ensure_int(self.n_components, "n_components")
            if not 1 <= n_components <= n_channels:
                raise ValueError(
                    "n_components must be between 1 and n_channels "
                    f"({n_channels}), got {n_components}"
                )

        reg = float(self.reg)
        if not 0 <= reg <= 1:
            raise ValueError(f"reg must be between 0 and 1, got {self.reg}")

        foreground_cov = np.cov(foreground, rowvar=True)
        background_cov = np.cov(background, rowvar=True)

        evals_bg = linalg.eigvalsh(background_cov)
        background_cov_reg = (1.0 - reg) * background_cov + reg * np.mean(
            evals_bg
        ) * np.eye(n_channels)

        try:
            eigenvalues, eigenvectors = linalg.eigh(foreground_cov, background_cov_reg)
        except np.linalg.LinAlgError as exp:
            raise ValueError(
                "Could not compute GED. The regularized background covariance "
                f"may be singular or ill-conditioned. Error: {exp}"
            )

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True)

        filters_all = eigenvectors.T

        mixing_all = linalg.pinv(filters_all)

        patterns_all = mixing_all.T

        eigenvalues_all = eigenvalues

        explained_ratio_all = eigenvalues_all / np.sum(eigenvalues_all)

        self._filters_all_ = filters_all

        self._mixing_all_ = mixing_all

        self._patterns_all_ = patterns_all

        self._eigenvalues_all_ = eigenvalues_all

        self._explained_ratio_all_ = explained_ratio_all

        self.filters_ = filters_all[:n_components]

        self.patterns_ = patterns_all[:n_components]

        self.eigenvalues_ = eigenvalues_all[:n_components]

        self.explained_ratio_ = explained_ratio_all[:n_components]

        self.foreground_cov_ = foreground_cov

        self.background_cov_ = background_cov

        self.background_cov_reg_ = background_cov_reg

        self.n_components_ = n_components

        self.n_channels_ = n_channels

        return self

    def transform(self, data):
        """Transform data into GED component space.

        Parameters
        ----------
        data : ndarray
            Data to transform. The shape must be ``(n_channels, n_times)`` or
            ``(n_epochs, n_channels, n_times)``.

        Returns
        -------
        transformed : ndarray
            The data transformed into GED component space.
        """
        self._check_is_fit()

        _validate_type(data, np.ndarray, "data")

        data = np.asarray(data, dtype=float)

        if data.ndim not in (2, 3):
            raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")

        if data.shape[-2] != self.filters_.shape[1]:
            raise ValueError(
                "data has the wrong number of channels. Expected "
                f"{self.filters_.shape[1]}, got {data.shape[-2]}"
            )

        return np.matmul(self.filters_, data)

    def apply(self, inst, exclude, copy=True):
        """Reconstruct data after excluding selected GED components.

        Parameters
        ----------
        inst : instance of Raw | ndarray, shape (n_channels, n_times)
            Data to clean. If Raw is provided, a Raw object is returned. If an
            ndarray is provided, an ndarray is returned.
        exclude : array-like of int
            GED component indices to remove before reconstruction.
        copy : bool
            If True, clean a copy of the input. If False, modify the input in
            place.

        Returns
        -------
        cleaned : instance of Raw | ndarray, shape (n_channels, n_times)
            Reconstructed data after excluding the selected components.
        """
        self._check_is_fit()

        if isinstance(inst, BaseRaw):
            if not inst.preload:
                raise ValueError("Raw data must be preloaded to apply GED")
            raw = inst.copy() if copy else inst
            data = raw.get_data()
            cleaned = self._apply_array(data, exclude)
            raw._data[:] = cleaned
            return raw

        _validate_type(inst, np.ndarray, "inst")
        data = np.array(inst, dtype=float, copy=copy)
        return self._apply_array(data, exclude)

    def _apply_array(self, data, exclude):
        """Apply GED cleaning to an array."""
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got {data.ndim}D")
        if data.shape[0] != self.filters_.shape[1]:
            raise ValueError(
                "data has the wrong number of channels. Expected "
                f"{self.filters_.shape[1]}, got {data.shape[0]}"
            )

        exclude = np.atleast_1d(exclude)
        if exclude.size == 0:
            return data
        if not np.issubdtype(exclude.dtype, np.integer):
            raise ValueError("exclude must contain integer component indices")
        if np.any(exclude < 0) or np.any(exclude >= self.n_components_):
            raise ValueError(
                "exclude contains invalid component indices. Valid indices are "
                f"0 to {self.n_components_ - 1}"
            )
        components = self._filters_all_ @ data
        components[exclude] = 0.0
        return self._mixing_all_ @ components

    def plot_components(self, info=None, picks=None, *, show=True):
        """Plot GED component topographies.

        Parameters
        ----------
        info : instance of Info | None
            Measurement info used to plot component topographies. If None, the
            info stored during fitting from Raw is used. If no info is
            available, channel weights are shown as line plots.
        picks : array-like of int | None
            GED component indices to plot. If None, all fitted components are
            plotted.
        show : bool
            Show figure if True.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure containing GED component topographies.
        """
        import matplotlib.pyplot as plt

        self._check_is_fit()
        picks = self._check_picks(picks)

        if info is None:
            info = getattr(self, "info_", None)

        n_components = len(picks)
        n_cols = min(5, n_components)
        n_rows = int(np.ceil(n_components / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3 * n_cols, 3 * n_rows),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, pick in zip(axes, picks):
            if info is not None:
                plot_topomap(self.patterns_[pick], info, axes=ax, show=False)
            else:
                ax.plot(self.patterns_[pick])
                ax.set_xlabel("Channel")
                ax.set_ylabel("Weight")
            ax.set_title(
                f"GED {pick}\n"
                f"λ={self.eigenvalues_[pick]:.2f}, "
                f"ratio={self.explained_ratio_[pick]:.2f}"
            )

        for ax in axes[len(picks) :]:
            ax.set_visible(False)

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_properties(
        self,
        inst,
        picks=None,
        *,
        info=None,
        sfreq=None,
        fmin=0.0,
        fmax=np.inf,
        show=True,
    ):
        """Plot GED component properties.

        Parameters
        ----------
        inst : instance of Raw | ndarray, shape (n_channels, n_times)
            Data used to compute GED component time series and power spectra.
        picks : array-like of int | None
            GED component indices to plot. If None, all fitted components are
            plotted.
        info : instance of Info | None
            Measurement info used to plot component topographies. If None, the
            info stored during fitting from Raw is used.
        sfreq : float | None
            Sampling frequency in Hz. Required when ``inst`` is an ndarray and
            no sampling frequency was stored during fitting.
        fmin : float
            Minimum frequency to show in the power spectrum.
        fmax : float
            Maximum frequency to show in the power spectrum.
        show : bool
            Show figure if True.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            One figure per GED component.
        """
        import matplotlib.pyplot as plt

        self._check_is_fit()
        picks = self._check_picks(picks)
        data, stored_info, stored_sfreq = self._get_plot_data(inst)

        if info is None:
            info = stored_info
        if sfreq is None:
            sfreq = stored_sfreq
        if sfreq is None:
            raise ValueError(
                "sfreq is required when inst is an ndarray and no sampling "
                "frequency was stored during fitting"
            )
        sfreq = float(sfreq)
        if sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {sfreq}")

        sources = self.transform(data)
        figs = list()

        for pick in picks:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

            if info is not None:
                plot_topomap(self.patterns_[pick], info, axes=axes[0], show=False)
            else:
                axes[0].plot(self.patterns_[pick])
                axes[0].set_xlabel("Channel")
                axes[0].set_ylabel("Weight")
            axes[0].set_title(f"GED {pick} topography")

            times = np.arange(data.shape[1]) / sfreq
            axes[1].plot(times, sources[pick])
            axes[1].set_title("Time series")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Amplitude")

            psd, freqs = psd_array_welch(
                sources[pick],
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                verbose=False,
            )
            axes[2].plot(freqs, psd)
            axes[2].set_title("Power spectrum")
            axes[2].set_xlabel("Frequency (Hz)")
            axes[2].set_ylabel("Power")

            fig.suptitle(f"GED component {pick}")
            fig.tight_layout()
            figs.append(fig)

        if show:
            plt.show()
        return figs

    def _check_picks(self, picks):
        """Validate component picks."""
        if picks is None:
            picks = np.arange(self.n_components_)
        else:
            picks = np.atleast_1d(picks)
            if not np.issubdtype(picks.dtype, np.integer):
                raise ValueError("picks must contain integer component indices")
            if np.any(picks < 0) or np.any(picks >= self.n_components_):
                raise ValueError(
                    "picks contains invalid component indices. Valid indices are "
                    f"0 to {self.n_components_ - 1}"
                )
        return picks

    def _get_plot_data(self, inst):
        """Get data, info, and sampling frequency for plotting."""
        if isinstance(inst, BaseRaw):
            data = inst.get_data()
            info = inst.info
            sfreq = inst.info["sfreq"]
        else:
            _validate_type(inst, np.ndarray, "inst")
            data = np.asarray(inst, dtype=float)
            info = getattr(self, "info_", None)
            sfreq = getattr(self, "sfreq_", None)

        if data.ndim != 2:
            raise ValueError(f"inst must contain 2D data, got {data.ndim}D")
        if data.shape[0] != self.filters_.shape[1]:
            raise ValueError(
                "inst has the wrong number of channels. Expected "
                f"{self.filters_.shape[1]}, got {data.shape[0]}"
            )
        return data, info, sfreq

    def plot_eigenvalues(self, picks=None, *, show=True):
        """Plot GED eigenvalues and normalized eigenvalue ratios.

        Parameters
        ----------
        picks : array-like of int | None
            GED component indices to plot. If None, all fitted components are
            plotted.
        show : bool
            Show figure if True.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure containing the eigenvalue plots.
        """
        self._check_is_fit()
        picks = self._check_picks(picks)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        axes[0].plot(picks, self.eigenvalues_[picks], marker="o")
        axes[0].set_title("GED eigenvalues")
        axes[0].set_xlabel("Component")
        axes[0].set_ylabel("Eigenvalue")

        axes[1].plot(picks, self.explained_ratio_[picks], marker="o")
        axes[1].set_title("Normalized eigenvalues")
        axes[1].set_xlabel("Component")
        axes[1].set_ylabel("Ratio")

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def _check_is_fit(self):
        """Check that GED has been fitted."""
        if not hasattr(self, "filters_"):
            raise RuntimeError("GED has not been fitted. Call fit() first.")
