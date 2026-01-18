# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import collections.abc as abc
from functools import partial

import numpy as np

from mne import __version__ as mne_version

from .._fiff.meas_info import Info, create_info
from .._fiff.pick import _picks_to_idx
from ..filter import filter_data
from ..utils import (
    _validate_type,
    fill_doc,
    logger,
)
from ..utils.check import _check_fname, _import_h5io_funcs, check_fname
from ._covs_ged import _ssd_estimate
from ._mod_ged import _get_spectral_ratio, _ssd_mod
from .base import _GEDTransformer


@fill_doc
class SSD(_GEDTransformer):
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
       length of FFT used. The default (None) will use 1 second of data.
       See :func:`mne.time_frequency.psd_array_welch` for more information.
    cov_method_params : dict | None (default None)
        As in :class:`mne.decoding.SPoC`
        The default is None.
    restr_type : "restricting" | "whitening" | None
        Restricting transformation for covariance matrices before performing
        generalized eigendecomposition.
        If "restricting" only restriction to the principal subspace of signal_cov
        will be performed.
        If "whitening", covariance matrices will be additionally rescaled according
        to the whitening for the signal_cov.
        If None, no restriction will be applied. Defaults to "whitening".

        .. versionadded:: 1.11
    rank : None | dict | ‘info’ | ‘full’
        As in :class:`mne.decoding.SPoC`
        This controls the rank computation that can be read from the
        measurement info or estimated from the data, which determines the
        maximum possible number of components.
        See Notes of :func:`mne.compute_rank` for details.
        We recommend to use 'full' when working with epoched data.

    Attributes
    ----------
    filters_ : array, shape (``n_channels or less``, n_channels)
        The spatial filters to be multiplied with the signal.
    patterns_ : array, shape (``n_channels or less``, n_channels)
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
        *,
        restr_type="whitening",
        rank=None,
    ):
        """Initialize instance."""
        self.info = info
        self.filt_params_signal = filt_params_signal
        self.filt_params_noise = filt_params_noise
        self.reg = reg
        self.n_components = n_components
        self.picks = picks
        self.sort_by_spectral_ratio = sort_by_spectral_ratio
        self.return_filtered = return_filtered
        self.n_fft = n_fft
        self.cov_method_params = cov_method_params
        self.restr_type = restr_type
        self.rank = rank

        cov_callable = partial(
            _ssd_estimate,
            reg=reg,
            cov_method_params=cov_method_params,
            info=info,
            picks=picks,
            n_fft=n_fft,
            filt_params_signal=filt_params_signal,
            filt_params_noise=filt_params_noise,
            rank=rank,
            sort_by_spectral_ratio=sort_by_spectral_ratio,
        )
        super().__init__(
            n_components=n_components,
            cov_callable=cov_callable,
            mod_ged_callable=_ssd_mod,
            restr_type=restr_type,
        )

    def __setstate__(self, state):
        """Restore state from serialization."""
        # Since read_ssd creates a new instance via __init__ first,
        # callables are already set correctly. We just restore fitted attributes.
        # Don't call super().__setstate__() as it would set callables to None.
        self.__dict__.update(state)
        return self

    def _validate_params(self, X):
        if isinstance(self.info, float):  # special case, mostly for testing
            self.sfreq_ = self.info
        else:
            _validate_type(self.info, Info, "info")
            self.sfreq_ = self.info["sfreq"]
        dicts = {"signal": self.filt_params_signal, "noise": self.filt_params_noise}
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
            self.filt_params_noise["l_freq"] > self.filt_params_signal["l_freq"]
            or self.filt_params_signal["h_freq"] > self.filt_params_noise["h_freq"]
        ):
            raise ValueError(
                "Wrongly specified frequency bands!\n"
                "The signal band-pass must be within the noise "
                "band-pass!"
            )
        self.freqs_signal_ = (
            self.filt_params_signal["l_freq"],
            self.filt_params_signal["h_freq"],
        )
        self.freqs_noise_ = (
            self.filt_params_noise["l_freq"],
            self.filt_params_noise["h_freq"],
        )
        _validate_type(self.sort_by_spectral_ratio, (bool,), "sort_by_spectral_ratio")
        _validate_type(self.n_fft, ("numeric", None), "n_fft")
        self.n_fft_ = min(
            int(self.n_fft if self.n_fft is not None else self.sfreq_),
            X.shape[-1],
        )
        _validate_type(self.return_filtered, (bool,), "return_filtered")
        if isinstance(self.info, Info):
            ch_types = self.info.get_channel_types(picks=self.picks, unique=True)
            if len(ch_types) > 1:
                raise ValueError(
                    "At this point SSD only supports fitting "
                    f"single channel types. Your info has {len(ch_types)} types."
                )
        _validate_type(self.cov_method_params, (abc.Mapping, None), "cov_method_params")

    def _check_X(self, X, *, y=None, fit=False):
        """Check input data."""
        X = self._check_data(X, y=y, fit=fit, atleast_3d=False)
        n_chan = X.shape[-2]
        if isinstance(self.info, Info) and n_chan != self.info["nchan"]:
            raise ValueError(
                "Info must match the input data."
                f"Found {n_chan} channels but expected {self.info['nchan']}."
            )
        return X

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
        X = self._check_X(X, y=y, fit=True)
        self._validate_params(X)
        if isinstance(self.info, Info):
            info = self.info
        else:
            info = create_info(X.shape[-2], self.sfreq_, ch_types="eeg")
        self.picks_ = _picks_to_idx(info, self.picks, none="data", exclude="bads")

        super().fit(X, y)

        logger.info("Done.")
        return self

    @fill_doc
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save the SSD decomposition to disk (in HDF5 format).

        Parameters
        ----------
        fname : path-like
            Path of file to save to. Should end with ``'.h5'`` or ``'.hdf5'``.
        %(overwrite)s
        %(verbose)s

        See Also
        --------
        mne.decoding.read_ssd
        """
        _, write_hdf5 = _import_h5io_funcs()
        check_fname(fname, "SSD", (".h5", ".hdf5"))
        fname = _check_fname(fname, overwrite=overwrite, verbose=verbose)
        state = self.__getstate__()
        state.update(
            class_name="SSD",
            mne_version=mne_version,
        )
        write_hdf5(
            fname, state, overwrite=overwrite, title="mnepython", slash="replace"
        )

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
        X = self._check_X(X)
        # For the case where n_epochs dimension is absent.
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        X_aux = X[..., self.picks_, :]
        if self.return_filtered:
            X_aux = filter_data(X_aux, self.sfreq_, **self.filt_params_signal)
        X_ssd = super().transform(X_aux).squeeze()

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
        spec_ratio, sorter_spec = _get_spectral_ratio(
            ssd_sources=ssd_sources,
            sfreq=self.sfreq_,
            n_fft=self.n_fft_,
            freqs_signal=self.freqs_signal_,
            freqs_noise=self.freqs_noise_,
        )
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
        pick_patterns = self.patterns_[: self.n_components].T
        X = pick_patterns @ X_ssd
        return X


def read_ssd(fname):
    """Read an SSD object from disk.

    Parameters
    ----------
    fname : path-like
        Path to an SSD file in HDF5 format, which should end with ``.h5`` or
        ``.hdf5``.

    Returns
    -------
    ssd : SSD
        The loaded SSD object.
    """
    read_hdf5, _ = _import_h5io_funcs()
    _validate_type(fname, "path-like", "fname")
    fname = _check_fname(fname=fname, overwrite="read", must_exist=False)
    state = read_hdf5(fname, title="mnepython", slash="replace")
    return SSD(info=None, filt_params_signal=None, filt_params_noise=None).__setstate__(
        state
    )
