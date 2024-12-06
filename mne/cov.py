# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import itertools as itt
from copy import deepcopy
from math import log

import numpy as np
from scipy.sparse import issparse

from . import viz
from ._fiff.constants import FIFF
from ._fiff.meas_info import _read_bad_channels, _write_bad_channels, create_info
from ._fiff.pick import (
    _DATA_CH_TYPES_SPLIT,
    _pick_data_channels,
    _picks_by_type,
    _picks_to_idx,
    pick_channels,
    pick_channels_cov,
    pick_info,
    pick_types,
)
from ._fiff.proj import (
    _check_projs,
    _has_eeg_average_ref_proj,
    _needs_eeg_average_ref_proj,
    _proj_equal,
    _read_proj,
    _write_proj,
)
from ._fiff.proj import activate_proj as _activate_proj
from ._fiff.proj import make_projector as _make_projector
from ._fiff.tag import find_tag
from ._fiff.tree import dir_tree_find
from .defaults import (
    _BORDER_DEFAULT,
    _EXTRAPOLATE_DEFAULT,
    _INTERPOLATION_DEFAULT,
    DEFAULTS,
    _handle_default,
)
from .epochs import Epochs
from .event import make_fixed_length_events
from .evoked import EvokedArray
from .fixes import (
    EmpiricalCovariance,
    _EstimatorMixin,
    _logdet,
    _safe_svd,
    empirical_covariance,
    log_likelihood,
)
from .rank import _compute_rank
from .utils import (
    _array_repr,
    _check_fname,
    _check_on_missing,
    _check_option,
    _on_missing,
    _pl,
    _scaled_array,
    _time_mask,
    _undo_scaling_cov,
    _validate_type,
    _verbose_safe_false,
    check_fname,
    check_version,
    copy_function_doc_to_method_doc,
    eigh,
    fill_doc,
    logger,
    verbose,
    warn,
)


def _check_covs_algebra(cov1, cov2):
    if cov1.ch_names != cov2.ch_names:
        raise ValueError("Both Covariance do not have the same list of channels.")
    projs1 = [str(c) for c in cov1["projs"]]
    projs2 = [str(c) for c in cov1["projs"]]
    if projs1 != projs2:
        raise ValueError(
            "Both Covariance do not have the same list of SSP projections."
        )


def _get_tslice(epochs, tmin, tmax):
    """Get the slice."""
    mask = _time_mask(epochs.times, tmin, tmax, sfreq=epochs.info["sfreq"])
    tstart = np.where(mask)[0][0] if tmin is not None else None
    tend = np.where(mask)[0][-1] + 1 if tmax is not None else None
    tslice = slice(tstart, tend, None)
    return tslice


@fill_doc
class Covariance(dict):
    """Noise covariance matrix.

    .. note::
        This class should not be instantiated directly via
        ``mne.Covariance(...)``. Instead, use one of the functions
        listed in the See Also section below.

    Parameters
    ----------
    data : array-like
        The data.
    names : list of str
        Channel names.
    bads : list of str
        Bad channels.
    projs : list
        Projection vectors.
    nfree : int
        Degrees of freedom.
    eig : array-like | None
        Eigenvalues.
    eigvec : array-like | None
        Eigenvectors.
    method : str | None
        The method used to compute the covariance.
    loglik : float
        The log likelihood.
    %(verbose)s

    Attributes
    ----------
    data : array of shape (n_channels, n_channels)
        The covariance.
    ch_names : list of str
        List of channels' names.
    nfree : int
        Number of degrees of freedom i.e. number of time points used.
    dim : int
        The number of channels ``n_channels``.

    See Also
    --------
    compute_covariance
    compute_raw_covariance
    make_ad_hoc_cov
    read_cov
    """

    @verbose
    def __init__(
        self,
        data,
        names,
        bads,
        projs,
        nfree,
        eig=None,
        eigvec=None,
        method=None,
        loglik=None,
        *,
        verbose=None,
    ):
        """Init of covariance."""
        diag = data.ndim == 1
        projs = _check_projs(projs)
        self.update(
            data=data,
            dim=len(data),
            names=names,
            bads=bads,
            nfree=nfree,
            eig=eig,
            eigvec=eigvec,
            diag=diag,
            projs=projs,
            kind=FIFF.FIFFV_MNE_NOISE_COV,
        )
        if method is not None:
            self["method"] = method
        if loglik is not None:
            self["loglik"] = loglik

    @property
    def data(self):
        """Numpy array of Noise covariance matrix."""
        return self["data"]

    @property
    def ch_names(self):
        """Channel names."""
        return self["names"]

    @property
    def nfree(self):
        """Number of degrees of freedom."""
        return self["nfree"]

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save covariance matrix in a FIF file.

        Parameters
        ----------
        fname : path-like
            Output filename.
        %(overwrite)s

            .. versionadded:: 1.0
        %(verbose)s
        """
        from ._fiff.write import start_and_end_file

        check_fname(
            fname, "covariance", ("-cov.fif", "-cov.fif.gz", "_cov.fif", "_cov.fif.gz")
        )
        fname = _check_fname(fname=fname, overwrite=overwrite)
        with start_and_end_file(fname) as fid:
            _write_cov(fid, self)

    def copy(self):
        """Copy the Covariance object.

        Returns
        -------
        cov : instance of Covariance
            The copied object.
        """
        return deepcopy(self)

    def as_diag(self):
        """Set covariance to be processed as being diagonal.

        Returns
        -------
        cov : dict
            The covariance.

        Notes
        -----
        This function allows creation of inverse operators
        equivalent to using the old "--diagnoise" mne option.

        This function operates in place.
        """
        if self["diag"]:
            return self
        self["diag"] = True
        self["data"] = np.diag(self["data"])
        self["eig"] = None
        self["eigvec"] = None
        return self

    def _as_square(self):
        # This is a hack but it works because np.diag() behaves nicely
        if self["diag"]:
            self["diag"] = False
            self.as_diag()
            self["diag"] = False
        return self

    def _get_square(self):
        if self["diag"] != (self.data.ndim == 1):
            raise RuntimeError(
                "Covariance attributes inconsistent, got data with "
                f"dimensionality {self.data.ndim} but diag={self['diag']}"
            )
        return np.diag(self.data) if self["diag"] else self.data.copy()

    def __repr__(self):  # noqa: D105
        s = "<Covariance | kind : "
        s += "full" if self.data.ndim == 2 else "diagonal"
        s += f", {_array_repr(self.data)}, n_samples : {self.nfree}>"
        return s

    def __add__(self, cov):
        """Add Covariance taking into account number of degrees of freedom."""
        _check_covs_algebra(self, cov)
        this_cov = cov.copy()
        this_cov["data"] = (
            (this_cov["data"] * this_cov["nfree"]) + (self["data"] * self["nfree"])
        ) / (self["nfree"] + this_cov["nfree"])
        this_cov["nfree"] += self["nfree"]

        this_cov["bads"] = list(set(this_cov["bads"]).union(self["bads"]))

        return this_cov

    def __iadd__(self, cov):
        """Add Covariance taking into account number of degrees of freedom."""
        _check_covs_algebra(self, cov)
        self["data"][:] = (
            (self["data"] * self["nfree"]) + (cov["data"] * cov["nfree"])
        ) / (self["nfree"] + cov["nfree"])
        self["nfree"] += cov["nfree"]

        self["bads"] = list(set(self["bads"]).union(cov["bads"]))

        return self

    @verbose
    @copy_function_doc_to_method_doc(viz.plot_cov)
    def plot(
        self,
        info,
        exclude=(),
        colorbar=True,
        proj=False,
        show_svd=True,
        show=True,
        verbose=None,
    ):
        return viz.plot_cov(
            self, info, exclude, colorbar, proj, show_svd, show, verbose
        )

    @verbose
    def plot_topomap(
        self,
        info,
        ch_type=None,
        *,
        scalings=None,
        proj=False,
        noise_cov=None,
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=1,
        cmap=None,
        vlim=(None, None),
        cnorm=None,
        colorbar=True,
        cbar_fmt="%3.1f",
        units=None,
        axes=None,
        show=True,
        verbose=None,
    ):
        """Plot a topomap of the covariance diagonal.

        Parameters
        ----------
        %(info_not_none)s
        %(ch_type_topomap)s

            .. versionadded:: 0.21
        %(scalings_topomap)s
        %(proj_plot)s
        noise_cov : instance of Covariance | None
            If not None, whiten the instance with ``noise_cov`` before
            plotting.
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s

            .. versionchanged:: 0.21

               - The default was changed to ``'local'`` for MEG sensors.
               - ``'local'`` was changed to use a convex hull mask
               - ``'head'`` was changed to extrapolate out to the clipping circle.
        %(border_topomap)s

            .. versionadded:: 0.20
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap)s

            .. versionadded:: 1.2
        %(cnorm)s

            .. versionadded:: 1.2
        %(colorbar_topomap)s
        %(cbar_fmt_topomap)s
        %(units_topomap_evoked)s
        %(axes_cov_plot_topomap)s
        %(show)s
        %(verbose)s

        Returns
        -------
        fig : instance of Figure
            The matplotlib figure.

        Notes
        -----
        .. versionadded:: 0.21
        """
        from .viz.misc import _index_info_cov

        info, C, _, _ = _index_info_cov(info, self, exclude=())
        evoked = EvokedArray(np.diag(C)[:, np.newaxis], info)
        if noise_cov is not None:
            # need to left and right multiply whitener, which for the diagonal
            # entries is the same as multiplying twice
            evoked = whiten_evoked(whiten_evoked(evoked, noise_cov), noise_cov)
            if units is None:
                units = "AU"
            if scalings is None:
                scalings = 1.0
        if units is None:
            units = {k: f"({v})²" for k, v in DEFAULTS["units"].items()}
        if scalings is None:
            scalings = {k: v * v for k, v in DEFAULTS["scalings"].items()}
        return evoked.plot_topomap(
            times=[0],
            ch_type=ch_type,
            vlim=vlim,
            cmap=cmap,
            sensors=sensors,
            cnorm=cnorm,
            colorbar=colorbar,
            scalings=scalings,
            units=units,
            res=res,
            size=size,
            cbar_fmt=cbar_fmt,
            proj=proj,
            show=show,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            outlines=outlines,
            contours=contours,
            image_interp=image_interp,
            axes=axes,
            extrapolate=extrapolate,
            sphere=sphere,
            border=border,
            time_format="",
        )

    @verbose
    def pick_channels(self, ch_names, ordered=True, *, verbose=None):
        """Pick channels from this covariance matrix.

        Parameters
        ----------
        ch_names : list of str
            List of channels to keep. All other channels are dropped.
        %(ordered)s
        %(verbose)s

        Returns
        -------
        cov : instance of Covariance.
            The modified covariance matrix.

        Notes
        -----
        Operates in-place.

        .. versionadded:: 0.20.0
        """
        return pick_channels_cov(
            self, ch_names, exclude=[], ordered=ordered, copy=False
        )


###############################################################################
# IO


@verbose
def read_cov(fname, verbose=None):
    """Read a noise covariance from a FIF file.

    Parameters
    ----------
    fname : path-like
        The path-like of file containing the covariance matrix. It should end
        with ``-cov.fif`` or ``-cov.fif.gz``.
    %(verbose)s

    Returns
    -------
    cov : Covariance
        The noise covariance matrix.

    See Also
    --------
    write_cov, compute_covariance, compute_raw_covariance
    """
    from ._fiff.open import fiff_open

    check_fname(
        fname, "covariance", ("-cov.fif", "-cov.fif.gz", "_cov.fif", "_cov.fif.gz")
    )
    fname = _check_fname(fname=fname, must_exist=True, overwrite="read")
    f, tree, _ = fiff_open(fname)
    with f as fid:
        return Covariance(
            **_read_cov(fid, tree, FIFF.FIFFV_MNE_NOISE_COV, limited=True)
        )


###############################################################################
# Estimate from data


@verbose
def make_ad_hoc_cov(info, std=None, *, verbose=None):
    """Create an ad hoc noise covariance.

    Parameters
    ----------
    %(info_not_none)s
    std : dict of float | None
        Standard_deviation of the diagonal elements. If dict, keys should be
        ``'grad'`` for gradiometers, ``'mag'`` for magnetometers and ``'eeg'``
        for EEG channels. If None, default values will be used (see Notes).
    %(verbose)s

    Returns
    -------
    cov : instance of Covariance
        The ad hoc diagonal noise covariance for the M/EEG data channels.

    Notes
    -----
    The default noise values are 5 fT/cm, 20 fT, and 0.2 µV for gradiometers,
    magnetometers, and EEG channels respectively.

    .. versionadded:: 0.9.0
    """
    picks = pick_types(info, meg=True, eeg=True, exclude=())
    std = _handle_default("noise_std", std)

    data = np.zeros(len(picks))
    for meg, eeg, val in zip(
        ("grad", "mag", False),
        (False, False, True),
        (std["grad"], std["mag"], std["eeg"]),
    ):
        these_picks = pick_types(info, meg=meg, eeg=eeg)
        data[np.searchsorted(picks, these_picks)] = val * val
    ch_names = [info["ch_names"][pick] for pick in picks]
    return Covariance(data, ch_names, info["bads"], info["projs"], nfree=0)


def _check_n_samples(n_samples, n_chan):
    """Check to see if there are enough samples for reliable cov calc."""
    n_samples_min = 10 * (n_chan + 1) // 2
    if n_samples <= 0:
        raise ValueError("No samples found to compute the covariance matrix")
    if n_samples < n_samples_min:
        warn(
            f"Too few samples (required : {n_samples_min} got : {n_samples}), "
            "covariance estimate may be unreliable"
        )


@verbose
def compute_raw_covariance(
    raw,
    tmin=0,
    tmax=None,
    tstep=0.2,
    reject=None,
    flat=None,
    picks=None,
    method="empirical",
    method_params=None,
    cv=3,
    scalings=None,
    n_jobs=None,
    return_estimators=False,
    reject_by_annotation=True,
    rank=None,
    verbose=None,
):
    """Estimate noise covariance matrix from a continuous segment of raw data.

    It is typically useful to estimate a noise covariance from empty room
    data or time intervals before starting the stimulation.

    .. note:: To estimate the noise covariance from epoched data, use
              :func:`mne.compute_covariance` instead.

    Parameters
    ----------
    raw : instance of Raw
        Raw data.
    tmin : float
        Beginning of time interval in seconds. Defaults to 0.
    tmax : float | None (default None)
        End of time interval in seconds. If None (default), use the end of the
        recording.
    tstep : float (default 0.2)
        Length of data chunks for artifact rejection in seconds.
        Can also be None to use a single epoch of (tmax - tmin)
        duration. This can use a lot of memory for large ``Raw``
        instances.
    reject : dict | None (default None)
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )

    flat : dict | None (default None)
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    %(picks_good_data_noref)s
    method : str | list | None (default 'empirical')
        The method used for covariance estimation.
        See :func:`mne.compute_covariance`.

        .. versionadded:: 0.12
    method_params : dict | None (default None)
        Additional parameters to the estimation procedure.
        See :func:`mne.compute_covariance`.

        .. versionadded:: 0.12
    cv : int | sklearn.model_selection object (default 3)
        The cross validation method. Defaults to 3, which will
        internally trigger by default :class:`sklearn.model_selection.KFold`
        with 3 splits.

        .. versionadded:: 0.12
    scalings : dict | None (default None)
        Defaults to ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale magnetometers and gradiometers
        at the same unit.

        .. versionadded:: 0.12
    %(n_jobs)s

        .. versionadded:: 0.12
    return_estimators : bool (default False)
        Whether to return all estimators or the best. Only considered if
        method equals 'auto' or is a list of str. Defaults to False.

        .. versionadded:: 0.12
    %(reject_by_annotation_epochs)s

        .. versionadded:: 0.14
    %(rank_none)s

        .. versionadded:: 0.17

        .. versionadded:: 0.18
           Support for 'info' mode.
    %(verbose)s

    Returns
    -------
    cov : instance of Covariance | list
        The computed covariance. If method equals 'auto' or is a list of str
        and return_estimators equals True, a list of covariance estimators is
        returned (sorted by log-likelihood, from high to low, i.e. from best
        to worst).

    See Also
    --------
    compute_covariance : Estimate noise covariance matrix from epoched data.

    Notes
    -----
    This function will:

    1. Partition the data into evenly spaced, equal-length epochs.
    2. Load them into memory.
    3. Subtract the mean across all time points and epochs for each channel.
    4. Process the :class:`Epochs` by :func:`compute_covariance`.

    This will produce a slightly different result compared to using
    :func:`make_fixed_length_events`, :class:`Epochs`, and
    :func:`compute_covariance` directly, since that would (with the recommended
    baseline correction) subtract the mean across time *for each epoch*
    (instead of across epochs) for each channel.
    """
    tmin = 0.0 if tmin is None else float(tmin)
    dt = 1.0 / raw.info["sfreq"]
    tmax = raw.times[-1] + dt if tmax is None else float(tmax)
    tstep = tmax - tmin if tstep is None else float(tstep)
    tstep_m1 = tstep - dt  # inclusive!
    events = make_fixed_length_events(raw, 1, tmin, tmax, tstep)
    logger.info(f"Using up to {len(events)} segment{_pl(events)}")

    # don't exclude any bad channels, inverses expect all channels present
    if picks is None:
        # Need to include all good channels e.g. if eog rejection is to be used
        picks = np.arange(raw.info["nchan"])
        pick_mask = np.isin(picks, _pick_data_channels(raw.info, with_ref_meg=False))
    else:
        pick_mask = slice(None)
        picks = _picks_to_idx(raw.info, picks)
    epochs = Epochs(
        raw,
        events,
        1,
        0,
        tstep_m1,
        baseline=None,
        picks=picks,
        reject=reject,
        flat=flat,
        verbose=_verbose_safe_false(),
        preload=False,
        proj=False,
        reject_by_annotation=reject_by_annotation,
    )
    if method is None:
        method = "empirical"
    if isinstance(method, str) and method == "empirical":
        # potentially *much* more memory efficient to do it the iterative way
        picks = picks[pick_mask]
        data = 0
        n_samples = 0
        mu = 0
        # Read data in chunks
        for raw_segment in epochs:
            raw_segment = raw_segment[pick_mask]
            mu += raw_segment.sum(axis=1)
            data += np.dot(raw_segment, raw_segment.T)
            n_samples += raw_segment.shape[1]
        _check_n_samples(n_samples, len(picks))
        data -= mu[:, None] * (mu[None, :] / n_samples)
        data /= n_samples - 1.0
        logger.info("Number of samples used : %d", n_samples)
        logger.info("[done]")
        ch_names = [raw.info["ch_names"][k] for k in picks]
        bads = [b for b in raw.info["bads"] if b in ch_names]
        return Covariance(data, ch_names, bads, raw.info["projs"], nfree=n_samples - 1)
    del picks, pick_mask

    # This makes it equivalent to what we used to do (and do above for
    # empirical mode), treating all epochs as if they were a single long one
    epochs.load_data()
    ch_means = epochs._data.mean(axis=0).mean(axis=1)
    epochs._data -= ch_means[np.newaxis, :, np.newaxis]
    # fake this value so there are no complaints from compute_covariance
    epochs.baseline = (None, None)
    return compute_covariance(
        epochs,
        keep_sample_mean=True,
        method=method,
        method_params=method_params,
        cv=cv,
        scalings=scalings,
        n_jobs=n_jobs,
        return_estimators=return_estimators,
        rank=rank,
    )


def _check_method_params(
    method,
    method_params,
    keep_sample_mean=True,
    name="method",
    allow_auto=True,
    rank=None,
):
    """Check that method and method_params are usable."""
    accepted_methods = (
        "auto",
        "empirical",
        "diagonal_fixed",
        "ledoit_wolf",
        "oas",
        "shrunk",
        "pca",
        "factor_analysis",
        "shrinkage",
    )
    _method_params = {
        "empirical": {"store_precision": False, "assume_centered": True},
        "diagonal_fixed": {"store_precision": False, "assume_centered": True},
        "ledoit_wolf": {"store_precision": False, "assume_centered": True},
        "oas": {"store_precision": False, "assume_centered": True},
        "shrinkage": {
            "shrinkage": 0.1,
            "store_precision": False,
            "assume_centered": True,
        },
        "shrunk": {
            "shrinkage": np.logspace(-4, 0, 30),
            "store_precision": False,
            "assume_centered": True,
        },
        "pca": {"iter_n_components": None},
        "factor_analysis": {"iter_n_components": None},
    }

    for ch_type in _DATA_CH_TYPES_SPLIT:
        _method_params["diagonal_fixed"][ch_type] = 0.1

    if isinstance(method_params, dict):
        for key, values in method_params.items():
            if key not in _method_params:
                raise ValueError(
                    'key ({}) must be "{}"'.format(key, '" or "'.join(_method_params))
                )

            _method_params[key].update(method_params[key])
        shrinkage = method_params.get("shrinkage", {}).get("shrinkage", 0.1)
        if not 0 <= shrinkage <= 1:
            raise ValueError(f"shrinkage must be between 0 and 1, got {shrinkage}")

    was_auto = False
    if method is None:
        method = ["empirical"]
    elif method == "auto" and allow_auto:
        was_auto = True
        method = ["shrunk", "diagonal_fixed", "empirical", "factor_analysis"]

    if not isinstance(method, list | tuple):
        method = [method]

    if not all(k in accepted_methods for k in method):
        raise ValueError(
            f"Invalid {name} ({method}). Accepted values (individually or "
            f"in a list) are any of '{accepted_methods}' or None."
        )
    if not (isinstance(rank, str) and rank == "full"):
        if was_auto:
            method.pop(method.index("factor_analysis"))
        for method_ in method:
            if method_ in ("pca", "factor_analysis"):
                raise ValueError(
                    f'{method_} can so far only be used with rank="full", got rank='
                    f"{rank!r}"
                )
    if not keep_sample_mean:
        if len(method) != 1 or "empirical" not in method:
            raise ValueError(
                f'`keep_sample_mean=False` is only supported with {name}="empirical"'
            )
        for p, v in _method_params.items():
            if v.get("assume_centered", None) is False:
                raise ValueError(
                    "`assume_centered` must be True if `keep_sample_mean` is False"
                )
    return method, _method_params


@verbose
def compute_covariance(
    epochs,
    keep_sample_mean=True,
    tmin=None,
    tmax=None,
    projs=None,
    method="empirical",
    method_params=None,
    cv=3,
    scalings=None,
    n_jobs=None,
    return_estimators=False,
    on_mismatch="raise",
    rank=None,
    verbose=None,
):
    """Estimate noise covariance matrix from epochs.

    The noise covariance is typically estimated on pre-stimulus periods
    when the stimulus onset is defined from events.

    If the covariance is computed for multiple event types (events
    with different IDs), the following two options can be used and combined:

        1. either an Epochs object for each event type is created and
           a list of Epochs is passed to this function.
        2. an Epochs object is created for multiple events and passed
           to this function.

    .. note:: To estimate the noise covariance from non-epoched raw data, such
              as an empty-room recording, use
              :func:`mne.compute_raw_covariance` instead.

    Parameters
    ----------
    epochs : instance of Epochs, or list of Epochs
        The epochs.
    keep_sample_mean : bool (default True)
        If False, the average response over epochs is computed for
        each event type and subtracted during the covariance
        computation. This is useful if the evoked response from a
        previous stimulus extends into the baseline period of the next.
        Note. This option is only implemented for method='empirical'.
    tmin : float | None (default None)
        Start time for baseline. If None start at first sample.
    tmax : float | None (default None)
        End time for baseline. If None end at last sample.
    projs : list of Projection | None (default None)
        List of projectors to use in covariance calculation, or None
        to indicate that the projectors from the epochs should be
        inherited. If None, then projectors from all epochs must match.
    method : str | list | None (default 'empirical')
        The method used for covariance estimation. If 'empirical' (default),
        the sample covariance will be computed. A list can be passed to
        perform estimates using multiple methods.
        If 'auto' or a list of methods, the best estimator will be determined
        based on log-likelihood and cross-validation on unseen data as
        described in :footcite:`EngemannGramfort2015`. Valid methods are
        'empirical', 'diagonal_fixed', 'shrunk', 'oas', 'ledoit_wolf',
        'factor_analysis', 'shrinkage', and 'pca' (see Notes). If ``'auto'``,
        it expands to::

             ['shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis']

        ``'factor_analysis'`` is removed when ``rank`` is not 'full'.
        The ``'auto'`` mode is not recommended if there are many
        segments of data, since computation can take a long time.

        .. versionadded:: 0.9.0
    method_params : dict | None (default None)
        Additional parameters to the estimation procedure. Only considered if
        method is not None. Keys must correspond to the value(s) of ``method``.
        If None (default), expands to the following (with the addition of
        ``{'store_precision': False, 'assume_centered': True} for all methods
        except ``'factor_analysis'`` and ``'pca'``)::

            {'diagonal_fixed': {'grad': 0.1, 'mag': 0.1, 'eeg': 0.1, ...},
             'shrinkage': {'shrinkage': 0.1},
             'shrunk': {'shrinkage': np.logspace(-4, 0, 30)},
             'pca': {'iter_n_components': None},
             'factor_analysis': {'iter_n_components': None}}

    cv : int | sklearn.model_selection object (default 3)
        The cross validation method. Defaults to 3, which will
        internally trigger by default :class:`sklearn.model_selection.KFold`
        with 3 splits.
    scalings : dict | None (default None)
        Defaults to ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale data to roughly the same order of
        magnitude.
    %(n_jobs)s
    return_estimators : bool (default False)
        Whether to return all estimators or the best. Only considered if
        method equals 'auto' or is a list of str. Defaults to False.
    on_mismatch : str
        What to do when the MEG<->Head transformations do not match between
        epochs. If "raise" (default) an error is raised, if "warn" then a
        warning is emitted, if "ignore" then nothing is printed. Having
        mismatched transforms can in some cases lead to unexpected or
        unstable results in covariance calculation, e.g. when data
        have been processed with Maxwell filtering but not transformed
        to the same head position.
    %(rank_none)s

        .. versionadded:: 0.17

        .. versionadded:: 0.18
           Support for 'info' mode.
    %(verbose)s

    Returns
    -------
    cov : instance of Covariance | list
        The computed covariance. If method equals ``'auto'`` or is a list of str
        and ``return_estimators=True``, a list of covariance estimators is
        returned (sorted by log-likelihood, from high to low, i.e. from best
        to worst).

    See Also
    --------
    compute_raw_covariance : Estimate noise covariance from raw data, such as
        empty-room recordings.

    Notes
    -----
    Baseline correction or sufficient high-passing should be used
    when creating the :class:`Epochs` to ensure that the data are zero mean,
    otherwise the computed covariance matrix will be inaccurate.

    Valid ``method`` strings are:

    * ``'empirical'``
        The empirical or sample covariance (default)
    * ``'diagonal_fixed'``
        A diagonal regularization based on channel types as in
        :func:`mne.cov.regularize`.
    * ``'shrinkage'``
        Fixed shrinkage.

      .. versionadded:: 0.16
    * ``'ledoit_wolf'``
        The Ledoit-Wolf estimator, which uses an
        empirical formula for the optimal shrinkage value :footcite:`LedoitWolf2004`.
    * ``'oas'``
        The OAS estimator :footcite:`ChenEtAl2010`, which uses a different
        empricial formula for the optimal shrinkage value.

      .. versionadded:: 0.16
    * ``'shrunk'``
        Like 'ledoit_wolf', but with cross-validation for optimal alpha.
    * ``'pca'``
        Probabilistic PCA with low rank :footcite:`TippingBishop1999`.
    * ``'factor_analysis'``
        Factor analysis with low rank :footcite:`Barber2012`.

    ``'ledoit_wolf'`` and ``'pca'`` are similar to ``'shrunk'`` and
    ``'factor_analysis'``, respectively, except that they use
    cross validation (which is useful when samples are correlated, which
    is often the case for M/EEG data). The former two are not included in
    the ``'auto'`` mode to avoid redundancy.

    For multiple event types, it is also possible to create a
    single :class:`Epochs` object with events obtained using
    :func:`mne.merge_events`. However, the resulting covariance matrix
    will only be correct if ``keep_sample_mean is True``.

    The covariance can be unstable if the number of samples is small.
    In that case it is common to regularize the covariance estimate.
    The ``method`` parameter allows to regularize the covariance in an
    automated way. It also allows to select between different alternative
    estimation algorithms which themselves achieve regularization.
    Details are described in :footcite:t:`EngemannGramfort2015`.

    For more information on the advanced estimation methods, see
    :ref:`the sklearn manual <sklearn:covariance>`.

    References
    ----------
    .. footbibliography::
    """
    # scale to natural unit for best stability with MEG/EEG
    scalings = _check_scalings_user(scalings)
    method, _method_params = _check_method_params(
        method, method_params, keep_sample_mean, rank=rank
    )
    del method_params

    # for multi condition support epochs is required to refer to a list of
    # epochs objects

    def _unpack_epochs(epochs):
        if len(epochs.event_id) > 1:
            epochs = [epochs[k] for k in epochs.event_id]
        else:
            epochs = [epochs]
        return epochs

    if not isinstance(epochs, list):
        epochs = _unpack_epochs(epochs)
    else:
        epochs = sum([_unpack_epochs(epoch) for epoch in epochs], [])

    # check for baseline correction
    if any(
        epochs_t.baseline is None
        and epochs_t.info["highpass"] < 0.5
        and keep_sample_mean
        for epochs_t in epochs
    ):
        warn("Epochs are not baseline corrected, covariance matrix may be inaccurate")

    orig = epochs[0].info["dev_head_t"]
    _check_on_missing(on_mismatch, "on_mismatch")
    for ei, epoch in enumerate(epochs):
        epoch.info._check_consistency()
        if (orig is None) != (epoch.info["dev_head_t"] is None) or (
            orig is not None
            and not np.allclose(orig["trans"], epoch.info["dev_head_t"]["trans"])
        ):
            msg = (
                "MEG<->Head transform mismatch between epochs[0]:\n{}\n\n"
                "and epochs[{}]:\n{}".format(orig, ei, epoch.info["dev_head_t"])
            )
            _on_missing(on_mismatch, msg, "on_mismatch")

    bads = epochs[0].info["bads"]
    if projs is None:
        projs = epochs[0].info["projs"]
        # make sure Epochs are compatible
        for epochs_t in epochs[1:]:
            if epochs_t.proj != epochs[0].proj:
                raise ValueError("Epochs must agree on the use of projections")
            for proj_a, proj_b in zip(epochs_t.info["projs"], projs):
                if not _proj_equal(proj_a, proj_b):
                    raise ValueError("Epochs must have same projectors")
    projs = _check_projs(projs)
    ch_names = epochs[0].ch_names

    # make sure Epochs are compatible
    for epochs_t in epochs[1:]:
        if epochs_t.info["bads"] != bads:
            raise ValueError("Epochs must have same bad channels")
        if epochs_t.ch_names != ch_names:
            raise ValueError("Epochs must have same channel names")
    picks_list = _picks_by_type(epochs[0].info)
    picks_meeg = np.concatenate([b for _, b in picks_list])
    picks_meeg = np.sort(picks_meeg)
    ch_names = [epochs[0].ch_names[k] for k in picks_meeg]
    info = epochs[0].info  # we will overwrite 'epochs'

    if not keep_sample_mean:
        # prepare mean covs
        n_epoch_types = len(epochs)
        data_mean = [0] * n_epoch_types
        n_samples = np.zeros(n_epoch_types, dtype=np.int64)
        n_epochs = np.zeros(n_epoch_types, dtype=np.int64)

        for ii, epochs_t in enumerate(epochs):
            tslice = _get_tslice(epochs_t, tmin, tmax)
            for e in epochs_t:
                e = e[picks_meeg, tslice]
                if not keep_sample_mean:
                    data_mean[ii] += e
                n_samples[ii] += e.shape[1]
                n_epochs[ii] += 1

        n_samples_epoch = n_samples // n_epochs
        norm_const = np.sum(n_samples_epoch * (n_epochs - 1))
        data_mean = [
            1.0 / n_epoch * np.dot(mean, mean.T)
            for n_epoch, mean in zip(n_epochs, data_mean)
        ]

    info = pick_info(info, picks_meeg)
    tslice = _get_tslice(epochs[0], tmin, tmax)
    epochs = [ee.get_data(picks=picks_meeg)[..., tslice] for ee in epochs]
    picks_meeg = np.arange(len(picks_meeg))
    picks_list = _picks_by_type(info)

    if len(epochs) > 1:
        epochs = np.concatenate(epochs, 0)
    else:
        epochs = epochs[0]

    epochs = np.hstack(epochs)
    n_samples_tot = epochs.shape[-1]
    _check_n_samples(n_samples_tot, len(picks_meeg))

    epochs = epochs.T  # sklearn | C-order
    cov_data = _compute_covariance_auto(
        epochs,
        method=method,
        method_params=_method_params,
        info=info,
        cv=cv,
        n_jobs=n_jobs,
        stop_early=True,
        picks_list=picks_list,
        scalings=scalings,
        rank=rank,
    )

    if keep_sample_mean is False:
        cov = cov_data["empirical"]["data"]
        # undo scaling
        cov *= n_samples_tot - 1
        # ... apply pre-computed class-wise normalization
        for mean_cov in data_mean:
            cov -= mean_cov
        cov /= norm_const

    covs = list()
    for this_method, data in cov_data.items():
        cov = Covariance(
            data.pop("data"), ch_names, info["bads"], projs, nfree=n_samples_tot - 1
        )

        # add extra info
        cov.update(method=this_method, **data)
        covs.append(cov)
    logger.info("Number of samples used : %d", n_samples_tot)
    covs.sort(key=lambda c: c["loglik"], reverse=True)

    if len(covs) > 1:
        msg = ["log-likelihood on unseen data (descending order):"]
        for c in covs:
            msg.append(f"{c['method']}: {c['loglik']:0.3f}")
        logger.info("\n   ".join(msg))
        if return_estimators:
            out = covs
        else:
            out = covs[0]
            logger.info("selecting best estimator: {}".format(out["method"]))
    else:
        out = covs[0]
    logger.info("[done]")

    return out


def _check_scalings_user(scalings):
    if isinstance(scalings, dict):
        for k, v in scalings.items():
            _check_option("the keys in `scalings`", k, ["mag", "grad", "eeg"])
    elif scalings is not None and not isinstance(scalings, np.ndarray):
        raise TypeError(
            f"scalings must be a dict, ndarray, or None, got {type(scalings)}"
        )
    scalings = _handle_default("scalings", scalings)
    return scalings


def _eigvec_subspace(eig, eigvec, mask):
    """Compute the subspace from a subset of eigenvectors."""
    # We do the same thing we do with projectors:
    P = np.eye(len(eigvec)) - np.dot(eigvec[~mask].conj().T, eigvec[~mask])
    eig, eigvec = eigh(P)
    eigvec = eigvec.conj().T
    return eig, eigvec


@verbose
def _compute_rank_raw_array(
    data, info, rank, scalings, *, log_ch_type=None, verbose=None
):
    from .io import RawArray

    return _compute_rank(
        RawArray(data, info, copy=None, verbose=_verbose_safe_false()),
        rank,
        scalings,
        info,
        log_ch_type=log_ch_type,
    )


def _compute_covariance_auto(
    data,
    method,
    info,
    method_params,
    cv,
    scalings,
    n_jobs,
    stop_early,
    picks_list,
    rank,
    *,
    cov_kind="",
    log_ch_type=None,
    log_rank=True,
):
    """Compute covariance auto mode."""
    # rescale to improve numerical stability
    orig_rank = rank
    rank = _compute_rank_raw_array(
        data.T,
        info,
        rank=rank,
        scalings=scalings,
        verbose=_verbose_safe_false(),
    )
    with _scaled_array(data.T, picks_list, scalings):
        C = np.dot(data.T, data)
        _, eigvec, mask = _smart_eigh(
            C,
            info,
            rank,
            proj_subspace=True,
            do_compute_rank=False,
            log_ch_type=log_ch_type,
            verbose=None if log_rank else _verbose_safe_false(),
        )
        eigvec = eigvec[mask]
        data = np.dot(data, eigvec.T)
        used = np.where(mask)[0]
        sub_picks_list = [
            (key, np.searchsorted(used, picks)) for key, picks in picks_list
        ]
        sub_info = pick_info(info, used) if len(used) != len(mask) else info
        if log_rank:
            logger.info(f"Reducing data rank from {len(mask)} -> {eigvec.shape[0]}")
        estimator_cov_info = list()

        ok_sklearn = check_version("sklearn")
        if not ok_sklearn and (len(method) != 1 or method[0] != "empirical"):
            raise ValueError(
                'scikit-learn is not installed, `method` must be "empirical", got '
                f"{repr(method)}"
            )

        for method_ in method:
            data_ = data.copy()
            name = method_.__name__ if callable(method_) else method_
            logger.info(
                f'Estimating {cov_kind + (" " if cov_kind else "")}'
                f"covariance using {name.upper()}"
            )
            mp = method_params[method_]
            _info = {}

            if method_ == "empirical":
                est = EmpiricalCovariance(**mp)
                est.fit(data_)
                estimator_cov_info.append((est, est.covariance_, _info))
                del est

            elif method_ == "diagonal_fixed":
                est = _RegCovariance(info=sub_info, **mp)
                est.fit(data_)
                estimator_cov_info.append((est, est.covariance_, _info))
                del est

            elif method_ == "ledoit_wolf":
                from sklearn.covariance import LedoitWolf

                shrinkages = []
                lw = LedoitWolf(**mp)

                for ch_type, picks in sub_picks_list:
                    lw.fit(data_[:, picks])
                    shrinkages.append((ch_type, lw.shrinkage_, picks))
                sc = _ShrunkCovariance(shrinkage=shrinkages, **mp)
                sc.fit(data_)
                estimator_cov_info.append((sc, sc.covariance_, _info))
                del lw, sc

            elif method_ == "oas":
                from sklearn.covariance import OAS

                shrinkages = []
                oas = OAS(**mp)

                for ch_type, picks in sub_picks_list:
                    oas.fit(data_[:, picks])
                    shrinkages.append((ch_type, oas.shrinkage_, picks))
                sc = _ShrunkCovariance(shrinkage=shrinkages, **mp)
                sc.fit(data_)
                estimator_cov_info.append((sc, sc.covariance_, _info))
                del oas, sc

            elif method_ == "shrinkage":
                sc = _ShrunkCovariance(**mp)
                sc.fit(data_)
                estimator_cov_info.append((sc, sc.covariance_, _info))
                del sc

            elif method_ == "shrunk":
                from sklearn.covariance import ShrunkCovariance
                from sklearn.model_selection import GridSearchCV

                shrinkage = mp.pop("shrinkage")
                tuned_parameters = [{"shrinkage": shrinkage}]
                shrinkages = []
                gs = GridSearchCV(ShrunkCovariance(**mp), tuned_parameters, cv=cv)
                for ch_type, picks in sub_picks_list:
                    gs.fit(data_[:, picks])
                    shrinkages.append((ch_type, gs.best_estimator_.shrinkage, picks))
                shrinkages = [c[0] for c in zip(shrinkages)]
                sc = _ShrunkCovariance(shrinkage=shrinkages, **mp)
                sc.fit(data_)
                estimator_cov_info.append((sc, sc.covariance_, _info))
                del shrinkage, sc

            elif method_ == "pca":
                assert orig_rank == "full"
                pca, _info = _auto_low_rank_model(
                    data_,
                    method_,
                    n_jobs=n_jobs,
                    method_params=mp,
                    cv=cv,
                    stop_early=stop_early,
                )
                pca.fit(data_)
                estimator_cov_info.append((pca, pca.get_covariance(), _info))
                del pca

            elif method_ == "factor_analysis":
                assert orig_rank == "full"
                fa, _info = _auto_low_rank_model(
                    data_,
                    method_,
                    n_jobs=n_jobs,
                    method_params=mp,
                    cv=cv,
                    stop_early=stop_early,
                )
                fa.fit(data_)
                estimator_cov_info.append((fa, fa.get_covariance(), _info))
                del fa
            else:
                raise ValueError("Oh no! Your estimator does not have a .fit method")
            logger.info("Done.")

        if len(method) > 1:
            logger.info("Using cross-validation to select the best estimator.")

        out = dict()
        for ei, (estimator, cov, runtime_info) in enumerate(estimator_cov_info):
            if len(method) > 1:
                loglik = _cross_val(data, estimator, cv, n_jobs)
            else:
                loglik = None
            # project back
            cov = np.dot(eigvec.T, np.dot(cov, eigvec))
            # undo bias
            cov *= data.shape[0] / (data.shape[0] - 1)
            # undo scaling
            _undo_scaling_cov(cov, picks_list, scalings)
            method_ = method[ei]
            name = method_.__name__ if callable(method_) else method_
            out[name] = dict(loglik=loglik, data=cov, estimator=estimator)
            out[name].update(runtime_info)

    return out


def _gaussian_loglik_scorer(est, X, y=None):
    """Compute the Gaussian log likelihood of X under the model in est."""
    # compute empirical covariance of the test set
    precision = est.get_precision()
    n_samples, n_features = X.shape
    log_like = -0.5 * (X * (np.dot(X, precision))).sum(axis=1)
    log_like -= 0.5 * (n_features * log(2.0 * np.pi) - _logdet(precision))
    out = np.mean(log_like)
    return out


def _cross_val(data, est, cv, n_jobs):
    """Compute cross validation."""
    from sklearn.model_selection import cross_val_score

    return np.mean(
        cross_val_score(
            est, data, cv=cv, n_jobs=n_jobs, scoring=_gaussian_loglik_scorer
        )
    )


def _auto_low_rank_model(
    data, mode, n_jobs, method_params, cv, stop_early=True, verbose=None
):
    """Compute latent variable models."""
    method_params = deepcopy(method_params)
    iter_n_components = method_params.pop("iter_n_components")
    if iter_n_components is None:
        iter_n_components = np.arange(5, data.shape[1], 5)
    from sklearn.decomposition import PCA, FactorAnalysis

    if mode == "factor_analysis":
        est = FactorAnalysis
    else:
        assert mode == "pca"
        est = PCA
    est = est(**method_params)
    est.n_components = 1
    scores = np.empty_like(iter_n_components, dtype=np.float64)
    scores.fill(np.nan)

    # make sure we don't empty the thing if it's a generator
    max_n = max(list(deepcopy(iter_n_components)))
    if max_n > data.shape[1]:
        warn(
            f"You are trying to estimate {max_n} components on matrix "
            f"with {data.shape[1]} features."
        )

    for ii, n in enumerate(iter_n_components):
        est.n_components = n
        try:  # this may fail depending on rank and split
            score = _cross_val(data=data, est=est, cv=cv, n_jobs=n_jobs)
        except ValueError:
            score = np.inf
        if np.isinf(score) or score > 0:
            logger.info("... infinite values encountered. stopping estimation")
            break
        logger.info("... rank: %i - loglik: %0.3f", n, score)
        if score != -np.inf:
            scores[ii] = score

        if ii >= 3 and np.all(np.diff(scores[ii - 3 : ii]) < 0) and stop_early:
            # early stop search when loglik has been going down 3 times
            logger.info("early stopping parameter search.")
            break

    # happens if rank is too low right form the beginning
    if np.isnan(scores).all():
        raise RuntimeError(
            "Oh no! Could not estimate covariance because all "
            "scores were NaN. Please contact the MNE-Python "
            "developers."
        )

    i_score = np.nanargmax(scores)
    best = est.n_components = iter_n_components[i_score]
    logger.info("... best model at rank = %i", best)
    runtime_info = {
        "ranks": np.array(iter_n_components),
        "scores": scores,
        "best": best,
        "cv": cv,
    }
    return est, runtime_info


###############################################################################
# Sklearn Estimators


class _RegCovariance(_EstimatorMixin):
    """Aux class."""

    def __init__(
        self,
        info,
        grad=0.1,
        mag=0.1,
        eeg=0.1,
        seeg=0.1,
        ecog=0.1,
        hbo=0.1,
        hbr=0.1,
        fnirs_cw_amplitude=0.1,
        fnirs_fd_ac_amplitude=0.1,
        fnirs_fd_phase=0.1,
        fnirs_od=0.1,
        csd=0.1,
        dbs=0.1,
        store_precision=False,
        assume_centered=False,
    ):
        self.info = info
        # For sklearn compat, these cannot (easily?) be combined into
        # a single dictionary
        self.grad = grad
        self.mag = mag
        self.eeg = eeg
        self.seeg = seeg
        self.dbs = dbs
        self.ecog = ecog
        self.hbo = hbo
        self.hbr = hbr
        self.fnirs_cw_amplitude = fnirs_cw_amplitude
        self.fnirs_fd_ac_amplitude = fnirs_fd_ac_amplitude
        self.fnirs_fd_phase = fnirs_fd_phase
        self.fnirs_od = fnirs_od
        self.csd = csd
        self.store_precision = store_precision
        self.assume_centered = assume_centered

    def fit(self, X):
        """Fit covariance model with classical diagonal regularization."""
        self.estimator_ = EmpiricalCovariance(
            store_precision=self.store_precision, assume_centered=self.assume_centered
        )

        self.covariance_ = self.estimator_.fit(X).covariance_
        self.covariance_ = 0.5 * (self.covariance_ + self.covariance_.T)
        cov_ = Covariance(
            data=self.covariance_,
            names=self.info["ch_names"],
            bads=self.info["bads"],
            projs=self.info["projs"],
            nfree=len(self.covariance_),
        )
        cov_ = regularize(
            cov_,
            self.info,
            proj=False,
            exclude="bads",
            grad=self.grad,
            mag=self.mag,
            eeg=self.eeg,
            ecog=self.ecog,
            seeg=self.seeg,
            dbs=self.dbs,
            hbo=self.hbo,
            hbr=self.hbr,
            rank="full",
        )
        self.estimator_.covariance_ = self.covariance_ = cov_.data
        return self

    def score(self, X_test, y=None):
        """Delegate call to modified EmpiricalCovariance instance."""
        return self.estimator_.score(X_test, y=y)

    def get_precision(self):
        """Delegate call to modified EmpiricalCovariance instance."""
        return self.estimator_.get_precision()


class _ShrunkCovariance(_EstimatorMixin):
    """Aux class."""

    def __init__(self, store_precision, assume_centered, shrinkage=0.1):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.shrinkage = shrinkage

    def fit(self, X):
        """Fit covariance model with oracle shrinkage regularization."""
        from sklearn.covariance import shrunk_covariance

        self.estimator_ = EmpiricalCovariance(
            store_precision=self.store_precision, assume_centered=self.assume_centered
        )

        cov = self.estimator_.fit(X).covariance_

        if not isinstance(self.shrinkage, list | tuple):
            shrinkage = [("all", self.shrinkage, np.arange(len(cov)))]
        else:
            shrinkage = self.shrinkage

        zero_cross_cov = np.zeros_like(cov, dtype=bool)
        for a, b in itt.combinations(shrinkage, 2):
            picks_i, picks_j = a[2], b[2]
            ch_ = a[0], b[0]
            if "eeg" in ch_:
                zero_cross_cov[np.ix_(picks_i, picks_j)] = True
                zero_cross_cov[np.ix_(picks_j, picks_i)] = True

        self.zero_cross_cov_ = zero_cross_cov

        # Apply shrinkage to blocks
        for ch_type, c, picks in shrinkage:
            sub_cov = cov[np.ix_(picks, picks)]
            cov[np.ix_(picks, picks)] = shrunk_covariance(sub_cov, shrinkage=c)

        # Apply shrinkage to cross-cov
        for a, b in itt.combinations(shrinkage, 2):
            shrinkage_i, shrinkage_j = a[1], b[1]
            picks_i, picks_j = a[2], b[2]
            c_ij = np.sqrt((1.0 - shrinkage_i) * (1.0 - shrinkage_j))
            cov[np.ix_(picks_i, picks_j)] *= c_ij
            cov[np.ix_(picks_j, picks_i)] *= c_ij

        # Set to zero the necessary cross-cov
        if np.any(zero_cross_cov):
            cov[zero_cross_cov] = 0.0

        self.estimator_.covariance_ = self.covariance_ = cov
        return self

    def score(self, X_test, y=None):
        """Delegate to modified EmpiricalCovariance instance."""
        # compute empirical covariance of the test set
        test_cov = empirical_covariance(
            X_test - self.estimator_.location_, assume_centered=True
        )
        if np.any(self.zero_cross_cov_):
            test_cov[self.zero_cross_cov_] = 0.0
        res = log_likelihood(test_cov, self.estimator_.get_precision())
        return res

    def get_precision(self):
        """Delegate to modified EmpiricalCovariance instance."""
        return self.estimator_.get_precision()


###############################################################################
# Writing


@verbose
def write_cov(fname, cov, *, overwrite=False, verbose=None):
    """Write a noise covariance matrix.

    Parameters
    ----------
    fname : path-like
        The name of the file. It should end with ``-cov.fif`` or
        ``-cov.fif.gz``.
    cov : Covariance
        The noise covariance matrix.
    %(overwrite)s

        .. versionadded:: 1.0
    %(verbose)s

    See Also
    --------
    read_cov
    """
    cov.save(fname, overwrite=overwrite, verbose=verbose)


###############################################################################
# Prepare for inverse modeling


def _unpack_epochs(epochs):
    """Aux Function."""
    if len(epochs.event_id) > 1:
        epochs = [epochs[k] for k in epochs.event_id]
    else:
        epochs = [epochs]

    return epochs


def _get_ch_whitener(A, pca, ch_type, rank):
    """Get whitener params for a set of channels."""
    # whitening operator
    eig, eigvec = eigh(A, overwrite_a=True)
    eigvec = eigvec.conj().T
    mask = np.ones(len(eig), bool)
    eig[:-rank] = 0.0
    mask[:-rank] = False

    logger.info(
        f"    Setting small {ch_type} eigenvalues to zero "
        f'({"using" if pca else "without"} PCA)'
    )
    if pca:  # No PCA case.
        # This line will reduce the actual number of variables in data
        # and leadfield to the true rank.
        eigvec = eigvec[:-rank].copy()
    return eig, eigvec, mask


@verbose
def prepare_noise_cov(
    noise_cov,
    info,
    ch_names=None,
    rank=None,
    scalings=None,
    on_rank_mismatch="ignore",
    verbose=None,
):
    """Prepare noise covariance matrix.

    Parameters
    ----------
    noise_cov : instance of Covariance
        The noise covariance to process.
    %(info_not_none)s (Used to get channel types and bad channels).
    ch_names : list | None
        The channel names to be considered. Can be None to use
        ``info['ch_names']``.
    %(rank_none)s

        .. versionadded:: 0.18
           Support for 'info' mode.
    scalings : dict | None
        Data will be rescaled before rank estimation to improve accuracy.
        If dict, it will override the following dict (default if None)::

            dict(mag=1e12, grad=1e11, eeg=1e5)
    %(on_rank_mismatch)s
    %(verbose)s

    Returns
    -------
    cov : instance of Covariance
        A copy of the covariance with the good channels subselected
        and parameters updated.
    """
    # reorder C and info to match ch_names order
    noise_cov_idx = list()
    missing = list()
    ch_names = info["ch_names"] if ch_names is None else ch_names
    for c in ch_names:
        # this could be try/except ValueError, but it is not the preferred way
        if c in noise_cov.ch_names:
            noise_cov_idx.append(noise_cov.ch_names.index(c))
        else:
            missing.append(c)
    if len(missing):
        raise RuntimeError(f"Not all channels present in noise covariance:\n{missing}")
    C = noise_cov._get_square()[np.ix_(noise_cov_idx, noise_cov_idx)]
    info = pick_info(info, pick_channels(info["ch_names"], ch_names, ordered=False))
    projs = info["projs"] + noise_cov["projs"]
    noise_cov = Covariance(
        data=C,
        names=ch_names,
        bads=list(noise_cov["bads"]),
        projs=deepcopy(noise_cov["projs"]),
        nfree=noise_cov["nfree"],
        method=noise_cov.get("method", None),
        loglik=noise_cov.get("loglik", None),
    )

    eig, eigvec, _ = _smart_eigh(
        noise_cov,
        info,
        rank,
        scalings,
        projs,
        ch_names,
        on_rank_mismatch=on_rank_mismatch,
    )
    noise_cov.update(eig=eig, eigvec=eigvec)
    return noise_cov


@verbose
def _smart_eigh(
    C,
    info,
    rank,
    scalings=None,
    projs=None,
    ch_names=None,
    proj_subspace=False,
    do_compute_rank=True,
    on_rank_mismatch="ignore",
    *,
    log_ch_type=None,
    verbose=None,
):
    """Compute eigh of C taking into account rank and ch_type scalings."""
    scalings = _handle_default("scalings_cov_rank", scalings)
    projs = info["projs"] if projs is None else projs
    ch_names = info["ch_names"] if ch_names is None else ch_names
    if info["ch_names"] != ch_names:
        info = pick_info(info, [info["ch_names"].index(c) for c in ch_names])
    assert info["ch_names"] == ch_names
    n_chan = len(ch_names)

    # Create the projection operator
    proj, ncomp, _ = _make_projector(projs, ch_names)

    if isinstance(C, Covariance):
        C = C["data"]
    if ncomp > 0:
        logger.info("    Created an SSP operator (subspace dimension = %d)", ncomp)
        C = np.dot(proj, np.dot(C, proj.T))

    noise_cov = Covariance(C, ch_names, [], projs, 0)
    if do_compute_rank:  # if necessary
        rank = _compute_rank(
            noise_cov,
            rank,
            scalings,
            info,
            on_rank_mismatch=on_rank_mismatch,
            log_ch_type=log_ch_type,
        )
    assert C.ndim == 2 and C.shape[0] == C.shape[1]

    # time saving short-circuit
    if proj_subspace and sum(rank.values()) == C.shape[0]:
        return np.ones(n_chan), np.eye(n_chan), np.ones(n_chan, bool)

    dtype = complex if C.dtype == np.complex128 else float
    eig = np.zeros(n_chan, dtype)
    eigvec = np.zeros((n_chan, n_chan), dtype)
    mask = np.zeros(n_chan, bool)
    for ch_type, picks in _picks_by_type(
        info, meg_combined=True, ref_meg=False, exclude=[]
    ):
        if len(picks) == 0:
            continue
        this_C = C[np.ix_(picks, picks)]

        if ch_type not in rank and ch_type in ("mag", "grad"):
            this_rank = rank["meg"]  # if there is only one or the other
        else:
            this_rank = rank[ch_type]

        if log_ch_type is not None:
            ch_type_ = log_ch_type
        else:
            ch_type_ = ch_type.upper()
        e, ev, m = _get_ch_whitener(this_C, False, ch_type_, this_rank)
        if proj_subspace:
            # Choose the subspace the same way we do for projections
            e, ev = _eigvec_subspace(e, ev, m)
        eig[picks], eigvec[np.ix_(picks, picks)], mask[picks] = e, ev, m
        largest, smallest = e[-1], e[m][0]
        if largest > 1e10 * smallest:
            warn(
                f"The largest eigenvalue of the {len(picks)}-channel {ch_type} "
                f"covariance (rank={this_rank}) is over 10 orders of magnitude "
                f"larger than the smallest ({largest:0.3g} > 1e10 * {smallest:0.3g}), "
                "the resulting whitener will likely be unstable"
            )

        # XXX : also handle ref for sEEG and ECoG
        if (
            ch_type == "eeg"
            and _needs_eeg_average_ref_proj(info)
            and not _has_eeg_average_ref_proj(info, projs=projs)
        ):
            warn(
                'No average EEG reference present in info["projs"], '
                "covariance may be adversely affected. Consider recomputing "
                "covariance using with an average eeg reference projector "
                "added."
            )
    return eig, eigvec, mask


@verbose
def regularize(
    cov,
    info,
    mag=0.1,
    grad=0.1,
    eeg=0.1,
    exclude="bads",
    proj=True,
    seeg=0.1,
    ecog=0.1,
    hbo=0.1,
    hbr=0.1,
    fnirs_cw_amplitude=0.1,
    fnirs_fd_ac_amplitude=0.1,
    fnirs_fd_phase=0.1,
    fnirs_od=0.1,
    csd=0.1,
    dbs=0.1,
    rank=None,
    scalings=None,
    verbose=None,
):
    """Regularize noise covariance matrix.

    This method works by adding a constant to the diagonal for each
    channel type separately. Special care is taken to keep the
    rank of the data constant.

    .. note:: This function is kept for reasons of backward-compatibility.
              Please consider explicitly using the ``method`` parameter in
              :func:`mne.compute_covariance` to directly combine estimation
              with regularization in a data-driven fashion. See the
              :ref:`FAQ <faq_how_should_i_regularize>` for more information.

    Parameters
    ----------
    cov : Covariance
        The noise covariance matrix.
    %(info_not_none)s (Used to get channel types and bad channels).
    mag : float (default 0.1)
        Regularization factor for MEG magnetometers.
    grad : float (default 0.1)
        Regularization factor for MEG gradiometers. Must be the same as
        ``mag`` if data have been processed with SSS.
    eeg : float (default 0.1)
        Regularization factor for EEG.
    exclude : list | 'bads' (default 'bads')
        List of channels to mark as bad. If 'bads', bads channels
        are extracted from both info['bads'] and cov['bads'].
    proj : bool (default True)
        Apply projections to keep rank of data.
    seeg : float (default 0.1)
        Regularization factor for sEEG signals.
    ecog : float (default 0.1)
        Regularization factor for ECoG signals.
    hbo : float (default 0.1)
        Regularization factor for HBO signals.
    hbr : float (default 0.1)
        Regularization factor for HBR signals.
    fnirs_cw_amplitude : float (default 0.1)
        Regularization factor for fNIRS CW raw signals.
    fnirs_fd_ac_amplitude : float (default 0.1)
        Regularization factor for fNIRS FD AC raw signals.
    fnirs_fd_phase : float (default 0.1)
        Regularization factor for fNIRS raw phase signals.
    fnirs_od : float (default 0.1)
        Regularization factor for fNIRS optical density signals.
    csd : float (default 0.1)
        Regularization factor for EEG-CSD signals.
    dbs : float (default 0.1)
        Regularization factor for DBS signals.
    %(rank_none)s

        .. versionadded:: 0.17

        .. versionadded:: 0.18
           Support for 'info' mode.
    scalings : dict | None
        Data will be rescaled before rank estimation to improve accuracy.
        See :func:`mne.compute_covariance`.

        .. versionadded:: 0.17
    %(verbose)s

    Returns
    -------
    reg_cov : Covariance
        The regularized covariance matrix.

    See Also
    --------
    mne.compute_covariance
    """  # noqa: E501
    cov = cov.copy()
    info._check_consistency()
    scalings = _handle_default("scalings_cov_rank", scalings)
    regs = dict(
        eeg=eeg,
        seeg=seeg,
        dbs=dbs,
        ecog=ecog,
        hbo=hbo,
        hbr=hbr,
        fnirs_cw_amplitude=fnirs_cw_amplitude,
        fnirs_fd_ac_amplitude=fnirs_fd_ac_amplitude,
        fnirs_fd_phase=fnirs_fd_phase,
        fnirs_od=fnirs_od,
        csd=csd,
    )

    if exclude is None:
        raise ValueError('exclude must be a list of strings or "bads"')

    if exclude == "bads":
        exclude = info["bads"] + cov["bads"]

    picks_dict = {ch_type: [] for ch_type in _DATA_CH_TYPES_SPLIT}
    meg_combined = "auto" if rank != "full" else False
    picks_dict.update(
        dict(
            _picks_by_type(
                info, meg_combined=meg_combined, exclude=exclude, ref_meg=False
            )
        )
    )
    if len(picks_dict.get("meg", [])) > 0 and rank != "full":  # combined
        if mag != grad:
            raise ValueError(
                "On data where magnetometers and gradiometers are dependent (e.g., "
                f"SSSed data), mag ({mag}) must equal grad ({grad})"
            )
        logger.info("Regularizing MEG channels jointly")
        regs["meg"] = mag
    else:
        regs.update(mag=mag, grad=grad)
    if rank != "full":
        rank = _compute_rank(cov, rank, scalings, info)

    info_ch_names = info["ch_names"]
    ch_names_by_type = dict()
    for ch_type, picks_type in picks_dict.items():
        ch_names_by_type[ch_type] = [info_ch_names[i] for i in picks_type]

    # This actually removes bad channels from the cov, which is not backward
    # compatible, so let's leave all channels in
    cov_good = pick_channels_cov(
        cov, include=info_ch_names, exclude=exclude, ordered=False
    )
    ch_names = cov_good.ch_names

    # Now get the indices for each channel type in the cov
    idx_cov = {ch_type: [] for ch_type in ch_names_by_type}
    for i, ch in enumerate(ch_names):
        for ch_type in ch_names_by_type:
            if ch in ch_names_by_type[ch_type]:
                idx_cov[ch_type].append(i)
                break
        else:
            raise Exception(f"channel {ch} is unknown type")

    C = cov_good["data"]

    assert len(C) == sum(map(len, idx_cov.values()))

    if proj:
        projs = info["projs"] + cov_good["projs"]
        projs = _activate_proj(projs)

    for ch_type in idx_cov:
        desc = ch_type.upper()
        idx = idx_cov[ch_type]
        if len(idx) == 0:
            continue
        reg = regs[ch_type]
        if reg == 0.0:
            logger.info(f"    {desc} regularization : None")
            continue
        logger.info(f"    {desc} regularization : {reg}")

        this_C = C[np.ix_(idx, idx)]
        U = np.eye(this_C.shape[0])
        this_ch_names = [ch_names[k] for k in idx]
        if rank == "full":
            if proj:
                P, ncomp, _ = _make_projector(projs, this_ch_names)
                if ncomp > 0:
                    # This adjustment ends up being redundant if rank is None:
                    U = _safe_svd(P)[0][:, :-ncomp]
                    logger.info(
                        f"    Created an SSP operator for {desc} (dimension = {ncomp})"
                    )
        else:
            this_picks = pick_channels(info["ch_names"], this_ch_names)
            this_info = pick_info(info, this_picks)
            # Here we could use proj_subspace=True, but this should not matter
            # since this is already in a loop over channel types
            _, eigvec, mask = _smart_eigh(this_C, this_info, rank)
            U = eigvec[mask].T
        this_C = np.dot(U.T, np.dot(this_C, U))

        sigma = np.mean(np.diag(this_C))
        this_C.flat[:: len(this_C) + 1] += reg * sigma  # modify diag inplace
        this_C = np.dot(U, np.dot(this_C, U.T))
        C[np.ix_(idx, idx)] = this_C

    # Put data back in correct locations
    idx = pick_channels(cov.ch_names, info_ch_names, exclude=exclude, ordered=False)
    cov["data"][np.ix_(idx, idx)] = C

    return cov


def _regularized_covariance(
    data,
    reg=None,
    method_params=None,
    info=None,
    rank=None,
    *,
    log_ch_type=None,
    log_rank=None,
    cov_kind="",
):
    """Compute a regularized covariance from data using sklearn.

    This is a convenience wrapper for mne.decoding functions, which
    adopted a slightly different covariance API.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        The covariance matrix.
    """
    _validate_type(reg, (str, "numeric", None))
    if reg is None:
        reg = "empirical"
    elif not isinstance(reg, str):
        reg = float(reg)
        if method_params is not None:
            raise ValueError(
                "If reg is a float, method_params must be None (got "
                f"{type(method_params)})"
            )
        method_params = dict(
            shrinkage=dict(shrinkage=reg, assume_centered=True, store_precision=False)
        )
        reg = "shrinkage"
    method, method_params = _check_method_params(
        reg, method_params, name="reg", allow_auto=False, rank=rank
    )
    # use mag instead of eeg here to avoid the cov EEG projection warning
    info = create_info(data.shape[-2], 1000.0, "mag") if info is None else info
    picks_list = _picks_by_type(info)
    scalings = _handle_default("scalings_cov_rank", None)
    cov = _compute_covariance_auto(
        data.T,
        method=method,
        method_params=method_params,
        info=info,
        cv=None,
        n_jobs=None,
        stop_early=True,
        picks_list=picks_list,
        scalings=scalings,
        rank=rank,
        cov_kind=cov_kind,
        log_ch_type=log_ch_type,
        log_rank=log_rank,
    )[reg]["data"]
    return cov


@verbose
def compute_whitener(
    noise_cov,
    info=None,
    picks=None,
    rank=None,
    scalings=None,
    return_rank=False,
    pca=False,
    return_colorer=False,
    on_rank_mismatch="warn",
    verbose=None,
):
    """Compute whitening matrix.

    Parameters
    ----------
    noise_cov : Covariance
        The noise covariance.
    %(info)s Can be None if ``noise_cov`` has already been
        prepared with :func:`prepare_noise_cov`.
    %(picks_good_data_noref)s
    %(rank_none)s

        .. versionadded:: 0.18
           Support for 'info' mode.
    scalings : dict | None
        The rescaling method to be applied. See documentation of
        ``prepare_noise_cov`` for details.
    return_rank : bool
        If True, return the rank used to compute the whitener.

        .. versionadded:: 0.15
    pca : bool | str
        Space to project the data into. Options:

        :data:`python:True`
            Whitener will be shape (n_nonzero, n_channels).
        ``'white'``
            Whitener will be shape (n_channels, n_channels), potentially rank
            deficient, and have the first ``n_channels - n_nonzero`` rows and
            columns set to zero.
        :data:`python:False` (default)
            Whitener will be shape (n_channels, n_channels), potentially rank
            deficient, and rotated back to the space of the original data.

        .. versionadded:: 0.18
    return_colorer : bool
        If True, return the colorer as well.
    %(on_rank_mismatch)s
    %(verbose)s

    Returns
    -------
    W : ndarray, shape (n_channels, n_channels) or (n_nonzero, n_channels)
        The whitening matrix.
    ch_names : list
        The channel names.
    rank : int
        Rank reduction of the whitener. Returned only if return_rank is True.
    colorer : ndarray, shape (n_channels, n_channels) or (n_channels, n_nonzero)
        The coloring matrix.
    """  # noqa: E501
    _validate_type(pca, (str, bool), "space")
    _valid_pcas = (True, "white", False)
    if pca not in _valid_pcas:
        raise ValueError(f"space must be one of {_valid_pcas}, got {pca}")
    if info is None:
        if "eig" not in noise_cov:
            raise ValueError(
                "info can only be None if the noise cov has already been prepared with "
                "prepare_noise_cov"
            )
        ch_names = deepcopy(noise_cov["names"])
    else:
        picks = _picks_to_idx(info, picks, with_ref_meg=False)
        ch_names = [info["ch_names"][k] for k in picks]
        del picks
        noise_cov = prepare_noise_cov(
            noise_cov, info, ch_names, rank, scalings, on_rank_mismatch=on_rank_mismatch
        )

    n_chan = len(ch_names)
    assert n_chan == len(noise_cov["eig"])

    #   Omit the zeroes due to projection
    eig = noise_cov["eig"].copy()
    nzero = eig > 0
    eig[~nzero] = 0.0  # get rid of numerical noise (negative) ones

    if noise_cov["eigvec"].dtype.kind == "c":
        dtype = np.complex128
    else:
        dtype = np.float64
    W = np.zeros((n_chan, 1), dtype)
    W[nzero, 0] = 1.0 / np.sqrt(eig[nzero])
    #   Rows of eigvec are the eigenvectors
    W = W * noise_cov["eigvec"]  # C ** -0.5
    C = np.sqrt(eig) * noise_cov["eigvec"].conj().T  # C ** 0.5
    n_nzero = nzero.sum()
    logger.info(
        "    Created the whitener using a noise covariance matrix "
        "with rank %d (%d small eigenvalues omitted)",
        n_nzero,
        noise_cov["dim"] - n_nzero,
    )

    # Do the requested projection
    if pca is True:
        W = W[nzero]
        C = C[:, nzero]
    elif pca is False:
        W = np.dot(noise_cov["eigvec"].conj().T, W)
        C = np.dot(C, noise_cov["eigvec"])

    # Triage return
    out = W, ch_names
    if return_rank:
        out += (n_nzero,)
    if return_colorer:
        out += (C,)
    return out


@verbose
def whiten_evoked(
    evoked, noise_cov, picks=None, diag=None, rank=None, scalings=None, verbose=None
):
    """Whiten evoked data using given noise covariance.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data.
    noise_cov : instance of Covariance
        The noise covariance.
    %(picks_good_data)s
    diag : bool (default False)
        If True, whiten using only the diagonal of the covariance.
    %(rank_none)s

        .. versionadded:: 0.18
           Support for 'info' mode.
    scalings : dict | None (default None)
        To achieve reliable rank estimation on multiple sensors,
        sensors have to be rescaled. This parameter controls the
        rescaling. If dict, it will override the
        following default dict (default if None):

            dict(mag=1e12, grad=1e11, eeg=1e5)
    %(verbose)s

    Returns
    -------
    evoked_white : instance of Evoked
        The whitened evoked data.
    """
    evoked = evoked.copy()
    picks = _picks_to_idx(evoked.info, picks)

    if diag:
        noise_cov = noise_cov.as_diag()

    W, _ = compute_whitener(
        noise_cov, evoked.info, picks=picks, rank=rank, scalings=scalings
    )

    evoked.data[picks] = np.sqrt(evoked.nave) * np.dot(W, evoked.data[picks])
    return evoked


@verbose
def _read_cov(fid, node, cov_kind, limited=False, verbose=None):
    """Read a noise covariance matrix."""
    #   Find all covariance matrices
    from ._fiff.write import _safe_name_list

    covs = dir_tree_find(node, FIFF.FIFFB_MNE_COV)
    if len(covs) == 0:
        raise ValueError("No covariance matrices found")

    #   Is any of the covariance matrices a noise covariance
    for p in range(len(covs)):
        tag = find_tag(fid, covs[p], FIFF.FIFF_MNE_COV_KIND)

        if tag is not None and int(tag.data.item()) == cov_kind:
            this = covs[p]

            #   Find all the necessary data
            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_DIM)
            if tag is None:
                raise ValueError("Covariance matrix dimension not found")
            dim = int(tag.data.item())

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_NFREE)
            if tag is None:
                nfree = -1
            else:
                nfree = int(tag.data.item())

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_METHOD)
            if tag is None:
                method = None
            else:
                method = tag.data

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_SCORE)
            if tag is None:
                score = None
            else:
                score = tag.data[0]

            tag = find_tag(fid, this, FIFF.FIFF_MNE_ROW_NAMES)
            if tag is None:
                names = []
            else:
                names = _safe_name_list(tag.data, "read", "names")
                if len(names) != dim:
                    raise ValueError(
                        "Number of names does not match covariance matrix dimension"
                    )

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV)
            if tag is None:
                tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_DIAG)
                if tag is None:
                    raise ValueError("No covariance matrix data found")
                else:
                    #   Diagonal is stored
                    data = tag.data
                    diag = True
                    logger.info(
                        "    %d x %d diagonal covariance (kind = " "%d) found.",
                        dim,
                        dim,
                        cov_kind,
                    )

            else:
                if not issparse(tag.data):
                    #   Lower diagonal is stored
                    vals = tag.data
                    data = np.zeros((dim, dim))
                    data[np.tril(np.ones((dim, dim))) > 0] = vals
                    data = data + data.T
                    data.flat[:: dim + 1] /= 2.0
                    diag = False
                    logger.info(
                        "    %d x %d full covariance (kind = %d) " "found.",
                        dim,
                        dim,
                        cov_kind,
                    )
                else:
                    diag = False
                    data = tag.data
                    logger.info(
                        "    %d x %d sparse covariance (kind = %d)" " found.",
                        dim,
                        dim,
                        cov_kind,
                    )

            #   Read the possibly precomputed decomposition
            tag1 = find_tag(fid, this, FIFF.FIFF_MNE_COV_EIGENVALUES)
            tag2 = find_tag(fid, this, FIFF.FIFF_MNE_COV_EIGENVECTORS)
            if tag1 is not None and tag2 is not None:
                eig = tag1.data
                eigvec = tag2.data
            else:
                eig = None
                eigvec = None

            #   Read the projection operator
            projs = _read_proj(fid, this)

            #   Read the bad channel list
            bads = _read_bad_channels(fid, this, None)

            #   Put it together
            assert dim == len(data)
            assert data.ndim == (1 if diag else 2)
            cov = dict(
                kind=cov_kind,
                diag=diag,
                dim=dim,
                names=names,
                data=data,
                projs=projs,
                bads=bads,
                nfree=nfree,
                eig=eig,
                eigvec=eigvec,
            )
            if score is not None:
                cov["loglik"] = score
            if method is not None:
                cov["method"] = method
            if limited:
                del cov["kind"], cov["dim"], cov["diag"]

            return cov

    logger.info("    Did not find the desired covariance matrix (kind = %d)", cov_kind)

    return None


def _write_cov(fid, cov):
    """Write a noise covariance matrix."""
    from ._fiff.write import (
        end_block,
        start_block,
        write_double,
        write_float_matrix,
        write_int,
        write_name_list_sanitized,
        write_string,
    )

    start_block(fid, FIFF.FIFFB_MNE_COV)

    #   Dimensions etc.
    write_int(fid, FIFF.FIFF_MNE_COV_KIND, cov["kind"])
    write_int(fid, FIFF.FIFF_MNE_COV_DIM, cov["dim"])
    if cov["nfree"] > 0:
        write_int(fid, FIFF.FIFF_MNE_COV_NFREE, cov["nfree"])

    #   Channel names
    if cov["names"] is not None and len(cov["names"]) > 0:
        write_name_list_sanitized(
            fid, FIFF.FIFF_MNE_ROW_NAMES, cov["names"], 'cov["names"]'
        )

    #   Data
    if cov["diag"]:
        write_double(fid, FIFF.FIFF_MNE_COV_DIAG, cov["data"])
    else:
        # Store only lower part of covariance matrix
        dim = cov["dim"]
        mask = np.tril(np.ones((dim, dim), dtype=bool)) > 0
        vals = cov["data"][mask].ravel()
        write_double(fid, FIFF.FIFF_MNE_COV, vals)

    #   Eigenvalues and vectors if present
    if cov["eig"] is not None and cov["eigvec"] is not None:
        write_float_matrix(fid, FIFF.FIFF_MNE_COV_EIGENVECTORS, cov["eigvec"])
        write_double(fid, FIFF.FIFF_MNE_COV_EIGENVALUES, cov["eig"])

    #   Projection operator
    if cov["projs"] is not None and len(cov["projs"]) > 0:
        _write_proj(fid, cov["projs"])

    #   Bad channels
    _write_bad_channels(fid, cov["bads"], None)

    # estimator method
    if "method" in cov:
        write_string(fid, FIFF.FIFF_MNE_COV_METHOD, cov["method"])

    # negative log-likelihood score
    if "loglik" in cov:
        write_double(fid, FIFF.FIFF_MNE_COV_SCORE, np.array(cov["loglik"]))

    #   Done!
    end_block(fid, FIFF.FIFFB_MNE_COV)


@verbose
def _ensure_cov(cov, name="cov", *, verbose=None):
    _validate_type(cov, ("path-like", Covariance), name)
    logger.info(f"Noise covariance  : {cov}")
    if not isinstance(cov, Covariance):
        cov = read_cov(cov, verbose=_verbose_safe_false())
    return cov
