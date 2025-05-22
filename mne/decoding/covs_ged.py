"""Covariance estimation for GED transformers."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import scipy.linalg

from .._fiff.meas_info import Info, create_info
from .._fiff.pick import _picks_to_idx
from ..cov import Covariance, _compute_rank_raw_array, _regularized_covariance
from ..filter import filter_data
from ..utils import _verbose_safe_false, logger, pinv


def _concat_cov(x_class, *, cov_kind, log_rank, reg, cov_method_params, rank, info):
    """Concatenate epochs before computing the covariance."""
    _, n_channels, _ = x_class.shape

    x_class = x_class.transpose(1, 0, 2).reshape(n_channels, -1)
    cov = _regularized_covariance(
        x_class,
        reg=reg,
        method_params=cov_method_params,
        rank=rank,
        info=info,
        cov_kind=cov_kind,
        log_rank=log_rank,
        log_ch_type="data",
    )
    weight = x_class.shape[0]

    return cov, weight


def _epoch_cov(x_class, *, cov_kind, log_rank, reg, cov_method_params, rank, info):
    """Mean of per-epoch covariances."""
    name = reg if isinstance(reg, str) else "empirical"
    name += " with shrinkage" if isinstance(reg, float) else ""
    logger.info(
        f"Estimating {cov_kind + (' ' if cov_kind else '')}"
        f"covariance (average over epochs; {name.upper()})"
    )
    cov = sum(
        _regularized_covariance(
            this_X,
            reg=reg,
            method_params=cov_method_params,
            rank=rank,
            info=info,
            cov_kind=cov_kind,
            log_rank=log_rank and ii == 0,
            log_ch_type="data",
            verbose=_verbose_safe_false(),
        )
        for ii, this_X in enumerate(x_class)
    )
    cov /= len(x_class)
    weight = len(x_class)

    return cov, weight


def _csp_estimate(X, y, reg, cov_method_params, cov_est, rank, norm_trace):
    _, n_channels, _ = X.shape
    classes_ = np.unique(y)
    if cov_est == "concat":
        cov_estimator = _concat_cov
    elif cov_est == "epoch":
        cov_estimator = _epoch_cov
    # Someday we could allow the user to pass this, then we wouldn't need to convert
    # but in the meantime they can use a pipeline with a scaler
    _info = create_info(n_channels, 1000.0, "mag")
    if isinstance(rank, dict):
        _rank = {"mag": sum(rank.values())}
    else:
        _rank = _compute_rank_raw_array(
            X.transpose(1, 0, 2).reshape(X.shape[1], -1),
            _info,
            rank=rank,
            scalings=None,
            log_ch_type="data",
        )

    covs = []
    sample_weights = []
    for ci, this_class in enumerate(classes_):
        cov, weight = cov_estimator(
            X[y == this_class],
            cov_kind=f"class={this_class}",
            log_rank=ci == 0,
            reg=reg,
            cov_method_params=cov_method_params,
            rank=_rank,
            info=_info,
        )

        if norm_trace:
            cov /= np.trace(cov)

        covs.append(cov)
        sample_weights.append(weight)

    covs = np.stack(covs)
    C_ref = covs.mean(0)

    return covs, C_ref, _info, _rank, dict(sample_weights=np.array(sample_weights))


def _construct_signal_from_epochs(epochs, events, sfreq, tmin):
    """Reconstruct pseudo continuous signal from epochs."""
    n_epochs, n_channels, n_times = epochs.shape
    tmax = tmin + n_times / float(sfreq)
    start = np.min(events[:, 0]) + int(tmin * sfreq)
    stop = np.max(events[:, 0]) + int(tmax * sfreq) + 1

    n_samples = stop - start
    n_epochs, n_channels, n_times = epochs.shape
    events_pos = events[:, 0] - events[0, 0]

    raw = np.zeros((n_channels, n_samples))
    for idx in range(n_epochs):
        onset = events_pos[idx]
        offset = onset + n_times
        raw[:, onset:offset] = epochs[idx]

    return raw


def _least_square_evoked(epochs_data, events, tmin, sfreq):
    """Least square estimation of evoked response from epochs data.

    Parameters
    ----------
    epochs_data : array, shape (n_channels, n_times)
        The epochs data to estimate evoked.
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be ignored.
    tmin : float
        Start time before event.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    evokeds : array, shape (n_class, n_components, n_times)
        An concatenated array of evoked data for each event type.
    toeplitz : array, shape (n_class * n_components, n_channels)
        An concatenated array of toeplitz matrix for each event type.
    """
    n_epochs, n_channels, n_times = epochs_data.shape
    tmax = tmin + n_times / float(sfreq)

    # Deal with shuffled epochs
    events = events.copy()
    events[:, 0] -= events[0, 0] + int(tmin * sfreq)

    # Construct raw signal
    raw = _construct_signal_from_epochs(epochs_data, events, sfreq, tmin)

    # Compute the independent evoked responses per condition, while correcting
    # for event overlaps.
    n_min, n_max = int(tmin * sfreq), int(tmax * sfreq)
    window = n_max - n_min
    n_samples = raw.shape[1]
    toeplitz = list()
    classes = np.unique(events[:, 2])
    for ii, this_class in enumerate(classes):
        # select events by type
        sel = events[:, 2] == this_class

        # build toeplitz matrix
        trig = np.zeros((n_samples,))
        ix_trig = (events[sel, 0]) + n_min
        trig[ix_trig] = 1
        toeplitz.append(scipy.linalg.toeplitz(trig[0:window], trig))

    # Concatenate toeplitz
    toeplitz = np.array(toeplitz)
    X = np.concatenate(toeplitz)

    # least square estimation
    predictor = np.dot(pinv(np.dot(X, X.T)), X)
    evokeds = np.dot(predictor, raw.T)
    evokeds = np.transpose(np.vsplit(evokeds, len(classes)), (0, 2, 1))
    return evokeds, toeplitz


def _xdawn_estimate(
    X,
    y,
    reg,
    cov_method_params,
    R=None,
    events=None,
    tmin=0,
    sfreq=1,
    info=None,
    rank="full",
):
    if not isinstance(X, np.ndarray) or X.ndim != 3:
        raise ValueError("X must be 3D ndarray")

    classes = np.unique(y)

    # XXX Eventually this could be made to deal with rank deficiency properly
    # by exposing this "rank" parameter, but this will require refactoring
    # the linalg.eigh call to operate in the lower-dimension
    # subspace, then project back out.

    # Retrieve or compute whitening covariance
    if R is None:
        R = _regularized_covariance(
            np.hstack(X), reg, cov_method_params, info, rank=rank
        )
    elif isinstance(R, Covariance):
        R = R.data
    if not isinstance(R, np.ndarray) or (
        not np.array_equal(R.shape, np.tile(X.shape[1], 2))
    ):
        raise ValueError(
            "R must be None, a covariance instance, "
            "or an array of shape (n_chans, n_chans)"
        )

    # Get prototype events
    if events is not None:
        evokeds, toeplitzs = _least_square_evoked(X, events, tmin, sfreq)
    else:
        evokeds, toeplitzs = list(), list()
        for c in classes:
            # Prototyped response for each class
            evokeds.append(np.mean(X[y == c, :, :], axis=0))
            toeplitzs.append(1.0)

    covs = []
    for evo, toeplitz in zip(evokeds, toeplitzs):
        # Estimate covariance matrix of the prototype response
        evo = np.dot(evo, toeplitz)
        evo_cov = _regularized_covariance(evo, reg, cov_method_params, info, rank=rank)
        covs.append(evo_cov)

    covs.append(R)
    covs = np.stack(covs)
    C_ref = None
    rank = None
    info = None
    return covs, C_ref, info, rank, dict()


def _ssd_estimate(
    X,
    y,
    reg,
    cov_method_params,
    info,
    picks,
    filt_params_signal,
    filt_params_noise,
    rank,
):
    if isinstance(info, Info):
        sfreq = info["sfreq"]
    elif isinstance(info, float):  # special case, mostly for testing
        sfreq = info
        info = create_info(X.shape[-2], sfreq, ch_types="eeg")
    picks = _picks_to_idx(info, picks, none="data", exclude="bads")
    X_aux = X[..., picks, :]
    X_signal = filter_data(X_aux, sfreq, **filt_params_signal)
    X_noise = filter_data(X_aux, sfreq, **filt_params_noise)
    X_noise -= X_signal
    if X.ndim == 3:
        X_signal = np.hstack(X_signal)
        X_noise = np.hstack(X_noise)

    # prevent rank change when computing cov with rank='full'
    S = _regularized_covariance(
        X_signal,
        reg=reg,
        method_params=cov_method_params,
        rank="full",
        info=info,
    )
    R = _regularized_covariance(
        X_noise,
        reg=reg,
        method_params=cov_method_params,
        rank="full",
        info=info,
    )
    covs = [S, R]
    C_ref = S
    return covs, C_ref, info, rank, dict()


def _spoc_estimate(X, y, reg, cov_method_params, rank):
    # Normalize target variable
    target = y.astype(np.float64)
    target -= target.mean()
    target /= target.std()

    n_epochs, n_channels = X.shape[:2]

    # Estimate single trial covariance
    covs = np.empty((n_epochs, n_channels, n_channels))
    for ii, epoch in enumerate(X):
        covs[ii] = _regularized_covariance(
            epoch,
            reg=reg,
            method_params=cov_method_params,
            rank=rank,
            log_ch_type="data",
            log_rank=ii == 0,
        )

    S = np.mean(covs * target[:, np.newaxis, np.newaxis], axis=0)
    R = covs.mean(0)

    covs = [S, R]
    C_ref = None
    info = None
    return covs, C_ref, info, rank, dict()
