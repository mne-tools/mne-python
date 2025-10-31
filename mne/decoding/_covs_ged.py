"""Covariance estimation for GED transformers."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.meas_info import Info, create_info
from .._fiff.pick import _picks_to_idx, pick_info
from ..cov import Covariance, _compute_rank_raw_array, _regularized_covariance
from ..defaults import _handle_default
from ..filter import filter_data
from ..rank import compute_rank
from ..utils import _verbose_safe_false, logger


def _concat_cov(x_class, *, cov_kind, log_rank, reg, cov_method_params, info, rank):
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

    return cov, n_channels  # the weight here is just the number of channels


def _epoch_cov(x_class, *, cov_kind, log_rank, reg, cov_method_params, info, rank):
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


def _handle_info_rank(X, info, rank):
    if info is None:
        # use mag instead of eeg to avoid the cov EEG projection warning
        info = create_info(X.shape[1], 1000.0, "mag")
        if isinstance(rank, dict):
            rank = dict(mag=sum(rank.values()))

    return info, rank


def _csp_estimate(X, y, reg, cov_method_params, cov_est, info, rank, norm_trace):
    _, n_channels, _ = X.shape
    classes_ = np.unique(y)
    if cov_est == "concat":
        cov_estimator = _concat_cov
    elif cov_est == "epoch":
        cov_estimator = _epoch_cov

    info, rank = _handle_info_rank(X, info, rank)
    if not isinstance(rank, dict):
        rank = _compute_rank_raw_array(
            np.hstack(X),
            info,
            rank=rank,
            scalings=None,
            log_ch_type="data",
            on_few_samples="ignore",
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
            info=info,
            rank=rank,
        )

        if norm_trace:
            cov /= np.trace(cov)

        covs.append(cov)
        sample_weights.append(weight)

    covs = np.stack(covs)
    C_ref = covs.mean(0)

    return covs, C_ref, info, rank, dict(sample_weights=np.array(sample_weights))


def _xdawn_estimate(
    X,
    y,
    reg,
    cov_method_params,
    R=None,
    info=None,
    rank="full",
):
    classes = np.unique(y)
    info, rank = _handle_info_rank(X, info, rank)

    # Retrieve or compute whitening covariance
    if R is None:
        R = _regularized_covariance(
            np.hstack(X), reg, cov_method_params, info, rank=rank
        )
    elif isinstance(R, Covariance):
        R = R.data

    # Get prototype events
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
    C_ref = R
    if not isinstance(rank, dict):
        rank = _compute_rank_raw_array(
            np.hstack(X),
            info,
            rank=rank,
            scalings=None,
            log_ch_type="data",
            on_few_samples="ignore",
        )
    return covs, C_ref, info, rank, dict()


def _ssd_estimate(
    X,
    y,
    reg,
    cov_method_params,
    info,
    picks,
    n_fft,
    filt_params_signal,
    filt_params_noise,
    rank,
    sort_by_spectral_ratio,
):
    if isinstance(info, Info):
        sfreq = info["sfreq"]
    elif isinstance(info, float):  # special case, mostly for testing
        sfreq = info
        info = create_info(X.shape[-2], sfreq, ch_types="eeg")
    picks_ = _picks_to_idx(info, picks, none="data", exclude="bads")
    X_aux = X[..., picks_, :]
    X_signal = filter_data(X_aux, sfreq, **filt_params_signal)
    X_noise = filter_data(X_aux, sfreq, **filt_params_noise)
    X_noise -= X_signal
    if X.ndim == 3:
        X_signal = np.hstack(X_signal)
        X_noise = np.hstack(X_noise)

    # prevent rank change when computing cov with rank='full'
    picked_info = pick_info(info, picks_)
    S = _regularized_covariance(
        X_signal,
        reg=reg,
        method_params=cov_method_params,
        rank="full",
        info=picked_info,
    )
    R = _regularized_covariance(
        X_noise,
        reg=reg,
        method_params=cov_method_params,
        rank="full",
        info=picked_info,
    )
    covs = [S, R]
    C_ref = S

    all_ranks = list()
    for cov in covs:
        r = list(
            compute_rank(
                Covariance(
                    cov,
                    picked_info.ch_names,
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
    all_ranks.append(r)
    rank = np.min(all_ranks)
    freqs_signal = (filt_params_signal["l_freq"], filt_params_signal["h_freq"])
    freqs_noise = (filt_params_noise["l_freq"], filt_params_noise["h_freq"])
    n_fft = min(
        int(n_fft if n_fft is not None else sfreq),
        X.shape[-1],
    )
    kwargs = dict(
        X=X,
        picks=picks_,
        sfreq=sfreq,
        n_fft=n_fft,
        freqs_signal=freqs_signal,
        freqs_noise=freqs_noise,
        sort_by_spectral_ratio=sort_by_spectral_ratio,
    )
    rank = dict(eeg=rank)
    info = picked_info
    return covs, C_ref, info, rank, kwargs


def _spoc_estimate(X, y, reg, cov_method_params, info, rank):
    info, rank = _handle_info_rank(X, info, rank)
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
    C_ref = R
    if not isinstance(rank, dict):
        rank = _compute_rank_raw_array(
            np.hstack(X),
            info,
            rank=rank,
            scalings=None,
            log_ch_type="data",
            on_few_samples="ignore",
        )
    return covs, C_ref, info, rank, dict()
