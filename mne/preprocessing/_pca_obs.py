"""Principle Component Analysis Optimal Basis Sets (PCA-OBS)."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import math

import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
from scipy.signal import detrend

from ..io.fiff.raw import Raw
from ..utils import _PCA, _validate_type, logger, verbose


@verbose
def apply_pca_obs(
    raw: Raw,
    picks: list[str],
    *,
    qrs_times: np.ndarray,
    n_components: int = 4,
    n_jobs: int | None = None,
    copy: bool = True,
    verbose: bool | str | int | None = None,
) -> Raw:
    """
    Apply the PCA-OBS algorithm to picks of a Raw object.

    Uses the optimal basis set (OBS) algorithm from :footcite:`NiazyEtAl2005`.

    Parameters
    ----------
    raw : instance of Raw
        The raw data to process.
    %(picks_all_data_noref)s
    qrs_times : ndarray, shape (n_peaks,)
        Array of times in the Raw data of detected R-peaks in ECG channel.
    n_components : int
        Number of PCA components to use to form the OBS (default 4).
    %(n_jobs)s
    copy : bool
        If False, modify the Raw instance in-place.
        If True (default), copy the raw instance before processing.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.

    Notes
    -----
    .. versionadded:: 1.10

    References
    ----------
    .. footbibliography::
    """
    # sanity checks
    _validate_type(qrs_times, np.ndarray, "qrs_times")
    if len(qrs_times.shape) > 1:
        raise ValueError("qrs_times must be a 1d array")
    if qrs_times.dtype not in [int, float]:
        raise ValueError("qrs_times must be an array of either integers or floats")
    if np.any(qrs_times < 0):
        raise ValueError("qrs_times must be strictly positive")
    if np.any(qrs_times >= raw.times[-1]):
        logger.warning("some out of bound qrs_times will be ignored..")

    if copy:
        raw = raw.copy()

    raw.apply_function(
        _pca_obs,
        picks=picks,
        n_jobs=n_jobs,
        # args sent to PCA_OBS, convert times to indices
        qrs=raw.time_as_index(qrs_times),
        n_components=n_components,
    )

    return raw


def _pca_obs(
    data: np.ndarray,
    qrs: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Algorithm to remove heart artefact from EEG data (array of length n_times)."""
    # set to baseline
    data = data - np.mean(data)

    # Allocate memory for artifact which will be subtracted from the data
    fitted_art = np.zeros(data.shape)

    # Extract QRS event indexes which are within out data timeframe
    peak_idx = qrs[qrs < len(data)]
    peak_count = len(peak_idx)

    ##################################################################
    # Preparatory work - reserving memory, configure sizes, de-trend #
    ##################################################################
    # define peak range based on RR
    mRR = np.median(np.diff(peak_idx))
    peak_range = round(mRR / 2)  # Rounds to an integer
    mid_p = peak_range + 1
    n_samples_fit = round(
        peak_range / 8
    )  # sample fit for interpolation between fitted artifact windows

    # make sure array is long enough for PArange (if not cut off last ECG peak)
    # NOTE: Here we previously checked for the last part of the window to be big enough.
    while peak_idx[peak_count - 1] + peak_range > len(data):
        peak_count = peak_count - 1  # reduce number of QRS complexes detected

    # build PCA matrix(heart-beat-epochs x window-length)
    pcamat = np.zeros((peak_count - 1, 2 * peak_range + 1))  # [epoch x time]
    # picking out heartbeat epochs
    for p in range(1, peak_count):
        pcamat[p - 1, :] = data[peak_idx[p] - peak_range : peak_idx[p] + peak_range + 1]

    # detrending matrix(twice)
    pcamat = detrend(
        pcamat, type="constant", axis=1
    )  # [epoch x time] - detrended along the epoch
    mean_effect: np.ndarray = np.mean(
        pcamat, axis=0
    )  # [1 x time], contains the mean over all epochs
    dpcamat = detrend(pcamat, type="constant", axis=1)  # [time x epoch]

    ############################
    # Perform PCA with sklearn #
    ############################
    # run PCA, perform singular value decomposition (SVD)
    pca = _PCA()
    pca.fit(dpcamat)
    factor_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # define selected number of components using profile likelihood

    #####################################
    # Make template of the ECG artefact #
    #####################################
    mean_effect = mean_effect.reshape(-1, 1)
    pca_template = np.c_[mean_effect, factor_loadings[:, :n_components]]

    ################
    # Data Fitting #
    ################
    window_start_idx = []
    window_end_idx = []
    post_idx_next_peak = None

    for p in range(peak_count):
        # if the current peak doesn't have enough data in the
        # start of the peak_range, skip fitting the artifact
        if peak_idx[p] - peak_range < 0:
            continue

        # Deals with start portion of data
        if p == 0:
            pre_range = peak_range
            post_range = math.floor((peak_idx[p + 1] - peak_idx[p]) / 2)
            if post_range > peak_range:
                post_range = peak_range

            fitted_art, post_idx_next_peak = _fit_ecg_template(
                data=data,
                pca_template=pca_template,
                a_peak_idx=peak_idx[p],
                peak_range=peak_range,
                pre_range=pre_range,
                post_range=post_range,
                mid_p=mid_p,
                fitted_art=fitted_art,
                post_idx_previous_peak=post_idx_next_peak,
                n_samples_fit=n_samples_fit,
            )
            # Appending to list instead of using counter
            window_start_idx.append(peak_idx[p] - peak_range)
            window_end_idx.append(peak_idx[p] + peak_range)

        # Deals with last edge of data
        elif p == peak_count - 1:
            pre_range = math.floor((peak_idx[p] - peak_idx[p - 1]) / 2)
            post_range = peak_range
            if pre_range > peak_range:
                pre_range = peak_range
            fitted_art, _ = _fit_ecg_template(
                data=data,
                pca_template=pca_template,
                a_peak_idx=peak_idx[p],
                peak_range=peak_range,
                pre_range=pre_range,
                post_range=post_range,
                mid_p=mid_p,
                fitted_art=fitted_art,
                post_idx_previous_peak=post_idx_next_peak,
                n_samples_fit=n_samples_fit,
            )
            window_start_idx.append(peak_idx[p] - peak_range)
            window_end_idx.append(peak_idx[p] + peak_range)

        # Deals with middle portion of data
        else:
            # ---------------- Processing of central data - --------------------
            # cycle through peak artifacts identified by peakplot
            pre_range = math.floor((peak_idx[p] - peak_idx[p - 1]) / 2)
            post_range = math.floor((peak_idx[p + 1] - peak_idx[p]) / 2)
            if pre_range >= peak_range:
                pre_range = peak_range
            if post_range > peak_range:
                post_range = peak_range

            a_template = pca_template[
                mid_p - peak_range - 1 : mid_p + peak_range + 1, :
            ]
            fitted_art, post_idx_next_peak = _fit_ecg_template(
                data=data,
                pca_template=a_template,
                a_peak_idx=peak_idx[p],
                peak_range=peak_range,
                pre_range=pre_range,
                post_range=post_range,
                mid_p=mid_p,
                fitted_art=fitted_art,
                post_idx_previous_peak=post_idx_next_peak,
                n_samples_fit=n_samples_fit,
            )
            window_start_idx.append(peak_idx[p] - peak_range)
            window_end_idx.append(peak_idx[p] + peak_range)

    # Actually subtract the artefact, return needs to be the same shape as input data
    data -= fitted_art
    return data


def _fit_ecg_template(
    data: np.ndarray,
    pca_template: np.ndarray,
    a_peak_idx: int,
    peak_range: int,
    pre_range: int,
    post_range: int,
    mid_p: float,
    fitted_art: np.ndarray,
    post_idx_previous_peak: int | None,
    n_samples_fit: int,
) -> tuple[np.ndarray, int]:
    """
    Fits the heartbeat artefact found in the data.

    Returns the fitted artefact and the index of the next peak.

    Parameters
    ----------
        data (ndarray): Data from the raw signal (n_channels, n_times)
        pca_template (ndarray): Mean heartbeat and first N (default 4)
            principal components of the heartbeat matrix
        a_peak_idx (int): Sample index of current R-peak
        peak_range (int): Half the median RR-interval
        pre_range (int): Number of samples to fit before the R-peak
        post_range (int): Number of samples to fit after the R-peak
        mid_p (float): Sample index marking middle of the median RR interval
            in the signal. Used to extract relevant part of PCA_template.
        fitted_art (ndarray): The computed heartbeat artefact computed to
            remove from the data
        post_idx_previous_peak (optional int): Sample index of previous R-peak
        n_samples_fit (int): Sample fit for interpolation in fitted artifact
            windows. Helps reduce sharp edges at end of fitted heartbeat events

    Returns
    -------
        tuple[np.ndarray, int]: the fitted artifact and the next peak index
    """
    # post_idx_next_peak is passed in in PCA_OBS, used here as post_idx_previous_peak
    # Then next_peak is returned at the end and the process repeats
    # select window of template
    template = pca_template[mid_p - peak_range - 1 : mid_p + peak_range + 1, :]

    # select window of data and detrend it
    slice_ = data[a_peak_idx - peak_range : a_peak_idx + peak_range + 1]

    detrended_data = detrend(slice_, type="constant")

    # maps data on template and then maps it again back to the sensor space
    least_square = np.linalg.lstsq(template, detrended_data, rcond=None)
    pad_fit = np.dot(template, least_square[0])

    # fit artifact
    fitted_art[a_peak_idx - pre_range - 1 : a_peak_idx + post_range] = pad_fit[
        mid_p - pre_range - 1 : mid_p + post_range
    ].T

    # if last peak, return
    if post_idx_previous_peak is None:
        return fitted_art, a_peak_idx + post_range

    # interpolate time between peaks
    intpol_window = np.ceil([post_idx_previous_peak, a_peak_idx - pre_range]).astype(
        int
    )  # interpolation window

    if intpol_window[0] < intpol_window[1]:
        # Piecewise Cubic Hermite Interpolating Polynomial(PCHIP) + replace EEG data

        # You have x_fit which is two slices on either side of the interpolation window
        #   endpoints
        # You have y_fit which is the y vals corresponding to x values above
        # You have x_interpol which is the time points between the two slices in x_fit
        #   that you want to interpolate
        # You have y_interpol which is values from pchip at the time points specified in
        #   x_interpol
        # points to be interpolated in pt - the gap between the endpoints of the window
        x_interpol = np.arange(intpol_window[0], intpol_window[1] + 1, 1)
        # Entire range of x values in this step (taking some
        # number of samples before and after the window)
        x_fit = np.concatenate(
            [
                np.arange(intpol_window[0] - n_samples_fit, intpol_window[0] + 1, 1),
                np.arange(intpol_window[1], intpol_window[1] + n_samples_fit + 1, 1),
            ]
        )
        y_fit = fitted_art[x_fit]
        y_interpol = pchip(x_fit, y_fit)(x_interpol)  # perform interpolation

        # make fitted artefact in the desired range equal to the completed fit above
        fitted_art[post_idx_previous_peak : a_peak_idx - pre_range + 1] = y_interpol

    return fitted_art, a_peak_idx + post_range
