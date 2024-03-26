# Authors: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from mne._fiff.pick import _picks_to_idx
from mne.parallel import parallel_func

from ..._fiff.constants import FIFF
from ...io import BaseRaw
from ...utils import (
    _check_option,
    _check_preload,
    _validate_type,
    fill_doc,
    logger,
    warn,
)


def interpolate_blinks(raw, buffer=0.05, match="BAD_blink", interpolate_gaze=False):
    """Interpolate eyetracking signals during blinks.

    This function uses the timing of blink annotations to estimate missing
    data. Operates in place.

    Parameters
    ----------
    raw : instance of Raw
        The raw data with at least one ``'pupil'`` or ``'eyegaze'`` channel.
    buffer : float | array-like of float, shape ``(2,))``
        The time in seconds before and after a blink to consider invalid and
        include in the segment to be interpolated over. Default is ``0.05`` seconds
        (50 ms). If array-like, the first element is the time before the blink and the
        second element is the time after the blink to consider invalid, for example,
        ``(0.025, .1)``.
    match : str | list of str
        The description of annotations to interpolate over. If a list, the data within
        all annotations that match any of the strings in the list will be interpolated
        over. Defaults to ``'BAD_blink'``.
    interpolate_gaze : bool
        If False, only apply interpolation to ``'pupil channels'``. If True, interpolate
        over ``'eyegaze'`` channels as well. Defaults to False, because eye position can
        change in unpredictable ways during blinks.

    Returns
    -------
    self : instance of Raw
        Returns the modified instance.

    Notes
    -----
    .. versionadded:: 1.5
    """
    _check_preload(raw, "interpolate_blinks")
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(buffer, (float, tuple, list, np.ndarray), "buffer")
    _validate_type(match, (str, tuple, list, np.ndarray), "match")

    # determine the buffer around blinks to include in the interpolation
    buffer = np.array(buffer, dtype=float)
    if buffer.size == 1:
        buffer = np.array([buffer, buffer])

    if isinstance(match, str):
        match = [match]

    # get the blink annotations
    blink_annots = [annot for annot in raw.annotations if annot["description"] in match]
    if not blink_annots:
        warn(f"No annotations matching {match} found. Aborting.")
        return raw
    _interpolate_blinks(raw, buffer, blink_annots, interpolate_gaze=interpolate_gaze)

    # remove bad from the annotation description
    for desc in match:
        if desc.startswith("BAD_"):
            logger.info(f"Removing 'BAD_' from {desc}.")
            raw.annotations.rename({desc: desc.replace("BAD_", "")})
    return raw


def _interpolate_blinks(raw, buffer, blink_annots, interpolate_gaze):
    """Interpolate eyetracking signals during blinks in-place."""
    logger.info("Interpolating missing data during blinks...")
    pre_buffer, post_buffer = buffer
    # iterate over each eyetrack channel and interpolate the blinks
    interpolated_chs = []
    for ci, ch_info in enumerate(raw.info["chs"]):
        if interpolate_gaze:  # interpolate over all eyetrack channels
            if ch_info["kind"] != FIFF.FIFFV_EYETRACK_CH:
                continue
        else:  # interpolate over pupil channels only
            if ch_info["coil_type"] != FIFF.FIFFV_COIL_EYETRACK_PUPIL:
                continue
        # Create an empty boolean mask
        mask = np.zeros_like(raw.times, dtype=bool)
        for annot in blink_annots:
            if "ch_names" not in annot or not annot["ch_names"]:
                msg = f"Blink annotation missing values for 'ch_names' key: {annot}"
                raise ValueError(msg)
            start = annot["onset"] - pre_buffer
            end = annot["onset"] + annot["duration"] + post_buffer
            if ch_info["ch_name"] not in annot["ch_names"]:
                continue  # skip if the channel is not in the blink annotation
            # Update the mask for times within the current blink period
            mask |= (raw.times >= start) & (raw.times <= end)
        blink_indices = np.where(mask)[0]
        non_blink_indices = np.where(~mask)[0]

        # Linear interpolation
        interpolated_samples = np.interp(
            raw.times[blink_indices],
            raw.times[non_blink_indices],
            raw._data[ci, non_blink_indices],
        )
        # Replace the samples at the blink_indices with the interpolated values
        raw._data[ci, blink_indices] = interpolated_samples
        interpolated_chs.append(ch_info["ch_name"])
    if interpolated_chs:
        logger.info(
            f"Interpolated {len(interpolated_chs)} channels: {interpolated_chs}"
        )
    else:
        warn("No channels were interpolated.")


@fill_doc
def pupil_zscores(epochs, baseline=(None, 0)):
    """Get normalized pupil data.

    This function normalizes pupil responses within each epoch by subtracting
    the mean pupil response during a specified baseline period and then dividing
    by the standard deviation of all data (across time). This may help to compare
    pupil responses across epochs or participants.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs with pupil channels.
    %(pupil_baseline)s

    Returns
    -------
    pupil_data : array
        An array of pupil size data, shape (n_epochs, n_channels, n_times).
    """
    from mne import BaseEpochs

    # Code ported from https://github.com/pyeparse/pyeparse
    _check_preload(epochs, "Z-score normalization")
    _validate_type(epochs, BaseEpochs, "epochs")
    _validate_type(baseline, (tuple, list, np.ndarray), "baseline")

    pupil_picks = _picks_to_idx(epochs.info, "pupil", allow_empty=False)
    if len(baseline) != 2:
        raise RuntimeError("baseline must be a 2-element list")
    baseline = np.array(baseline)
    if baseline[0] is None:
        baseline[0] = epochs.times[0]
    if baseline[1] is None:
        baseline[1] = epochs.times[-1]
    baseline = epochs.time_as_index(baseline)
    zs = epochs.get_data(pupil_picks)
    std = np.nanstd(zs.flat)
    bl = np.nanmean(zs[..., baseline[0] : baseline[1] + 1], axis=-1)
    zs -= bl[:, np.newaxis, :]
    zs /= std
    return zs


@fill_doc
def deconvolve(
    epochs,
    spacing=0.1,
    baseline=(None, 0),
    bounds=None,
    max_iter=500,
    kernel=None,
    n_jobs=1,
    acc=1e-6,
    method="minimize",
    reg=100,
):
    r"""Deconvolve pupillary responses.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs with pupil data to deconvolve.
    spacing : float | array
        Spacing of time points to use for deconvolution. Can also
        be an array to directly specify time points to use.
    %(pupil_baseline)s
        This is passed to :func:`~mne.preprocessing.eyetracking.pupil_zscores`.
    bounds : array of shape (2,) | None
        Limits for deconvolution values. Can be, e.g. ``(0, np.inf)`` to
        constrain to positive values. If ``None``, no bounds are used. Default is
        ``None``.
    max_iter : int
        Maximum number of iterations of minimization algorithm. Default is ``500``.
    kernel : array | None
        Kernel to assume when doing deconvolution. If ``None``, the
        Hoeks and Levelt (1993)\ :footcite:p:`Hoeks1993` kernel will be used.
    %(n_jobs)s
    acc : float
        The requested accuracy. Lower accuracy generally means smoother
        fits.
    method : ``"minimize"`` | ``"inverse"``
        Can be ``"minimize"`` to use SLSQP or ``"inverse"`` to use
        Tikhonov-regularized pseudoinverse. Default is ``"minimize"``.
    reg : float
        Regularization factor for pseudoinverse calculation. Only used if method is
        ``"inverse"``. Default is 100.

    Returns
    -------
    fit : array,  shape (n_epochs, n_channels, n_fit_times)
        Array of fits.
    times : array, shape (n_fit_times,)
        The array of times at which points were fit.

    Notes
    -----
    This method is adapted from Wierda et al., 2012, "Pupil dilation
    deconvolution reveals the dynamics of attention at high temporal
    resolution."\ :footcite:p:`Wierda2012`

    Our implementation does not, by default, force all weights to be
    greater than zero. It also does not do first-order detrending,
    which the Wierda paper discusses implementing.

    References
    ----------
    .. footbibliography::
    """
    from scipy import linalg

    # Code ported from https://github.com/pyeparse/pyeparse
    _validate_type(spacing, (float, np.ndarray, tuple, list), "spacing")
    _validate_type(bounds, (type(None), tuple, list, np.ndarray), "bounds")
    _validate_type(max_iter, int, "max_iter")
    _validate_type(kernel, (np.ndarray, type(None)), "kernel")
    _validate_type(n_jobs, int, "n_jobs")
    _validate_type(acc, float, "acc")
    _validate_type(method, str, "method")
    _check_option("method", method, ["minimize", "inverse"])
    _validate_type(reg, (int, float), "reg")

    if bounds is not None:
        bounds = np.array(bounds)
        if bounds.ndim != 1 or bounds.size != 2:
            raise RuntimeError("bounds must be 2-element array or None")
    if kernel is None:
        kernel = pupil_kernel(epochs.info["sfreq"])
    else:
        kernel = np.array(kernel, np.float64)
        if kernel.ndim != 1:
            raise TypeError("kernel must be 1D")

    # get the data (and make sure it exists)
    pupil_data = pupil_zscores(epochs, baseline=baseline)

    # set up parallel function (and check n_jobs)
    parallel, p_fun, n_jobs = parallel_func(_do_deconv, n_jobs)

    # figure out where the samples go
    n_samp = len(epochs.times)
    if not isinstance(spacing, (np.ndarray, tuple, list)):
        times = np.arange(epochs.times[0], epochs.times[-1], spacing)
        times = np.unique(times)
    else:
        times = np.asanyarray(spacing)
    samples = epochs.time_as_index(times)
    if len(samples) == 0:
        warn("No usable samples")
        return np.array([]), np.array([])

    # convert bounds to slsqp representation
    if bounds is not None:
        bounds = np.array([bounds for _ in range(len(samples))])
    else:
        bounds = []  # compatible with old version of scipy

    # Build the convolution matrix
    conv_mat = np.zeros((n_samp, len(samples)))
    for li, loc in enumerate(samples):
        eidx = min(loc + len(kernel), n_samp)
        conv_mat[loc:eidx, li] = kernel[: eidx - loc]

    # do the fitting
    if method == "inverse":
        u, s, v = linalg.svd(conv_mat, full_matrices=False)
        # Threshold small singular values
        s[s < 1e-7 * s[0]] = 0
        # Regularize non-zero singular values
        s[s > 0] /= s[s > 0] ** 2 + reg
        inv_conv_mat = np.dot(v.T, s[:, np.newaxis] * u.T)
        fit = np.dot(pupil_data, inv_conv_mat.T)
    else:  # minimize
        fit_fails = parallel(
            p_fun(data, conv_mat, bounds, max_iter, acc)
            for data in np.array_split(pupil_data, n_jobs)
        )
        fit = np.concatenate([f[0] for f in fit_fails])
        fails = np.concatenate([f[1] for f in fit_fails])
        if np.any(fails):
            reasons = ", ".join(str(r) for r in np.setdiff1d(np.unique(fails), [0]))
            warn(
                f"{np.sum(fails != 0)} out of {len(fails)} fits "
                f"did not converge (reasons: {reasons})"
            )
    return fit, times


def _do_deconv(pupil_data, conv_mat, bounds, max_iter, acc):
    """Parallelize deconvolution helper function."""
    # Code ported from https://github.com/pyeparse/pyeparse
    from scipy.optimize import fmin_slsqp

    x0 = np.zeros(conv_mat.shape[1])
    fit = np.empty((pupil_data.shape[0], pupil_data.shape[1], conv_mat.shape[1]))
    failed = np.empty(fit.shape)
    for ei, data in enumerate(pupil_data):
        out = fmin_slsqp(
            _score,
            x0,
            args=(data, conv_mat),
            epsilon=1e-4,
            bounds=bounds,
            disp=False,
            full_output=True,
            iter=max_iter,
            acc=acc,
        )
        fit[ei, :, :] = out[0]
        failed[ei, :, :] = out[3]
    return fit, failed


def _score(vals, x_0, conv_mat):
    return np.mean((x_0 - conv_mat.dot(vals)) ** 2)


def pupil_kernel(sfreq, dur=4.0, t_max=0.930, n=10.1, s=1.0):
    r"""Generate pupil response kernel modeled as an Erlang gamma function.

    Parameters
    ----------
    sfreq : int
        Sampling frequency (samples/second) to use in generating the kernel.
    dur : float
        Length (in seconds) of the generated kernel. Default is ``4.0`` seconds.
    t_max : float
        Time (in seconds) where the response maximum is stipulated to occur. Default is
        ``0.930`` seconds, as in Hoeks and Levelt (1993)\ :footcite:p:`Hoeks1993`.
    n : float
        Number of negative-exponential layers in the cascade defining the
        gamma function. Default is ``10.1``, as in Hoeks and Levelt (1993)\
        :footcite:p:`Hoeks1993`.
    s : float | None
        Desired value for the area under the kernel. If ``None``, no scaling is
        performed. Default is ``1.0``.

    Returns
    -------
    h : array
        The generated kernel.

    References
    ----------
    .. footbibliography::
    """
    # Code ported from https://github.com/pyeparse/pyeparse
    n_samp = int(np.round(sfreq * dur))
    t = np.arange(n_samp, dtype=float) / sfreq
    h = (t**n) * np.exp(-n * t / t_max)
    scal = 1.0 if s is None else float(s) / (np.sum(h) * (t[1] - t[0]))
    h = scal * h
    return h
