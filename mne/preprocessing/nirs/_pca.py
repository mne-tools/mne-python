# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# The core logic for this implementation was adapted from the Cedalion project
# (https://github.com/ibs-lab/cedalion), which is originally based on Homer3
# (https://github.com/BUNPC/Homer3).

import numpy as np
from scipy.linalg import svd

from ...io import BaseRaw
from ...utils import _validate_type, verbose
from ..nirs import _validate_nirs_info


@verbose
def motion_correct_pca(raw, tInc, nSV=0.97, *, verbose=None):
    """Apply PCA-based motion correction to fNIRS data.

    Extracts the motion-artifact portions of the signal, performs a singular
    value decomposition across all fNIRS channels, removes the dominant
    principal components, and reinserts the cleaned segments.

    Based on Homer3 v1.80.2 ``hmrR_MotionCorrectPCA.m``
    :footcite:`HuppertEtAl2009`.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    tInc : array-like of bool, shape (n_times,)
        Global motion-artifact mask.  ``True`` = clean sample,
        ``False`` = motion artifact.
    nSV : float | int
        Number of principal components to remove.

        * ``0 < nSV < 1`` – remove the fewest components whose cumulative
          explained variance reaches ``nSV`` (e.g. ``0.97`` removes
          components explaining the top 97 %% of variance).
        * ``nSV >= 1`` – remove exactly ``int(nSV)`` components.

        Default is ``0.97``.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        Data with PCA motion correction applied (copy).
    svs : ndarray
        Normalised singular values (sum = 1) of the PCA decomposition.
    nSV : int
        Actual number of principal components removed.

    Notes
    -----
    There is a shorter alias ``mne.preprocessing.nirs.pca`` that
    can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(raw, BaseRaw, "raw")
    raw = raw.copy().load_data()
    picks = _validate_nirs_info(raw.info)

    if not len(picks):
        raise RuntimeError(
            "PCA motion correction should be run on optical density or hemoglobin data."
        )

    tInc = np.asarray(tInc, dtype=bool)
    n_times = raw._data.shape[1]

    # (n_picks, n_times)
    y = raw._data[picks, :]

    # Motion-artifact samples only, shape (n_motion_samples, n_picks)
    y_motion = y[:, ~tInc].T

    if y_motion.shape[0] == 0:
        raise ValueError(
            "No motion-artifact samples found (tInc is all True). "
            "Mark artifact regions with False before calling this function."
        )

    # Z-score across time within the motion segments
    y_mean = y_motion.mean(axis=0)
    y_std = y_motion.std(axis=0)
    y_std[y_std == 0] = 1.0  # avoid divide-by-zero for flat channels
    y_zscore = (y_motion - y_mean) / y_std

    # PCA via SVD of the covariance-like matrix
    yo = y_zscore.copy()
    c = np.dot(y_zscore.T, y_zscore)
    V, St, _ = svd(c)

    svs = St / np.sum(St)

    # Cumulative variance for fractional nSV selection
    svsc = svs.copy()
    for i in range(1, len(svs)):
        svsc[i] = svsc[i - 1] + svs[i]

    if 0 < nSV < 1:
        mask_keep = svsc < nSV
        nSV = int(np.where(~mask_keep)[0][0])
    else:
        nSV = int(nSV)

    ev = np.zeros((len(svs), 1))
    ev[:nSV] = 1
    ev = np.diag(np.squeeze(ev))

    # Remove top PCs from motion-artifact segments
    yc = yo - np.dot(np.dot(yo, V), np.dot(ev, V.T))
    yc = (yc * y_std) + y_mean  # back to original scale

    # Identify motion-segment boundaries (1=good → 0=bad transitions)
    lst_ms = np.where(np.diff(tInc.astype(int)) == -1)[0]  # motion starts
    lst_mf = np.where(np.diff(tInc.astype(int)) == 1)[0]  # motion ends

    if len(lst_ms) == 0:
        lst_ms = np.asarray([0])
    if len(lst_mf) == 0:
        lst_mf = np.asarray([n_times - 1])
    if lst_ms[0] > lst_mf[0]:
        lst_ms = np.insert(lst_ms, 0, 0)
    if lst_ms[-1] > lst_mf[-1]:
        lst_mf = np.append(lst_mf, n_times - 1)

    # Cumulative lengths used to index into yc
    lst_mb = lst_mf - lst_ms
    for ii in range(1, len(lst_mb)):
        lst_mb[ii] = lst_mb[ii - 1] + lst_mb[ii]
    lst_mb = lst_mb - 1

    n_picks = len(picks)
    cleaned_ts = y.T.copy()  # (n_times, n_picks)
    orig_ts = y.T.copy()

    for jj in range(n_picks):
        # First motion segment
        lst = np.arange(lst_ms[0], lst_mf[0])
        if lst_ms[0] > 0:
            cleaned_ts[lst, jj] = (
                yc[: lst_mb[0] + 1, jj] - yc[0, jj] + cleaned_ts[lst[0], jj]
            )
        else:
            cleaned_ts[lst, jj] = (
                yc[: lst_mb[0] + 1, jj] - yc[lst_mb[0], jj] + cleaned_ts[lst[-1], jj]
            )

        # Intermediate non-motion and motion segments
        for kk in range(len(lst_mf) - 1):
            # Non-motion gap between MA[kk] and MA[kk+1]
            lst = np.arange(lst_mf[kk] - 1, lst_ms[kk + 1] + 1)
            cleaned_ts[lst, jj] = (
                orig_ts[lst, jj] - orig_ts[lst[0], jj] + cleaned_ts[lst[0], jj]
            )

            # Next motion segment
            lst = np.arange(lst_ms[kk + 1], lst_mf[kk + 1])
            cleaned_ts[lst, jj] = (
                yc[lst_mb[kk] + 1 : lst_mb[kk + 1] + 1, jj]
                - yc[lst_mb[kk] + 1, jj]
                + cleaned_ts[lst[0], jj]
            )

        # Trailing non-motion segment
        if lst_mf[-1] < n_times - 1:
            lst = np.arange(lst_mf[-1] - 1, n_times)
            cleaned_ts[lst, jj] = (
                orig_ts[lst, jj] - orig_ts[lst[0], jj] + cleaned_ts[lst[0], jj]
            )

    raw._data[picks, :] = cleaned_ts.T
    return raw, svs, nSV


# provide a short alias
pca = motion_correct_pca
