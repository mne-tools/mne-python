# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.pick import _picks_to_idx
from ..evoked import Evoked
from ..utils import _ensure_int, _validate_type, verbose


def _temp_proj(ref_2, ref_1, raw_data, n_proj=6):
    # Orthonormalize gradiometer and magnetometer data by a QR decomposition
    ref_1_orth = np.linalg.qr(ref_1.T)[0]
    ref_2_orth = np.linalg.qr(ref_2.T)[0]

    # Calculate cross-correlation
    cross_corr = np.dot(ref_1_orth.T, ref_2_orth)

    # Channel weights for common temporal subspace by SVD of cross-correlation
    ref_1_ch_weights, _, _ = np.linalg.svd(cross_corr)

    # Get temporal signals from channel weights
    proj_mat = ref_1_orth @ ref_1_ch_weights

    # Project out common subspace
    filtered_data = raw_data
    proj_vec = proj_mat[:, :n_proj]
    weights = filtered_data @ proj_vec
    filtered_data -= weights @ proj_vec.T


@verbose
def cortical_signal_suppression(
    evoked, picks=None, mag_picks=None, grad_picks=None, n_proj=6, *, verbose=None
):
    """Apply cortical signal suppression (CSS) to evoked data.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked object to use for CSS. Must contain magnetometer,
        gradiometer, and EEG channels.
    %(picks_good_data)s
    mag_picks : array-like of int
        Array of the first set of channel indices that will be used to find
        the common temporal subspace. If None (default), all magnetometers will
        be used.
    grad_picks : array-like of int
        Array of the second set of channel indices that will be used to find
        the common temporal subspace. If None (default), all gradiometers will
        be used.
    n_proj : int
        The number of projection vectors.
    %(verbose)s

    Returns
    -------
    evoked_subcortical : instance of Evoked
        The evoked object with contributions from the ``mag_picks`` and ``grad_picks``
        channels removed from the ``picks`` channels.

    Notes
    -----
    This method removes the common signal subspace between two sets of
    channels (``mag_picks`` and ``grad_picks``) from a set of channels
    (``picks``) via a temporal projection using ``n_proj`` number of
    projection vectors. In the reference publication :footcite:`Samuelsson2019`,
    the joint subspace between magnetometers and gradiometers is used to
    suppress the cortical signal in the EEG data. In principle, other
    combinations of sensor types (or channels) could be used to suppress
    signals from other sources.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(evoked, Evoked, "evoked")
    n_proj = _ensure_int(n_proj, "n_proj")
    picks = _picks_to_idx(evoked.info, picks, none="data", exclude="bads")
    mag_picks = _picks_to_idx(evoked.info, mag_picks, none="mag", exclude="bads")
    grad_picks = _picks_to_idx(evoked.info, grad_picks, none="grad", exclude="bads")
    evoked_subcortical = evoked.copy()

    # Get data
    all_data = evoked.data
    mag_data = all_data[mag_picks]
    grad_data = all_data[grad_picks]

    # Process data with temporal projection algorithm
    data = all_data[picks]
    _temp_proj(mag_data, grad_data, data, n_proj=n_proj)
    evoked_subcortical.data[picks, :] = data

    return evoked_subcortical
