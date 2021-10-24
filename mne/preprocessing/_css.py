# Author: John Samuelsson <johnsam@mit.edu>

import numpy as np

from ..evoked import Evoked
from ..io.pick import _picks_to_idx
from ..utils import verbose, _validate_type, _ensure_int


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
def cortical_signal_suppression(evoked, picks=None, mag_picks=None,
                                grad_picks=None, n_proj=6, *, verbose=None):
    """Apply cortical signal suppression (CSS) to evoked data.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked object to use for CSS. Must contain magnetometer,
        gradiometer, and EEG channels.
    %(picks_good_data)s
    mag_picks : array-like of int
        Array of the magnetometer channel indices that will be used to find
        the reference data. If None (default), all magnetometers will
        be used.
    grad_picks : array-like of int
        Array of the gradiometer channel indices that will be used to find
        the reference data. If None (default), all gradiometers will
        be used.
    n_proj : int
        The number of projection vectors.
    %(verbose)s

    Returns
    -------
    evoked_subcortical : instance of Evoked
        The evoked object with cortical contributions to the EEG data
        suppressed.

    Notes
    -----
    This method removes the common signal subspace between the magnetometer
    data and the gradiometer data from the EEG data. This is done by a temporal
    projection using ``n_proj`` number of projection vectors. For reference,
    see :footcite:`Samuelsson2019`.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(evoked, Evoked, 'evoked')
    n_proj = _ensure_int(n_proj, 'n_proj')
    picks = _picks_to_idx(
        evoked.info, picks, none='data', exclude='bads')
    mag_picks = _picks_to_idx(
        evoked.info, mag_picks, none='mag', exclude='bads')
    grad_picks = _picks_to_idx(
        evoked.info, grad_picks, none='grad', exclude='bads')
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
