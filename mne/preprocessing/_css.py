# Author: John Samuelsson <johnsam@mit.edu>

import numpy as np
from ..utils import logger
from ..evoked import EvokedArray, Evoked
from .. import pick_types
from numpy import linalg


def _temp_proj(ref_2, ref_1, raw_data, n_proj=6):
    """Remove common signal subspace of ref_2 and ref_1 from raw_data.

    Parameters
    ----------
    ref_2 : np.ndarray of float, shape (n_sensors_2, n_times)
        The magnetometer data for CSS. Can use either all magnetometer data or
        a few selected sensors close to a region to be suppressed.
    ref_1 : np.ndarray of float, shape (n_sensors_1, n_times)
        The gradiometer data for CSS. Can use either all gradiometer data or
        a few selected sensors close to a region to be suppressed.
    raw_data : np.ndarray of float, shape (n_sensors_raw, n_times)
        The data to be filtered, typically the EEG data.
    n_proj : int
        The number of projection vectors.

    Returns
    -------
    filtered_data : np.ndarray of float, shape (n_sensors_raw, n_times)
        The filtered data.

    Notes
    -----
    This temporal projection procedure removes the common signal subspace
    between ref_2 and ref_1 from raw_data using n_proj number of
    projection vectors. Normally used for cortical signal suppression, where
    ref_1 is gradiometer data, ref_2 is magnetometer data and
    raw_data is EEG data.

    """
    # Orthonormalize gradiometer and magnetometer data by a QR decomposition
    ref_1_orth = linalg.qr(ref_1.T)[0]
    ref_2_orth = linalg.qr(ref_2.T)[0]

    # Calculate cross-correlation
    cross_corr = np.dot(ref_1_orth.T, ref_2_orth)

    # Channel weights for common temporal subspace by SVD of cross-correlation
    ref_1_ch_weights, sing_vals, ref_2_ch_weights = linalg.svd(cross_corr)

    # Get temporal signals from channel weights
    proj_mat = np.dot(ref_1_orth, ref_1_ch_weights)

    # Project out common subspace
    filtered_data = raw_data
    for i in range(n_proj):
        proj_vec = proj_mat[:, i].reshape(proj_mat.shape[0], 1)
        weights = np.dot(filtered_data, proj_vec)
        filtered_data = filtered_data - np.dot(weights, proj_vec.T)

    return filtered_data


def cortical_signal_suppression(evoked, mag_inds=None, grad_inds=None,
                                n_proj=6):
    """Remove the cortical signals in EEG by cortical signal suppression (CSS).

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked object to use for CSS. Must contain magnetometer,
        gradiometer, and EEG channels.
    mag_inds : np.ndarray of int
        Array of the magnetometer channel indicies that will be used to find
        the reference data. If None (default), all channels will be used.
    grad_inds : np.ndarray of int
        Array of the gradiometer channel indicies that will be used to find
        the reference data. If None (default), all channels will be used.
    n_proj : int
        The number of projection vectors.

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
    if not isinstance(evoked, (EvokedArray, Evoked)):
        raise ValueError('evoked needs to an instance of Evoked, however type \
                         {} was passed'.format(type(evoked)))
    if mag_inds is None:
        mag_inds = pick_types(evoked.info, meg='mag')
    if grad_inds is None:
        grad_inds = pick_types(evoked.info, meg='grad')
    evoked_subcortical = evoked.copy()

    # Load data if not preloaded
    if not evoked.preload:
        logger.info('Data not preloaded. Loading data now...')
        evoked.load_data()

    # Get data
    eeg_inds = pick_types(evoked.info, meg=False, eeg=True)
    all_data = evoked.data
    mag_data = all_data[mag_inds]
    grad_data = all_data[grad_inds]

    # Process data with temporal projection algorithm
    eeg_data = all_data[eeg_inds]
    eeg_subcortical = _temp_proj(mag_data, grad_data, eeg_data, n_proj=n_proj)
    evoked_subcortical.data[eeg_inds, :] = eeg_subcortical

    return evoked_subcortical
